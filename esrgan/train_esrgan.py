# esrgan/train_esrgan.py
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
import torch.nn.functional as F
import torchvision.utils
from pathlib import Path
from tqdm import tqdm
import wandb
import argparse
import os
import sys

# --- Path logic and imports ---
SCRIPT_DIR = Path(__file__).parent.absolute()
sys.path.append(str(SCRIPT_DIR.parent))

from esrgan.generator import GeneratorESRGAN
from esrgan.discriminator import Discriminator
from utils.datasets import TrainDataset, ValDataset
from utils.loss import VGGLoss
from config import config

# --- Checkpoint directory ---
CHECKPOINT_DIR = SCRIPT_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# @torch.no_grad()
# def validate_one_epoch(gen, loader):
#     """Calculates the average validation PSNR for the generator."""
#     gen.eval()
#     total_psnr = 0.0
#     with autocast(device_type=config.DEVICE, dtype=torch.float16, enabled=(config.DEVICE == 'cuda')):
#         for lr, hr in tqdm(loader, desc="Validating", leave=False):
#             lr, hr = lr.to(config.DEVICE), hr.to(config.DEVICE)
#             fake_hr = gen(lr)
#             mse = F.mse_loss(fake_hr, hr)
#             psnr = 10 * torch.log10(1 / mse)
#             total_psnr += psnr.item()
#     gen.train()
#     return total_psnr / len(loader)
@torch.no_grad()
def validate_one_epoch(gen, loader):
    """
    Calculates and compares the average validation PSNR for the generator's SR output
    and a standard bicubic upscale.
    """
    gen.eval()
    total_sr_psnr = 0.0
    total_bicubic_psnr = 0.0

    with autocast(device_type=config.DEVICE, dtype=torch.float16, enabled=(config.DEVICE == 'cuda')):
        for lr, hr in tqdm(loader, desc="Validating", leave=False):
            lr, hr = lr.to(config.DEVICE), hr.to(config.DEVICE)

            # --- 1. Super-Resolution (Generator's Output) ---
            fake_hr = gen(lr)
            sr_mse = F.mse_loss(fake_hr, hr)
            sr_psnr = 10 * torch.log10(1 / sr_mse)
            total_sr_psnr += sr_psnr.item()

            # --- 2. Bicubic Upscale (Baseline) ---
            # Upscale the low-res image to match the high-res dimensions
            bicubic_hr = F.interpolate(lr, scale_factor=config.SCALE, mode='bicubic', align_corners=False)
            bicubic_mse = F.mse_loss(bicubic_hr, hr)
            bicubic_psnr = 10 * torch.log10(1 / bicubic_mse)
            total_bicubic_psnr += bicubic_psnr.item()

    gen.train()

    # Calculate averages
    avg_sr_psnr = total_sr_psnr / len(loader)
    avg_bicubic_psnr = total_bicubic_psnr / len(loader)
    improvement = 100 * (1 - avg_sr_psnr / avg_bicubic_psnr)
    print("val_sr_psnr:", avg_sr_psnr, ", val_bicubic_psnr:", avg_bicubic_psnr, ", improvement %:", improvement)
    return avg_sr_psnr


def train_one_epoch(gen, disc, loader, opt_g, opt_d, scaler_g, scaler_d, l1_loss, vgg_loss, adv_loss):
    """A single training loop for one epoch of GAN training."""
    loop = tqdm(loader, leave=True)
    for lr, hr in loop:
        lr, hr = lr.to(config.DEVICE), hr.to(config.DEVICE)

        # --- Train Discriminator ---
        with autocast(device_type=config.DEVICE, dtype=torch.float16, enabled=(config.DEVICE == 'cuda')):
            fake = gen(lr)
            disc_real = disc(hr)
            disc_fake = disc(fake.detach())
            loss_disc_real = adv_loss(disc_real, torch.ones_like(disc_real))
            loss_disc_fake = adv_loss(disc_fake, torch.zeros_like(disc_fake))
            loss_disc = (loss_disc_real + loss_disc_fake) / 2

        disc.zero_grad(set_to_none=True)
        scaler_d.scale(loss_disc).backward()
        scaler_d.step(opt_d)
        scaler_d.update()

        # --- Train Generator ---
        with autocast(device_type=config.DEVICE, dtype=torch.float16, enabled=(config.DEVICE == 'cuda')):
            disc_fake_for_gen = disc(fake)
            gen_adv_loss = adv_loss(disc_fake_for_gen, torch.ones_like(disc_fake_for_gen))
            gen_l1_loss = l1_loss(fake, hr)
            gen_vgg_loss = vgg_loss(fake, hr)
            loss_gen = (config.LAMBDA_L1 * gen_l1_loss) + (config.LAMBDA_ADV * gen_adv_loss) + (config.LAMBDA_PERCEP * gen_vgg_loss)

        gen.zero_grad(set_to_none=True)
        scaler_g.scale(loss_gen).backward()
        scaler_g.step(opt_g)
        scaler_g.update()

        wandb.log({
            "Train/discriminator_loss": loss_disc.item(), "Train/generator_loss": loss_gen.item(),
            "Train/gen_adv_loss": gen_adv_loss.item(), "Train/gen_l1_loss": gen_l1_loss.item(),
            "Train/gen_vgg_loss": gen_vgg_loss.item(),
        })
        loop.set_postfix(d_loss=loss_disc.item(), g_loss=loss_gen.item())


def main(args):
    wandb.init(project="SuperResolution-ESRGAN", config=vars(config), resume="allow", id=args.wandb_id)

    train_dataset = TrainDataset(hr_dir=config.DATA_DIR/"train"/"HR", lr_dir=config.DATA_DIR/"train"/"LR")
    val_dataset = ValDataset(hr_dir=config.DATA_DIR/"val"/"HR", lr_dir=config.DATA_DIR/"val"/"LR")
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True,
                              num_workers=config.NUM_WORKERS, pin_memory=(config.DEVICE == "cuda"),
                              persistent_workers=(config.NUM_WORKERS > 0), 
                              prefetch_factor=2 if config.NUM_WORKERS > 0 else None)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    gen = GeneratorESRGAN(num_features=config.ESRGAN_NUM_FEATURES, num_blocks=config.ESRGAN_NUM_RRDB).to(config.DEVICE)
    disc = Discriminator().to(config.DEVICE)
    opt_g = optim.Adam(gen.parameters(), lr=config.ESRGAN_LR, betas=(0.9, 0.999))
    opt_d = optim.Adam(disc.parameters(), lr=config.ESRGAN_LR, betas=(0.9, 0.999))
    scaler_g = GradScaler(enabled=(config.DEVICE == 'cuda'))
    scaler_d = GradScaler(enabled=(config.DEVICE == 'cuda'))

    start_epoch = 0
    best_psnr = 0.0

    # --- Automatic Resume Logic ---
    if args.resume:
        ckpt_files = list(CHECKPOINT_DIR.glob("latest_checkpoint_*.pth"))
        if ckpt_files:
            latest_ckpt_path = max(ckpt_files, key=os.path.getctime)
            print(f"--- Resuming from latest checkpoint: {latest_ckpt_path.name} ---")
            checkpoint = torch.load(latest_ckpt_path, map_location=config.DEVICE)
            gen.load_state_dict(checkpoint['gen_state_dict'])
            disc.load_state_dict(checkpoint['disc_state_dict'])
            opt_g.load_state_dict(checkpoint['opt_g_state_dict'])
            opt_d.load_state_dict(checkpoint['opt_d_state_dict'])
            start_epoch = checkpoint['epoch']
            best_psnr = checkpoint.get('best_psnr', 0.0)
            print(f"Resumed from epoch {start_epoch} (best PSNR = {best_psnr:.2f} dB)")
        else:
            print("--- No checkpoint found, starting fresh. ---")

    scheduler_g = optim.lr_scheduler.CosineAnnealingLR(opt_g, T_max=config.ESRGAN_EPOCHS, eta_min=1e-7, last_epoch=start_epoch - 1)
    scheduler_d = optim.lr_scheduler.CosineAnnealingLR(opt_d, T_max=config.ESRGAN_EPOCHS, eta_min=1e-7, last_epoch=start_epoch - 1)

    l1_loss, vgg_loss, adversarial_loss = nn.L1Loss(), VGGLoss(), nn.BCEWithLogitsLoss()

    # --- Pre-training Logic ---
    pretrained_file = CHECKPOINT_DIR / "generator_pretrained.pth"
    if args.mode in ["pretrain", "all"] and not pretrained_file.exists() and start_epoch == 0:
        print("--- Starting Generator Pre-training (L1 Loss only) ---")
        for epoch in range(config.PRETRAIN_EPOCHS):
            # Pre-training loop implementation...
            torch.save(gen.state_dict(), pretrained_file)
        print(f"--- Pre-training Finished. Model saved to '{pretrained_file}' ---")

    if args.mode in ["train", "all"] and start_epoch == 0 and pretrained_file.exists():
         print(f"--- Loading pre-trained generator from {pretrained_file} ---")
         gen.load_state_dict(torch.load(pretrained_file, map_location=config.DEVICE))

    # --- Main GAN Training Loop ---
    if args.mode in ["train", "all"]:
        print("\n--- Starting Main ESRGAN Training ---")
        for epoch in range(start_epoch, config.ESRGAN_EPOCHS):
            train_one_epoch(gen, disc, train_loader, opt_g, opt_d, scaler_g, scaler_d, l1_loss, vgg_loss, adversarial_loss)

            # --- Validation and Checkpointing ---
            avg_psnr = validate_one_epoch(gen, val_loader)
            print(f"Epoch {epoch+1} | Validation PSNR: {avg_psnr:.2f} dB")

            wandb.log({"epoch": epoch + 1, "Validation/psnr": avg_psnr, "Train/lr": scheduler_g.get_last_lr()[0]})
            scheduler_g.step(); scheduler_d.step()

            # Save best model
            if avg_psnr > best_psnr:
                best_psnr = avg_psnr
                torch.save(gen.state_dict(), CHECKPOINT_DIR / "best_generator.pth")
                print(f"New best generator saved with PSNR: {best_psnr:.2f} dB")

            # Save latest checkpoint
            for old_ckpt in CHECKPOINT_DIR.glob("latest_checkpoint_*.pth"): old_ckpt.unlink()
            latest_ckpt_path = CHECKPOINT_DIR / f"latest_checkpoint_epoch_{epoch + 1}.pth"
            checkpoint_data = {
                'epoch': epoch + 1,
                'gen_state_dict': gen.state_dict(),
                'disc_state_dict': disc.state_dict(),
                'opt_g_state_dict': opt_g.state_dict(),
                'opt_d_state_dict': opt_d.state_dict(),
                'best_psnr': best_psnr
            }
            torch.save(checkpoint_data, latest_ckpt_path)
            print(f"Checkpoint saved for epoch {epoch + 1}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Super-Resolution GAN.")
    parser.add_argument("--mode", type=str, default="all", choices=["pretrain", "train", "all"])
    parser.add_argument("--resume", action="store_true", help="Resume training from the latest checkpoint.")
    parser.add_argument("--wandb_id", type=str, default=None, help="Weights & Biases run ID to resume logging.")
    args = parser.parse_args()
    main(args)