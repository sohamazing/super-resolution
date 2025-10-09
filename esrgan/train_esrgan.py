# esrgan/train_esrgan.py
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor
from torchvision.models import vgg19
import torchvision.utils
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import wandb
import argparse
import os
import sys

SCRIPT_DIR = Path(__file__).parent.absolute()
sys.path.append(str(SCRIPT_DIR.parent)) # .../super-res/
CHECKPOINT_DIR = SCRIPT_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

from esrgan.generator import GeneratorRRDB
from esrgan.discriminator import Discriminator
from utils.datasets import TrainDataset, ValDataset
from utils.loss import VGGLoss
from config import config

def train_one_epoch(gen, disc, loader, opt_g, opt_d, l1_loss, vgg_loss, adv_loss):
    """A single training loop for one epoch of GAN training."""
    loop = tqdm(loader, leave=True)
    for i, (lr, hr) in enumerate(loop):
        lr = lr.to(config.DEVICE)
        hr = hr.to(config.DEVICE)

        # --- Train Discriminator ---
        fake = gen(lr)
        disc_real = disc(hr)
        disc_fake = disc(fake.detach())
        loss_disc_real = adv_loss(disc_real, torch.ones_like(disc_real))
        loss_disc_fake = adv_loss(disc_fake, torch.zeros_like(disc_fake))
        loss_disc = (loss_disc_real + loss_disc_fake) / 2

        disc.zero_grad()
        loss_disc.backward()
        opt_d.step()

        # --- Train Generator ---
        disc_fake_for_gen = disc(fake)
        gen_adv_loss = adv_loss(disc_fake_for_gen, torch.ones_like(disc_fake_for_gen))
        gen_l1_loss = l1_loss(fake, hr)
        gen_vgg_loss = vgg_loss(fake, hr)
        loss_gen = (config.LAMBDA_L1 * gen_l1_loss) + (config.LAMBDA_ADV * gen_adv_loss) + (config.LAMBDA_PERCEP * gen_vgg_loss)

        gen.zero_grad()
        loss_gen.backward()
        opt_g.step()

        # Log losses to wandb
        wandb.log({
            "discriminator_loss": loss_disc.item(),
            "generator_loss": loss_gen.item(),
            "gen_adv_loss": gen_adv_loss.item(),
            "gen_l1_loss": gen_l1_loss.item(),
            "gen_vgg_loss": gen_vgg_loss.item(),
        })

def main(args):
    wandb.init(project="SuperResolution-ESRGAN", config=vars(config), resume="allow", id=args.wandb_id)

    train_dataset = TrainDataset(
        hr_dir=config.DATA_DIR / "train" / "HR",
        lr_dir=config.DATA_DIR / "train" / "LR"
    )
    val_dataset = ValDataset(
        hr_dir=config.DATA_DIR / "val" / "HR",
        lr_dir=config.DATA_DIR / "val" / "LR"
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS, # Uses multiple cores for fetching
        pin_memory=(config.DEVICE == "cuda") # Pin memory only if CUDA is selected
    )
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    generator = GeneratorRRDB().to(config.DEVICE)
    discriminator = Discriminator().to(config.DEVICE)
    optimizer_g = optim.Adam(generator.parameters(), lr=config.ESRGAN_LR, betas=(0.9, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=config.ESRGAN_LR, betas=(0.9, 0.999))

    start_epoch = 0
    if args.resume_epoch > 0:
        start_epoch = args.resume_epoch
        print(f"--- Resuming GAN training from Epoch {start_epoch} ---")
        gen_path = CHECKPOINT_DIR / f"generator_epoch_{start_epoch}.pth"
        disc_path = CHECKPOINT_DIR / f"discriminator_epoch_{start_epoch}.pth"
        generator.load_state_dict(torch.load(gen_path, map_location=config.DEVICE))
        discriminator.load_state_dict(torch.load(disc_path, map_location=config.DEVICE))

        try:
            opt_g_path = CHECKPOINT_DIR / f"optimizer_g_epoch_{start_epoch}.pth"
            opt_d_path = CHECKPOINT_DIR / f"optimizer_d_epoch_{start_epoch}.pth"
            optimizer_g.load_state_dict(torch.load(opt_g_path))
            optimizer_d.load_state_dict(torch.load(opt_d_path))
            print("Optimizer states loaded successfully.")
        except FileNotFoundError:
            print("Optimizer state files not found. Initializing optimizers from scratch.")

    scheduler_g = optim.lr_scheduler.CosineAnnealingLR(optimizer_g, T_max=config.ESRGAN_EPOCHS, eta_min=1e-7, last_epoch=start_epoch - 1)
    scheduler_d = optim.lr_scheduler.CosineAnnealingLR(optimizer_d, T_max=config.ESRGAN_EPOCHS, eta_min=1e-7, last_epoch=start_epoch - 1)

    l1_loss = nn.L1Loss()
    vgg_loss = VGGLoss()
    adversarial_loss = nn.BCEWithLogitsLoss()

    pretrained_file = CHECKPOINT_DIR / "generator_pretrained.pth"
    if not pretrained_file.exists() and args.mode != 'train' and args.resume_epoch == 0:
        print("--- Starting Generator Pre-training (L1 Loss only) ---")
        for epoch in range(config.PRETRAIN_EPOCHS):
            loop = tqdm(train_loader, leave=True, desc=f"Pre-train Epoch {epoch+1}/{config.PRETRAIN_EPOCHS}")
            g_loss_accum = 0
            for lr, hr in loop:
                lr, hr = lr.to(config.DEVICE), hr.to(config.DEVICE)
                fake = generator(lr)
                loss = l1_loss(fake, hr)
                g_loss_accum += loss.item()
                optimizer_g.zero_grad()
                loss.backward()
                optimizer_g.step()
                loop.set_postfix(pretrain_l1_loss=loss.item())
            wandb.log({"pretrain_epoch": epoch, "pretrain_g_loss_avg": g_loss_accum / len(train_loader)})
        torch.save(generator.state_dict(), pretrained_file)
        print(f"--- Pre-training Finished. Model saved to '{pretrained_file}' ---")
    else:
        print(f"--- Found existing '{pretrained_file}' or resuming. Skipping pre-training. ---")

    if args.mode == "train" or args.mode == "all":
        if args.resume_epoch == 0:
            generator.load_state_dict(torch.load(pretrained_file, map_location=config.DEVICE))

        print("--- Starting Main GAN Training ---")
        for epoch in range(start_epoch, config.ESRGAN_EPOCHS):
            print(f"\n--- Main Training Epoch {epoch+1}/{config.ESRGAN_EPOCHS} ---")
            train_one_epoch(generator, discriminator, train_loader, optimizer_g, optimizer_d, l1_loss, vgg_loss, adversarial_loss)
            scheduler_g.step()
            scheduler_d.step()

            if (epoch + 1) % 10 == 0:
                print("...Saving models and logging validation images...")
                torch.save(generator.state_dict(), CHECKPOINT_DIR / f"generator_epoch_{epoch+1}.pth")
                torch.save(discriminator.state_dict(), CHECKPOINT_DIR / f"discriminator_epoch_{epoch+1}.pth")
                torch.save(optimizer_g.state_dict(), CHECKPOINT_DIR / f"optimizer_g_epoch_{epoch+1}.pth")
                torch.save(optimizer_d.state_dict(), CHECKPOINT_DIR / f"optimizer_d_epoch_{epoch+1}.pth")

                generator.eval()
                with torch.no_grad():
                    # Validation images
                    lr_batch, hr_batch = next(iter(val_loader))
                    lr_batch, hr_batch = lr_batch.to(config.DEVICE), hr_batch.to(config.DEVICE)
                    # Generate the super-resolved image
                    fake_hr = generator(lr_batch) 
                    # Bicubic upscale as a baseline comparison
                    bicubic_hr = torch.nn.functional.interpolate( 
                        lr_batch,
                        scale_factor=config.SCALE,
                        mode='bicubic',
                        align_corners=False
                    )
                    # First 4 images only (for a cleaner visual)
                    num_samples = min(4, lr_batch.size(0))
                    # [Baseline | Model Output | Ground Truth]
                    image_grid = torch.cat([bicubic_hr[:num_samples], fake_hr[:num_samples], hr_batch[:num_samples]], dim=0)
                    grid = torchvision.utils.make_grid(image_grid, normalize=True, nrow=num_samples)
                    wandb.log({"validation_samples": wandb.Image(grid, caption=f"Epoch {epoch+1}")})

                generator.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Super-Resolution GAN.")
    parser.add_argument("--mode", type=str, default="all", choices=["pretrain", "train", "all"])
    parser.add_argument("--resume_epoch", type=int, default=0, help="Epoch to resume GAN training from.")
    parser.add_argument("--wandb_id", type=str, default=None, help="Weights & Biases run ID to resume logging.")
    args = parser.parse_args()
    main(args)