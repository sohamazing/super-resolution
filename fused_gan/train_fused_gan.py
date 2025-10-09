# fused_gan/train_fused_gan.py
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import argparse
import sys
from pathlib import Path
import torchvision
import torch.nn.functional as F
import contextlib

# --- Project Structure Setup (unchanged) ---
SCRIPT_DIR = Path(__file__).parent.absolute()
sys.path.append(str(SCRIPT_DIR.parent))

# --- Import all our custom components (unchanged) ---
from fused_gan.generator_hcast import HCASTGenerator
from fused_gan.discriminator_unet import DiscriminatorUNet
from utils.loss import FusedGANLoss
from utils.datasets import TrainDataset, ValDataset
from config import config

# Define the directory for saving model checkpoints.
CHECKPOINT_DIR = SCRIPT_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

class EMA:
    """Exponential Moving Average for model weights."""
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (
                    self.decay * self.shadow[name].data + (1 - self.decay) * param.data )

    def apply_shadow(self):
        self.backup = {} # Clear previous backup
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name].clone()

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {} # Clear backup

def pretrain_generator(gen, loader, opt_g, scheduler_g, scaler_g, autocast_context, ema=None):
    """
    Pre-trains the generator using only L1 pixel loss.
    """
    l1_loss = nn.L1Loss()
    print("--- Starting Generator Pre-training (L1 Loss only) ---")

    for epoch in range(config.PRETRAIN_EPOCHS):
        gen.train()
        loop = tqdm(loader, desc=f"Pre-train Epoch {epoch+1}/{config.PRETRAIN_EPOCHS}")
        epoch_loss = 0.0

        for i, (lr, hr) in enumerate(loop):
            lr = lr.to(config.DEVICE, non_blocking=True)
            hr = hr.to(config.DEVICE, non_blocking=True)

            with autocast_context:
                fake_hr = gen(lr)
                loss = l1_loss(fake_hr, hr)

            # Backpropagation
            opt_g.zero_grad(set_to_none=True)
            scaler_g.scale(loss).backward()

            scaler_g.unscale_(opt_g) # Must unscale before clipping
            torch.nn.utils.clip_grad_norm_(gen.parameters(), max_norm=1.0) # Gradient clipping for stability

            scaler_g.step(opt_g) # Apply gradients
            scaler_g.update()

            # Update EMA
            if ema is not None:
                ema.update()

            epoch_loss += loss.item()
            loop.set_postfix(l1_loss=loss.item())

            if i % 200 == 0:
                wandb.log({
                    "pretrain/l1_loss": loss.item(),
                    "pretrain/step": epoch * len(loader) + i
                })

        # Log epoch metrics and step scheduler
        avg_loss = epoch_loss / len(loader)
        wandb.log({
            "pretrain/epoch_loss": avg_loss,
            "pretrain/epoch": epoch,
            "pretrain/lr": scheduler_g.get_last_lr()[0]
        })

        scheduler_g.step()

        print(f"Epoch {epoch+1}/{config.PRETRAIN_EPOCHS} - Loss: {avg_loss:.6f} - LR: {scheduler_g.get_last_lr()[0]:.6f}")

def train_one_epoch(gen, disc, loader, opt_g, opt_d, loss_fn, scaler_g, scaler_d, autocast_context, ema=None, epoch=0):
    """A single training loop for one epoch of alternating GAN training."""
    gen.train()
    disc.train()

    loop = tqdm(loader, leave=True, desc=f"Epoch {epoch+1}")

    d_losses, g_losses, pix_losses, adv_losses, percep_losses = [], [], [], [], []

    for i, (lr, hr) in enumerate(loop):
        lr = lr.to(config.DEVICE, non_blocking=True)
        hr = hr.to(config.DEVICE, non_blocking=True)

        with autocast_context:
            fake_hr = gen(lr)

        # --- Phase 1: Train the Discriminator ---
        with autocast_context:
            disc_real = disc(hr)
            disc_fake = disc(fake_hr.detach())
            d_loss = loss_fn.calculate_d_loss(disc_real, disc_fake)

        opt_d.zero_grad(set_to_none=True)
        scaler_d.scale(d_loss).backward()
        scaler_d.unscale_(opt_d)
        torch.nn.utils.clip_grad_norm_(disc.parameters(), max_norm=1.0)
        scaler_d.step(opt_d)
        scaler_d.update()

        # --- Phase 2: Train the Generator ---
        with autocast_context:
            disc_fake_for_g = disc(fake_hr)
            g_loss, pix_loss, adv_loss, percep_loss = loss_fn.calculate_g_loss(
                disc_fake_for_g, fake_hr, hr
            )

        opt_g.zero_grad(set_to_none=True)
        scaler_g.scale(g_loss).backward()
        scaler_g.unscale_(opt_g)
        torch.nn.utils.clip_grad_norm_(gen.parameters(), max_norm=1.0)
        scaler_g.step(opt_g)
        scaler_g.update()

        # Update EMA
        if ema is not None:
            ema.update()

        # Track losses
        d_losses.append(d_loss.item())
        g_losses.append(g_loss.item())
        pix_losses.append(pix_loss.item())
        adv_losses.append(adv_loss.item())
        percep_losses.append(percep_loss.item())

        if i % 100 == 0:
            wandb.log({
                "train/d_loss": d_loss.item(), "train/g_loss": g_loss.item(),
                "train/pixel_loss": pix_loss.item(), "train/adversarial_loss": adv_loss.item(),
                "train/perceptual_loss": percep_loss.item(), "train/step": epoch * len(loader) + i
            })

        loop.set_postfix(
            d_loss=f"{d_loss.item():.4f}", g_loss=f"{g_loss.item():.4f}", pix=f"{pix_loss.item():.4f}"
        )

    return {
        'd_loss': sum(d_losses) / len(d_losses), 'g_loss': sum(g_losses) / len(g_losses),
        'pix_loss': sum(pix_losses) / len(pix_losses), 'adv_loss': sum(adv_losses) / len(adv_losses),
        'percep_loss': sum(percep_losses) / len(percep_losses)
    }

@torch.no_grad()
def validate_and_visualize(gen, val_sample_lr, val_sample_hr, epoch, ema=None):
    """Run validation and create visualization."""
    gen.eval()

    if ema is not None:
        ema.apply_shadow()

    with torch.cuda.amp.autocast(enabled=(config.DEVICE == "cuda")):
        fake_hr = gen(val_sample_lr)

    mse = F.mse_loss(fake_hr, val_sample_hr)
    psnr = 10 * torch.log10(1 / mse)

    grid = torchvision.utils.make_grid(
        torch.cat([fake_hr, val_sample_hr], dim=0), normalize=True, nrow=fake_hr.shape[0]
    )

    wandb.log({
        "validation/samples": wandb.Image(grid, caption=f"Epoch {epoch+1} | PSNR: {psnr:.2f} dB"),
        "validation/psnr": psnr.item(),
        "validation/epoch": epoch
    })

    if ema is not None:
        ema.restore()

    gen.train()
    return psnr.item()

def main(args):
    wandb.init(
        project="SuperResolution-FusedGAN", config=vars(config),
        resume="allow" if args.wandb_id else None, id=args.wandb_id
    )
    # Use TrainDataset for the training set (already cropped, adds rotations)
    train_dataset = TrainDataset(
        hr_dir=config.DATA_DIR / "train" / "HR",
        lr_dir=config.DATA_DIR / "train" / "LR",
    ) 
    # Use ValDataset for the validation set (with center cropping)
    val_dataset = ValDataset(
        hr_dir=config.DATA_DIR / "val" / "HR",
        lr_dir=config.DATA_DIR / "val" / "LR",
    )

    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE, shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=(config.DEVICE == "cuda"),
        persistent_workers=(config.NUM_WORKERS > 0),
        prefetch_factor=2 if config.NUM_WORKERS > 0 else None
    )
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    gen = HCASTGenerator().to(config.DEVICE)
    disc = DiscriminatorUNet().to(config.DEVICE)
    loss_fn = FusedGANLoss()

    opt_g = optim.AdamW(gen.parameters(), lr=config.FUSEDGAN_LR, betas=(0.9, 0.999), weight_decay=1e-4)
    opt_d = optim.AdamW(disc.parameters(), lr=config.FUSEDGAN_LR, betas=(0.9, 0.999), weight_decay=1e-4)

    scheduler_g = optim.lr_scheduler.CosineAnnealingLR(opt_g, T_max=config.FUSEDGAN_EPOCHS, eta_min=1e-6)
    scheduler_d = optim.lr_scheduler.CosineAnnealingLR(opt_d, T_max=config.FUSEDGAN_EPOCHS, eta_min=1e-6)

    # Scheduler for pretrain phase
    scheduler_g_pretrain = optim.lr_scheduler.CosineAnnealingLR(opt_g, T_max=config.PRETRAIN_EPOCHS, eta_min=1e-6)

    # Cross-platform Automatic Mixed Precision (AMP) setup
    amp_enabled = config.DEVICE != 'cpu'
    # Use bfloat16 for MPS (Apple Silicon) if available, otherwise float16 for CUDA
    dtype = torch.bfloat16 if config.DEVICE == 'mps' and torch.cuda.is_bf16_supported() else torch.float16
    
    scaler_g = torch.amp.GradScaler(device_type=config.DEVICE, enabled=amp_enabled)
    scaler_d = torch.amp.GradScaler(device_type=config.DEVICE, enabled=amp_enabled)
    autocast_context = torch.amp.autocast(device_type=config.DEVICE, dtype=dtype, enabled=amp_enabled)

    ema = EMA(gen) if args.use_ema else None
    if ema:
        ema.register()

    print("Caching validation sample...")
    val_sample_lr, val_sample_hr = next(iter(val_loader))
    val_sample_lr, val_sample_hr = val_sample_lr.to(config.DEVICE), val_sample_hr.to(config.DEVICE)

    pretrained_file = CHECKPOINT_DIR / "generator_pretrained.pth"
    start_epoch, best_psnr = 0, 0.0

    if args.resume_epoch > 0:
        # Resume from checkpoint
        start_epoch = args.resume_epoch
        latest_checkpoint_path = CHECKPOINT_DIR / "latest_checkpoint.pth"
        print(f"--- Resuming GAN training from Epoch {start_epoch} ---")
        if latest_checkpoint_path.exists():
            checkpoint = torch.load(latest_checkpoint_path, map_location=config.DEVICE)
            gen.load_state_dict(checkpoint['gen_state_dict'])
            disc.load_state_dict(checkpoint['disc_state_dict'])
            opt_g.load_state_dict(checkpoint['opt_g_state_dict'])
            opt_d.load_state_dict(checkpoint['opt_d_state_dict'])
            scheduler_g.load_state_dict(checkpoint['scheduler_g_state_dict'])
            scheduler_d.load_state_dict(checkpoint['scheduler_d_state_dict'])
            best_psnr = checkpoint.get('best_psnr', 0.0)
            if ema and 'ema_state_dict' in checkpoint: ema.shadow = checkpoint['ema_state_dict']
            print(f"Checkpoint loaded. Best PSNR so far: {best_psnr:.2f} dB")
        else:
            print(f"Warning: Checkpoint not found at {latest_checkpoint_path}")

    elif args.mode in ["train", "all"] and pretrained_file.exists():
        print(f"Loading pretrained generator from '{pretrained_file}'")
        gen.load_state_dict(torch.load(pretrained_file, map_location=config.DEVICE))

    if args.mode in ["pretrain", "all"] and not pretrained_file.exists() and start_epoch == 0:
        pretrain_generator(gen, train_loader, opt_g, scheduler_g_pretrain, scaler_g, autocast_context, ema)
        torch.save(gen.state_dict(), pretrained_file)
        print(f"Pre-training complete. Model saved to '{pretrained_file}'")

        # Reset optimizers and schedulers for GAN phase
        opt_g = optim.AdamW(gen.parameters(), lr=config.FUSEDGAN_LR, betas=(0.9, 0.999), weight_decay=1e-4)
        opt_d = optim.AdamW(disc.parameters(), lr=config.FUSEDGAN_LR, betas=(0.9, 0.999), weight_decay=1e-4)
        scheduler_g = optim.lr_scheduler.CosineAnnealingLR(opt_g, T_max=config.FUSEDGAN_EPOCHS, eta_min=1e-6)
        scheduler_d = optim.lr_scheduler.CosineAnnealingLR(opt_d, T_max=config.FUSEDGAN_EPOCHS, eta_min=1e-6)

    if args.mode in ["train", "all"]:
        print("\n--- Starting Main Fused-GAN Training ---")
        for epoch in range(start_epoch, config.FUSEDGAN_EPOCHS):
            metrics = train_one_epoch(gen, disc, train_loader, opt_g, opt_d, loss_fn, scaler_g, scaler_d, autocast_context, ema, epoch)
            wandb.log({
                "train/epoch_d_loss": metrics['d_loss'], 
                "train/epoch_g_loss": metrics['g_loss'], 
                "train/lr_g": scheduler_g.get_last_lr()[0], 
                "epoch": epoch + 1
            })
            scheduler_g.step(); scheduler_d.step()

            if (epoch + 1) % 5 == 0:
                print(f"\nRunning validation...")
                psnr = validate_and_visualize(gen, val_sample_lr, val_sample_hr, epoch, ema)
                print(f"  Validation PSNR: {psnr:.2f} dB")

                checkpoint = {'epoch': epoch + 1, 'gen_state_dict': gen.state_dict(), 'disc_state_dict': disc.state_dict(), 'opt_g_state_dict': opt_g.state_dict(), 'opt_d_state_dict': opt_d.state_dict(), 'scheduler_g_state_dict': scheduler_g.state_dict(), 'scheduler_d_state_dict': scheduler_d.state_dict(), 'best_psnr': best_psnr}
                if ema: checkpoint['ema_state_dict'] = ema.shadow

                torch.save(checkpoint, CHECKPOINT_DIR / "latest_checkpoint.pth")

                if psnr > best_psnr:
                    best_psnr = psnr
                    checkpoint['best_psnr'] = best_psnr
                    torch.save(checkpoint, CHECKPOINT_DIR / "best_model.pth")
                    print(f"New best model! PSNR: {psnr:.2f} dB")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Fused-GAN model.")
    parser.add_argument("--mode", type=str, default="all", choices=["pretrain", "train", "all"], help="Set training mode.")
    parser.add_argument("--resume_epoch", type=int, default=0, help="Epoch to resume GAN training from.")
    parser.add_argument("--wandb_id", type=str, default=None, help="Weights & Biases run ID to resume logging.")
    parser.add_argument("--use_ema", action="store_true", help="Use Exponential Moving Average.")
    args = parser.parse_args()
    main(args)
