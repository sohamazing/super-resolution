# diffusion/train_diffusion.py
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torch.nn.functional as F
import torchvision.utils
from torch.amp import autocast, GradScaler
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import wandb
import argparse
import sys
import os

# --- Path logic and imports ---
SCRIPT_DIR = Path(__file__).parent.absolute()
sys.path.append(str(SCRIPT_DIR.parent))  # .../super-res/

from diffusion.diffusion_model import DiffusionUNet
from diffusion.scheduler import Scheduler
from utils.datasets import TrainDataset, ValDataset
from config import config  # Central config file

# --- Checkpoint directory ---
CHECKPOINT_DIR = SCRIPT_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


@torch.no_grad()
def validate_one_epoch(model, loader, scheduler, loss_fn, device):
    """Calculate average validation loss for one epoch."""
    model.eval()
    total_val_loss = 0.0
    with autocast(device_type=device, dtype=torch.float16, enabled=(device != 'cpu')):
        for lr_batch, hr_batch in tqdm(loader, desc="Validating", leave=False):
            lr_batch, hr_batch = lr_batch.to(device), hr_batch.to(device)
            t = torch.randint(0, scheduler.timesteps, (hr_batch.shape[0],), device=device).long()
            noise = torch.randn_like(hr_batch)
            noisy_hr_batch = scheduler.add_noise(x_start=hr_batch, t=t, noise=noise)
            lr_upscaled = F.interpolate(lr_batch, scale_factor=config.SCALE, mode='bicubic', align_corners=False)
            predicted_noise = model(noisy_hr_batch, t, lr_upscaled)
            total_val_loss += loss_fn(noise, predicted_noise).item()
    model.train()
    return total_val_loss / len(loader)


@torch.no_grad()
def sample_and_log_images(model, scheduler, loader, epoch):
    """Run full denoising loop on validation patch and log result."""
    model.eval()
    lr_batch, hr_batch = next(iter(loader))
    lr_batch, hr_batch = lr_batch.to(config.DEVICE), hr_batch.to(config.DEVICE)

    img = torch.randn_like(hr_batch)
    lr_upscaled = F.interpolate(lr_batch, scale_factor=config.SCALE, mode='bicubic', align_corners=False)

    for i in tqdm(reversed(range(0, scheduler.timesteps)), desc="Sampling", total=scheduler.timesteps, leave=False):
        t = torch.full((img.shape[0],), i, device=config.DEVICE, dtype=torch.long)
        predicted_noise = model(img, t, lr_upscaled)
        img = scheduler.sample_previous_timestep(img, t, predicted_noise)

    all_images = torch.cat([lr_upscaled, img, hr_batch], dim=0)
    grid = torchvision.utils.make_grid(all_images, normalize=True, nrow=lr_batch.size(0))
    wandb.log({"validation_sample": wandb.Image(grid, caption=f"Epoch {epoch + 1}")})
    model.train()


def main(args):
    wandb.init(project="SuperResolution-Diffusion", config=vars(config), resume="allow", id=args.wandb_id)

    # --- Data ---
    train_dataset = TrainDataset(hr_dir=config.DATA_DIR / "train" / "HR", lr_dir=config.DATA_DIR / "train" / "LR")
    val_dataset = ValDataset(hr_dir=config.DATA_DIR / "val" / "HR", lr_dir=config.DATA_DIR / "val" / "LR")
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True,
                              num_workers=config.NUM_WORKERS, pin_memory=(config.DEVICE == "cuda"),
                              persistent_workers=(config.NUM_WORKERS > 0), 
                              prefetch_factor=2 if config.NUM_WORKERS > 0 else None)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    # --- Model + Optimizer ---
    model = DiffusionUNet(
        features=config.DIFFUSION_FEATURES,
        time_emb_dim=config.DIFFUSION_TIME_EMB_DIM
    ).to(config.DEVICE)

    scheduler = Scheduler(timesteps=config.DIFFUSION_TIMESTEPS, device=config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=config.DIFFUSION_LR)
    loss_fn = nn.L1Loss()
    scaler = GradScaler(enabled=(config.DEVICE == 'cuda'))

    start_epoch = 0
    best_val_loss = float('inf')

    # --- Resume logic ---
    if args.resume:
        ckpt_files = list(CHECKPOINT_DIR.glob("latest_checkpoint_*.pth"))
        if ckpt_files:
            latest_ckpt_path = max(ckpt_files, key=os.path.getctime)
            print(f"--- Resuming from {latest_ckpt_path.name} ---")
            checkpoint = torch.load(latest_ckpt_path, map_location=config.DEVICE)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            best_val_loss = checkpoint.get('best_val_loss', best_val_loss)
            print(f"Resumed from epoch {start_epoch} (best val loss = {best_val_loss:.4f})")
        else:
            print("--- No checkpoint found, starting fresh. ---")

    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.DIFFUSION_EPOCHS, eta_min=1e-7, last_epoch=start_epoch - 1
    )

    # --- Configurable checkpoint interval ---
    checkpoint_interval = getattr(config, "CHECKPOINT_INTERVAL", 1)

    print("--- Starting Diffusion Model Training ---")
    for epoch in range(start_epoch, config.DIFFUSION_EPOCHS):
        model.train()
        loop = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{config.DIFFUSION_EPOCHS}")
        avg_loss = 0.0

        # --- Training loop ---
        for lr_batch, hr_batch in loop:
            lr_batch, hr_batch = lr_batch.to(config.DEVICE), hr_batch.to(config.DEVICE)
            optimizer.zero_grad()
            with autocast(device_type=config.DEVICE, dtype=torch.float16, enabled=(config.DEVICE == 'cuda')):
                t = torch.randint(0, scheduler.timesteps, (hr_batch.shape[0],), device=config.DEVICE).long()
                noise = torch.randn_like(hr_batch)
                noisy_hr_batch = scheduler.add_noise(x_start=hr_batch, t=t, noise=noise)
                lr_upscaled = F.interpolate(
                    lr_batch, scale_factor=config.SCALE, mode='bicubic', align_corners=False
                )
                predicted_noise = model(noisy_hr_batch, t, lr_upscaled)
                loss = loss_fn(noise, predicted_noise)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            avg_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_train_loss = avg_loss / len(train_loader)
        avg_val_loss = validate_one_epoch(model, val_loader, scheduler, loss_fn, config.DEVICE)

        # --- Logging ---
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "learning_rate": lr_scheduler.get_last_lr()[0]
        })
        lr_scheduler.step()

        # --- Checkpoint ---
        if (epoch + 1) % checkpoint_interval == 0:
            # Save best model first
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), CHECKPOINT_DIR / "best_model.pth")
                print(f"Best model updated (val_loss={best_val_loss:.4f})")

            # Save latest checkpoint for resuming
            latest_ckpt_path = CHECKPOINT_DIR / f"latest_checkpoint_epoch_{epoch + 1}.pth"
            checkpoint_data = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss
            }
            torch.save(checkpoint_data, latest_ckpt_path)
            print(f"Saved checkpoint for epoch {epoch + 1}")

            # Now safely delete old checkpoints
            for old_ckpt in CHECKPOINT_DIR.glob("latest_checkpoint_*.pth"):
                if old_ckpt != latest_ckpt_path:
                    old_ckpt.unlink()

            # Log validation sample
            sample_and_log_images(model, scheduler, val_loader, epoch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Super-Resolution Diffusion Model.")
    parser.add_argument("--resume", action="store_true", help="Resume training from the latest checkpoint.")
    parser.add_argument("--wandb_id", type=str, default=None, help="Weights & Biases run ID to resume logging.")
    args = parser.parse_args()
    main(args)
