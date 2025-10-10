# swinir/train_swinir.py
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import wandb
import argparse
import sys
from pathlib import Path
import torchvision

# --- Project Structure Setup ---
# Ensures we can import from the parent directory (e.g., utils, config)
SCRIPT_DIR = Path(__file__).parent.absolute()
sys.path.append(str(SCRIPT_DIR.parent))
CHECKPOINT_DIR = SCRIPT_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

from swinir.fused_swinir_model import FusedSwinIR
from utils.datasets import TrainDataset, ValDataset
from config import config

def train_one_epoch(model, loader, optimizer, loss_fn, scaler):
    """
    A single training loop for one epoch, featuring SOTA practices like
    Automatic Mixed Precision (AMP) for faster training on modern GPUs.
    """
    loop = tqdm(loader, leave=True)
    avg_loss = 0.0

    for i, (lr, hr) in enumerate(loop):
        lr = lr.to(config.DEVICE)
        hr = hr.to(config.DEVICE)

        # Use torch.cuda.amp.autocast to run the forward pass in mixed precision.
        # This significantly speeds up training on NVIDIA L4, T4, V100, A100 GPUs.
        with autocast(device_type=config.DEVICE, dtype=torch.float16, enabled=(config.DEVICE != 'cpu')):
            sr = model(lr)
            loss = loss_fn(sr, hr)

        # Backpropagation with GradScaler
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        avg_loss += loss.item()
        loop.set_postfix(l1_loss=loss.item())
        
    return avg_loss / len(loader)

@torch.no_grad()
def validate_one_epoch(model, loader):
    """Calculates the average validation PSNR for the model."""
    model.eval()
    total_psnr = 0.0
    with autocast(device_type=config.DEVICE, dtype=torch.float16, enabled=(config.DEVICE == 'cuda')):
        for lr, hr in tqdm(loader, desc="Validating", leave=False):
            lr, hr = lr.to(config.DEVICE), hr.to(config.DEVICE)
            sr = model(lr)
            mse = F.mse_loss(sr, hr)
            psnr = 10 * torch.log10(1 / mse)
            total_psnr += psnr.item()
    model.train()
    return total_psnr / len(loader)

@torch.no_grad()
def sample_and_log_images(model, loader, epoch):
    """Logs a visual sample of [Bicubic | Generated | Ground Truth]."""
    model.eval()
    lr, hr = next(iter(loader))
    lr, hr = lr.to(config.DEVICE), hr.to(config.DEVICE)
    with autocast(device_type=config.DEVICE, dtype=torch.float16, enabled=(config.DEVICE == 'cuda')):
        sr = model(lr)
    bicubic = F.interpolate(lr, scale_factor=config.SCALE, mode='bicubic', align_corners=False)
    image_grid = torchvision.utils.make_grid(torch.cat([bicubic, sr, hr], dim=0), normalize=True)
    wandb.log({"validation_samples": wandb.Image(image_grid, caption=f"Epoch {epoch+1}")})
    model.train()

def main(args):
    # Initialize Weights & Biases for experiment tracking
    wandb.init(project="SuperResolution-FusedSwinIR", config=vars(config), resume="allow", id=args.wandb_id)
    
    train_dataset = TrainDataset(
        hr_dir=config.DATA_DIR / "train" / "HR", 
        lr_dir=config.DATA_DIR / "train" / "LR",
    )
    val_dataset = ValDataset(
        hr_dir=config.DATA_DIR / "val" / "HR", 
        lr_dir=config.DATA_DIR / "val" / "LR",
    )

    train_loader = DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, 
        num_workers=config.NUM_WORKERS, pin_memory=(config.DEVICE == "cuda")
    )
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=True)

    # Initialize our custom FusedSwinIR model
    model = FusedSwinIR(
        embed_dim=config.SWIN_EMBED_DIM,
        num_heads=config.SWIN_NUM_HEADS,
        window_size=config.SWIN_WINDOW_SIZE,
        num_layers=config.SWIN_NUM_LAYERS,
        num_blocks_per_layer=config.SWIN_NUM_BLOCKS_PER_LAYER,
        scale=config.SCALE
    ).to(config.DEVICE)
    
    # AdamW is the standard, state-of-the-art optimizer for transformer-based models.
    optimizer = optim.AdamW(model.parameters(), lr=config.SWIN_LR, weight_decay=1e-2)
    
    # L1 loss is a robust choice for image-to-image tasks, focusing on pixel-wise accuracy.
    loss_fn = nn.L1Loss()
    
    # GradScaler is essential for stable mixed-precision training.
    scaler = GradScaler(device_type=config.DEVICE, enabled=(config.DEVICE != 'cpu'))

    start_epoch = 0
    best_psnr = 0.0

    # --- Full Automatic Resume Logic ---
    if args.resume:
        ckpt_files = list(CHECKPOINT_DIR.glob("latest_checkpoint_*.pth"))
        if ckpt_files:
            latest_ckpt_path = max(ckpt_files, key=os.path.getctime)
            print(f"--- Resuming from latest checkpoint: {latest_ckpt_path.name} ---")
            checkpoint = torch.load(latest_ckpt_path, map_location=config.DEVICE)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            best_psnr = checkpoint.get('best_psnr', 0.0)
            print(f"Resumed from epoch {start_epoch} (best PSNR = {best_psnr:.2f} dB)")
        else:
            print("--- No checkpoint found, starting fresh. ---")

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.SWIN_EPOCHS, eta_min=1e-7, last_epoch=start_epoch - 1)

    print("--- Starting FusedSwinIR Training ---")
    for epoch in range(start_epoch, config.SWIN_EPOCHS):
        avg_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, scaler)
        
        # --- Full Validation and Checkpointing Block ---
        avg_psnr = validate_one_epoch(model, val_loader)
        print(f"Epoch {epoch+1} | Train Loss: {avg_loss:.4f} | Validation PSNR: {avg_psnr:.2f} dB")
        
        wandb.log({
            "epoch": epoch + 1, 
            "Train/loss": avg_loss, 
            "Validation/psnr": avg_psnr,
            "Train/lr": scheduler.get_last_lr()[0]
        })
        scheduler.step()
        
        sample_and_log_images(model, val_loader, epoch)
        
        # Save best model
        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            torch.save(model.state_dict(), CHECKPOINT_DIR / "best_model.pth")
            print(f"New best model saved with PSNR: {best_psnr:.2f} dB")

        # Save latest checkpoint
        for old_ckpt in CHECKPOINT_DIR.glob("latest_checkpoint_*.pth"): old_ckpt.unlink()
        latest_ckpt_path = CHECKPOINT_DIR / f"latest_checkpoint_epoch_{epoch + 1}.pth"
        checkpoint_data = {
            'epoch': epoch + 1, 'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_psnr': best_psnr
        }
        torch.save(checkpoint_data, latest_ckpt_path)
        print(f"Checkpoint saved for epoch {epoch + 1}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the FusedSwinIR model.")
    parser.add_argument("--resume", action="store_true", help="Resume training from the latest checkpoint.") # <-- MODIFIED
    parser.add_argument("--wandb_id", type=str, default=None, help="Weights & Biases run ID to resume logging.")
    args = parser.parse_args()
    main(args)