# diffusion/train_diffusion.py
"""
Training script for Conditional Diffusion Super-Resolution Model.
Clean, modular, and production-ready with proper checkpointing and logging.
"""
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision
from torch.amp import autocast, GradScaler
from pathlib import Path
from tqdm import tqdm
import wandb
import argparse
import sys

# Setup paths
SCRIPT_DIR = Path(__file__).parent.absolute()
sys.path.append(str(SCRIPT_DIR.parent))

from diffusion.diffusion_model import DiffusionUNet # or import DiffusionUNetLite as DiffusionUNet
from diffusion.scheduler import create_scheduler, DDPMScheduler, DDIMScheduler
from utils.datasets import TrainDataset, ValDataset
from config import config

CHECKPOINT_DIR = SCRIPT_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

class EMA:
    """Exponential Moving Average for model weights."""
    def __init__(self, model, decay=0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (
                    self.decay * self.shadow[name].data +
                    (1 - self.decay) * param.data
                )

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name].clone()

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name].clone()
        self.backup = {}


@torch.no_grad()
def validate(model, val_loader, scheduler, loss_fn, device, use_amp):
    """
    Validate the model on the validation set.

    Returns:
        dict: Validation metrics
    """
    model.eval()
    total_loss = 0.0
    total_bicubic_mse = 0.0
    total_model_mse = 0.0
    num_batches = 0

    for lr_batch, hr_batch in tqdm(val_loader, desc="Validating", leave=False):
        lr_batch = lr_batch.to(device)
        hr_batch = hr_batch.to(device)

        with autocast(device_type=device, enabled=use_amp):
            # Random timestep for validation
            t = torch.randint(0, scheduler.timesteps, (hr_batch.shape[0],), device=device)

            # Add noise
            noise = torch.randn_like(hr_batch)
            noisy_hr = scheduler.add_noise(hr_batch, t, noise)

            # Upscale LR for conditioning
            lr_upscaled = F.interpolate(
                lr_batch,
                scale_factor=config.SCALE,
                mode='bicubic',
                align_corners=False
            )

            # Predict noise
            pred_noise = model(noisy_hr, t, lr_upscaled)

            # Calculate losses
            loss = loss_fn(noise, pred_noise)
            bicubic_mse = F.mse_loss(lr_upscaled, hr_batch)
            model_mse = F.mse_loss(pred_noise, noise)

            total_loss += loss.item()
            total_bicubic_mse += bicubic_mse.item()
            total_model_mse += model_mse.item()
            num_batches += 1

    model.train()

    avg_loss = total_loss / num_batches
    avg_bicubic_mse = total_bicubic_mse / num_batches
    avg_model_mse = total_model_mse / num_batches

    # Calculate improvement
    improvement = ((avg_bicubic_mse - avg_model_mse) / avg_bicubic_mse) * 100

    return {
        'val_loss': avg_loss,
        'bicubic_mse': avg_bicubic_mse,
        'model_mse': avg_model_mse,
        'improvement_pct': improvement
    }


@torch.no_grad()
def sample_images(model, scheduler, val_loader, device, scheduler_type='ddpm',
                 num_inference_steps=None, ddim_eta=0.0):
    """
    Generate SR images using the full denoising process.

    Args:
        scheduler_type: 'ddpm' or 'ddim'
        num_inference_steps: Number of denoising steps
                            (DDPM: must equal training steps, DDIM: can be less)
        ddim_eta: DDIM stochasticity (0=deterministic, 1=stochastic like DDPM)
    """
    model.eval()

    # Get one batch
    lr_batch, hr_batch = next(iter(val_loader))
    lr_batch = lr_batch.to(device)
    hr_batch = hr_batch.to(device)

    # Upscale LR for conditioning
    lr_upscaled = F.interpolate(
        lr_batch,
        scale_factor=config.SCALE,
        mode='bicubic',
        align_corners=False
    )

    # Start from pure noise
    img = torch.randn_like(hr_batch)

    # Get sampling timesteps based on scheduler type
    if scheduler_type == 'ddim' and isinstance(scheduler, DDIMScheduler):
        # DDIM: Can use fewer steps
        if num_inference_steps is None:
            num_inference_steps = config.DIFFUSION_DDIM_STEPS

        timesteps = scheduler.get_sampling_timesteps(num_inference_steps)

        # DDIM sampling loop
        for i, t_idx in enumerate(tqdm(timesteps, desc=f"Sampling (DDIM, {num_inference_steps} steps)", leave=False)):
            t = torch.full((img.shape[0],), t_idx, device=device, dtype=torch.long)

            # Get previous timestep
            if i < len(timesteps) - 1:
                t_prev = torch.full((img.shape[0],), timesteps[i+1], device=device, dtype=torch.long)
            else:
                t_prev = torch.full((img.shape[0],), -1, device=device, dtype=torch.long)

            # Predict noise
            pred_noise = model(img, t, lr_upscaled)

            # DDIM step
            img = scheduler.sample_prev_timestep(img, t, t_prev, pred_noise, eta=ddim_eta)

    else:
        # DDPM: Must use all training steps
        if num_inference_steps is None:
            num_inference_steps = scheduler.timesteps

        timesteps = list(range(0, scheduler.timesteps))[::-1]

        # DDPM sampling loop
        for t_idx in tqdm(timesteps, desc=f"Sampling (DDPM, {num_inference_steps} steps)", leave=False):
            t = torch.full((img.shape[0],), t_idx, device=device, dtype=torch.long)

            # Predict noise
            pred_noise = model(img, t, lr_upscaled)

            # DDPM step (stochastic)
            img = scheduler.sample_prev_timestep(img, t, pred_noise)

    model.train()

    # Create comparison grid: [LR_upscaled | Model_SR | GT_HR]
    comparison = torch.cat([lr_upscaled, img, hr_batch], dim=0)

    return comparison


def train_one_epoch(model, train_loader, scheduler, optimizer, loss_fn,
                   scaler, ema, device, use_amp, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0

    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=True)

    for step, (lr_batch, hr_batch) in enumerate(loop):
        lr_batch = lr_batch.to(device, non_blocking=True)
        hr_batch = hr_batch.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type=device, enabled=use_amp):
            # Sample random timesteps
            t = torch.randint(0, scheduler.timesteps, (hr_batch.shape[0],), device=device)

            # Add noise to clean images
            noise = torch.randn_like(hr_batch)
            noisy_hr = scheduler.add_noise(hr_batch, t, noise)

            # Upscale LR for conditioning
            lr_upscaled = F.interpolate(
                lr_batch,
                scale_factor=config.SCALE,
                mode='bicubic',
                align_corners=False
            )

            # Predict noise
            pred_noise = model(noisy_hr, t, lr_upscaled)

            # Calculate loss (simple MSE on noise prediction)
            loss = loss_fn(noise, pred_noise)

        # Backward pass
        scaler.scale(loss).backward()

        # Gradient clipping for stability
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        scaler.step(optimizer)
        scaler.update()

        # Update EMA
        if ema:
            ema.update()

        total_loss += loss.item()

        # Update progress bar
        loop.set_postfix(loss=f"{loss.item():.4f}")

        # Log to wandb
        if step % 100 == 0:
            wandb.log({
                'train/loss': loss.item(),
                'train/step': epoch * len(train_loader) + step
            })

    avg_loss = total_loss / len(train_loader)
    return avg_loss


def main(args):
    # Initialize wandb
    wandb.init(
        project="SuperResolution-Diffusion",
        config=vars(config),
        resume="allow" if args.wandb_id else None,
        id=args.wandb_id,
        name=f"diffusion-{config.DIFFUSION_TIMESTEPS}steps"
    )

    # Device setup
    device = config.DEVICE
    use_amp = (device == 'cuda')
    print(f"Using device: {device}")
    print(f"Mixed precision: {use_amp}")

    # Data loaders
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
        num_workers=config.NUM_WORKERS,
        pin_memory=(device == "cuda"),
        persistent_workers=(config.NUM_WORKERS > 0),
        prefetch_factor=2 if config.NUM_WORKERS > 0 else None
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=2,
        pin_memory=(device == "cuda")
    )

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # Initialize model
    model = DiffusionUNet(
        in_channels=6,  # 3 (noisy image) + 3 (LR condition)
        out_channels=3,
        features=config.DIFFUSION_FEATURES,
        time_emb_dim=config.DIFFUSION_TIME_EMB_DIM,
        dropout=0.1
    ).to(device)

    print(f"Model parameters: {model.get_num_params():,}")

    # Initialize scheduler based on config
    scheduler = create_scheduler(
        scheduler_type=config.DIFFUSION_SCHEDULER_TYPE,
        timesteps=config.DIFFUSION_TIMESTEPS,
        schedule=config.DIFFUSION_SCHEDULE,
        device=device
    )

    print(f"Scheduler: {scheduler}")
    print(f"Sampling method: {config.DIFFUSION_SCHEDULER_TYPE.upper()}")
    if config.DIFFUSION_SCHEDULER_TYPE == 'ddim':
        print(f"DDIM inference steps: {config.DIFFUSION_DDIM_STEPS}")
        print(f"DDIM eta: {config.DIFFUSION_DDIM_ETA}")

    # Initialize EMA
    ema = EMA(model, decay=0.9999) if args.use_ema else None
    if ema:
        print("✅ EMA enabled")

    # Optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.DIFFUSION_LR,
        betas=(0.9, 0.999),
        weight_decay=1e-4
    )

    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.DIFFUSION_EPOCHS,
        eta_min=1e-7
    )

    # Loss function and scaler
    loss_fn = nn.MSELoss()  # L2 loss on noise prediction
    scaler = GradScaler(enabled=use_amp)

    # Resume from checkpoint
    start_epoch = 0
    best_val_loss = float('inf')

    if args.resume:
        checkpoint_path = CHECKPOINT_DIR / "checkpoint_latest.pth"
        if checkpoint_path.exists():
            print(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)

            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch']
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))

            if ema and 'ema_state_dict' in checkpoint:
                ema.shadow = checkpoint['ema_state_dict']

            print(f"✅ Resumed from epoch {start_epoch}")
            print(f"   Best validation loss: {best_val_loss:.6f}")
        else:
            print("No checkpoint found, starting from scratch")

    # Training loop
    print("\n" + "="*60)
    print("Starting Diffusion Model Training")
    print("="*60 + "\n")

    for epoch in range(start_epoch, config.DIFFUSION_EPOCHS):
        # Train one epoch
        avg_train_loss = train_one_epoch(
            model, train_loader, scheduler, optimizer, loss_fn,
            scaler, ema, device, use_amp, epoch
        )

        # Validate
        if (epoch + 1) % 5 == 0 or epoch == 0:
            # Apply EMA weights for validation
            if ema:
                ema.apply_shadow()

            val_metrics = validate(model, val_loader, scheduler, loss_fn, device, use_amp)

            # Restore original weights
            if ema:
                ema.restore()

            # Log metrics
            print(f"\nEpoch {epoch+1}/{config.DIFFUSION_EPOCHS}")
            print(f"  Train Loss: {avg_train_loss:.6f}")
            print(f"  Val Loss: {val_metrics['val_loss']:.6f}")
            print(f"  Bicubic MSE: {val_metrics['bicubic_mse']:.6f}")
            print(f"  Model MSE: {val_metrics['model_mse']:.6f}")
            print(f"  Improvement: {val_metrics['improvement_pct']:.2f}%")
            print(f"  LR: {lr_scheduler.get_last_lr()[0]:.2e}")

            wandb.log({
                'epoch': epoch + 1,
                'train/epoch_loss': avg_train_loss,
                'val/loss': val_metrics['val_loss'],
                'val/bicubic_mse': val_metrics['bicubic_mse'],
                'val/model_mse': val_metrics['model_mse'],
                'val/improvement_pct': val_metrics['improvement_pct'],
                'train/lr': lr_scheduler.get_last_lr()[0]
            })

            # Save checkpoint
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': lr_scheduler.state_dict(),
                'val_loss': val_metrics['val_loss'],
                'best_val_loss': best_val_loss
            }

            if ema:
                checkpoint['ema_state_dict'] = ema.shadow

            # Save latest checkpoint
            torch.save(checkpoint, CHECKPOINT_DIR / "checkpoint_latest.pth")

            # Save best model
            if val_metrics['val_loss'] < best_val_loss:
                best_val_loss = val_metrics['val_loss']
                checkpoint['best_val_loss'] = best_val_loss
                torch.save(checkpoint, CHECKPOINT_DIR / "best_model.pth")
                print(f"  🌟 New best model saved! Loss: {best_val_loss:.6f}")

            # Generate sample images
            if (epoch + 1) % 10 == 0:
                print("  📸 Generating sample images...")
                if ema:
                    ema.apply_shadow()

                sample_grid = sample_images(
                    model, scheduler, val_loader, device,
                    num_inference_steps=50  # Faster sampling
                )

                if ema:
                    ema.restore()

                grid = torchvision.utils.make_grid(
                    sample_grid,
                    nrow=sample_grid.shape[0] // 3,
                    normalize=True
                )

                wandb.log({
                    'samples': wandb.Image(
                        grid,
                        caption=f"Epoch {epoch+1} | Val Loss: {val_metrics['val_loss']:.4f}"
                    )
                })

        # Step learning rate scheduler
        lr_scheduler.step()

    print("\n" + "="*60)
    print("Training Complete!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print("="*60 + "\n")

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Conditional Diffusion Model for Super-Resolution"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from latest checkpoint"
    )
    parser.add_argument(
        "--wandb_id",
        type=str,
        default=None,
        help="W&B run ID for resuming"
    )
    parser.add_argument(
        "--use_ema",
        action="store_true",
        help="Use Exponential Moving Average"
    )

    args = parser.parse_args()
    main(args)