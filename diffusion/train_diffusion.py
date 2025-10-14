# diffusion/train_diffusion.py
"""
Training script for Conditional Diffusion Super-Resolution Model.
Clean, modular, and production-ready with proper checkpointing and logging.
"""
import os
import sys
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
import psutil

# Setup paths
SCRIPT_DIR = Path(__file__).parent.absolute()
sys.path.append(str(SCRIPT_DIR.parent))

from config import config
from diffusion.diffusion_model import DiffusionUNet, DiffusionUNetLite
from diffusion.scheduler import create_scheduler, DDPMScheduler, DDIMScheduler
from utils.datasets import TrainDataset, TrainDatasetAugmented, ValDataset, ValDatasetGrid, ValDatasetCenterGrid


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

            # 1. Extract necessary alpha values for timestep t
            # NOTE: Assumes sqrt_alphas_cumprod and sqrt_one_minus_alphas_cumprod are defined on scheduler
            sqrt_alphas_cumprod_t = scheduler._extract(scheduler.sqrt_alphas_cumprod, t, hr_batch.shape)
            sqrt_one_minus_alphas_cumprod_t = scheduler._extract(scheduler.sqrt_one_minus_alphas_cumprod, t, hr_batch.shape)

            # 2. DDPM formula to estimate x_0 (predicted clean image):
            # x_0 = (x_t - sqrt(1 - alpha_cumprod) * pred_noise) / sqrt(alpha_cumprod)
            pred_x0 = (noisy_hr - sqrt_one_minus_alphas_cumprod_t * pred_noise) / sqrt_alphas_cumprod_t

            # 3. Clip predicted x_0 for stability (important for diffusion models)
            pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)

            # Calculate the training loss (MSE on noise)
            loss = loss_fn(noise, pred_noise)

            # Calculate the Bicubic baseline MSE
            bicubic_mse = F.mse_loss(lr_upscaled, hr_batch)

            # Calculate the TRUE Reconstruction MSE (Model vs. Ground Truth)
            reconstruction_mse = F.mse_loss(pred_x0, hr_batch)

            total_loss += loss.item()
            total_bicubic_mse += bicubic_mse.item()
            total_model_mse += reconstruction_mse.item() # Now tracks Reconstruction MSE
            num_batches += 1

    model.train()

    avg_loss = total_loss / num_batches
    avg_bicubic_mse = total_bicubic_mse / num_batches
    avg_model_mse = total_model_mse / num_batches # True Model Reconstruction MSE

    # Calculate improvement: should now be positive if the model is better than Bicubic
    improvement = ((avg_bicubic_mse - avg_model_mse) / avg_bicubic_mse) * 100

    return {
        'val_loss': avg_loss,          # Noise Prediction Loss
        'bicubic_mse': avg_bicubic_mse,
        'model_mse': avg_model_mse,    # TRUE Reconstruction MSE
        'improvement_pct': improvement
    }

@torch.no_grad()
def sample_images(model, scheduler, val_loader_samples, device, scheduler_type='ddpm',
                 num_inference_steps=None, ddim_eta=0.0):
    """Generate SR images using the full denoising process."""
    model.eval()

    # Get one batch
    lr_batch, hr_batch = next(iter(val_loader_samples))
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

    # Determine sampling method
    is_ddim = isinstance(scheduler, DDIMScheduler)

    if is_ddim:
        # DDIM: Can use fewer steps
        if num_inference_steps is None:
            num_inference_steps = config.DIFFUSION_DDIM_STEPS
        timesteps = scheduler.get_sampling_timesteps(num_inference_steps)

        # DDIM sampling loop
        for i, t_idx in enumerate(tqdm(timesteps, desc=f"DDIM ({num_inference_steps} steps)", leave=False)):
            t = torch.full((img.shape[0],), t_idx, device=device, dtype=torch.long)

            # Get previous timestep
            if i < len(timesteps) - 1:
                t_prev = torch.full((img.shape[0],), timesteps[i+1], device=device, dtype=torch.long)
            else:
                t_prev = torch.full((img.shape[0],), -1, device=device, dtype=torch.long)

            # Predict and denoise
            pred_noise = model(img, t, lr_upscaled)
            img = scheduler.sample_prev_timestep(img, t, t_prev, pred_noise, eta=ddim_eta)

    else:
        # DDPM: Must use all training steps
        timesteps = list(range(0, scheduler.timesteps))[::-1]

        # DDPM sampling loop
        for t_idx in tqdm(timesteps, desc=f"DDPM ({len(timesteps)} steps)", leave=False):
            t = torch.full((img.shape[0],), t_idx, device=device, dtype=torch.long)

            # Predict and denoise
            pred_noise = model(img, t, lr_upscaled)
            img = scheduler.sample_prev_timestep(img, t, pred_noise)

    model.train()

    # Create comparison grid: [LR_upscaled | Model_SR | GT_HR]
    comparison = torch.cat([lr_upscaled, img, hr_batch], dim=0)
    return comparison


def train_one_epoch(model, train_loader, diffusion_scheduler, optimizer, loss_fn,
                    scaler, ema, device, use_amp, epoch, lr_scheduler=None, grad_clip=1.0):
    """
    Train the diffusion model for one epoch with:
    - Mixed precision (torch.amp)
    - EMA updates
    - Robust checkpointing
    - Memory-safe gradient accumulation
    """
    model.train()
    total_loss = 0.0
    step_global = epoch * len(train_loader)

    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=True)

    for step, (lr_batch, hr_batch) in enumerate(loop):
        lr_batch = lr_batch.to(device, non_blocking=True)
        hr_batch = hr_batch.to(device, non_blocking=True)

        # Bicubic upscale for conditioning
        lr_upscaled = F.interpolate(
            lr_batch,
            scale_factor=config.SCALE,
            mode='bicubic',
            align_corners=False
        )

        # Random timesteps
        t = torch.randint(0, diffusion_scheduler.timesteps, (hr_batch.size(0),), device=device, dtype=torch.long)

        # Add noise
        noise = torch.randn_like(hr_batch)
        noisy_hr = diffusion_scheduler.add_noise(hr_batch, t, noise)

        optimizer.zero_grad(set_to_none=True)

        # === Forward pass with mixed precision ===
        with autocast(device_type=device, dtype=torch.float16 if use_amp else torch.float32):
            pred_noise = model(noisy_hr, t, lr_upscaled)
            loss = loss_fn(noise, pred_noise)

        # === Backprop ===
        scaler.scale(loss).backward()

        # Gradient clipping (important for stability with diffusion)
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        scaler.step(optimizer)
        scaler.update()

        # EMA update
        if ema:
            ema.update()

        total_loss += loss.item()
        loop.set_postfix(loss=f"{loss.item():.4f}")

        # Log training progress
        if step % 100 == 0:
            wandb.log({
                'train/loss': loss.item(),
                'train/step': step_global + step,
                'train/lr': optimizer.param_groups[0]['lr']
            })

        if step % 500 == 0:
            torch.cuda.empty_cache()

        # Periodic checkpointing (safe for long runs)
        # checkpoint_interval = 2000
        # if checkpoint_interval and (step + 1) % checkpoint_interval == 0:
        #     ckpt_path = CHECKPOINT_DIR / f"checkpoint_ep{epoch+1}_step{step+1}.pth"
        #     torch.save({
        #         'epoch': epoch + 1,
        #         'step': step + 1,
        #         'model_state_dict': model.state_dict(),
        #         'optimizer_state_dict': optimizer.state_dict(),
        #         'scaler_state_dict': scaler.state_dict(),
        #         'ema_state_dict': ema.shadow if ema else None,
        #         'lr_scheduler_state_dict': lr_scheduler.state_dict() if lr_scheduler else None
        #     }, ckpt_path)
        #     print(f"Intermediate checkpoint saved at step {step+1}")

    avg_loss = total_loss / len(train_loader)
    return avg_loss

def print_training_summary(model, scheduler, config, device, use_amp, num_train_samples):
    """Prints a structured summary of the model, config, and environment."""
    
    # Check if a model method exists to get architecture details
    model_arch_summary = "No architecture summary available."
    try:
        if hasattr(model, 'get_architecture_summary'):
            model_arch_summary = model.get_architecture_summary()
        elif hasattr(model, '__repr__'):
            model_arch_summary = str(model)
    except Exception as e:
        model_arch_summary = f"Error retrieving architecture summary: {e}"

    summary = []
    summary.append("=" * 80)
    summary.append(f"{'TRAINING SUMMARY':^80}")
    summary.append("=" * 80)

    # 1. Environment and Device
    summary.append(f"1. ENVIRONMENT & DEVICE:")
    summary.append(f"  Device: {str(device).upper()}")
    summary.append(f"  Mixed Precision (AMP): {'Enabled' if use_amp else 'Disabled'}")
    
    try:
        if device == 'cuda':
            total_memory = torch.cuda.get_device_properties(device).total_memory / (1024**3)
            summary.append(f"  GPU VRAM (Total): {total_memory:.2f} GiB")
        elif device == 'mps':
            total_memory = psutil.virtual_memory().total / (1024**3)
            summary.append(f"  Unified Memory (Total): {total_memory:.2f} GiB (Apple Silicon)")
    except ImportError:
        summary.append("  (Install `psutil` for memory summary: pip install psutil)")
    except Exception as e:
        summary.append(f"  Could not retrieve memory info: {e}")

    summary.append(f"  vCPUs (Workers): {config.NUM_WORKERS} / {os.cpu_count() or 4}")
    summary.append("-" * 80)
    
    # 2. Model & Data
    summary.append(f"2. MODEL & DATA CONFIG:")
    summary.append(f"  Model Parameters: {model.get_num_params():,}")
    summary.append(f"  Training Samples: {num_train_samples:,}")
    summary.append(f"  Patch Size (HR): {config.HR_CROP_SIZE}x{config.HR_CROP_SIZE}")
    summary.append(f"  Batch Size (Train): {config.BATCH_SIZE}")
    summary.append(f"  Learning Rate: {config.DIFFUSION_LR:.2e}")
    summary.append(f"  LR Scheduler: CosineAnnealingLR (T_max={config.DIFFUSION_EPOCHS})")
    summary.append(f"  EMA Enabled: {'Yes' if 'ema' in globals() and ema is not None else 'No'}")
    summary.append(f"  Grad Checkpoint: {config.DIFFUSION_GRAD_CHECKPOINT}")
    summary.append("-" * 80)

    # 3. Diffusion Schedule
    summary.append(f"3. DIFFUSION SCHEDULE:")
    summary.append(f"  Training Timesteps: {config.DIFFUSION_TIMESTEPS}")
    summary.append(f"  Beta Schedule: {config.DIFFUSION_SCHEDULE.capitalize()}")
    summary.append(f"  Sampling Method: {config.DIFFUSION_SCHEDULER_TYPE.upper()}")
    if config.DIFFUSION_SCHEDULER_TYPE == 'ddim':
        summary.append(f"  DDIM Inference Steps: {config.DIFFUSION_DDIM_STEPS}")
        summary.append(f"  DDIM Eta: {config.DIFFUSION_DDIM_ETA}")
    summary.append("-" * 80)

    # 4. MODEL ARCHITECTURE
    if config.DISPLAY_MODEL_ARCH:
        summary.append("4. MODEL ARCHITECTURE:")
        summary.append(model_arch_summary)
        summary.append("=" * 80)
    
    # Print the full summary
    print("\n".join(summary))

def main(args):
    # Initialize wandb
    wandb.init(
        project="SuperResolution-Diffusion",
        config=vars(config),
        resume="allow" if args.wandb_id else None,
        id=args.wandb_id,
        name=f"diffusion-{config.DIFFUSION_TIMESTEPS}steps"
    )

    if args.model_type:
        config.DIFFUSION_MODEL_TYPE = args.model_type

    # Device setup
    device = config.DEVICE
    use_amp = (device == 'cuda')
    print(f"Using device: {device}")
    print(f"Mixed precision: {use_amp}")

    # Select the train and val dataset dirs
    train_hr_dir = config.DATA_DIR / "train" / "HR"
    train_lr_dir = config.DATA_DIR / "train" / "LR"
    val_hr_dir = config.DATA_DIR / "val" / "HR"
    val_lr_dir = config.DATA_DIR / "val" / "LR"

    # Select the training dataset class in (TrainData)
    train_kwargs = {"hr_dir": train_hr_dir, "lr_dir": train_lr_dir}
    TrainData = TrainDatasetAugmented if config.TRAIN_AUGMENT_FACTOR > 1 else TrainDataset
    if config.TRAIN_AUGMENT_FACTOR > 1:
        train_kwargs["augment_factor"] = config.TRAIN_AUGMENT_FACTOR

    # Select the validation dataset class (ValData)
    val_kwargs = {"hr_dir": val_hr_dir, "lr_dir": val_lr_dir}
    if config.VAL_GRID_MODE:
        if config.VAL_SAMPLE_CENTER:
            ValData = ValDatasetCenterGrid
            val_kwargs["augment_factor"] = config.VAL_AUGMENT_FACTOR
        else:
            ValData = ValDatasetGrid
    else:
        ValData = ValDataset

    # Data Loaders
    train_dataset = TrainData(**train_kwargs)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=(device == "cuda"),
        persistent_workers=(config.NUM_WORKERS > 0),
        prefetch_factor=2 if config.NUM_WORKERS > 0 else None
    )

    val_dataset = ValData(**val_kwargs) # many patches per image
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=(device == "cuda")
    )
    val_dataset_samples = ValDataset(hr_dir=val_hr_dir, lr_dir=val_lr_dir) # 1 patch per image
    val_loader_samples = DataLoader(
        val_dataset_samples,
        batch_size=4,   # Get 4 unique images as requested. 
        shuffle=False,   # Shuffle to see different images each time you sample
        num_workers=2,
        pin_memory=(device == "cuda")
    )

    # Initialize model
    model_type = getattr(config, "DIFFUSION_MODEL_TYPE", None)

    # 2. Build the correct model with proper kwargs
    if model_type == "custom":
        model = DiffusionUNet(
            in_channels=6,  # (3 noisy + 3 LR conditional)
            out_channels=3,
            features=config.DIFFUSION_FEATURES,
            time_emb_dim=config.DIFFUSION_TIME_EMB_DIM,
            dropout=0.1,
            grad_ckpt=config.DIFFUSION_GRAD_CHECKPOINT
        )
    elif model_type == "default":
        model = DiffusionUNetLite(
            in_channels=6,
            out_channels=3
        )
    else:
        raise ValueError(f"Unknown DIFFUSION_MODEL_TYPE '{model_type}'. Valid options: ['custom', 'default'].")

    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"-- Model type --: {model_type}")
    print(f"Model parameters: {total_params:,}")
    wandb.config.update({
        "model_type": model_type,
        "model_parameters": total_params
    }, allow_val_change=True)

    print(f"Model get_num_params: {model.get_num_params():,}")

    # Initialize scheduler based on config
    scheduler = create_scheduler(
        scheduler_type=config.DIFFUSION_SCHEDULER_TYPE,
        timesteps=config.DIFFUSION_TIMESTEPS,
        schedule=config.DIFFUSION_SCHEDULE,
        device=device
    )

    print(f"-- Scheduler --: {scheduler}")
    print(f"Sampling method: {config.DIFFUSION_SCHEDULER_TYPE.upper()}")
    if config.DIFFUSION_SCHEDULER_TYPE == 'ddim':
        print(f"DDIM inference steps: {config.DIFFUSION_DDIM_STEPS}")
        print(f"DDIM eta: {config.DIFFUSION_DDIM_ETA}")

    # Initialize EMA
    ema = EMA(model, decay=0.9999) if args.use_ema else None
    if ema:
        print("EMA enabled")

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
            if 'scheduler_state_dict' in checkpoint:
                lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            else:
                print("No scheduler state found in checkpoint. Reinitializing scheduler.")

            if ema and 'ema_state_dict' in checkpoint:
                ema.shadow = checkpoint['ema_state_dict']
            else:
                print("No EMA state found in checkpoint.")

            start_epoch = checkpoint['epoch']
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            print(f"   Resumed from epoch {start_epoch}")
            print(f"   Best validation loss: {best_val_loss:.6f}")
        else:
            print("No checkpoint found, starting from scratch")

    # Print Overview
    num_train_patches = len(train_dataset)
    num_val_patches = len(val_dataset)
    total_patches = num_train_patches + num_val_patches
    train_ratio = num_train_patches / total_patches
    val_ratio = num_val_patches / total_patches

    print_training_summary(model, scheduler, config, device, use_amp, num_train_patches)

    print(f"\n{'='*80}")
    print(f"DATASET PATCH SPLIT SUMMARY")
    print(f"{'-'*80}")
    print(f"  Training patches (random crops):     {num_train_patches:,}")
    print(f"  Validation patches (deterministic):  {num_val_patches:,}")
    print(f"  Ratio (train : val):                 {train_ratio:.2f} : {val_ratio:.2f}")
    print(f"{'='*80}\n")

    # Training loop
    print("\n" + "="*60)
    print("Starting Diffusion Model Training")
    print("="*60 + "\n")

    for epoch in range(start_epoch, config.DIFFUSION_EPOCHS):
        # Train one epoch
        avg_train_loss = train_one_epoch(
            model, train_loader, scheduler, optimizer, loss_fn,
            scaler, ema, device, use_amp, epoch, lr_scheduler=lr_scheduler
        )

        # Validate
        if (epoch + 1) % config.CHECKPOINT_INTERVAL == 0 or epoch == 0:
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
                'val/epoch': epoch + 1,
                'val/avg_train_loss': avg_train_loss,
                'val/avg_val_loss': val_metrics['val_loss'],
                'val/bicubic_mse': val_metrics['bicubic_mse'],
                'val/model_mse': val_metrics['model_mse'],
                'val/improvement_pct': val_metrics['improvement_pct'],
                # 'train/lr': lr_scheduler.get_last_lr()[0]
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
                print(f" New best model saved! Loss: {best_val_loss:.6f}")

            # Generate sample images
            if (epoch + 1) % config.SAMPLE_INTERVAL == 0:
                print(" Generating sample images...")
                if ema:
                    ema.apply_shadow()

                sample_grid = sample_images(
                    model, scheduler, val_loader_samples, device,
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
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        choices=["default", "custom"],
        help="Override config to select model type"
    )


    args = parser.parse_args()
    main(args)