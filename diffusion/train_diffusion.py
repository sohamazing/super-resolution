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
    Validate the model's noise prediction performance on the validation set.
    This is the core training objective.
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0

    for lr_batch, hr_batch in tqdm(val_loader, desc="Validating (Noise Loss)", leave=False):
        lr_batch = lr_batch.to(device)
        hr_batch = hr_batch.to(device)

        with autocast(device_type=device, enabled=use_amp):
            # Random timestep for validation
            t = torch.randint(1, scheduler.timesteps, (hr_batch.shape[0],), device=device)

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

            # Calculate the training loss (MSE on noise)
            loss = loss_fn(pred_noise, noise)
            total_loss += loss.item()
            num_batches += 1

    model.train()
    avg_loss = total_loss / num_batches
    return {'val_noise_loss': avg_loss}


@torch.no_grad()
def sample_and_evaluate(model, scheduler, val_loader_samples, device, use_amp,
                        num_inference_steps=None, ddim_eta=0.0):
    """
    Generate SR images using the full denoising process and evaluate their quality
    against the ground truth and a bicubic baseline.
    """
    model.eval()

    # Get one batch for sampling
    lr_batch, hr_batch = next(iter(val_loader_samples))
    lr_batch = lr_batch.to(device)
    hr_batch = hr_batch.to(device)

    # Upscale LR for conditioning and as a baseline
    lr_upscaled = F.interpolate(
        lr_batch,
        scale_factor=config.SCALE,
        mode='bicubic',
        align_corners=False
    )

    # Start the denoising process from pure noise
    img = torch.randn_like(hr_batch)

    # --- Full Denoising Loop (DDIM or DDPM) ---
    is_ddim = isinstance(scheduler, DDIMScheduler)
    if is_ddim:
        if num_inference_steps is None:
            num_inference_steps = config.DIFFUSION_DDIM_STEPS
        timesteps = scheduler.get_sampling_timesteps(num_inference_steps)
        desc = f"DDIM Sampling ({num_inference_steps} steps)"
    else:
        timesteps = list(range(0, scheduler.timesteps))[::-1]
        desc = f"DDPM Sampling ({len(timesteps)} steps)"

    # Wrap the loop in autocast for mixed precision inference
    with autocast(device_type=device, enabled=use_amp):
        for i, t_idx in enumerate(tqdm(timesteps, desc=desc, leave=False)):
            t = torch.full((img.shape[0],), t_idx, device=device, dtype=torch.long)
            pred_noise = model(img, t, lr_upscaled)

            if is_ddim:
                t_prev = torch.full((img.shape[0],), timesteps[i+1], device=device, dtype=torch.long) if i < len(timesteps) - 1 else torch.full((img.shape[0],), -1, device=device, dtype=torch.long)
                img = scheduler.sample_prev_timestep(img, t, t_prev, pred_noise, eta=ddim_eta)
            else:
                img = scheduler.sample_prev_timestep(img, t, pred_noise)
    # --- End of Loop ---

    # --- Calculate Meaningful Metrics ---
    # Ensure metrics are calculated in float32 for precision
    bicubic_mse = F.mse_loss(lr_upscaled.float(), hr_batch.float())
    model_mse = F.mse_loss(img.float(), hr_batch.float())
    improvement = ((bicubic_mse.item() - model_mse.item()) / bicubic_mse.item()) * 100 if bicubic_mse.item() > 0 else 0.0

    # --- Create Visualization Grid ---
    lr_upscaled_vis = denormalize(lr_upscaled)
    img_vis = denormalize(img)
    hr_batch_vis = denormalize(hr_batch)
    comparison_grid = torch.cat([lr_upscaled_vis, img_vis, hr_batch_vis], dim=0)
    
    model.train()

    return {
        'comparison_grid': comparison_grid,
        'bicubic_mse': bicubic_mse.item(),
        'model_mse': model_mse.item(),
        'improvement_pct': improvement
    }


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """
    Converts [-1, 1] range tensors back to [0, 1] for visualization or saving.
    Supports batched tensors too.
    """
    return tensor.mul(0.5).add(0.5).clamp(0, 1)


def train_one_epoch(model, train_loader, diffusion_scheduler, optimizer, loss_fn,
                    scaler, ema, device, use_amp, epoch, lr_scheduler=None, grad_clip=1.0):
    """
    Train the diffusion model for one epoch.
    """
    model.train()
    total_loss = 0.0
    step_global = epoch * len(train_loader)
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=True)

    for step, (lr_batch, hr_batch) in enumerate(loop):
        lr_batch = lr_batch.to(device, non_blocking=True)
        hr_batch = hr_batch.to(device, non_blocking=True)

        with torch.no_grad():
            lr_upscaled = F.interpolate(
                lr_batch,
                scale_factor=config.SCALE,
                mode='bicubic',
                align_corners=False
            )

        t = torch.randint(0, diffusion_scheduler.timesteps, (hr_batch.size(0),), device=device, dtype=torch.long)
        noise = torch.randn_like(hr_batch)
        noisy_hr = diffusion_scheduler.add_noise(hr_batch, t, noise)

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type=device, dtype=torch.float16 if use_amp else torch.float32):
            pred_noise = model(noisy_hr, t, lr_upscaled)
            loss = loss_fn(noise, pred_noise)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        if ema:
            ema.update()

        total_loss += loss.item()
        loop.set_postfix(loss=f"{loss.item():.4f}")

        if step % 100 == 0:
            wandb.log({
                'train/loss': loss.item(),
                'train/step': step_global + step,
                'train/lr': optimizer.param_groups[0]['lr']
            })
    
    return total_loss / len(train_loader)


def print_training_summary(model, scheduler, config, device, use_amp, num_train_samples, num_val_samples):
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
    summary.append(f"  Patch Size (HR): {config.HR_CROP_SIZE}x{config.HR_CROP_SIZE}")
    summary.append(f"  Training Samples: {num_train_samples:,}")
    summary.append(f"  Validation Samples: {num_val_samples:,}")
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
    wandb.init(
        project="SuperResolution-Diffusion",
        config=vars(config),
        resume="allow" if args.wandb_id else None,
        id=args.wandb_id,
        name=f"diffusion-{config.DIFFUSION_TIMESTEPS}steps"
    )

    if args.model_type:
        config.DIFFUSION_MODEL_TYPE = args.model_type

    device = config.DEVICE
    use_amp = (device == 'cuda')
    print(f"Using device: {device}")
    print(f"Mixed precision: {use_amp}")

    train_hr_dir = config.DATA_DIR / "train" / "HR"
    train_lr_dir = config.DATA_DIR / "train" / "LR"
    val_hr_dir = config.DATA_DIR / "val" / "HR"
    val_lr_dir = config.DATA_DIR / "val" / "LR"

    train_kwargs = {"hr_dir": train_hr_dir, "lr_dir": train_lr_dir}
    TrainData = TrainDatasetAugmented if config.TRAIN_AUGMENT_FACTOR > 1 else TrainDataset
    if config.TRAIN_AUGMENT_FACTOR > 1:
        train_kwargs["augment_factor"] = config.TRAIN_AUGMENT_FACTOR

    val_kwargs = {"hr_dir": val_hr_dir, "lr_dir": val_lr_dir}
    ValData = ValDatasetGrid if config.VAL_GRID_MODE else ValDataset
    if config.VAL_GRID_MODE: # uses patch size and step size to make grid ()
        if config.VAL_SAMPLE_CENTER:
            ValData = ValDatasetCenterGrid # augment_factor patches from a cenetered grid 
            val_kwargs["augment_factor"] = config.VAL_AUGMENT_FACTOR

    train_dataset = TrainData(**train_kwargs)
    train_loader = DataLoader(train_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=(device == "cuda"),
        persistent_workers=(config.NUM_WORKERS > 0),
        prefetch_factor=2 if config.NUM_WORKERS > 0 else None
    )

    val_dataset = ValData(**val_kwargs)
    val_loader = DataLoader(val_dataset, 
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=(device == "cuda")
    )
    
    val_dataset_samples = ValDataset(hr_dir=val_hr_dir, lr_dir=val_lr_dir)
    val_loader_samples = DataLoader(val_dataset_samples, 
        batch_size=4,
        shuffle=False,
        num_workers=2,
        pin_memory=(device=="cuda")
    )

    model_type = getattr(config, "DIFFUSION_MODEL_TYPE", "default")
    if model_type == "custom":
        model = DiffusionUNet(
            in_channels=6, out_channels=3, 
            features=config.DIFFUSION_FEATURES,
            time_emb_dim=config.DIFFUSION_TIME_EMB_DIM, 
            dropout=0.1,
            grad_ckpt=config.DIFFUSION_GRAD_CHECKPOINT
        )
    else:
        model = DiffusionUNetLite(in_channels=6, out_channels=3)

    model = model.to(device)
    print(f"-- Model type --: {model_type}")
    print(f"Model parameters: {model.get_num_params():,}")
    wandb.config.update({"model_type": model_type, "model_parameters": model.get_num_params()}, allow_val_change=True)

    scheduler = create_scheduler(
        scheduler_type=config.DIFFUSION_SCHEDULER_TYPE,
        timesteps=config.DIFFUSION_TIMESTEPS,
        schedule=config.DIFFUSION_SCHEDULE,
        device=device
    )
    print(f"-- Scheduler --: {scheduler}")

    ema = EMA(model, decay=0.9999) if args.use_ema else None
    if ema: print("EMA enabled")

    optimizer = optim.AdamW(model.parameters(), lr=config.DIFFUSION_LR, betas=(0.9, 0.999), weight_decay=1e-4)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.DIFFUSION_EPOCHS, eta_min=1e-7)
    loss_fn = nn.MSELoss()
    scaler = GradScaler(enabled=use_amp)

    start_epoch = 0
    best_val_loss = float('inf')
    if args.resume:
        checkpoint_path = CHECKPOINT_DIR / "checkpoint_latest.pth"
        if checkpoint_path.exists():
            print(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint: lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            if ema and 'ema_state_dict' in checkpoint: ema.shadow = checkpoint['ema_state_dict']
            start_epoch = checkpoint['epoch']
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            print(f"   Resumed from epoch {start_epoch}, Best val loss: {best_val_loss:.6f}")

    print_training_summary(model, scheduler, config, device, use_amp, len(train_dataset), len(val_dataset))

    print("\n" + "="*60 + "\nStarting Diffusion Model Training\n" + "="*60 + "\n")

    for epoch in range(start_epoch, config.DIFFUSION_EPOCHS):
        avg_train_loss = train_one_epoch(
            model, train_loader, scheduler, optimizer, loss_fn,
            scaler, ema, device, use_amp, epoch, lr_scheduler=lr_scheduler
        )

        if (epoch + 1) % config.CHECKPOINT_INTERVAL == 0:
            if ema: ema.apply_shadow()

            val_metrics = validate(model, val_loader, scheduler, loss_fn, device, use_amp)
            
            # --- Perform full evaluation and sampling on a subset of validation data ---
            eval_results = sample_and_evaluate(
                model, scheduler, val_loader_samples, device, use_amp,
                num_inference_steps=config.DIFFUSION_DDIM_STEPS
            )

            if ema: ema.restore()

            print(f"\n--- Epoch {epoch+1}/{config.DIFFUSION_EPOCHS} ---")
            print(f"  Avg Train Loss:       {avg_train_loss:.6f}")
            print(f"  Val Noise Loss:       {val_metrics['val_noise_loss']:.6f}")
            print(f"  Bicubic MSE (sample): {eval_results['bicubic_mse']:.6f}")
            print(f"  Model MSE (sample):   {eval_results['model_mse']:.6f}")
            print(f"  Improvement %:        {eval_results['improvement_pct']:.2f}%")
            print(f"  Learning Rate:        {lr_scheduler.get_last_lr()[0]:.2e}")

            wandb.log({
                'epoch': epoch + 1,
                'val/avg_train_loss': avg_train_loss,
                'val/avg_val_loss': val_metrics['val_noise_loss'],
                'visual/bicubic_mse': eval_results['bicubic_mse'],
                'visual/model_mse': eval_results['model_mse'],
                'visual/improvement_pct': eval_results['improvement_pct'],
                'visual/samples': wandb.Image(
                    torchvision.utils.make_grid(
                        eval_results['comparison_grid'],
                        nrow=eval_results['comparison_grid'].shape[0] // 3
                    ),
                    caption=f"Epoch {epoch+1} | [Top: Bicubic, Mid: Model, Btm: Ground Truth]"
                )
            })

            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': lr_scheduler.state_dict(),
                'val_loss': val_metrics['val_noise_loss'],
                'best_val_loss': best_val_loss
            }
            if ema: checkpoint['ema_state_dict'] = ema.shadow

            torch.save(checkpoint, CHECKPOINT_DIR / "checkpoint_latest.pth")

            if val_metrics['val_noise_loss'] < best_val_loss:
                best_val_loss = val_metrics['val_noise_loss']
                checkpoint['best_val_loss'] = best_val_loss
                torch.save(checkpoint, CHECKPOINT_DIR / "best_model.pth")
                print(f"  New best model saved! Val Noise Loss: {best_val_loss:.6f}")

        lr_scheduler.step()

    print("\n" + "="*60 + "\nTraining Complete!\n" + f"Best validation noise loss: {best_val_loss:.6f}\n" + "="*60 + "\n")
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Conditional Diffusion Model for Super-Resolution")
    parser.add_argument("--resume", action="store_true", help="Resume training from latest checkpoint")
    parser.add_argument("--wandb_id", type=str, default=None, help="W&B run ID for resuming")
    parser.add_argument("--use_ema", action="store_true", help="Use Exponential Moving Average")
    parser.add_argument("--model_type", type=str, default=None, choices=["default", "custom"], help="Override config to select model type")
    args = parser.parse_args()
    main(args)