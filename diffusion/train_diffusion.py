# diffusion/train_diffusion.py
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor
import torchvision.utils
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import wandb
import argparse
import sys
import os

# Path logic and imports at the top
SCRIPT_DIR = Path(__file__).parent.absolute()
sys.path.append(str(SCRIPT_DIR.parent)) # .../super-res/

from diffusion.diffusion_model import DiffusionUNet
from diffusion.scheduler import Scheduler
from utils.datasets import SuperResDataset
from config import config # Import the new central config

# Define and create a local checkpoint directory inside the diffusion folder
CHECKPOINT_DIR = SCRIPT_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

@torch.no_grad()
def sample_and_log_images(model, scheduler, loader, epoch):
    """Performs the full denoising loop to generate a sample image and logs it."""
    model.eval() # Set the model to evaluation mode
    
    # Get one sample from the validation set
    hr_sample, lr_sample = next(iter(loader))
    lr_sample = lr_sample.to(config.DEVICE)
    hr_sample = hr_sample.to(config.DEVICE)
    
    # Start the generation process from pure random noise of the target HR shape
    img = torch.randn_like(hr_sample).to(config.DEVICE)
    # Create a simple bicubic upscale of the LR image for visual comparison
    lr_upscaled = nn.functional.interpolate(lr_sample, scale_factor=config.SCALE, mode='bicubic', align_corners=False)

    # The reverse diffusion loop (from t=T-1 down to 0)
    for i in tqdm(reversed(range(0, scheduler.timesteps)), desc="Sampling", total=scheduler.timesteps, leave=False):
        t = torch.full((1,), i, device=config.DEVICE, dtype=torch.long)
        # Predict the noise in the current image
        predicted_noise = model(img, t, lr_upscaled)
        # Use the scheduler to take one step back towards a cleaner image
        img = scheduler.sample_previous_timestep(img, t, predicted_noise)

    # Crop the original HR image to match the output size, just in case of rounding errors
    c, h, w = img.shape[1:]
    hr_cropped = hr_sample[:, :, :h, :w]

    # Create a grid comparing the Bicubic upscale, our model's output, and the ground truth
    grid = torchvision.utils.make_grid(torch.cat([lr_upscaled, img, hr_cropped], dim=0), normalize=True)
    wandb.log({"validation_sample": wandb.Image(grid, caption=f"Epoch {epoch+1}")})
    
    model.train() # Set the model back to training mode

def main(args):
    # Initialize Weights & Biases for experiment tracking
    wandb.init(project="SuperResolution-Diffusion", config=vars(config), resume="allow", id=args.wandb_id)
    # Setup data loaders
    train_dataset = SuperResDataset(config.DATA_DIR / "train" / "HR", config.DATA_DIR / "train" / "LR")
    val_dataset = SuperResDataset(config.DATA_DIR / "val" / "HR", config.DATA_DIR / "val" / "LR")
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)

    # Initialize model, scheduler, optimizer, and loss function
    model = DiffusionUNet().to(config.DEVICE)
    scheduler = Scheduler(timesteps=config.TIMESTEPS)
    optimizer = optim.Adam(model.parameters(), lr=config.DIFFUSION_LR)
    loss_fn = nn.L1Loss() # L1 loss (Mean Absolute Error) is often more stable for image tasks
    
    start_epoch = 0
    # Check if resuming from a checkpoint
    if args.resume_epoch > 0:
        start_epoch = args.resume_epoch
        print(f"--- Resuming Diffusion training from Epoch {start_epoch} ---")
        model.load_state_dict(torch.load(CHECKPOINT_DIR / f"diffusion_model_epoch_{start_epoch}.pth"))
        try:
            optimizer.load_state_dict(torch.load(CHECKPOINT_DIR / f"optimizer_diffusion_epoch_{start_epoch}.pth"))
            print("Optimizer state loaded successfully.")
        except FileNotFoundError:
            print("Optimizer state not found. Initializing from scratch.")

    # A learning rate scheduler helps the model converge better
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.DIFFUSION_EPOCHS, eta_min=1e-7, last_epoch=start_epoch-1)

    print("--- Starting Diffusion Model Training ---")
    for epoch in range(start_epoch, config.DIFFUSION_EPOCHS):
        loop = tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{config.DIFFUSION_EPOCHS}")
        avg_loss = 0.0
        
        for i, (hr_batch, lr_batch) in enumerate(loop):
            hr_batch, lr_batch = hr_batch.to(config.DEVICE), lr_batch.to(config.DEVICE)
            
            # 1. Pick a random timestep 't' for each image in the batch
            t = torch.randint(0, scheduler.timesteps, (hr_batch.shape[0],), device=config.DEVICE).long()
            # 2. Create the ground-truth noise
            noise = torch.randn_like(hr_batch)
            # 3. Use the scheduler to create the noisy image for timestep 't'
            noisy_hr_batch = scheduler.add_noise(x_start=hr_batch, t=t, noise=noise)
            # 4. Create the LR image condition, upscaled to the target size
            lr_upscaled = nn.functional.interpolate(lr_batch, scale_factor=config.SCALE, mode='bicubic', align_corners=False)
            
            # 5. Get the model's prediction of the noise
            predicted_noise = model(noisy_hr_batch, t, lr_upscaled)
            
            # 6. Calculate the loss between the actual noise and the predicted noise
            loss = loss_fn(noise, predicted_noise)

            # 7. Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            avg_loss += loss.item()
            loop.set_postfix(loss=loss.item()) # Update the progress bar
        
        # Log metrics for the epoch and step the learning rate scheduler
        wandb.log({"epoch_loss": avg_loss / len(train_loader), "learning_rate": lr_scheduler.get_last_lr()[0]})
        lr_scheduler.step()

        # Save a checkpoint and log a visual sample every 25 epochs
        if (epoch + 1) % 25 == 0:
            print("...Saving model checkpoint and logging validation sample...")
            torch.save(model.state_dict(), CHECKPOINT_DIR / f"diffusion_model_epoch_{epoch+1}.pth")
            torch.save(optimizer.state_dict(), CHECKPOINT_DIR / f"optimizer_diffusion_epoch_{epoch+1}.pth")
            sample_and_log_images(model, scheduler, val_loader, epoch)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Super-Resolution Diffusion Model.")
    parser.add_argument("--resume_epoch", type=int, default=0, help="Epoch to resume training from.")
    parser.add_argument("--wandb_id", type=str, default=None, help="Weights & Biases run ID to resume logging.")
    args = parser.parse_args()
    main(args)

