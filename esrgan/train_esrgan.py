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

# Now these imports will work
from esrgan.generator import GeneratorRRDB
from esrgan.discriminator import Discriminator

## --- CONFIGURATION --- ##
from dataclasses import dataclass, field
@dataclass
class Config:
    # This single line correctly checks for CUDA, then MPS, then CPU
    DEVICE: str = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    
    # Absolute path to the shared dataset
    DATA_DIR: Path = Path("/Users/soham/Documents/super-res/div2K-flickr2K-data")
    # DATA_DIR: Path = Path("/Volumes/LaCie/SuperResolution/div2K-flickr2K-data")

    # Directory to save checkpoints, anchored to the script's location
    CHECKPOINT_DIR: Path = SCRIPT_DIR 

    NUM_EPOCHS: int = 2000
    PRETRAIN_EPOCHS: int = 20
    SCALE: int = 4
    BATCH_SIZE: int = 16
    LEARNING_RATE: float = 2e-4
    LAMBDA_L1: float = 1.0
    LAMBDA_ADV: float = 5e-3
    LAMBDA_PERCEP: float = 1.0
config = Config()
## --------------------- ##

class VGGLoss(nn.Module):
    """Calculates the VGG Perceptual Loss."""
    def __init__(self):
        super().__init__()
        vgg = vgg19(weights='IMAGENET1K_V1').features[:36].eval().to(config.DEVICE)
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg
        self.loss = nn.L1Loss()

    def forward(self, input, target):
        return self.loss(self.vgg(input), self.vgg(target))

class ImageDataset(Dataset):
    """Custom PyTorch Dataset for loading our patches."""
    def __init__(self, hr_dir, lr_dir):
        self.hr_paths = sorted(list(Path(hr_dir).glob("*.png")))
        self.lr_paths = sorted(list(Path(lr_dir).glob("*.png")))
        self.transform = ToTensor()

    def __len__(self):
        return len(self.hr_paths)

    def __getitem__(self, index):
        hr_image = Image.open(self.hr_paths[index]).convert("RGB")
        lr_image = Image.open(self.lr_paths[index]).convert("RGB")
        return self.transform(lr_image), self.transform(hr_image)

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
    # --- Weights & Biases Setup ---
    # Moved inside main() to prevent issues with multiprocessing
    wandb.init(project="SuperResolution-ESRGAN", config=vars(config), resume="allow", id=args.wandb_id)
    
    # Set num_workers=0 for macOS to ensure stability and clean logs
    train_dataset = ImageDataset(config.DATA_DIR / "train" / "HR", config.DATA_DIR / "train" / "LR")
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0) # macOS 
    # train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=4) 

    generator = GeneratorRRDB().to(config.DEVICE)
    discriminator = Discriminator().to(config.DEVICE)
    optimizer_g = optim.Adam(generator.parameters(), lr=config.LEARNING_RATE, betas=(0.9, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=config.LEARNING_RATE, betas=(0.9, 0.999))
    
    # --- Resume from checkpoint logic (made more robust) ---
    start_epoch = 0
    if args.resume_epoch > 0:
        start_epoch = args.resume_epoch
        print(f"--- Resuming GAN training from Epoch {start_epoch} ---")
        gen_path = config.CHECKPOINT_DIR / f"generator_epoch_{start_epoch}.pth"
        disc_path = config.CHECKPOINT_DIR / f"discriminator_epoch_{start_epoch}.pth"
        generator.load_state_dict(torch.load(gen_path, map_location=config.DEVICE))
        discriminator.load_state_dict(torch.load(disc_path, map_location=config.DEVICE))
        
        # Safely try to load optimizer states
        try:
            opt_g_path = config.CHECKPOINT_DIR / f"optimizer_g_epoch_{start_epoch}.pth"
            opt_d_path = config.CHECKPOINT_DIR / f"optimizer_d_epoch_{start_epoch}.pth"
            optimizer_g.load_state_dict(torch.load(opt_g_path))
            optimizer_d.load_state_dict(torch.load(opt_d_path))
            print("Optimizer states loaded successfully.")
        except FileNotFoundError:
            print("Optimizer state files not found. Initializing optimizers from scratch.")

    scheduler_g = optim.lr_scheduler.CosineAnnealingLR(optimizer_g, T_max=config.NUM_EPOCHS, eta_min=1e-7, last_epoch=start_epoch - 1)
    scheduler_d = optim.lr_scheduler.CosineAnnealingLR(optimizer_d, T_max=config.NUM_EPOCHS, eta_min=1e-7, last_epoch=start_epoch - 1)
    
    l1_loss = nn.L1Loss()
    vgg_loss = VGGLoss()
    adversarial_loss = nn.BCEWithLogitsLoss()

    # --- Pre-training Phase ---
    pretrained_file = config.CHECKPOINT_DIR / "generator_pretrained.pth"
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

    # --- Main GAN Training Loop ---
    if args.mode == "train" or args.mode == "all":
        if args.resume_epoch == 0:
            print("--- Loading Pre-trained Generator Weights for GAN Training ---")
            generator.load_state_dict(torch.load(pretrained_file, map_location=config.DEVICE))
            
        print("--- Starting Main GAN Training ---")
        for epoch in range(start_epoch, config.NUM_EPOCHS):
            print(f"\n--- Main Training Epoch {epoch+1}/{config.NUM_EPOCHS} ---")
            train_one_epoch(generator, discriminator, train_loader, optimizer_g, optimizer_d, l1_loss, vgg_loss, adversarial_loss)
            scheduler_g.step()
            scheduler_d.step()

            if (epoch + 1) % 5 == 0:
                print("...Saving models and logging validation images...")
                torch.save(generator.state_dict(), config.CHECKPOINT_DIR / f"generator_epoch_{epoch+1}.pth")
                torch.save(discriminator.state_dict(), config.CHECKPOINT_DIR / f"discriminator_epoch_{epoch+1}.pth")
                torch.save(optimizer_g.state_dict(), config.CHECKPOINT_DIR / f"optimizer_g_epoch_{epoch+1}.pth")
                torch.save(optimizer_d.state_dict(), config.CHECKPOINT_DIR / f"optimizer_d_epoch_{epoch+1}.pth")
                
                val_dataset = ImageDataset(config.DATA_DIR / "val" / "HR", config.DATA_DIR / "val" / "LR")
                if len(val_dataset) > 0:
                    generator.eval()
                    with torch.no_grad():
                        lr, hr = val_dataset[0]
                        lr_batch = lr.unsqueeze(0).to(config.DEVICE)
                        fake_hr = generator(lr_batch)
                        bicubic_hr = torch.nn.functional.interpolate(lr_batch, scale_factor=config.SCALE, mode='bicubic', align_corners=False)
                        
                        c, h, w = fake_hr.shape[1:]
                        hr_cropped = hr.unsqueeze(0)[:, :, :h, :w].to(config.DEVICE)

                        image_grid = torch.cat([bicubic_hr, fake_hr, hr_cropped], dim=0)
                        grid = torchvision.utils.make_grid(image_grid, normalize=True)
                        wandb.log({"validation_grid": wandb.Image(grid)})
                    generator.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Super-Resolution GAN.")
    parser.add_argument("--mode", type=str, default="all", choices=["pretrain", "train", "all"])
    parser.add_argument("--resume_epoch", type=int, default=0, help="Epoch to resume GAN training from.")
    parser.add_argument("--wandb_id", type=str, default=None, help="Weights & Biases run ID to resume logging.")
    args = parser.parse_args()
    main(args)