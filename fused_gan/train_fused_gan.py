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

# --- Project Structure Setup ---
# Add the project's root directory to the Python path to allow for module imports.
SCRIPT_DIR = Path(__file__).parent.absolute()
sys.path.append(str(SCRIPT_DIR.parent))

# --- Import all our custom components ---
from fused_gan.generator_hcast import HCASTGenerator
from fused_gan.discriminator_unet import DiscriminatorUNet
from utils.loss import FusedGANLoss
from utils.datasets import SuperResDataset
from config import config

# Define the directory for saving model checkpoints.
CHECKPOINT_DIR = SCRIPT_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

class EMA:
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
                    self.decay * self.shadow[name].data +
                    (1 - self.decay) * param.data
                )

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]


def pretrain_generator(gen, loader, opt_g, scaler_g, autocast_context):
    """
    Pre-trains the generator using only L1 pixel loss to provide a stable
    starting point before introducing the complexity of adversarial training.
    """
    l1_loss = nn.L1Loss()
    print("--- Starting Generator Pre-training (L1 Loss only) ---")
    for epoch in range(config.PRETRAIN_EPOCHS):
        loop = tqdm(loader, desc=f"Pre-train Epoch {epoch+1}/{config.PRETRAIN_EPOCHS}")
        for i, (lr, hr) in enumerate(loop):
            lr, hr = lr.to(config.DEVICE), hr.to(config.DEVICE)

            # Use Automatic Mixed Precision for faster computation.
            with autocast_context:
                fake_hr = gen(lr)
                # Calculate the direct pixel-wise difference.
                loss = l1_loss(fake_hr, hr)

            # Standard backpropagation for the generator.
            opt_g.zero_grad(set_to_none=True)
            opt_d.zero_grad(set_to_none=True)
            scaler_g.step(opt_g)
            scaler_g.update()

            loop.set_postfix(l1_loss=loss.item())
            if i % 200 == 0:
                wandb.log({"pretrain_l1_loss": loss.item()})

def train_one_epoch(gen, disc, loader, opt_g, opt_d, loss_fn, scaler_g, scaler_d, autocast_context):
    """A single training loop for one epoch of alternating GAN training."""
    loop = tqdm(loader, leave=True)

    for i, (lr, hr) in enumerate(loop):
        # lr = lr.to(config.DEVICE)
        # hr = hr.to(config.DEVICE)
        lr = lr.to(config.DEVICE, non_blocking=True)
        hr = hr.to(config.DEVICE, non_blocking=True)

        with autocast_context:
            fake_hr = gen(lr)

        # --- Phase 1: Train the Discriminator ---
        # The goal is to teach the discriminator to distinguish real images from fake ones.
        with autocast_context:
            disc_real = disc(hr)
            # We .detach() the fake_hr so that gradients from the discriminator's loss
            # do NOT flow backward into the generator. We only want to update the discriminator here.
            disc_fake = disc(fake_hr.detach())
            d_loss = loss_fn.calculate_d_loss(disc_real, disc_fake)

        # Update discriminator weights.
        opt_d.zero_grad()
        scaler_d.scale(d_loss).backward()
        scaler_d.unscale_(opt_d)
        torch.nn.utils.clip_grad_norm_(disc.parameters(), max_norm=1.0)
        scaler_d.step(opt_d)
        scaler_d.update()

        # --- Phase 2: Train the Generator ---
        # The goal is to teach the generator to fool the discriminator.
        with autocast_context:
            # Run the discriminator again on the fake image, but this time we DO NOT detach it.
            # We need the gradients to flow all the way back to the generator.
            disc_fake_for_g = disc(fake_hr)
            # Calculate the generator's combined loss (adversarial + pixel + perceptual).
            g_loss, pix_loss, adv_loss, percep_loss = loss_fn.calculate_g_loss(
                disc_fake_for_g, fake_hr, hr
            )

        # Update generator weights.
        opt_g.zero_grad(set_to_none=True)
        scaler_g.scale(g_loss).backward()
        scaler_g.unscale_(opt_g)
        torch.nn.utils.clip_grad_norm_(gen.parameters(), max_norm=1.0)  # ✅ ADD
        scaler_g.step(opt_g)
        scaler_g.update()

        # Periodically log the individual loss components for detailed monitoring.
        if i % 100 == 0:
            wandb.log({
                "d_loss": d_loss.item(), "g_loss": g_loss.item(), "pixel_loss": pix_loss.item(),
                "adversarial_loss": adv_loss.item(), "perceptual_loss": percep_loss.item(),
            })

        loop.set_postfix(d_loss=d_loss.item(), g_loss=g_loss.item())

def main(args):
    # Initialize Weights & Biases for experiment tracking and visualization.
    wandb.init(project="SuperResolution-FusedGAN", config=vars(config), resume="allow", id=args.wandb_id)

    # Setup data loaders. pin_memory=True can speed up data transfer to the GPU.
    train_dataset = SuperResDataset(config.DATA_DIR / "train" / "HR", config.DATA_DIR / "train" / "LR")
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=(config.DEVICE == "cuda"),
        persistent_workers=True,
        prefetch_factor=2
    )
    val_loader = DataLoader(SuperResDataset(config.DATA_DIR / "val" / "HR", config.DATA_DIR / "val" / "LR"), batch_size=1)

    # Initialize our custom Generator and Discriminator models.
    gen = HCASTGenerator().to(config.DEVICE)
    disc = DiscriminatorUNet().to(config.DEVICE)
    loss_fn = FusedGANLoss()

    # GANs require two separate optimizers, one for each model.
    opt_g = optim.AdamW(gen.parameters(), lr=config.FUSEDGAN_LR, weight_decay=1e-4)
    opt_d = optim.AdamW(disc.parameters(), lr=config.FUSEDGAN_LR, weight_decay=1e-4)

    # For mixed-precision training.
    if config.DEVICE == "cuda":
        scaler_g = torch.cuda.amp.GradScaler()
        scaler_d = torch.cuda.amp.GradScaler()
        autocast_context = torch.cuda.amp.autocast()
    else: # No CUDA.
        class DummyScaler:
            def scale(self, loss): return loss
            def step(self, optimizer): optimizer.step()
            def update(self): pass
        scaler_g = DummyScaler()
        scaler_d = DummyScaler()
        autocast_context = contextlib.nullcontext()

    #
    ema = EMA(gen, decay=0.999)
    ema.register()

    # --- Pre-training & Resuming Logic ---
    pretrained_file = CHECKPOINT_DIR / "generator_pretrained.pth"
    start_epoch = 0

    if args.resume_epoch > 0:
        # If resuming, load all model and optimizer states to continue seamlessly.
        start_epoch = args.resume_epoch
        print(f"--- Resuming GAN training from Epoch {start_epoch} ---")
        gen.load_state_dict(torch.load(CHECKPOINT_DIR / f"generator_epoch_{start_epoch}.pth"))
        disc.load_state_dict(torch.load(CHECKPOINT_DIR / f"discriminator_epoch_{start_epoch}.pth"))
        # You would also load optimizer states here in a full implementation.
    elif args.mode in ["train", "all"]:
        # If not resuming, check for a pre-trained generator.
        if pretrained_file.exists():
            print(f"--- Found pretrained generator '{pretrained_file}', loading weights. ---")
            gen.load_state_dict(torch.load(pretrained_file, map_location=config.DEVICE))
        elif args.mode == "train":
            print("--- WARNING: No pretrained model found. Starting GAN training from scratch. ---")

    # Conditionally run the pre-training phase.
    if args.mode in ["pretrain", "all"] and not pretrained_file.exists() and not args.resume_epoch > 0:
        pretrain_generator(gen, train_loader, opt_g, scaler_g, autocast_context)
        torch.save(gen.state_dict(), pretrained_file)
        print(f"--- Pre-training finished. Model saved to '{pretrained_file}' ---")

    # Conditionally run the main GAN training phase.
    if args.mode in ["train", "all"]:
        print("\n--- Starting Main Fused-GAN Training ---")
        for epoch in range(start_epoch, config.FUSEDGAN_EPOCHS):
            train_one_epoch(gen, disc, train_loader, opt_g, opt_d, loss_fn, scaler_g, scaler_d, autocast_context)
            scheduler_g.step()
            scheduler_d.step()

            # Periodically save model checkpoints.
            if (epoch + 1) % 5 == 0:
                torch.save(gen.state_dict(), CHECKPOINT_DIR / f"generator_epoch_{epoch+1}.pth")
                torch.save(disc.state_dict(), CHECKPOINT_DIR / f"discriminator_epoch_{epoch+1}.pth")

                # Generate and log a validation image sample to wandb for visual feedback.
                gen.eval()
                with torch.no_grad():
                    lr, hr = next(iter(val_loader))
                    lr, hr = lr.to(config.DEVICE), hr.to(config.DEVICE)
                    fake_hr = gen(lr)
                    grid = torchvision.utils.make_grid(torch.cat([fake_hr, hr], dim=0), normalize=True)
                    wandb.log({"validation_samples": wandb.Image(grid, caption=f"Epoch {epoch+1}")})
                gen.train()

if __name__ == "__main__":
    # A professional argument parser to control the script's behavior.
    parser = argparse.ArgumentParser(description="Train the Fused-GAN model.")
    parser.add_argument("--mode", type=str, default="all", choices=["pretrain", "train", "all"], help="Set training mode: pretrain only, train only, or all.")
    parser.add_argument("--resume_epoch", type=int, default=0, help="Epoch to resume GAN training from (skips pre-training).")
    parser.add_argument("--wandb_id", type=str, default=None, help="Weights & Biases run ID to resume logging.")
    args = parser.parse_args()
    main(args)