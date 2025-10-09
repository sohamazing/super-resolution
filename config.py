# config.py
from pathlib import Path
from dataclasses import dataclass, field
import torch
import os
import sys

class SuperResConfig:
    """A single source of truth for all model and training parameters."""
    
    # --- Device and Data Paths ---
    TRAIN_VAL_SOURCES: list = field(default_factory=lambda: [
        Path.home() / "Desktop" / "Photos" / "Flickr2K" / "Flickr2K_HR",
        Path.home() / "Desktop" / "Photos" / "Div2K" / "DIV2K_train_HR",
    ])
    TEST_SOURCES: list = field(default_factory=lambda: [
        Path.home() / "Desktop" / "Photos" / "Div2K" / "DIV2K_valid_HR"
    ])
    DATA_DIR: Path = Path("div2K-flickr2K-data") # relative local path 
    # DATA_DIR: Path = Path("/Users/soham/Documents/super-res/div2K-flickr2K-data") # absolute path
    # DATA_DIR: Path = Path("/Volumes/LaCie/SuperResolution/div2K-flickr2K-data") # for external drive
    SUPPORTED_EXTENSIONS: tuple = (".png", ".jpg", ".jpeg", ".heic", ".heif", ".dng", ".cr2", ".arw")
    VAL_SPLIT: float = 0.1

    # --- Data Processing Parameters (for create_patches.py) ---
    PATCH_SIZE: int = 256
    STEP: int = 128
    SCALE: int = 4
    HR_CROP_SIZE = PATCH_SIZE
    LR_CROP_SIZE = PATCH_SIZE // SCALE
    SUPPORTED_EXTENSIONS: tuple = (".png", ".jpg", ".jpeg", ".heic", ".heif", ".dng", ".cr2", ".arw")
    
    # --- Common Training Parameters ---
    DEVICE: str = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    NUM_WORKERS: int = 0 if sys.platform == "darwin" else os.cpu_count() or 4
    BATCH_SIZE: int = 16 # 8 
    PRETRAIN_EPOCHS: int = 20

    # --- ESRGAN Specific ---
    ESRGAN_EPOCHS: int = 2000
    ESRGAN_LR: float = 2e-4 # 1e-4 
    LAMBDA_L1: float = 1.0
    LAMBDA_ADV: float = 5e-3
    LAMBDA_PERCEP: float = 1.0

    # --- Diffusion Specific ---
    DIFFUSION_EPOCHS: int = 2000 # 1000
    DIFFUSION_LR: float = 1e-4
    TIMESTEPS: int = 1000

    # --- SwinIR Specific ---
    SWIN_EMBED_DIM: int = 180
    SWIN_NUM_LAYERS: int = 4 # Number of RSTL layers + Fusion Blocks.
    SWIN_NUM_BLOCKS_PER_LAYER: int = 6 # Number of Swin blocks within each RSTL.
    SWIN_NUM_HEADS: int = 6
    SWIN_WINDOW_SIZE: int = 8
    SWIN_LR: float = 2e-4
    SWIN_EPOCHS: int = 1000

    # --- Fused-GAN Specific ---
    FUSEDGAN_EPOCHS: int = 1000
    FUSEDGAN_LR: float = 1e-4

# Create an instance for easy importing across the project
config = SuperResConfig()
