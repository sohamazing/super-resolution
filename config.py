# config.py
from pathlib import Path
import torch
import os

class GlobalConfig:
    """A single source of truth for all model and training parameters."""
    
    # --- Project Structure ---

    # --- Device and Data Paths ---
    DEVICE: str = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    DATA_DIR: Path = Path("/Users/soham/Documents/super-res/div2K-flickr2K-data")
    # DATA_DIR: Path = Path("/Volumes/LaCie/SuperResolution/div2K-flickr2K-data") # Example for external drive

    # --- Data Processing Parameters (for create_patches.py) ---
    PATCH_SIZE: int = 256
    SCALE: int = 4
    STEP: int = 128
    VAL_SPLIT: float = 0.1
    SUPPORTED_EXTENSIONS: tuple = (".png", ".jpg", ".jpeg", ".heic", ".heif", ".dng", ".cr2", ".arw")
    NUM_WORKERS: int = os.cpu_count() or 4
    
    # --- Common Training Parameters ---
    BATCH_SIZE: int = 16
    
    # --- ESRGAN Specific ---
    ESRGAN_EPOCHS: int = 2000
    PRETRAIN_EPOCHS: int = 20
    ESRGAN_LR: float = 2e-4
    LAMBDA_L1: float = 1.0
    LAMBDA_ADV: float = 5e-3
    LAMBDA_PERCEP: float = 1.0

    # --- Diffusion Specific ---
    DIFFUSION_EPOCHS: int = 500
    DIFFUSION_LR: float = 1e-4
    TIMESTEPS: int = 1000

# Create an instance for easy importing across the project
config = GlobalConfig()
