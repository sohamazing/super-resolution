# config.py
import os
import sys
from pathlib import Path
from dataclasses import dataclass, field
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

@dataclass
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
    # DATA_DIR: Path = Path("/Users/soham/Documents/super-res/data") # absolute path
    # DATA_DIR: Path = Path("/Volumes/LaCie/SuperResolution/data") # for external drive
    SUPPORTED_EXTENSIONS: tuple = (".png", ".jpg", ".jpeg", ".heic", ".heif", ".dng", ".cr2", ".arw")
    VAL_SPLIT: float = 0.1

    # --- Data Processing Parameters ---
    PATCH_SIZE: int = 256
    STEP: int = 128
    SCALE: int = 4
    HR_CROP_SIZE: int = PATCH_SIZE
    LR_CROP_SIZE: int = PATCH_SIZE // SCALE
    SUPPORTED_EXTENSIONS: tuple = (".png", ".jpg", ".jpeg", ".heic", ".heif", ".dng", ".cr2", ".arw")

    # --- Common Training Parameters ---
    DEVICE: str = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    NUM_WORKERS: int = 0 if sys.platform == "darwin" else os.cpu_count() or 4
    BATCH_SIZE: int = 8 if sys.platform == "darwin" else 16
    CHECKPOINT_INTERVAL: int = 1 # in epochs
    PRETRAIN_EPOCHS: int = 10

    # --- Diffusion Specific ---
    DIFFUSION_EPOCHS: int = 1000
    DIFFUSION_LR: float = 2e-4
    DIFFUSION_TIMESTEPS: int = 1000
    DIFFUSION_TIME_EMB_DIM: int = 24 # larger = 32
    DIFFUSION_FEATURES: list = field(default_factory=lambda: [48, 96, 192]) # larger = [64, 128, 256]

    # --- ESRGAN Specific ---
    ESRGAN_EPOCHS: int = 1000
    ESRGAN_LR: float = 2e-4
    ESRGAN_NUM_FEATURES: int = 64
    ESRGAN_NUM_RRDB: int = 23

    # --- Fusion-SRGAN Specific ---
    FUSION_SRGAN_EPOCHS: int = 1000
    FUSION_SRGAN_LR: float = 2e-4 # 1e-4
    FUSION_SRGAN_GEN_FEATURES: list = field(default_factory=lambda: [48, 96, 128]) # larger = [64, 128, 256]
    FUSION_SRGAN_DIS_FEATURES: list = field(default_factory=lambda: [32, 64, 128, 256]) # larger = [64, 128, 256, 512]
    FUSION_SRGAN_EMBED_DIM: int = 180
    FUSION_SRGAN_NUM_HEADS: int = 6 # num attention heads
    FUSION_SRGAN_WINDOW_SIZE: int = 8
    FUSION_SRGAN_NUM_SWIN_BLOCKS: int = 4 # larger = 6
    FUSION_SRGAN_DROPOUT: float = 0.1

    # --- Loss Specific ---
    LAMBDA_L1: float = 1.0
    LAMBDA_ADV: float = 5e-3
    LAMBDA_PERCEP: float = 1.0

    # --- SwinIR Specific ---
    SWIN_EPOCHS: int = 1000
    SWIN_LR: float = 2e-4
    SWIN_EMBED_DIM: int = 180
    SWIN_NUM_HEADS: int = 6
    SWIN_WINDOW_SIZE: int = 8
    SWIN_NUM_LAYERS: int = 4 # Number of RSTL layers + Fusion Blocks.
    SWIN_NUM_BLOCKS_PER_LAYER: int = 6 # Number of Swin blocks within each RSTL.

# Create an instance for easy importing across the project
config = SuperResConfig()
