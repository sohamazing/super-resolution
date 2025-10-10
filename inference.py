#!/usr/bin/env python3
"""
Unified Super-Resolution Inference Script
==========================================
A professional, unified inference script for all project models, featuring:
- Support for ESRGAN, Diffusion, SwinIR, and Fusion-SRGAN models.
- Automatic 'best' checkpoint detection and robust loading.
- Tiled inference for large images with seamless blending.
- Batch processing for directories.
- Device-agnostic half-precision (FP16/BF16) support.

Usage:
    # Single image, auto-detecting the best checkpoint
    python inference.py --model fusion_srgan --input image.png --output result.png

    # Batch process an entire directory
    python inference.py --model swinir --input_dir ./images --output_dir ./results

    # Process a large image using tiled inference
    python inference.py --model esrgan --input large.jpg --output large_sr.jpg --tile_size 256
"""

import torch
from torch import nn
from torch.amp import autocast
import torch.nn.functional as F
from pathlib import Path
import torchvision.transforms.functional as TF
from tqdm import tqdm
from PIL import Image
import argparse
from typing import Optional
import sys

# --- Project Setup ---
ROOT_DIR = Path(__file__).parent.absolute()
sys.path.append(str(ROOT_DIR))
from config import config

# --- Model Imports ---
from diffusion.diffusion_model import DiffusionUNet
from esrgan.generator import GeneratorESRGAN
from fusion_srgan.generator_swinunet import SwinUNetGenerator
from swinir.fused_swinir_model import FusedSwinIR

# ==================================
# MODEL REGISTRY
# ==================================
MODEL_REGISTRY = {
    "fusion_srgan": {
        "class": SwinUNetGenerator,
        "args": {
            "features": config.FUSION_SRGAN_GEN_FEATURES,
            "embed_dim": config.FUSION_SRGAN_EMBED_DIM,
            "num_heads": config.FUSION_SRGAN_NUM_HEADS,
            "window_size": config.FUSION_SRGAN_WINDOW_SIZE,
            "num_swin_blocks": config.FUSION_SRGAN_NUM_SWIN_BLOCKS,
            "scale": config.SCALE,
            "dropout": config.FUSION_SRGAN_DROPOUT,
        },
        "description": "Hybrid CNN U-Net with a Swin Transformer bottleneck, trained as a GAN.",
        "checkpoint_dir": ROOT_DIR / "fusion_srgan" / "checkpoints",
        "best_filename": "best_generator.pth"
    },
    "diffusion": {
        "class": DiffusionUNet,
        "args": {
            "features": config.DIFFUSION_FEATURES,
            "time_emb_dim": config.DIFFUSION_TIME_EMB_DIM,
        },
        "description": "Denoising Diffusion Probabilistic Model (DDPM) using a U-Net to predict noise.",
        "checkpoint_dir": ROOT_DIR / "diffusion" / "checkpoints",
        "best_filename": "best_model.pth"
    },
    "esrgan": {
        "class": GeneratorESRGAN, # Assuming this is the correct class name from your esrgan/generator.py
        "args": {
            "num_features": config.ESRGAN_NUM_FEATURES,
            "num_blocks": config.ESRGAN_NUM_RRDB,
        },
        "description": "Classic ESRGAN architecture using Residual-in-Residual Dense Blocks (RRDBs).",
        "checkpoint_dir": ROOT_DIR / "esrgan" / "checkpoints",
        "best_filename": "best_generator.pth"
    },
    "swinir": {
        "class": FusedSwinIR,
        "args": {
            "embed_dim": config.SWIN_EMBED_DIM,
            "num_heads": config.SWIN_NUM_HEADS,
            "window_size": config.SWIN_WINDOW_SIZE,
            "num_layers": config.SWIN_NUM_LAYERS,
            "num_blocks_per_layer": config.SWIN_NUM_BLOCKS_PER_LAYER,
            "scale": config.SCALE,
        },
        "description": "Swin Transformer with parallel CNN feature fusion for deep feature extraction.",
        "checkpoint_dir": ROOT_DIR / "swinir" / "checkpoints",
        "best_filename": "best_model.pth"
    },
}

# ==================================
# INFERENCE ENGINE
# ==================================
class SuperResolutionInference:
    """
    Unified inference engine that handles model loading, image processing,
    and tiled inference for large images.
    """
    def __init__(self, model_name: str, checkpoint_path: Optional[Path] = None, half_precision: bool = False):
        self.model_name = model_name
        self.device = config.DEVICE
        self.half = half_precision and (self.device != 'cpu')
        
        print(f"Initializing '{model_name.upper()}' engine on device '{self.device}' (AMP: {self.half})...")
        self.model = self._load_model(checkpoint_path)
        self.model.eval()
        if self.half:
            self.model.half()

    def _load_model(self, checkpoint_path: Optional[Path]) -> nn.Module:
        """Loads model architecture and weights from the registry."""
        if self.model_name not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model: '{self.model_name}'. Choose from {list(MODEL_REGISTRY.keys())}")
        
        entry = MODEL_REGISTRY[self.model_name]
        
        if checkpoint_path is None:
            checkpoint_path = entry['checkpoint_dir'] / entry['best_filename']
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Auto-detection failed: Cannot find '{checkpoint_path}'. Please train the model or specify a path with --checkpoint.")
        
        print(f"Loading checkpoint: {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        
        if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        elif isinstance(state_dict, dict) and 'gen_state_dict' in state_dict:
            state_dict = state_dict['gen_state_dict']
            
        model = entry['class'](**entry['args'])
        model.load_state_dict(state_dict, strict=True)
        return model.to(self.device)

    @torch.no_grad()
    def run(self, lr_image: Image.Image, tile_size: int = 0, tile_overlap: int = 32) -> Image.Image:
        """Performs super-resolution on a single PIL image."""
        lr_tensor = TF.to_tensor(lr_image).unsqueeze(0).to(self.device)
        if self.half:
            lr_tensor = lr_tensor.half()
            
        use_tiling = tile_size > 0 and (lr_tensor.shape[2] > tile_size or lr_tensor.shape[3] > tile_size)

        with autocast(device_type=self.device.split(':')[0], enabled=self.half):
            if self.model_name == 'diffusion':
                if use_tiling:
                    print("Warning: Tiled inference is not supported for diffusion models. Performing full inference.")
                sr_tensor = self._infer_diffusion(lr_tensor)
            elif use_tiling:
                sr_tensor = self._infer_tiled(lr_tensor, tile_size, tile_overlap)
            else:
                sr_tensor = self._infer_full(lr_tensor)

        sr_tensor = sr_tensor.squeeze(0).float().cpu().clamp(0, 1)
        return TF.to_pil_image(sr_tensor)
        
    def _infer_full(self, lr_tensor: torch.Tensor) -> torch.Tensor:
        """Inference on the entire image at once."""
        return self.model(lr_tensor)

    def _infer_diffusion(self, lr_tensor: torch.Tensor) -> torch.Tensor:
        """CRITICAL: Inference using the special denoising loop for diffusion models."""
        from diffusion.scheduler import Scheduler
        scheduler = Scheduler(timesteps=config.DIFFUSION_TIMESTEPS)
        
        output_shape = (1, 3, lr_tensor.shape[2] * config.SCALE, lr_tensor.shape[3] * config.SCALE)
        img = torch.randn(output_shape, device=self.device)
        if self.half: img = img.half()

        lr_upscaled = F.interpolate(lr_tensor, scale_factor=config.SCALE, mode='bicubic', align_corners=False)
        
        for i in tqdm(reversed(range(0, scheduler.timesteps)), desc="Denoising", total=scheduler.timesteps, leave=False):
            t = torch.full((1,), i, device=self.device, dtype=torch.long)
            predicted_noise = self.model(img, t, lr_upscaled)
            img = scheduler.sample_previous_timestep(img, t, predicted_noise)
        return img

    def _infer_tiled(self, lr_tensor: torch.Tensor, tile_size: int, tile_overlap: int) -> torch.Tensor:
        """Robust inference in patches with reflection padding and a Hann window for smooth blending."""
        b, c, h, w = lr_tensor.shape
        scale = config.SCALE
        stride = tile_size - tile_overlap
        
        pad_h = (stride - (h - tile_overlap) % stride) % stride
        pad_w = (stride - (w - tile_overlap) % stride) % stride
        lr_padded = F.pad(lr_tensor, [0, pad_w, 0, pad_h], 'reflect')
        
        sr_shape = (b, c, (h + pad_h) * scale, (w + pad_w) * scale)
        sr_padded = torch.zeros(sr_shape, device=self.device)
        blend_weights = torch.zeros(sr_shape, device=self.device)
        
        window = torch.hann_window(tile_size * scale, periodic=False).unsqueeze(1) * \
                 torch.hann_window(tile_size * scale, periodic=False).unsqueeze(0)
        window = window.to(self.device).expand(1, c, -1, -1)
        if self.half: window = window.half()
        
        ph, pw = lr_padded.shape[2:]
        for y in range(0, ph - tile_overlap, stride):
            for x in range(0, pw - tile_overlap, stride):
                lr_tile = lr_padded[:, :, y:y+tile_size, x:x+tile_size]
                sr_tile = self.model(lr_tile)
                
                y_sr, x_sr = y * scale, x * scale
                sr_padded[:, :, y_sr:y_sr+tile_size*scale, x_sr:x_sr+tile_size*scale] += sr_tile * window
                blend_weights[:, :, y_sr:y_sr+tile_size*scale, x_sr:x_sr+tile_size*scale] += window

        sr_image = sr_padded / (blend_weights + 1e-8)
        return sr_image[:, :, :h*scale, :w*scale]

# ==================================
# COMMAND-LINE INTERFACE
# ==================================
def main():
    parser = argparse.ArgumentParser(
        description="Super-Resolution Inference Script",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Examples:
  # Basic inference, auto-detecting the best checkpoint
  python inference.py --model diffusion --input images/XX.png  # creates images/XX_SR.png
  python inference.py --model swinir --input images/input.png --output results/output.png

  # Batch process a directory
  python inference.py --model fusion_srgan --input_dir images/ --output_dir results/

  # Use tiled inference for a very large image
  python inference.py --model esrgan --input large.jpg --output large_sr.jpg --tile_size 256
"""
    )
    parser.add_argument('--model', type=str, required=True, choices=list(MODEL_REGISTRY.keys()), help="Model to use.")
    parser.add_argument('--checkpoint', type=Path, default=None, help="Path to a specific model checkpoint. (Optional)")
    
    io_group = parser.add_argument_group('Input/Output')
    io_group.add_argument('--input', type=Path, default=None, help="Path to a single input image.")
    io_group.add_argument('--output', type=Path, default=None, help="Path to save the output image. (Optional)")
    io_group.add_argument('--input_dir', type=Path, default=None, help="Directory of images for batch processing.")
    io_group.add_argument('--output_dir', type=Path, default=None, help="Directory to save batch results. (Default: input_dir/sr_results)")
    
    tile_group = parser.add_argument_group('Tiling (for large images)')
    tile_group.add_argument('--tile_size', type=int, default=0, help="Tile size for tiled inference (e.g., 256). 0 disables.")
    tile_group.add_argument('--tile_overlap', type=int, default=32, help="Overlap between tiles in pixels.")

    perf_group = parser.add_argument_group('Performance')
    perf_group.add_argument('--half', action='store_true', help="Enable half-precision (FP16/BF16) for faster inference.")

    args = parser.parse_args()

    if not args.input and not args.input_dir:
        parser.error("Either --input or --input_dir must be specified.")
    
    engine = SuperResolutionInference(model_name=args.model, checkpoint_path=args.checkpoint, half_precision=args.half)

    if args.input:
        output_path = args.output or args.input.parent / f"{args.input.stem}_SR.png"
        print(f"Processing single image: {args.input} -> {output_path}")
        lr_image = Image.open(args.input)
        sr_image = engine.run(lr_image, tile_size=args.tile_size, tile_overlap=args.tile_overlap)
        sr_image.save(output_path)

    if args.input_dir:
        output_dir = args.output_dir or args.input_dir / "sr_results"
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Processing directory: {args.input_dir} -> {output_dir}")
        image_paths = sorted([p for p in args.input_dir.iterdir() if p.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']])
        for path in tqdm(image_paths, desc="Batch Processing"):
            try:
                lr_image = Image.open(path)
                sr_image = engine.run(lr_image, tile_size=args.tile_size, tile_overlap=args.tile_overlap)
                output_path = output_dir / f"{path.stem}_SR.png"
                sr_image.save(output_path)
            except Exception as e:
                print(f"\nERROR: Failed to process {path.name}: {e}")
    
    print("\nInference complete!")

if __name__ == '__main__':
    main()