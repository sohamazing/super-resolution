# diffusion/inference.py
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import argparse
import sys
import torchvision.transforms as T

# Path logic to allow imports from the parent directory
SCRIPT_DIR = Path(__file__).parent.absolute()
sys.path.append(str(SCRIPT_DIR.parent))

from diffusion.diffusion_model import DiffusionUNet
from diffusion.scheduler import Scheduler
from config import config

def run_inference(args):
    """Runs a tiled inference process for the Diffusion model."""
    # --- 1. Setup Model and Scheduler ---
    print("--- Loading model and scheduler... ---")
    device = config.DEVICE
    model = DiffusionUNet().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    scheduler = Scheduler(timesteps=config.TIMESTEPS)
    
    # --- 2. Load and Prepare Input Image ---
    print(f"--- Loading input image: {args.input} ---")
    input_img = Image.open(args.input).convert("RGB")
    transform = T.ToTensor()
    lr_tensor = transform(input_img).unsqueeze(0).to(device)
    
    C, H, W = lr_tensor.shape[1:]
    hr_h, hr_w = H * config.SCALE, W * config.SCALE
    
    # --- 3. Tiled Inference Logic ---
    patch_size_lr = args.patch_size
    overlap = args.overlap
    
    # Create an empty canvas for the final output and a counter for averaging overlaps
    output_canvas = torch.zeros((1, C, hr_h, hr_w), device=device)
    overlap_counter = torch.zeros((1, C, hr_h, hr_w), device=device)

    print("--- Running tiled inference... ---")
    with torch.no_grad():
        # Iterate over the large image with overlapping patches
        for y in tqdm(range(0, H, patch_size_lr - overlap), desc="Image Tiling"):
            for x in range(0, W, patch_size_lr - overlap):
                y_start, x_start = y, x
                y_end, x_end = min(y + patch_size_lr, H), min(x + patch_size_lr, W)
                
                lr_patch = lr_tensor[:, :, y_start:y_end, x_start:x_end]
                
                # --- Run the full diffusion denoising loop on this single patch ---
                hr_patch_h, hr_patch_w = lr_patch.shape[2] * config.SCALE, lr_patch.shape[3] * config.SCALE
                img = torch.randn((1, C, hr_patch_h, hr_patch_w), device=device)
                lr_upscaled = F.interpolate(lr_patch, scale_factor=config.SCALE, mode='bicubic', align_corners=False)

                for i in reversed(range(0, scheduler.timesteps)):
                    t = torch.full((1,), i, device=device, dtype=torch.long)
                    predicted_noise = model(img, t, lr_upscaled)
                    img = scheduler.sample_previous_timestep(img, t, predicted_noise)
                
                # --- Place the generated high-resolution patch onto the canvas ---
                y_start_hr, x_start_hr = y_start * config.SCALE, x_start * config.SCALE
                output_canvas[:, :, y_start_hr:y_start_hr+hr_patch_h, x_start_hr:x_start_hr+hr_patch_w] += img
                overlap_counter[:, :, y_start_hr:y_start_hr+hr_patch_h, x_start_hr:x_start_hr+hr_patch_w] += 1
    
    # Average the pixels in the overlapping regions to create a seamless image
    final_output = output_canvas / overlap_counter
    
    # --- 4. Save the Final Image ---
    print(f"--- Saving final image to: {args.output} ---")
    output_img = T.ToPILImage()(final_output.squeeze(0).clamp(0, 1).cpu())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_img.save(args.output)
    print("--- Inference complete! ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run tiled inference with a Diffusion model.")
    parser.add_argument("--input", type=Path, required=True, help="Path to the low-resolution input image.")
    parser.add_argument("--output", type=Path, required=True, help="Path to save the high-resolution output image.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to the model checkpoint .pth file.")
    parser.add_argument("--patch_size", type=int, default=64, help="Size of the LR patches to process.")
    parser.add_argument("--overlap", type=int, default=16, help="Overlap between patches in pixels to avoid seams.")
    args = parser.parse_args()
    run_inference(args)