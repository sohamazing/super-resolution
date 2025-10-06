# swin_ir/inference.py
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

from swin_ir.swin_ir_model import FusedSwinIR
from config import config

def run_inference(args):
    """Runs a tiled inference process for the FusedSwinIR model."""
    # --- 1. Setup Model ---
    print("--- Loading FusedSwinIR model... ---")
    device = config.DEVICE
    model = FusedSwinIR(
        embed_dim=config.SWIN_EMBED_DIM,
        num_heads=config.SWIN_NUM_HEADS,
        window_size=config.SWIN_WINDOW_SIZE,
        num_layers=config.SWIN_NUM_LAYERS,
        num_blocks_per_layer=config.SWIN_NUM_BLOCKS_PER_LAYER
    ).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    
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
    
    output_canvas = torch.zeros((1, C, hr_h, hr_w), device=device)
    overlap_counter = torch.zeros((1, C, hr_h, hr_w), device=device)

    print("--- Running tiled inference... ---")
    with torch.no_grad():
        for y in tqdm(range(0, H, patch_size_lr - overlap), desc="Image Tiling"):
            for x in range(0, W, patch_size_lr - overlap):
                y_start, x_start = y, x
                y_end, x_end = min(y + patch_size_lr, H), min(x + patch_size_lr, W)
                
                lr_patch = lr_tensor[:, :, y_start:y_end, x_start:x_end]
                
                # --- Run a single forward pass on the patch ---
                sr_patch = model(lr_patch)
                
                # --- Place the generated high-resolution patch onto the canvas ---
                hr_patch_h, hr_patch_w = sr_patch.shape[2], sr_patch.shape[3]
                y_start_hr, x_start_hr = y_start * config.SCALE, x_start * config.SCALE
                output_canvas[:, :, y_start_hr:y_start_hr+hr_patch_h, x_start_hr:x_start_hr+hr_patch_w] += sr_patch
                overlap_counter[:, :, y_start_hr:y_start_hr+hr_patch_h, x_start_hr:x_start_hr+hr_patch_w] += 1
    
    # Average the pixels in the overlapping regions
    final_output = output_canvas / overlap_counter
    
    # --- 4. Save the Final Image ---
    print(f"--- Saving final image to: {args.output} ---")
    output_img = T.ToPILImage()(final_output.squeeze(0).clamp(0, 1).cpu())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_img.save(args.output)
    print("--- Inference complete! ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run tiled inference with a FusedSwinIR model.")
    parser.add_argument("--input", type=Path, required=True, help="Path to the low-resolution input image.")
    parser.add_argument("--output", type=Path, required=True, help="Path to save the high-resolution output image.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to the model checkpoint .pth file.")
    parser.add_argument("--patch_size", type=int, default=64, help="Size of the LR patches to process (e.g., 64 for a 256px HR model).")
    parser.add_argument("--overlap", type=int, default=16, help="Overlap between patches in pixels to avoid seams.")
    args = parser.parse_args()
    run_inference(args)