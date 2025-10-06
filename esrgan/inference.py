# esrgan/inference.py
import torch
from pathlib import Path
from PIL import Image
import argparse
import sys
import torchvision.transforms as T

# Path logic to allow imports from the parent directory
SCRIPT_DIR = Path(__file__).parent.absolute()
sys.path.append(str(SCRIPT_DIR.parent))

from esrgan.generator import GeneratorRRDB
from config import config

def run_inference(args):
    """Runs a direct, single-pass inference for the ESRGAN model."""
    # --- 1. Setup Model ---
    print("--- Loading ESRGAN model... ---")
    device = config.DEVICE
    model = GeneratorRRDB(num_features=64, num_blocks=23).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    
    # --- 2. Load and Prepare Input Image ---
    print(f"--- Loading input image: {args.input} ---")
    input_img = Image.open(args.input).convert("RGB")
    transform = T.ToTensor()
    lr_tensor = transform(input_img).unsqueeze(0).to(device)
    
    # --- 3. Run Inference ---
    print("--- Running inference... ---")
    with torch.no_grad():
        # Pass the entire image through the model in one go.
        # For very large images on low-memory GPUs, a tiled approach could be added here.
        sr_tensor = model(lr_tensor)
        
    # --- 4. Save the Final Image ---
    print(f"--- Saving final image to: {args.output} ---")
    output_img = T.ToPILImage()(sr_tensor.squeeze(0).clamp(0, 1).cpu())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_img.save(args.output)
    print("--- Inference complete! ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference with an ESRGAN model.")
    parser.add_argument("--input", type=Path, required=True, help="Path to the low-resolution input image.")
    parser.add_argument("--output", type=Path, required=True, help="Path to save the high-resolution output image.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to the generator checkpoint .pth file.")
    args = parser.parse_args()
    run_inference(args)