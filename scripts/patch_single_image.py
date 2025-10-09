import os
import cv2
import numpy as np
import rawpy
import imageio.v2 as imageio
from pathlib import Path
import argparse
import pillow_heif

# Register the HEIC opener
pillow_heif.register_heif_opener()

# --- Hardcoded settings for simplicity ---
PATCH_SIZE = 256
SCALE = 4
STEP = 128
SUPPORTED_EXTENSIONS = (".png", ".jpg", ".jpeg", ".heic", ".heif", ".dng", ".cr2", ".arw")

def read_image(path: Path) -> np.ndarray | None:
    """Reads various image formats into a NumPy array."""
    try:
        ext = path.suffix.lower()
        if ext in ['.arw', '.cr2', '.dng']:
            with rawpy.imread(str(path)) as raw:
                return raw.postprocess(output_bps=16)
        else:
            return imageio.imread(path)
    except Exception as e:
        print(f"Error reading {path.name}: {e}")
        return None

def create_patches(image: np.ndarray, patch_size: int, stride: int) -> list:
    """Extracts patches from an image using a sliding window."""
    patches = []
    h, w = image.shape[:2]
    for i in range(0, h - patch_size + 1, stride):
        for j in range(0, w - patch_size + 1, stride):
            patches.append(image[i:i + patch_size, j:j + patch_size])
    return patches

def process_single_image(source_path: Path, output_dir: Path):
    """Processes one image into HR and LR patches."""
    hr_dir = output_dir / "HR"
    lr_dir = output_dir / "LR"
    hr_dir.mkdir(parents=True, exist_ok=True)
    lr_dir.mkdir(parents=True, exist_ok=True)

    lr_patch_size = PATCH_SIZE // SCALE
    
    print(f"Reading image: {source_path.name}...")
    hr_image_rgb = read_image(source_path)
    if hr_image_rgb is None or hr_image_rgb.ndim < 3:
        print("Could not read or process image.")
        return

    hr_image = cv2.cvtColor(hr_image_rgb, cv2.COLOR_RGB2BGR)
    
    if hr_image.dtype == np.uint16:
        print("Converting 16-bit RAW image to 8-bit.")
        hr_image = (hr_image / 256).astype(np.uint8)
        
    lr_image = cv2.resize(hr_image, (hr_image.shape[1] // SCALE, hr_image.shape[0] // SCALE), interpolation=cv2.INTER_CUBIC)
    
    print("Creating patches...")
    hr_patches = create_patches(hr_image, PATCH_SIZE, STEP)
    lr_patches = create_patches(lr_image, lr_patch_size, STEP // SCALE)

    patch_count = 0
    for i, (hr_patch, lr_patch) in enumerate(zip(hr_patches, lr_patches)):
        unique_id = f"{source_path.stem}_{i}"
        cv2.imwrite(str(hr_dir / f"{unique_id}.png"), hr_patch)
        cv2.imwrite(str(lr_dir / f"{unique_id}.png"), lr_patch)
        patch_count += 1
        
    print(f"✅ Done! Created {patch_count} HR/LR patch pairs in '{output_dir}'.")
    return patch_count

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create HR/LR patches for a SINGLE source image.")
    parser.add_argument("source_image", type=str, help="Path to the single high-resolution source image.")
    parser.add_argument("output_dir", type=str, help="Directory to save the 'HR' and 'LR' subfolders with patches.")
    
    args = parser.parse_args()
    
    source_path = Path(args.source_image).expanduser()
    output_path = Path(args.output_dir).expanduser()

    if not source_path.exists():
        print(f"❌ ERROR: Source file not found: {source_path}")
    else:
        process_single_image(source_path, output_path)
