import os
import sys
import cv2
import numpy as np
import rawpy
import imageio.v2 as imageio # Use v2 to silence the deprecation warning
from tqdm import tqdm
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import random
# Imports for HEIC support
import pillow_heif
pillow_heif.register_heif_opener()

SCRIPT_DIR = Path(__file__).parent.absolute()
sys.path.append(str(SCRIPT_DIR.parent))
from config import config


def read_image(path: Path) -> np.ndarray | None:
    """Reads various image formats into a NumPy array."""
    try:
        ext = path.suffix.lower()
        if ext in ['.arw', '.cr2', '.dng']:
            with rawpy.imread(str(path)) as raw:
                # Returns an RGB numpy array
                return raw.postprocess(output_bps=16)
        else:
            # Returns an RGB numpy array
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

def process_image_to_patches(args):
    """Processes a single image into corresponding HR and LR patches."""
    path, hr_dir, lr_dir = args
    lr_patch_size = config.PATCH_SIZE // config.SCALE
    
    hr_image_rgb = read_image(path)
    if hr_image_rgb is None or hr_image_rgb.ndim < 3: return 0

    hr_image = cv2.cvtColor(hr_image_rgb, cv2.COLOR_RGB2BGR)
    
    if hr_image.dtype == np.uint16:
        hr_image = (hr_image / 256).astype(np.uint8)
        
    lr_image = cv2.resize(hr_image, (hr_image.shape[1] // config.SCALE, hr_image.shape[0] // config.SCALE), interpolation=cv2.INTER_CUBIC)
    
    hr_patches = create_patches(hr_image, config.PATCH_SIZE, config.STEP)
    
    # --- FIX 2: Scale the step for the low-resolution image to ensure alignment ---
    lr_patches = create_patches(lr_image, lr_patch_size, config.STEP // config.SCALE)

    patch_count = 0
    for i, (hr_patch, lr_patch) in enumerate(zip(hr_patches, lr_patches)):
        unique_id = f"{path.stem}_{i}"
        cv2.imwrite(str(hr_dir / f"{unique_id}.png"), hr_patch)
        cv2.imwrite(str(lr_dir / f"{unique_id}.png"), lr_patch)
        patch_count += 1
        
    return patch_count

def process_image_pair_wrapper(args):
    """Wrapper function for process_image_pair to work with ProcessPoolExecutor."""
    path, hr_dir, lr_dir = args
    process_image_pair(path, hr_dir, lr_dir)

def process_image_pair(path, hr_dir, lr_dir):
    """Processes a single image into one HR/LR pair (for validation/test)."""
    hr_image_rgb = read_image(path)
    if hr_image_rgb is None or hr_image_rgb.ndim < 3: return

    # --- FIX 1: Convert from RGB (from imageio) to BGR (for OpenCV) ---
    hr_image = cv2.cvtColor(hr_image_rgb, cv2.COLOR_RGB2BGR)

    if hr_image.dtype == np.uint16:
        hr_image = (hr_image / 256).astype(np.uint8)
        
    lr_image = cv2.resize(hr_image, (hr_image.shape[1] // config.SCALE, hr_image.shape[0] // config.SCALE), interpolation=cv2.INTER_CUBIC)
    
    output_name = f"{path.stem}.png"
    cv2.imwrite(str(hr_dir / output_name), hr_image)
    cv2.imwrite(str(lr_dir / output_name), lr_image)


def main():
    # --- 1. Setup Directories ---
    print("Setting up directories...")
    hr_train_dir = config.DATA_DIR / "train" / "HR"
    lr_train_dir = config.DATA_DIR / "train" / "LR"
    hr_val_dir = config.DATA_DIR / "val" / "HR"
    lr_val_dir = config.DATA_DIR / "val" / "LR"
    hr_test_dir = config.DATA_DIR / "test" / "HR"
    lr_test_dir = config.DATA_DIR / "test" / "LR"
    for path in [hr_train_dir, lr_train_dir, hr_val_dir, lr_val_dir, hr_test_dir, lr_test_dir]:
        path.mkdir(parents=True, exist_ok=True)

    # --- 2. Gather and Split File Paths ---
    def gather_files(source_dirs):
        """Helper to scan directories for all supported file types, case-insensitively."""
        all_files = []
        for source_path in source_dirs:
            print(f"...scanning {source_path}")
            for ext in config.SUPPORTED_EXTENSIONS:
                all_files.extend(source_path.rglob(f"*{ext.lower()}"))
                all_files.extend(source_path.rglob(f"*{ext.upper()}"))
        return list(set(all_files)) # Remove duplicates

    print("Scanning for Train/Validation images...")
    train_val_paths = gather_files(config.TRAIN_VAL_SOURCES)
    
    print("\nScanning for Test images...")
    test_paths = gather_files(config.TEST_SOURCES)
    
    random.shuffle(train_val_paths)
    split_index = int(len(train_val_paths) * config.VAL_SPLIT)
    val_paths = train_val_paths[:split_index]
    train_paths = train_val_paths[split_index:]

    print(f"\nTotal files found: {len(train_paths)} training, {len(val_paths)} validation, {len(test_paths)} testing.")

    # --- 3. Process Datasets ---
    print("\nProcessing training images into patches...")
    total_patches = 0
    tasks = [(path, hr_train_dir, lr_train_dir) for path in train_paths]
    with ProcessPoolExecutor(max_workers=config.NUM_WORKERS) as executor:
        with tqdm(total=len(tasks), desc="Training Patches") as pbar:
            for count in executor.map(process_image_to_patches, tasks):
                total_patches += count
                pbar.update(1)

    print("\nProcessing validation images...")
    with ProcessPoolExecutor(max_workers=config.NUM_WORKERS) as executor:
        tasks = [(path, hr_val_dir, lr_val_dir) for path in val_paths]
        list(tqdm(executor.map(process_image_pair_wrapper, tasks), total=len(tasks), desc="Validation Images"))

    print("\nProcessing test images...")
    with ProcessPoolExecutor(max_workers=config.NUM_WORKERS) as executor:
        tasks = [(path, hr_test_dir, lr_test_dir) for path in test_paths]
        list(tqdm(executor.map(process_image_pair_wrapper, tasks), total=len(tasks), desc="Test Images"))

    print(f"\nDone! Created {total_patches} training patches, {len(val_paths)} validation pairs, and {len(test_paths)} test pairs.")

if __name__ == '__main__':
    main()