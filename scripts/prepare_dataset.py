#!/usr/bin/env python3
"""
prepare_dataset.py - Dataset Preparation for On-the-Fly Cropping

Processes source images into HR/LR pairs WITHOUT creating patches.
Much simpler and faster than create_patches.py since we only downsample once per image.

Directory structure created:
    DATA_DIR/
    ├── train/
    │   ├── HR/  ← Full-size HR images
    │   └── LR/  ← Full-size LR images (downsampled 4x)
    ├── val/
    │   ├── HR/
    │   └── LR/
    └── test/
        ├── HR/
        └── LR/

Usage:
    python prepare_dataset.py
    python prepare_dataset.py --force  # Overwrite existing files
"""

import os
import sys
import cv2
import numpy as np
import rawpy
import imageio.v2 as imageio
from tqdm import tqdm
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import random
import argparse

# HEIC support
import pillow_heif
pillow_heif.register_heif_opener()

# Import config
SCRIPT_DIR = Path(__file__).parent.absolute()
sys.path.append(str(SCRIPT_DIR.parent))
from config import config


def read_image(path: Path) -> np.ndarray | None:
    """
    Reads various image formats (JPG, PNG, HEIC, RAW) into NumPy array.
    
    Returns:
        RGB numpy array (H, W, 3) or None if error
    """
    try:
        ext = path.suffix.lower()
        
        # RAW formats (DNG, CR2, ARW)
        if ext in ['.arw', '.cr2', '.dng']:
            with rawpy.imread(str(path)) as raw:
                return raw.postprocess(output_bps=16)
        
        # Standard formats (JPG, PNG, HEIC, etc.)
        else:
            return imageio.imread(path)
            
    except Exception as e:
        print(f"Error reading {path.name}: {e}")
        return None


def process_image_pair(args):
    """
    Process a single source image into HR/LR pair.
    
    Args:
        args: Tuple of (source_path, hr_output_dir, lr_output_dir, force_overwrite)
    
    Returns:
        bool: True if processed successfully
    """
    source_path, hr_dir, lr_dir, force = args
    
    # Output paths
    output_name = f"{source_path.stem}.png"
    hr_output_path = hr_dir / output_name
    lr_output_path = lr_dir / output_name
    
    # Skip if already exists (unless force)
    if not force and hr_output_path.exists() and lr_output_path.exists():
        return True
    
    # Read source image
    hr_image_rgb = read_image(source_path)
    if hr_image_rgb is None or hr_image_rgb.ndim < 3:
        return False
    
    # Convert RGB to BGR (OpenCV format)
    hr_image = cv2.cvtColor(hr_image_rgb, cv2.COLOR_RGB2BGR)
    
    # Convert 16-bit to 8-bit if needed
    if hr_image.dtype == np.uint16:
        hr_image = (hr_image / 256).astype(np.uint8)
    
    # Create LR image by downsampling
    h, w = hr_image.shape[:2]
    lr_w, lr_h = w // config.SCALE, h // config.SCALE
    lr_image = cv2.resize(
        hr_image,
        (lr_w, lr_h),
        interpolation=cv2.INTER_CUBIC
    )
    
    # Save both images
    try:
        cv2.imwrite(str(hr_output_path), hr_image)
        cv2.imwrite(str(lr_output_path), lr_image)
        return True
    except Exception as e:
        print(f"Error saving {output_name}: {e}")
        return False


def gather_source_files(source_dirs):
    """
    Scan source directories for all supported image files.
    
    Args:
        source_dirs: List of Path objects to scan
    
    Returns:
        List of Path objects for all found images
    """
    all_files = []
    
    for source_path in source_dirs:
        if not source_path.exists():
            print(f"Warning: Source directory not found: {source_path}")
            continue
            
        print(f"Scanning: {source_path}")
        
        # Search for each supported extension (case-insensitive)
        for ext in config.SUPPORTED_EXTENSIONS:
            all_files.extend(source_path.rglob(f"*{ext.lower()}"))
            all_files.extend(source_path.rglob(f"*{ext.upper()}"))
    
    # Remove duplicates
    return list(set(all_files))


def main():
    parser = argparse.ArgumentParser(
        description="Prepare dataset for on-the-fly cropping training"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing files"
    )
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("-- Dataset Preparation for On-the-Fly Cropping --")
    print("="*70 + "\n")
    
    # --- 1. Setup Output Directories ---
    print("Setting up output directories...")
    
    hr_train_dir = config.DATA_DIR / "train" / "HR"
    lr_train_dir = config.DATA_DIR / "train" / "LR"
    hr_val_dir = config.DATA_DIR / "val" / "HR"
    lr_val_dir = config.DATA_DIR / "val" / "LR"
    hr_test_dir = config.DATA_DIR / "test" / "HR"
    lr_test_dir = config.DATA_DIR / "test" / "LR"
    
    for path in [hr_train_dir, lr_train_dir, hr_val_dir, lr_val_dir, hr_test_dir, lr_test_dir]:
        path.mkdir(parents=True, exist_ok=True)
    
    print(f"   Output directory: {config.DATA_DIR}")
    print()
    
    # --- 2. Gather Source Files ---
    print("Scanning source directories...")
    print(f"Supported formats: {', '.join(config.SUPPORTED_EXTENSIONS)}")
    print()
    
    train_val_paths = gather_source_files(config.TRAIN_VAL_SOURCES)
    test_paths = gather_source_files(config.TEST_SOURCES)
    
    if not train_val_paths:
        print("Error: No training/validation images found!")
        print("Check your TRAIN_VAL_SOURCES paths in config.py")
        sys.exit(1)
    
    print(f"   Found: {len(train_val_paths)} train/val images")
    print(f"   Found: {len(test_paths)} test images")
    print()
    
    # --- 3. Split Train/Val ---
    print("Splitting train/validation sets...")
    
    random.seed(42)  # For reproducibility
    random.shuffle(train_val_paths)
    
    split_index = int(len(train_val_paths) * config.VAL_SPLIT)
    val_paths = train_val_paths[:split_index]
    train_paths = train_val_paths[split_index:]
    
    print(f"-- Training: {len(train_paths)} images ({100-config.VAL_SPLIT*100:.0f}%)")
    print(f"-- Validation: {len(val_paths)} images ({config.VAL_SPLIT*100:.0f}%)")
    print(f"-- Test: {len(test_paths)} images")
    print()
    
    # --- 4. Process Images ---
    num_workers = config.NUM_WORKERS if config.NUM_WORKERS > 0 else 1
    print(f"Processing with {num_workers} workers...")
    print()
    
    # Process training images
    print("Processing training images...")
    train_tasks = [(path, hr_train_dir, lr_train_dir, args.force) for path in train_paths]
    train_success = 0
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm(
            executor.map(process_image_pair, train_tasks),
            total=len(train_tasks),
            desc="   Training",
            unit="img"
        ))
        train_success = sum(results)
    
    print(f"Successfully processed: {train_success}/{len(train_paths)}")
    print()
    
    # Process validation images
    print("Processing validation images...")
    val_tasks = [(path, hr_val_dir, lr_val_dir, args.force) for path in val_paths]
    val_success = 0
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm(
            executor.map(process_image_pair, val_tasks),
            total=len(val_tasks),
            desc="   Validation",
            unit="img"
        ))
        val_success = sum(results)
    
    print(f"Successfully processed: {val_success}/{len(val_paths)}")
    print()
    
    # Process test images
    if test_paths:
        print("Processing test images...")
        test_tasks = [(path, hr_test_dir, lr_test_dir, args.force) for path in test_paths]
        test_success = 0
        
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = list(tqdm(
                executor.map(process_image_pair, test_tasks),
                total=len(test_tasks),
                desc="   Test",
                unit="img"
            ))
            test_success = sum(results)
        
        print(f"Successfully processed: {test_success}/{len(test_paths)}")
        print()
    
    # --- 5. Summary ---
    print("="*70)
    print("-- Dataset Preparation Complete! --")
    print("="*70)
    print()
    print("Summary:")
    print(f"   Training images:   {train_success:4d} HR/LR pairs")
    print(f"   Validation images: {val_success:4d} HR/LR pairs")
    if test_paths:
        print(f"   Test images:       {test_success:4d} HR/LR pairs")
    print()
    print("Output structure:")
    print(f"   {config.DATA_DIR}/")
    print("   ├── train/")
    print(f"   │   ├── HR/  ({train_success} images)")
    print(f"   │   └── LR/  ({train_success} images)")
    print("   ├── val/")
    print(f"   │   ├── HR/  ({val_success} images)")
    print(f"   │   └── LR/  ({val_success} images)")
    if test_paths:
        print("   └── test/")
        print(f"       ├── HR/  ({test_success} images)")
        print(f"       └── LR/  ({test_success} images)")
    print()
    print("Ready for training with on-the-fly cropping!")
    print("Use TrainDatasetAugmented for best results.")
    print()
    
    # Check for failures
    total_failed = (len(train_paths) - train_success + 
                   len(val_paths) - val_success + 
                   len(test_paths) - test_success if test_paths else 0)
    
    if total_failed > 0:
        print(f"Warning: {total_failed} images failed to process")
        print("Check error messages above for details")
        print()


if __name__ == '__main__':
    main()