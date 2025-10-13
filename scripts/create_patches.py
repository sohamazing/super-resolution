# create_patches_unified.py
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
import shutil
from typing import List, Tuple, Optional

# --- HEIC/HEIF Support ---
import pillow_heif
pillow_heif.register_heif_opener()

# --- Path Setup ---
SCRIPT_DIR = Path(__file__).parent.absolute()
sys.path.append(str(SCRIPT_DIR.parent))

# --- Config Import ---
try:
    from config import config
except ImportError:
    print("Warning: config.py not found. Using hardcoded defaults.")
    from dataclasses import dataclass, field

    @dataclass
    class DefaultConfig:
        PATCH_SIZE: int = 256
        SCALE: int = 4
        STEP: int = 128
        NUM_WORKERS: int = os.cpu_count() or 4
        VAL_SPLIT: float = 0.1
        DATA_DIR: Path = Path("data")
        TRAIN_VAL_SOURCES: list = field(default_factory=lambda: [Path("./raw_images")])
        TEST_SOURCES: list = field(default_factory=lambda: [])
        SUPPORTED_EXTENSIONS: tuple = (
            ".png", ".jpg", ".jpeg", ".heic", ".heif", ".dng", ".cr2", ".arw"
        )

    config = DefaultConfig()


# ==============================================================================
#                               CORE UTILITIES
# ==============================================================================

def load_image(filepath: Path) -> Optional[np.ndarray]:
    """Loads an image (RAW, HEIC, or standard) into an RGB NumPy array."""
    try:
        ext = filepath.suffix.lower()
        if ext in ['.arw', '.cr2', '.dng']:
            with rawpy.imread(str(filepath)) as raw:
                return raw.postprocess(output_bps=16)
        return imageio.imread(filepath)
    except Exception as e:
        print(f"Error reading {filepath.name}: {e}")
        return None


def crop_image_to_scale(image: np.ndarray, scale_factor: int) -> np.ndarray:
    """Center-crops image to have height/width divisible by the scale factor."""
    h, w = image.shape[:2]
    new_h, new_w = (h // scale_factor) * scale_factor, (w // scale_factor) * scale_factor
    if new_h != h or new_w != w:
        start_h, start_w = (h - new_h) // 2, (w - new_w) // 2
        image = image[start_h:start_h + new_h, start_w:start_w + new_w]
    return image


def convert_uint16_to_uint8(image: np.ndarray) -> np.ndarray:
    """Converts 16-bit image to 8-bit."""
    if image.dtype == np.uint16:
        return (image / 256).astype(np.uint8)
    return image


def downsample_bicubic(image: np.ndarray, scale_factor: int) -> np.ndarray:
    """Downsamples image using bicubic interpolation (OpenCV)."""
    return cv2.resize(
        image,
        (image.shape[1] // scale_factor, image.shape[0] // scale_factor),
        interpolation=cv2.INTER_CUBIC,
    )


def create_patches(image: np.ndarray, patch_size: int, stride: int) -> List[np.ndarray]:
    """Extracts sliding-window patches from an image."""
    patches = []
    h, w = image.shape[:2]
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patches.append(image[y:y + patch_size, x:x + patch_size])
    return patches


def save_patch_pair(hr_patch: np.ndarray, lr_patch: np.ndarray,
                    hr_dir: Path, lr_dir: Path, name: str) -> bool:
    """Saves HR/LR patch pair."""
    try:
        cv2.imwrite(str(hr_dir / f"{name}.png"), hr_patch)
        cv2.imwrite(str(lr_dir / f"{name}.png"), lr_patch)
        return True
    except Exception as e:
        print(f"Error saving patch {name}: {e}")
        return False


# ==============================================================================
#                        IMAGE PROCESSING LOGIC
# ==============================================================================

def process_image(filepath: Path, hr_dir: Path, lr_dir: Path, use_patches: bool) -> int:
    """
    Processes one image into HR/LR pairs (patches or full images).
    """
    hr_rgb = load_image(filepath)
    if hr_rgb is None or hr_rgb.ndim < 3:
        return 0

    # Convert to OpenCV BGR for consistency
    hr_img = cv2.cvtColor(hr_rgb, cv2.COLOR_RGB2BGR)
    hr_img = convert_uint16_to_uint8(hr_img)
    hr_img = crop_image_to_scale(hr_img, config.SCALE)

    # Create LR version
    lr_img = downsample_bicubic(hr_img, config.SCALE)

    if use_patches:
        hr_patches = create_patches(hr_img, config.PATCH_SIZE, config.STEP)
        lr_patch_size = config.PATCH_SIZE // config.SCALE
        lr_patches = create_patches(lr_img, lr_patch_size, config.STEP // config.SCALE)

        count = 0
        for i, (hr_p, lr_p) in enumerate(zip(hr_patches, lr_patches)):
            name = f"{filepath.stem}_patch{i:04d}"
            if save_patch_pair(hr_p, lr_p, hr_dir, lr_dir, name):
                count += 1
        return count

    else:
        # Full image pair
        name = filepath.stem
        return 1 if save_patch_pair(hr_img, lr_img, hr_dir, lr_dir, name) else 0


# ==============================================================================
#                              PARALLEL WRAPPERS
# ==============================================================================

def process_training_image(args: Tuple[Path, Path, Path]) -> int:
    filepath, hr_dir, lr_dir = args
    return process_image(filepath, hr_dir, lr_dir, use_patches=True)


def process_validation_image(args: Tuple[Path, Path, Path]) -> int:
    filepath, hr_dir, lr_dir = args
    return process_image(filepath, hr_dir, lr_dir, use_patches=False)


# ==============================================================================
#                         OUTPUT DIRECTORY MANAGEMENT
# ==============================================================================

def clean_directory(path: Path):
    """Deletes and recreates a directory."""
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def setup_output_dirs(base: Path, subdirs=("HR", "LR"), clean=True):
    """Creates HR/LR subdirectories."""
    if clean:
        clean_directory(base)
    for sub in subdirs:
        (base / sub).mkdir(parents=True, exist_ok=True)


# ==============================================================================
#                             PIPELINE MODES
# ==============================================================================

def patch_image(source: Path, output: Path):
    """Processes a single HR image into HR/LR patches."""
    print(f"\n{'='*70}")
    print(f"Single Image Patch Generation")
    print(f"{'='*70}\n")

    setup_output_dirs(output)
    hr_dir, lr_dir = output / "HR", output / "LR"

    count = process_image(source, hr_dir, lr_dir, use_patches=True)
    print(f"\nComplete! Generated {count} HR/LR patch pairs at {output}")


def patch_bulk(input_dir: Path, output_dir: Path):
    """Processes all images in a directory into HR/LR patches."""
    print(f"\n{'='*70}")
    print(f"Bulk Directory Patch Generation")
    print(f"{'='*70}\n")

    setup_output_dirs(output_dir)
    hr_dir, lr_dir = output_dir / "HR", output_dir / "LR"

    image_files = []
    for ext in config.SUPPORTED_EXTENSIONS:
        image_files.extend(input_dir.rglob(f"*{ext.lower()}"))
        image_files.extend(input_dir.rglob(f"*{ext.upper()}"))

    if not image_files:
        print(f"ERROR: No images found in {input_dir}")
        return

    print(f"Found {len(image_files)} images. Processing...\n")

    total_patches = 0
    num_workers = max(1, config.NUM_WORKERS)
    tasks = [(path, hr_dir, lr_dir) for path in image_files]

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        with tqdm(total=len(tasks), desc="Images") as pbar:
            for count in executor.map(process_training_image, tasks):
                total_patches += count
                pbar.update(1)

    print(f"\nBulk Complete! Generated {total_patches} patches at {output_dir}")


def patch_dataset():
    """Full dataset generation: train (patches), val/test (full images)."""
    print(f"\n{'='*70}")
    print(f"Full Dataset Generation Pipeline")
    print(f"{'='*70}\n")

    train_base = config.DATA_DIR / f"train_{config.PATCH_SIZE}"
    val_base = config.DATA_DIR / "val"
    test_base = config.DATA_DIR / "test"

    for base in [train_base, val_base, test_base]:
        setup_output_dirs(base, clean=False)  # don't wipe existing processed images

    # Helper: find all images
    def find_all(paths):
        files = []
        for p in paths:
            for ext in config.SUPPORTED_EXTENSIONS:
                files.extend(p.rglob(f"*{ext.lower()}"))
                files.extend(p.rglob(f"*{ext.upper()}"))
        return sorted(list(set(files)))

    train_val_files = find_all(config.TRAIN_VAL_SOURCES)
    test_files = find_all(config.TEST_SOURCES)

    # Fallback to test split
    if not test_files:
        print("No test images found — creating test split from training set.")
        random.shuffle(train_val_files)
        test_split = int(len(train_val_files) * 0.05)  # 5% test split
        test_files = train_val_files[:test_split]
        train_val_files = train_val_files[test_split:]

    if not train_val_files:
        print("ERROR: No training/validation images found.")
        sys.exit(1)

    # Split train/val
    random.shuffle(train_val_files)
    val_split = int(len(train_val_files) * config.VAL_SPLIT)
    val_files = train_val_files[:val_split]
    train_files = train_val_files[val_split:]

    print(f"Training: {len(train_files)} | Validation: {len(val_files)} | Test: {len(test_files)}")

    num_workers = max(1, config.NUM_WORKERS)

    # Helper: skip processed
    def skip_existing(tasks, hr_dir):
        remaining = []
        existing = {p.stem for p in hr_dir.glob("*.png")}
        for (f, hr, lr) in tasks:
            if f.stem in existing:
                continue
            remaining.append((f, hr, lr))
        skipped = len(tasks) - len(remaining)
        if skipped > 0:
            print(f"Skipping {skipped} already-processed images in {hr_dir.parent.name}")
        return remaining

    # Training
    print("\nProcessing training images...")
    train_tasks = [(f, train_base / "HR", train_base / "LR") for f in train_files]
    train_tasks = skip_existing(train_tasks, train_base / "HR")

    total_train_patches = 0
    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        with tqdm(total=len(train_tasks), desc="Training") as pbar:
            for count in ex.map(process_training_image, train_tasks):
                total_train_patches += count
                pbar.update(1)

    # Validation
    if val_files:
        print("\nProcessing validation images...")
        val_tasks = [(f, val_base / "HR", val_base / "LR") for f in val_files]
        val_tasks = skip_existing(val_tasks, val_base / "HR")

        with ProcessPoolExecutor(max_workers=num_workers) as ex:
            list(tqdm(ex.map(process_validation_image, val_tasks),
                      total=len(val_tasks), desc="Validation"))

    # Testing
    if test_files:
        print("\nProcessing test images...")
        test_tasks = [(f, test_base / "HR", test_base / "LR") for f in test_files]
        test_tasks = skip_existing(test_tasks, test_base / "HR")

        with ProcessPoolExecutor(max_workers=num_workers) as ex:
            list(tqdm(ex.map(process_validation_image, test_tasks),
                      total=len(test_tasks), desc="Testing"))

    print(f"\nDataset Generation Complete!")
    print(f"  → {total_train_patches} training patches")
    print(f"  → {len(val_files)} validation pairs")
    print(f"  → {len(test_files)} test pairs")
    print(f"\nOutput root: {config.DATA_DIR}")



# ==============================================================================
#                               CLI ENTRYPOINT
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Unified Super-Resolution Patch Generation Script",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--single_image", type=str, help="Path to a single HR image file.")
    group.add_argument("--input_dir", type=str, help="Directory containing HR images.")
    parser.add_argument("--output_dir", type=str, help="Output directory (required for single/bulk modes).")
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.single_image and not args.input_dir:
        print("Running full dataset pipeline (train/val/test)...")
        patch_dataset()
        return

    if not args.output_dir:
        print("ERROR: --output_dir is required for single_image or input_dir modes.")
        sys.exit(1)

    output = Path(args.output_dir).expanduser()

    if args.single_image:
        source = Path(args.single_image).expanduser()
        if not source.is_file():
            print(f"File not found: {source}")
            sys.exit(1)
        patch_image(source, output)

    elif args.input_dir:
        input_dir = Path(args.input_dir).expanduser()
        if not input_dir.is_dir():
            print(f"Directory not found: {input_dir}")
            sys.exit(1)
        patch_bulk(input_dir, output)


if __name__ == "__main__":
    main()
