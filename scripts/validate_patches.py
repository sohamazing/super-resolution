import argparse
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from multiprocessing import Pool
import os

def validate_train_patch_pair(args_tuple):
    """
    Worker function for the TRAINING set.
    Validates HR/LR patch pairs for:
    - File existence
    - Corruption
    - Dimension correctness
    - Scale consistency
    """
    hr_patch_path, lq_train_dir, expected_hq_size, expected_lq_size, scale = args_tuple
    
    lq_patch_path = lq_train_dir / hr_patch_path.name
    if not lq_patch_path.exists():
        return f"TRAIN_ERROR: Missing LQ pair for {hr_patch_path.name}"

    # --- Validate HR patch ---
    try:
        with Image.open(hr_patch_path) as hr_img:
            hr_img.verify()
            hr_size = hr_img.size
            if hr_size != expected_hq_size:
                return f"TRAIN_ERROR: HQ dimension mismatch for {hr_patch_path.name}. Expected {expected_hq_size}, got {hr_size}"
    except Exception as e:
        return f"TRAIN_ERROR: Corrupted HQ file {hr_patch_path.name} - {e}"

    # --- Validate LR patch ---
    try:
        with Image.open(lq_patch_path) as lr_img:
            lr_img.verify()
            lr_size = lr_img.size
            if lr_size != expected_lq_size:
                return f"TRAIN_ERROR: LQ dimension mismatch for {lq_patch_path.name}. Expected {expected_lq_size}, got {lr_size}"
            # --- Scale Consistency Check ---
            if hr_size[0] != lr_size[0] * scale or hr_size[1] != lr_size[1] * scale:
                return (f"TRAIN_ERROR: SCALE_MISMATCH for {hr_patch_path.name} — "
                        f"HR: {hr_size}, LR: {lr_size}, expected HR=LR×{scale}")
    except Exception as e:
        return f"TRAIN_ERROR: Corrupted LQ file {lq_patch_path.name} - {e}"
        
    return None


def validate_full_image_pair(args_tuple):
    """
    Worker function for the VALIDATION/TEST sets.
    Validates existence, corruption, and dimensional consistency.
    """
    hr_image_path, lq_dir, scale = args_tuple

    lq_image_path = lq_dir / hr_image_path.name
    if not lq_image_path.exists():
        return f"VAL/TEST_ERROR: Missing LQ pair for {hr_image_path.name}"

    # --- HR image ---
    try:
        with Image.open(hr_image_path) as hr_img:
            hr_img.verify()
            hr_size = hr_img.size
    except Exception as e:
        return f"VAL/TEST_ERROR: Corrupted HQ file {hr_image_path.name} - {e}"

    # --- LR image ---
    try:
        with Image.open(lq_image_path) as lr_img:
            lr_img.verify()
            lr_size = lr_img.size
            expected_lr_size = (hr_size[0] // scale, hr_size[1] // scale)
            if lr_size != expected_lr_size:
                return (f"VAL/TEST_ERROR: SCALE_MISMATCH for {hr_image_path.name} — "
                        f"HR: {hr_size}, LR: {lr_size}, expected LR={expected_lr_size}")
    except Exception as e:
        return f"VAL/TEST_ERROR: Corrupted LQ file {lq_image_path.name} - {e}"
    
    return None


def run_validation(desc, worker_func, tasks, num_workers):
    """Helper to run validation stage with progress bar."""
    print(f"\n--- Validating {desc} Set ---")
    if not tasks:
        print("No files found to validate.")
        return []
    
    print(f"Found {len(tasks)} pairs to validate...")
    errors = []
    with Pool(processes=num_workers) as pool:
        for result in tqdm(pool.imap_unordered(worker_func, tasks), total=len(tasks), desc=f"Validating {desc}"):
            if result:
                errors.append(result)
    return errors


def main(args):
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"❌ ERROR: Data directory '{data_dir}' not found.")
        return

    paths = {
        "hr_train": data_dir / "train" / "HR", "lr_train": data_dir / "train" / "LR",
        "hr_val": data_dir / "val" / "HR", "lr_val": data_dir / "val" / "LR",
        "hr_test": data_dir / "test" / "HR", "lr_test": data_dir / "test" / "LR",
    }

    all_errors = []

    # --- 1. Validate Training Patches ---
    expected_hq_size = (args.patch_size, args.patch_size)
    expected_lq_size = (args.patch_size // args.scale, args.patch_size // args.scale)
    hr_train_patches = list(paths["hr_train"].glob("*.png"))
    train_tasks = [(p, paths["lr_train"], expected_hq_size, expected_lq_size, args.scale)
                   for p in hr_train_patches]
    all_errors.extend(run_validation("Training", validate_train_patch_pair, train_tasks, args.num_workers))

    # --- 2. Validate Validation Images ---
    hr_val_images = list(paths["hr_val"].glob("*.png"))
    val_tasks = [(p, paths["lr_val"], args.scale) for p in hr_val_images]
    all_errors.extend(run_validation("Validation", validate_full_image_pair, val_tasks, args.num_workers))

    # --- 3. Validate Test Images ---
    hr_test_images = list(paths["hr_test"].glob("*.png"))
    test_tasks = [(p, paths["lr_test"], args.scale) for p in hr_test_images]
    all_errors.extend(run_validation("Test", validate_full_image_pair, test_tasks, args.num_workers))
    
    # --- 4. Summary ---
    print("\n--- Final Validation Summary ---")
    if not all_errors:
        print("VALIDATION SUCCESS! Dataset is clean and scale-consistent.")
    else:
        print(f"INVALID DATASET! Found {len(all_errors)} issues.")
        log_file = data_dir / "validation_errors.log"
        with open(log_file, "w") as f:
            for e in sorted(all_errors):
                f.write(f"{e}\n")
        print(f"Details saved to: {log_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate Super-Resolution dataset (HR/LR pairs, scale, corruption).")
    parser.add_argument("--data_dir", type=str, default="div2K-flickr2K-data", help="Root dataset directory.")
    parser.add_argument("--patch_size", type=int, default=256, help="Expected HR patch size for training set.")
    parser.add_argument("--scale", type=int, default=4, help="Super-resolution scale factor.")
    parser.add_argument("--num_workers", type=int, default=os.cpu_count() or 4, help="Parallel worker count.")
    
    args = parser.parse_args()
    main(args)
