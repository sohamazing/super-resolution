# utils/datasets.py
import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import random
from abc import ABC, abstractmethod
import sys

SCRIPT_DIR = Path(__file__).parent.absolute()
sys.path.append(str(SCRIPT_DIR.parent))
from config import config

class SuperResDataset(Dataset, ABC):
    """
    Abstract base class for Super-Resolution datasets.
    Handles file discovery, normalization, and common utilities.
    """
    def __init__(self, hr_dir: str, lr_dir: str, scale_factor: int = 4):
        self.hr_dir = Path(hr_dir)
        self.lr_dir = Path(lr_dir)
        self.scale_factor = scale_factor
        # Collect HR–LR pairs by filename matching
        hr_candidates = sorted(list(self.hr_dir.glob("*.png")) + list(self.hr_dir.glob("*.jpg")))
        self.pairs = [
            (self.lr_dir / p.name, p)
            for p in hr_candidates
            if (self.lr_dir / p.name).exists()
        ]
        if not self.pairs:
            raise FileNotFoundError(f"No matching HR–LR pairs found in {hr_dir} and {lr_dir}")
        # Shared transforms
        self.to_tensor = T.ToTensor()
        self.normalize = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    def __len__(self):
        return len(self.pairs)

    @abstractmethod
    def __getitem__(self, index):
        pass

class TrainDatasetCropped(SuperResDataset):
    """
    Training dataset for super-resolution.
    Uses pre-cropped aligned patches; applies only random flips/rotations.
    """
    def __getitem__(self, index: int):
        lr_path, hr_path = self.pairs[index]
        hr = Image.open(hr_path).convert("RGB")
        lr = Image.open(lr_path).convert("RGB")
        # Convert to tensors early for easy augmentation
        hr, lr = self.to_tensor(hr), self.to_tensor(lr)
        # Apply identical random augmentations (no cropping)
        if random.random() > 0.5:
            hr, lr = TF.hflip(hr), TF.hflip(lr)
        if random.random() > 0.5:
            hr, lr = TF.vflip(hr), TF.vflip(lr)
        if random.random() > 0.5:
            angle = random.choice([90, 180, 270])
            hr = TF.rotate(hr, angle, interpolation=TF.InterpolationMode.BILINEAR)
            lr = TF.rotate(lr, angle, interpolation=TF.InterpolationMode.BILINEAR)

        return self.normalize(lr), self.normalize(hr)


class TrainDataset(SuperResDataset):
    """
    Training dataset that performs SYNCHRONIZED random cropping on HR/LR pairs.
    This allows training with different patch sizes without re-generating the dataset.
    It also acts as a powerful data augmentation.
    """
    def __getitem__(self, index: int):
        lr_path, hr_path = self.pairs[index]
        hr_image = Image.open(hr_path).convert("RGB")
        lr_image = Image.open(lr_path).convert("RGB")

        # Get the parameters for a random crop
        i, j, h, w = T.RandomCrop.get_params(
            hr_image, output_size=(config.HR_CROP_SIZE, config.HR_CROP_SIZE)
        )

        # Apply the exact same crop to the HR image
        hr_image = TF.crop(hr_image, i, j, h, w)

        # Apply the corresponding crop to the LR image
        # We must scale the parameters by the scale factor
        lr_i = i // self.scale_factor
        lr_j = j // self.scale_factor
        lr_h = h // self.scale_factor
        lr_w = w // self.scale_factor
        lr_image = TF.crop(lr_image, lr_i, lr_j, lr_h, lr_w)

        # --- Standard Augmentations (flips, rotates) ---
        if random.random() > 0.5:
            hr_image, lr_image = TF.hflip(hr_image), TF.hflip(lr_image)
        if random.random() > 0.5:
            hr_image, lr_image = TF.vflip(hr_image), TF.vflip(lr_image)
        
        # ADD THIS BLOCK BACK IN
        if random.random() > 0.5:
            angle = random.choice([90, 180, 270])
            hr_image = TF.rotate(hr_image, angle)
            lr_image = TF.rotate(lr_image, angle)
        
        # Convert to tensor and normalize
        hr_tensor = self.normalize(self.to_tensor(hr_image))
        lr_tensor = self.normalize(self.to_tensor(lr_image))

        return lr_tensor, hr_tensor


class TrainDatasetAugmented(TrainDataset):
    """
    Virtually augments the dataset by a multiplier factor.
    For each original image, it provides 'multiplier' number of unique random crops per epoch.
    """
    def __init__(self, hr_dir: str, lr_dir: str, multiplier: int = 64):
        super().__init__(hr_dir, lr_dir)
        self.multiplier = multiplier
        print(f"Dataset augmented by factor of {multiplier}. Original size: {len(self.pairs)}, New size: {self.__len__()}")

    def __len__(self):
        # Return the "multiplied" length of the dataset
        return len(self.pairs) * self.multiplier

    def __getitem__(self, index: int):
        # Use the new index to find the original image's index
        # For example, with a multiplier of 64:
        # indices 0-63 will map to original_index 0
        # indices 64-127 will map to original_index 1
        original_index = index // self.multiplier
        
        # Now, call the parent class's __getitem__ method with the original index.
        # Because it's called separately for each of the 64 indices,
        # the random crop will be different each time.
        return super().__getitem__(original_index)



class ValDataset(SuperResDataset):
    """
    Validation/Test dataset for super-resolution.
    Center-crops non-cropped validation images to fixed size.
    """
    def __init__(self, hr_dir: str, lr_dir: str):
        super().__init__(hr_dir, lr_dir)
        self.hr_transform = T.Compose([
            T.CenterCrop(config.HR_CROP_SIZE),
            T.ToTensor(),
            self.normalize
        ])
        self.lr_transform = T.Compose([
            T.CenterCrop(config.LR_CROP_SIZE),
            T.ToTensor(),
            self.normalize
        ])

    def __getitem__(self, index: int):
        lr_path, hr_path = self.pairs[index]
        hr = Image.open(hr_path).convert("RGB")
        lr = Image.open(lr_path).convert("RGB")
        return self.lr_transform(lr), self.hr_transform(hr)


class ValDatasetGrid(SuperResDataset):
    """
    Generates a deterministic grid of patches from full-size validation images on-the-fly.
    Used for comprehensive evaluation across the entire image.
    """
    def __init__(self, hr_dir: str, lr_dir: str):
        super().__init__(hr_dir, lr_dir)
        self.patch_size_hr = config.HR_CROP_SIZE
        self.stride_hr = config.HR_CROP_SIZE # no overlap. 50% overlap: config.HR_CROP_SIZE//2
        
        self.patch_map = []
        print("Pre-calculating validation grid patches...")
        for img_index, (lr_path, hr_path) in enumerate(self.pairs):
            # We only need the dimensions, so open with PIL
            with Image.open(hr_path) as hr_image:
                w, h = hr_image.size
            
            # Calculate how many patches fit in the image
            for i in range(0, h - self.patch_size_hr + 1, self.stride_hr):
                for j in range(0, w - self.patch_size_hr + 1, self.stride_hr):
                    # Store the image index and the top-left corner of the crop
                    self.patch_map.append((img_index, i, j))
        
        print(f"Validation set: {len(self.pairs)} images, {len(self.patch_map)} total grid patches.")

    def __len__(self):
        # The length is the total number of patches we can extract
        return len(self.patch_map)

    def __getitem__(self, index: int):
        # Look up the image index and its specific crop coordinates
        img_index, i, j = self.patch_map[index]
        
        lr_path, hr_path = self.pairs[img_index]
        hr_image = Image.open(hr_path).convert("RGB")
        lr_image = Image.open(lr_path).convert("RGB")
        
        # --- Perform the DETERMINISTIC crop ---
        h, w = self.patch_size_hr, self.patch_size_hr
        hr_image = TF.crop(hr_image, i, j, h, w)
        
        # Apply the corresponding crop to the LR image
        lr_i, lr_j = i // self.scale_factor, j // self.scale_factor
        lr_h, lr_w = h // self.scale_factor, w // self.scale_factor
        lr_image = TF.crop(lr_image, lr_i, lr_j, lr_h, lr_w)
        
        # No random augmentations for validation
        hr_tensor = self.normalize(self.to_tensor(hr_image))
        lr_tensor = self.normalize(self.to_tensor(lr_image))

        return lr_tensor, hr_tensor
