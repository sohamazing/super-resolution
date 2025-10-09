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

class TrainDataset(SuperResDataset):
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


