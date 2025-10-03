# utils/datasets.py
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from pathlib import Path
from PIL import Image

class SuperResDataset(Dataset):
    """A versatile dataset for super-resolution tasks that returns (LR, HR) image pairs."""
    def __init__(self, hr_dir, lr_dir):
        self.hr_paths = sorted(list(Path(hr_dir).glob("*.png")))
        self.lr_paths = sorted(list(Path(lr_dir).glob("*.png")))
        self.transform = ToTensor()

    def __len__(self):
        return len(self.hr_paths)

    def __getitem__(self, index):
        hr_image = Image.open(self.hr_paths[index]).convert("RGB")
        lr_image = Image.open(self.lr_paths[index]).convert("RGB")
        # Return as (input, ground_truth) -> (lr, hr)
        return self.transform(lr_image), self.transform(hr_image)