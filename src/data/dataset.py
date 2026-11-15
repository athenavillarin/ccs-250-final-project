"""Shared FloodNet loading so every method reads the same splits and binary masks."""
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.transforms import v2

from src.data.classes import to_binary
from src.utils.image import resize_long
from src.utils.paths import DATA_DIR, SPLITS_DIR


def read_split(name: str) -> list[str]:
    """Return the image ids listed in splits/<name>.txt."""
    return (SPLITS_DIR / f"{name}.txt").read_text().split()


def split_dir(name: str) -> Path:
    """Map a split name (e.g. 'train_25') to its official folder on disk."""
    return DATA_DIR / "floodnet" / name.split("_")[0]


def load_pair(split: str, image_id: str) -> tuple[np.ndarray, np.ndarray]:
    """Load an HxWx3 uint8 RGB image and its HxW uint8 binary flood mask."""
    root = split_dir(split)
    image = np.asarray(Image.open(root / "images" / f"{image_id}.jpg").convert("RGB"))
    mask = to_binary(np.asarray(Image.open(root / "masks" / f"{image_id}.png")))
    return image, mask


TRAIN_SIDE = 512
CROP = 384


class FloodDataset(Dataset):
    """Torch dataset of (image in [0,1], binary mask) pairs at TRAIN_SIDE, cached in memory."""

    def __init__(self, split: str, train: bool = False):
        """Load and resize every pair once; enable augmentation when train is True."""
        self.items = []
        for image_id in read_split(split):
            image, mask = load_pair(split, image_id)
            image = np.array(resize_long(image, TRAIN_SIDE))
            mask = np.array(resize_long(mask, TRAIN_SIDE, Image.NEAREST))
            self.items.append((image, mask))
        self.augment = v2.Compose([
            v2.RandomResizedCrop(CROP, scale=(0.3, 1.0), antialias=True),
            v2.RandomHorizontalFlip(),
            v2.RandomVerticalFlip(),
            v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.03),
        ]) if train else None

    def __len__(self) -> int:
        """Number of image/mask pairs."""
        return len(self.items)

    def __getitem__(self, i: int):
        """Return a float CxHxW image in [0,1] and a float 1xHxW mask."""
        image, mask = self.items[i]
        x = tv_tensors.Image(torch.from_numpy(image).permute(2, 0, 1))
        y = tv_tensors.Mask(torch.from_numpy(mask)[None])
        if self.augment:
            x, y = self.augment(x, y)
        return x.float() / 255, y.float()
