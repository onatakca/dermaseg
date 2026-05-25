"""ISIC 2018 torch Dataset with albumentations augmentation.

Pairs each dermoscopic image with its binary ground-truth mask, applies a
reproducible train/val/test split (config.SEED / VAL_SPLIT / TEST_SPLIT), and
returns (image, mask) tensors for SegFormer training. Run
`python scripts/download_data.py` first to populate config.RAW_DATA_DIR.
"""

from __future__ import annotations

import random
from pathlib import Path

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset

from src import config

IMAGES_DIR = config.RAW_DATA_DIR / "images"
MASKS_DIR = config.RAW_DATA_DIR / "masks"
MASK_SUFFIX = "_segmentation.png"


def list_pairs(
    images_dir: Path = IMAGES_DIR, masks_dir: Path = MASKS_DIR
) -> list[tuple[Path, Path]]:
    """Return (image, mask) paths matched by ISIC id; unmatched images are skipped."""
    pairs = [
        (img, masks_dir / f"{img.stem}{MASK_SUFFIX}")
        for img in sorted(images_dir.glob("*.jpg"))
        if (masks_dir / f"{img.stem}{MASK_SUFFIX}").exists()
    ]
    if not pairs:
        raise FileNotFoundError(
            f"No image/mask pairs under {images_dir} and {masks_dir}. "
            "Run `python scripts/download_data.py` first."
        )
    return pairs


def split_pairs(
    pairs: list[tuple[Path, Path]],
    seed: int = config.SEED,
    val_split: float = config.VAL_SPLIT,
    test_split: float = config.TEST_SPLIT,
) -> dict[str, list[tuple[Path, Path]]]:
    """Deterministically shuffle and split pairs into train/val/test."""
    pairs = list(pairs)
    random.Random(seed).shuffle(pairs)
    n = len(pairs)
    n_test = int(n * test_split)
    n_val = int(n * val_split)
    return {
        "test": pairs[:n_test],
        "val": pairs[n_test : n_test + n_val],
        "train": pairs[n_test + n_val :],
    }


def build_transforms(train: bool) -> A.Compose:
    """Albumentations pipeline. A. resizes masks with nearest-neighbor, keeping them binary."""
    aug: list = [A.Resize(config.IMAGE_SIZE, config.IMAGE_SIZE)]
    if train:
        aug += [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Rotate(limit=20, border_mode=0, p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.5),
        ]
    aug += [A.Normalize(mean=config.IMAGE_MEAN, std=config.IMAGE_STD), ToTensorV2()]
    return A.Compose(aug)


class ISICDataset(Dataset):
    """ISIC 2018 lesion segmentation: yields (image CHW float, mask HW int64)."""

    def __init__(self, pairs: list[tuple[Path, Path]], train: bool) -> None:
        self.pairs = pairs
        self.transforms = build_transforms(train)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img_path, mask_path = self.pairs[idx]
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"))
        mask = (mask > 127).astype(np.uint8)  # {0, 255} -> {0, 1}
        out = self.transforms(image=image, mask=mask)
        return out["image"], out["mask"].long()


def get_datasets() -> dict[str, ISICDataset]:
    """Build train/val/test datasets from the on-disk ISIC data."""
    splits = split_pairs(list_pairs())
    return {
        "train": ISICDataset(splits["train"], train=True),
        "val": ISICDataset(splits["val"], train=False),
        "test": ISICDataset(splits["test"], train=False),
    }
