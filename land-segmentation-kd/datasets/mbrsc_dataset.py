"""
PyTorch Dataset class for the MBRSC Dubai segmentation dataset.
Expects pre-processed 224×224 patches (images as PNG, masks as .npy).
"""
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path

import albumentations as A
from albumentations.pytorch import ToTensorV2

NUM_CLASSES = 6

CLASS_NAMES = ["building", "land", "road", "vegetation", "water", "unlabeled"]


def get_train_transforms(img_size=224):
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, p=0.3),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def get_val_transforms(img_size=224):
    return A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


class MBRSCDataset(Dataset):
    """MBRSC Dubai aerial segmentation dataset.

    Parameters
    ----------
    img_dir : str | Path
        Directory containing patch PNG images.
    mask_dir : str | Path
        Directory containing patch .npy mask files.
    split_file : str | Path
        Text file listing patch IDs (one per line, without extension).
    transform : albumentations.Compose, optional
        Augmentation / preprocessing pipeline.
    """

    def __init__(self, img_dir, mask_dir, split_file, transform=None):
        self.img_dir = Path(img_dir)
        self.mask_dir = Path(mask_dir)
        self.transform = transform
        with open(split_file) as f:
            self.ids = [line.strip() for line in f.readlines() if line.strip()]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        name = self.ids[idx]
        img = cv2.cvtColor(
            cv2.imread(str(self.img_dir / f"{name}.png")),
            cv2.COLOR_BGR2RGB,
        )
        mask = np.load(str(self.mask_dir / f"{name}.npy")).astype(np.int64)

        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img = augmented["image"]
            mask = augmented["mask"]

        return img, mask.long() if isinstance(mask, torch.Tensor) else torch.from_numpy(mask).long()
