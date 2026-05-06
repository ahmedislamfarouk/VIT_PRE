"""
Dataset loading, transformations, and custom PyTorch Dataset classes.

This module provides:
  - Dataset path resolution (local cache or Kaggle download).
  - Standard ImageNet-normalized image transforms for train/val.
  - Custom Dataset classes for distillation (teacher logits + image pairs).
  - Stratified subset selection for data-efficiency experiments.

All Datasets return (image, label) tuples, with distillation variants
adding a third element for pre-computed teacher logits.
"""

import os
from pathlib import Path

import torch
from torch.utils.data import Dataset, Subset
from torchvision import transforms, datasets
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

from mstd.config import DATASET_NAME, IMG_SIZE, IMAGENET_STATS, NUM_CLASSES


def get_dataset_path() -> str:
    """
    Resolve the path to the Intel Image Classification dataset.

    Checks a well-known local cache path first; falls back to downloading
    via kagglehub if not found.
    """
    cached = "/home/skyvision/.cache/kagglehub/datasets/puneet6060/intel-image-classification/versions/2"
    if os.path.exists(cached):
        return cached
    import kagglehub
    return kagglehub.dataset_download(DATASET_NAME)


def download_dataset() -> str:
    """Download the Intel Image Classification dataset via kagglehub."""
    import kagglehub
    path = kagglehub.dataset_download(DATASET_NAME)
    return path


def make_train_transform(augment: bool = True):
    """
    Build a training image transform pipeline.

    When augment=True (default), applies: resize, random horizontal flip,
    random rotation, color jitter, tensor conversion, ImageNet normalization,
    and random erasing. Without augmentation, only resize + normalize.
    """
    if augment:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(**IMAGENET_STATS),
            transforms.RandomErasing(p=0.25),
        ])
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(**IMAGENET_STATS),
    ])


def make_val_transform():
    """Build a validation/test transform pipeline (resize + normalize only)."""
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(**IMAGENET_STATS),
    ])


def make_augment_transform():
    """Build a lightweight augmentation pipeline (flip + normalize)."""
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(**IMAGENET_STATS),
    ])


class DistillDataset(Dataset):
    """
    Dataset for distillation training.

    Each sample returns (augmented_image, label, teacher_logits).

    The teacher_logits are pre-computed (by precompute.py) and served
    alongside each image so the student can compute the KL-divergence
    loss against the teacher ensemble during training.
    """

    def __init__(self, base_dataset, teacher_logits, transform_strong=None):
        self.base = base_dataset
        self.logits = teacher_logits
        self.transform_strong = transform_strong

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, label = self.base[idx]
        if self.transform_strong:
            img = self.transform_strong(img)
        return img, label, self.logits[idx]


class LogitsDataset(Dataset):
    """
    Dataset variant where the transform is applied inside __getitem__
    (rather than being baked into the base_dataset).

    Used for data-efficiency experiments where we need to pre-compute
    teacher logits on a subset with a validation transform, then apply
    augmentation on-the-fly during training.
    """

    def __init__(self, base_ds, teacher_logits, transform):
        self.base = base_ds
        self.logits = teacher_logits
        self.transform = transform

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, label = self.base[idx]
        return self.transform(img), label, self.logits[idx]


class SimpleDataset(Dataset):
    """
    Minimal dataset that applies a transform and returns (img, label).

    Used for standard (non-distillation) training loops.
    """

    def __init__(self, base_ds, transform):
        self.base = base_ds
        self.transform = transform

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, label = self.base[idx]
        return self.transform(img), label


def get_subset_indices(dataset, ratio, seed):
    """
    Perform stratified random sampling to select a subset of indices.

    Uses StratifiedShuffleSplit to preserve class distribution.
    If ratio >= 1.0, returns all indices.

    Args:
        dataset: torchvision Dataset with a .targets attribute.
        ratio: Fraction of data to keep (0.0 to 1.0).
        seed: Random seed for reproducibility.

    Returns:
        NumPy array of selected indices.
    """
    targets = np.array([dataset.targets[i] for i in range(len(dataset))])
    if ratio >= 1.0:
        return np.arange(len(targets))
    sss = StratifiedShuffleSplit(n_splits=1, train_size=ratio, random_state=seed)
    indices, _ = next(sss.split(np.zeros(len(targets)), targets))
    return indices
