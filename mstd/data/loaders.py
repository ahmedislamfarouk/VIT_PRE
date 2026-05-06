"""
DataLoader factory functions.

Provides standardized DataLoader creation for different training regimes:
  - create_dataloaders:       Standard baseline training (compare.py).
  - create_distill_loaders:   Distillation training with teacher logits.
  - create_subset_loaders:    Data-efficiency experiments (stratified subsets
                              with live teacher logit computation).
  - create_scratch_loaders:   From-scratch training with simple datasets.
"""

import os

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets
from tqdm import tqdm

from mstd.config import IMG_SIZE, IMAGENET_STATS, NUM_CLASSES, DEVICE
from mstd.data.dataset import (
    make_train_transform, make_val_transform,
    DistillDataset, LogitsDataset, SimpleDataset,
    get_subset_indices,
)
from mstd.models.teacher import TeacherEnsemble


def create_dataloaders(dataset_path: str, batch_size: int = 32):
    """
    Create train/val DataLoaders for standard baseline training.

    Uses moderate augmentation (flip, rotation) for training and
    a simple resize + normalize for validation.
    """
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(**IMAGENET_STATS),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(**IMAGENET_STATS),
    ])

    train_dir = os.path.join(dataset_path, "seg_train", "seg_train")
    val_dir = os.path.join(dataset_path, "seg_test", "seg_test")

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True,
    )

    return train_loader, val_loader


def create_distill_loaders(dataset_path, teacher_logits, batch_size, use_augment):
    """
    Create DataLoaders for distillation training.

    Uses DistillDataset which returns (image, label, teacher_logits) tuples.
    The teacher_logits are pre-computed soft targets from the teacher ensemble.
    """
    train_transform = make_train_transform(use_augment)
    val_transform = make_val_transform()

    train_dir = os.path.join(dataset_path, "seg_train", "seg_train")
    val_dir = os.path.join(dataset_path, "seg_test", "seg_test")

    train_base = datasets.ImageFolder(train_dir)
    val_base = datasets.ImageFolder(val_dir, transform=val_transform)

    train_ds = DistillDataset(train_base, teacher_logits, transform_strong=train_transform)
    val_ds = DistillDataset(val_base, torch.zeros(len(val_base), NUM_CLASSES))

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )
    return train_loader, val_loader


def create_subset_loaders(dataset_path, subset_ratio, batch_size=64, seed=42):
    """
    Create DataLoaders for data-efficiency experiments.

    A stratified subset of the training data is selected and teacher logits
    are computed on-the-fly for that subset using the TeacherEnsemble.
    The validation set is always full-size.
    """
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(),
        transforms.Normalize(**IMAGENET_STATS),
        transforms.RandomErasing(p=0.25),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(**IMAGENET_STATS),
    ])

    train_base = datasets.ImageFolder(os.path.join(dataset_path, "seg_train", "seg_train"))
    val_base = datasets.ImageFolder(os.path.join(dataset_path, "seg_test", "seg_test"), transform=val_transform)

    indices = get_subset_indices(train_base, subset_ratio, seed)
    subset_base = Subset(train_base, indices)

    tmp_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(**IMAGENET_STATS),
    ])
    tmp_base = datasets.ImageFolder(os.path.join(dataset_path, "seg_train", "seg_train"), transform=tmp_transform)
    subset_loader = DataLoader(
        Subset(tmp_base, indices), batch_size=64, shuffle=False, num_workers=4, pin_memory=True
    )

    teacher_model = TeacherEnsemble().to(DEVICE)
    all_logits = []
    for imgs, _ in tqdm(subset_loader, leave=False):
        imgs = imgs.to(DEVICE)
        with torch.no_grad():
            logits = teacher_model(imgs)
        all_logits.append(logits.cpu())
    teacher_logits = torch.cat(all_logits, dim=0)

    train_ds = LogitsDataset(subset_base, teacher_logits, train_transform)
    val_ds = LogitsDataset(val_base, torch.zeros(len(val_base), NUM_CLASSES), val_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, val_loader


def create_scratch_loaders(dataset_path, subset_ratio, batch_size=64, seed=42):
    """
    Create DataLoaders for from-scratch training experiments.

    Returns a tuple of (train_loader, val_loader, n_samples).
    Uses SimpleDataset (no teacher logits), with subset selection.
    """
    train_t = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(),
        transforms.Normalize(**IMAGENET_STATS),
    ])
    val_t = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(**IMAGENET_STATS),
    ])

    train_base = datasets.ImageFolder(os.path.join(dataset_path, "seg_train", "seg_train"))
    val_base = datasets.ImageFolder(os.path.join(dataset_path, "seg_test", "seg_test"), transform=val_t)

    indices = get_subset_indices(train_base, subset_ratio, seed)
    train_ds = SimpleDataset(Subset(train_base, indices), train_t)
    val_ds = SimpleDataset(val_base, val_t)

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True),
        len(indices),
    )
