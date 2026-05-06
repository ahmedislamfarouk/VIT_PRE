"""
Teacher logit pre-computation and caching.

Knowledge distillation requires the teacher's logits for every training
sample. Computing them on-the-fly each epoch is wasteful since the teachers
are frozen. This module computes them once, caches to disk, and provides
a fast load path for subsequent runs.

The cached file (teacher_logits.pkl) contains all three teachers' logits
plus the ground-truth labels, making TPE hyperparameter search and
final training much faster.
"""

import os
import pickle

import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from tqdm import tqdm

from mstd.config import BACKBONE_REGISTRY, CACHE_DIR, DEVICE, IMG_SIZE, IMAGENET_STATS
from mstd.models.teacher import TeacherModel


def precompute_teacher_logits(dataset_path: str, batch_size: int = 64, force: bool = False):
    """
    Pre-compute and cache teacher logits for the full training set.

    Each teacher (DeiT, DINOv2, SigLIP2) runs a single forward pass over
    the training data. The logits are concatenated and saved to a pickle
    file in CACHE_DIR.

    Args:
        dataset_path: Path to the Intel dataset root.
        batch_size: Batch size for teacher forward passes.
        force: If True, re-compute even if cache exists.

    Returns:
        Tuple of (deit_logits, dinov2_logits, siglip2_logits, labels).
        Each logits tensor is shape (N, NUM_CLASSES).
    """
    cache_file = CACHE_DIR / "teacher_logits.pkl"
    if cache_file.exists() and not force:
        with open(cache_file, "rb") as f:
            data = pickle.load(f)
        return data["deit"], data["dinov2"], data["siglip2"], data["labels"]

    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(**IMAGENET_STATS),
    ])
    train_dir = os.path.join(dataset_path, "seg_train", "seg_train")
    train_ds = datasets.ImageFolder(train_dir, transform=val_transform)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    teachers = {name: TeacherModel(name).to(DEVICE) for name in BACKBONE_REGISTRY}
    all_logits = {name: [] for name in BACKBONE_REGISTRY}
    all_labels = []

    for images, labels in tqdm(train_loader, desc="Teachers"):
        images = images.to(DEVICE)
        all_labels.append(labels)
        for name, model in teachers.items():
            logits = model(images)
            all_logits[name].append(logits.cpu())

    all_labels = torch.cat(all_labels, dim=0)
    for name in all_logits:
        all_logits[name] = torch.cat(all_logits[name], dim=0)

    with open(cache_file, "wb") as f:
        pickle.dump(
            {
                "deit": all_logits["deit"],
                "dinov2": all_logits["dinov2"],
                "siglip2": all_logits["siglip2"],
                "labels": all_labels,
            },
            f,
        )
    return all_logits["deit"], all_logits["dinov2"], all_logits["siglip2"], all_labels
