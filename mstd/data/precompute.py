import os
import pickle

import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from tqdm import tqdm

from ..config import BACKBONE_REGISTRY, CACHE_DIR, DEVICE, IMG_SIZE, IMAGENET_STATS
from ..models.teacher import TeacherModel


def precompute_teacher_logits(dataset_path: str, batch_size: int = 64, force: bool = False):
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
