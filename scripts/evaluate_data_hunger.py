#!/usr/bin/env python3
"""
evaluate_data_hunger.py — Data-Efficiency Evaluation
"""
import json
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from tqdm import tqdm

from mstd.config import (
    BACKBONE_REGISTRY, CHECKPOINT_DIR, RESULTS_DIR, DEVICE,
    IMG_SIZE, NUM_CLASSES, IMAGENET_STATS,
)
from mstd.models.backbone import load_backbone
from mstd.models.teacher import TeacherEnsemble
from mstd.models.student import AMTDStudentScratch
from mstd.data.dataset import get_subset_indices, LogitsDataset
from mstd.evaluation.metrics import evaluate_model
from mstd.visualization.plots import plot_data_hunger_curves

BASE_DIR = RESULTS_DIR.parent
DATASET_PATH = "/home/skyvision/.cache/kagglehub/datasets/puneet6060/intel-image-classification/versions/2"

TEACHER_CHECKPOINTS = {
    "deit": CHECKPOINT_DIR / "deit_best.pth",
    "dinov2": CHECKPOINT_DIR / "dinov2_best.pth",
    "siglip2": CHECKPOINT_DIR / "siglip2_best.pth",
}

SUBSET_RATIOS = [0.05, 0.10, 0.25, 0.50, 1.00]
SEED = 42


class BaselineModel(nn.Module):
    def __init__(self, name):
        super().__init__()
        import timm
        self.name = name
        timm_name, feature_dim = BACKBONE_REGISTRY[name][:2]
        self.backbone = timm.create_model(timm_name, pretrained=True, num_classes=0, dynamic_img_size=True)
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, NUM_CLASSES),
        )
        ckpt = torch.load(TEACHER_CHECKPOINTS[name], map_location="cpu", weights_only=False)
        state = ckpt["head_state_dict"]
        stripped = {k.replace("head.", "", 1): v for k, v in state.items()}
        self.head.load_state_dict(stripped)

    def forward(self, x):
        with torch.no_grad():
            feats = self.backbone(x)
        return self.head(feats)


def make_loaders(subset_ratio, seed=SEED):
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

    train_base = datasets.ImageFolder(os.path.join(DATASET_PATH, "seg_train", "seg_train"))
    val_base = datasets.ImageFolder(os.path.join(DATASET_PATH, "seg_test", "seg_test"))

    indices = get_subset_indices(train_base, subset_ratio, seed)
    from torch.utils.data import Subset
    subset_base = Subset(train_base, indices)

    tmp_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(**IMAGENET_STATS),
    ])
    tmp_base = datasets.ImageFolder(os.path.join(DATASET_PATH, "seg_train", "seg_train"), transform=tmp_transform)
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

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, val_loader


def train_baseline(model, loader, epochs, lr):
    model = model.to(DEVICE)
    optimizer = optim.AdamW(model.head.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    for _ in range(epochs):
        model.train()
        for images, labels, _ in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        scheduler.step()


def train_amtd(model, loader, epochs, lr, T=2.4, alpha=0.75, ls=0.1):
    model = model.to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    for _ in range(epochs):
        model.train()
        for images, labels, t_logits in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            t_logits = t_logits.to(DEVICE)
            optimizer.zero_grad()
            s_logits = model(images)
            hard = F.cross_entropy(s_logits, labels, label_smoothing=ls)
            soft = F.kl_div(
                F.log_softmax(s_logits / T, dim=1),
                F.softmax(t_logits / T, dim=1),
                reduction="batchmean",
            ) * (T ** 2)
            loss = alpha * hard + (1 - alpha) * soft
            loss.backward()
            optimizer.step()
        scheduler.step()


def main():
    results = {name: {} for name in list(BACKBONE_REGISTRY.keys()) + ["amtd"]}

    for ratio in SUBSET_RATIOS:
        print("=" * 70)
        print(f"Subset: {ratio*100:.0f}% of training data")
        print("=" * 70)
        train_loader, val_loader = make_loaders(ratio)

        for name in BACKBONE_REGISTRY:
            print(f"  Training {name}...")
            model = BaselineModel(name)
            epochs = 15 if ratio < 1.0 else 10
            train_baseline(model, train_loader, epochs, lr=1e-4)
            metrics = evaluate_model(model, val_loader, DEVICE)
            print(f"    -> Acc: {metrics['accuracy']*100:.2f}%")
            results[name][ratio] = metrics

        print(f"  Training AMTD...")
        model = AMTDStudentScratch()
        epochs = 20 if ratio < 1.0 else 15
        train_amtd(model, train_loader, epochs, lr=4e-4, T=2.4, alpha=0.75, ls=0.1)
        metrics = evaluate_model(model, val_loader, DEVICE)
        print(f"    -> Acc: {metrics['accuracy']*100:.2f}%")
        results["amtd"][ratio] = metrics

    out_json = RESULTS_DIR / "data_hunger_evaluation.json"
    serializable = {}
    for model_name, ratios in results.items():
        serializable[model_name] = {f"{r:.2f}": v for r, v in ratios.items()}
    with open(out_json, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\n[Results] Saved -> {out_json}")

    plot_data_hunger_curves(results, SUBSET_RATIOS, RESULTS_DIR / "data_hunger_curves.png")

    print("\n" + "=" * 70)
    print("DATA-HUNGER SUMMARY")
    print("=" * 70)
    labels_map = {"deit": "DeiT-Tiny", "dinov2": "DINOv2-S/14", "siglip2": "SigLIP2-B/16", "amtd": "AMTD (Ours)"}
    header = f"{'Model':<18}" + "".join([f"{r*100:>7.0f}%" for r in SUBSET_RATIOS])
    print(header)
    print("-" * len(header))
    for name in results:
        row = f"{labels_map[name]:<18}" + "".join([f"{results[name][r]['accuracy']*100:>7.2f}" for r in SUBSET_RATIOS])
        print(row)


if __name__ == "__main__":
    main()
