#!/usr/bin/env python3
"""
compare_backbones.py — Backbone Comparison for Intel Image Classification

Compares DeiT, SAM 2, and DINOv3 as frozen backbones with a shared
classification head on the Intel Image Classification dataset.

Usage:
    python3 compare_backbones.py              # run all 3 models
    python3 compare_backbones.py --model deit # run only DeiT
    python3 compare_backbones.py --download-only
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
DATASET_NAME = "puneet6060/intel-image-classification"
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-4
NUM_CLASSES = 6
CLASS_NAMES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]

CHECKPOINT_DIR = Path("checkpoints")
RESULTS_DIR = Path("results")
CHECKPOINT_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGENET_STATS = {
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
}


# ── Dataset Download ─────────────────────────────────────────────────────────
def download_dataset() -> str:
    import kagglehub
    print("Downloading Intel Image Classification dataset...")
    path = kagglehub.dataset_download(DATASET_NAME)
    print(f"Dataset path: {path}")
    return path


# ── DataLoaders ───────────────────────────────────────────────────────────────
def create_dataloaders(dataset_path: str, batch_size: int = BATCH_SIZE):
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

    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        print(f"Error: Dataset directories not found at {dataset_path}")
        print(f"Expected: {train_dir} and {val_dir}")
        sys.exit(1)

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    return train_loader, val_loader


# ── Backbone Loading ─────────────────────────────────────────────────────────
def load_backbone(name: str) -> tuple[nn.Module, int]:
    if name == "deit":
        import timm
        backbone = timm.create_model("deit_tiny_patch16_224", pretrained=True, num_classes=0)
        backbone.eval()
        for param in backbone.parameters():
            param.requires_grad = False
        feature_dim = 192
        print("Loaded DeiT Tiny backbone (dim=192)")

    elif name == "dinov3":
        backbone = torch.hub.load("facebookresearch/dinov3", "dinov3_vits14", pretrained=True)
        backbone.eval()
        for param in backbone.parameters():
            param.requires_grad = False
        feature_dim = 384
        print("Loaded DINOv3 ViT-S/14 backbone (dim=384)")

    elif name == "sam2":
        try:
            from sam2.build_sam import build_sam2
        except ImportError:
            print("SAM 2 not installed. Install with: pip install segment-anything-2")
            sys.exit(1)

        sam2_checkpoint = "sam2_hiera_tiny.pt"
        model_cfg = "sam2_hiera_t.yaml"

        if not os.path.exists(sam2_checkpoint):
            print("Downloading SAM 2 checkpoint...")
            os.system(
                "wget https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt"
            )

        sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=DEVICE)
        backbone = sam2_model.image_encoder
        backbone.eval()
        for param in backbone.parameters():
            param.requires_grad = False
        feature_dim = 384
        print("Loaded SAM 2 Hiera-Tiny image encoder backbone (dim=384)")

    else:
        raise ValueError(f"Unknown backbone: {name}")

    return backbone, feature_dim


# ── Classification Head ──────────────────────────────────────────────────────
class ClassificationHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.head(x)


# ── Full Model ────────────────────────────────────────────────────────────────
class BackboneClassifier(nn.Module):
    def __init__(self, backbone_name: str):
        super().__init__()
        self.backbone_name = backbone_name
        self.backbone, feature_dim = load_backbone(backbone_name)
        self.head = ClassificationHead(feature_dim)

    def forward(self, x):
        with torch.no_grad():
            if self.backbone_name == "sam2":
                features = self.backbone(x)
                if isinstance(features, list):
                    features = features[-1]
                features = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1)).squeeze(-1).squeeze(-1)
            elif self.backbone_name == "deit":
                features = self.backbone(x)
            elif self.backbone_name == "dinov3":
                features = self.backbone(x)
            else:
                raise ValueError(f"Unknown backbone: {self.backbone_name}")
        return self.head(features)


# ── Trainer ───────────────────────────────────────────────────────────────────
class ModelTrainer:
    def __init__(self, model_name: str, train_loader: DataLoader, val_loader: DataLoader):
        self.model_name = model_name
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.model = BackboneClassifier(model_name).to(DEVICE)

        self.optimizer = optim.AdamW(self.model.head.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=EPOCHS)
        self.criterion = nn.CrossEntropyLoss()

        self.history = {"train_loss": [], "val_loss": [], "val_acc": []}
        self.best_val_acc = 0.0

    def train_epoch(self) -> float:
        self.model.train()
        total_loss = 0
        for images, labels in tqdm(self.train_loader, desc=f"  [{self.model_name}] Train", leave=False):
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * images.size(0)

        return total_loss / len(self.train_loader.dataset)

    @torch.no_grad()
    def validate(self) -> tuple[float, float]:
        self.model.eval()
        total_loss = 0
        all_preds, all_labels = [], []

        for images, labels in tqdm(self.val_loader, desc=f"  [{self.model_name}] Val", leave=False):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(self.val_loader.dataset)
        accuracy = accuracy_score(all_labels, all_preds)
        return avg_loss, accuracy

    def train(self):
        print(f"\n{'='*60}")
        print(f"  Training {self.model_name.upper()}")
        print(f"{'='*60}")

        for epoch in range(1, EPOCHS + 1):
            train_loss = self.train_epoch()
            val_loss, val_acc = self.validate()
            self.scheduler.step()

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)

            print(f"  Epoch {epoch}/{EPOCHS} — "
                  f"Train Loss: {train_loss:.4f}, "
                  f"Val Loss: {val_loss:.4f}, "
                  f"Val Acc: {val_acc:.4f}")

            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.save_checkpoint()
                print(f"  ** New best accuracy: {val_acc:.4f} **")

        print(f"\n  Best Val Accuracy: {self.best_val_acc:.4f}")
        return self.history

    def save_checkpoint(self):
        path = CHECKPOINT_DIR / f"{self.model_name}_best.pth"
        torch.save({
            "model_name": self.model_name,
            "head_state_dict": self.model.head.state_dict(),
            "best_val_acc": self.best_val_acc,
        }, path)
        print(f"  Checkpoint saved: {path}")

    @torch.no_grad()
    def evaluate(self) -> dict:
        self.model.eval()
        all_preds, all_labels = [], []

        for images, labels in tqdm(self.val_loader, desc=f"  [{self.model_name}] Evaluate"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = self.model(images)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average="macro", zero_division=0
        )
        per_class_precision, per_class_recall, per_class_f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, zero_division=0
        )
        cm = confusion_matrix(all_labels, all_preds)

        results = {
            "model": self.model_name,
            "accuracy": round(accuracy, 4),
            "precision_macro": round(precision, 4),
            "recall_macro": round(recall, 4),
            "f1_macro": round(f1, 4),
            "per_class": {
                CLASS_NAMES[i]: {
                    "precision": round(per_class_precision[i], 4),
                    "recall": round(per_class_recall[i], 4),
                    "f1": round(per_class_f1[i], 4),
                }
                for i in range(NUM_CLASSES)
            },
            "confusion_matrix": cm.tolist(),
        }

        print(f"\n  [{self.model_name.upper()}] Evaluation Results:")
        print(f"    Accuracy:  {accuracy:.4f}")
        print(f"    Precision: {precision:.4f}")
        print(f"    Recall:    {recall:.4f}")
        print(f"    F1-Score:  {f1:.4f}")

        return results


# ── Plotting ──────────────────────────────────────────────────────────────────
def plot_training_curves(all_histories: dict):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for name, hist in all_histories.items():
        axes[0].plot(hist["train_loss"], label=f"{name} train", marker="o")
        axes[0].plot(hist["val_loss"], label=f"{name} val", marker="s", linestyle="--")
        axes[1].plot(hist["val_acc"], label=name, marker="o")

    axes[0].set_title("Loss over Epochs")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title("Validation Accuracy over Epochs")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "training_curves.png", dpi=150)
    plt.close()
    print(f"Training curves saved: {RESULTS_DIR / 'training_curves.png'}")


def plot_confusion_matrices(all_results: list):
    n = len(all_results)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    for idx, result in enumerate(all_results):
        cm = np.array(result["confusion_matrix"])
        im = axes[idx].imshow(cm, interpolation="nearest", cmap="Blues")
        axes[idx].set_title(f"{result['model'].upper()} (Acc: {result['accuracy']:.4f})")
        axes[idx].set_xlabel("Predicted")
        axes[idx].set_ylabel("True")

        tick_marks = np.arange(NUM_CLASSES)
        axes[idx].set_xticks(tick_marks)
        axes[idx].set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
        axes[idx].set_yticks(tick_marks)
        axes[idx].set_yticklabels(CLASS_NAMES)

        thresh = cm.max() / 2.0
        for i in range(NUM_CLASSES):
            for j in range(NUM_CLASSES):
                axes[idx].text(j, i, f"{cm[i, j]}",
                               ha="center", va="center",
                               color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "confusion_matrices.png", dpi=150)
    plt.close()
    print(f"Confusion matrices saved: {RESULTS_DIR / 'confusion_matrices.png'}")


def print_comparison_table(all_results: list):
    print(f"\n{'='*70}")
    print(f"  BACKBONE COMPARISON SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Model':<12} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print(f"  {'-'*60}")
    for r in all_results:
        print(f"  {r['model']:<12} {r['accuracy']:<12.4f} {r['precision_macro']:<12.4f} "
              f"{r['recall_macro']:<12.4f} {r['f1_macro']:<12.4f}")
    print(f"{'='*70}\n")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Backbone Comparison for Intel Image Classification")
    parser.add_argument("--model", type=str, choices=["deit", "sam2", "dinov3"],
                        help="Run only one model (default: all three)")
    parser.add_argument("--download-only", action="store_true", help="Only download the dataset")
    parser.add_argument("--dataset-path", type=str, default=None, help="Path to existing dataset")
    args = parser.parse_args()

    if args.download_only:
        download_dataset()
        return

    if args.dataset_path:
        dataset_path = args.dataset_path
    else:
        dataset_path = download_dataset()

    train_loader, val_loader = create_dataloaders(dataset_path)

    models_to_run = [args.model] if args.model else ["deit", "sam2", "dinov3"]

    all_histories = {}
    all_results = []

    for model_name in models_to_run:
        trainer = ModelTrainer(model_name, train_loader, val_loader)
        history = trainer.train()
        all_histories[model_name] = history

        result = trainer.evaluate()
        all_results.append(result)

    if len(all_results) > 1:
        print_comparison_table(all_results)

    comparison = {
        "models": all_results,
        "hyperparameters": {
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "image_size": IMG_SIZE,
            "optimizer": "AdamW",
            "scheduler": "CosineAnnealingLR",
        },
    }

    results_path = RESULTS_DIR / "comparison_results.json"
    with open(results_path, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"Comparison results saved: {results_path}")

    if len(all_histories) > 1:
        plot_training_curves(all_histories)
        plot_confusion_matrices(all_results)


if __name__ == "__main__":
    main()
