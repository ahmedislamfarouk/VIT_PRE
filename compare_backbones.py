#!/usr/bin/env python3
"""
compare_backbones.py — Backbone Comparison for Intel Image Classification

Compares DeiT, DINOv2, and SigLIP2 as frozen backbones with a shared
classification head on the Intel Image Classification dataset.

Usage:
    python3 compare_backbones.py                  # run all 3 models
    python3 compare_backbones.py --model deit      # run only DeiT
    python3 compare_backbones.py --model dinov2    # run only DINOv2
    python3 compare_backbones.py --model siglip2   # run only SigLIP2
    python3 compare_backbones.py --download-only

Dependencies:
%uv pip install timm torch torchvision kagglehub scikit-learn matplotlib tqdm
"""

import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
DATASET_NAME  = "puneet6060/intel-image-classification"
IMG_SIZE      = 224
BATCH_SIZE    = 32
EPOCHS        = 10
LEARNING_RATE = 1e-4
NUM_CLASSES   = 6
CLASS_NAMES   = ["buildings", "forest", "glacier", "mountain", "sea", "street"]

CHECKPOINT_DIR = Path("checkpoints")
RESULTS_DIR    = Path("results")
CHECKPOINT_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGENET_STATS = {
    "mean": [0.485, 0.456, 0.406],
    "std":  [0.229, 0.224, 0.225],
}

# ── Backbone registry ─────────────────────────────────────────────────────────
# Maps model key -> (timm_name, feature_dim, description)
BACKBONE_REGISTRY = {
    "deit": (
        "deit_tiny_patch16_224",
        192,
        "DeiT-Tiny  | patch16/224 | dim=192",
    ),
    "dinov2": (
        "vit_small_patch14_reg4_dinov2.lvd142m",
        384,
        "DINOv2 ViT-S/14 reg4 | LVD-142M pretrain | dim=384 | native 224",
    ),
    "siglip2": (
        "vit_base_patch16_siglip_224.v2_webli",
        768,
        "SigLIP 2 ViT-B/16 | WebLI pretrain (Feb 2025) | dim=768",
    ),
}


# ── Dataset Download ──────────────────────────────────────────────────────────
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
    val_dir   = os.path.join(dataset_path, "seg_test",  "seg_test")

    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        print(f"Error: Dataset directories not found at {dataset_path}")
        print(f"Expected: {train_dir}  and  {val_dir}")
        sys.exit(1)

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset   = datasets.ImageFolder(val_dir,   transform=val_transform)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True,
    )

    print(f"Train samples: {len(train_dataset)},  Val samples: {len(val_dataset)}")
    return train_loader, val_loader


# ── Backbone Loading ──────────────────────────────────────────────────────────
def load_backbone(name: str) -> tuple[nn.Module, int]:
    """
    Load a frozen pretrained ViT backbone via timm.
    Returns (backbone, feature_dim).
    """
    import timm

    if name not in BACKBONE_REGISTRY:
        raise ValueError(f"Unknown backbone '{name}'. "
                         f"Choose from: {list(BACKBONE_REGISTRY.keys())}")

    timm_name, feature_dim, description = BACKBONE_REGISTRY[name]

    # num_classes=0 removes the classification head; timm returns the pooled
    # [CLS] token (or GAP) as a feature vector of shape (B, feature_dim).
    backbone = timm.create_model(
        timm_name, pretrained=True, num_classes=0, dynamic_img_size=True
    )
    backbone.eval()
    for param in backbone.parameters():
        param.requires_grad = False

    print(f"Loaded backbone: {description}")
    return backbone, feature_dim


# ── Classification Head ───────────────────────────────────────────────────────
class ClassificationHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


# ── Full Model ────────────────────────────────────────────────────────────────
class BackboneClassifier(nn.Module):
    def __init__(self, backbone_name: str):
        super().__init__()
        self.backbone_name = backbone_name
        self.backbone, feature_dim = load_backbone(backbone_name)
        self.head = ClassificationHead(feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            # All three backbones (DeiT, DINOv2, SigLIP2) loaded with
            # num_classes=0 return a pooled (B, D) feature vector directly.
            features = self.backbone(x)
        return self.head(features)


# ── Trainer ───────────────────────────────────────────────────────────────────
class ModelTrainer:
    def __init__(self, model_name: str, train_loader: DataLoader, val_loader: DataLoader):
        self.model_name   = model_name
        self.train_loader = train_loader
        self.val_loader   = val_loader

        self.model     = BackboneClassifier(model_name).to(DEVICE)
        self.optimizer = optim.AdamW(
            self.model.head.parameters(), lr=LEARNING_RATE, weight_decay=1e-4
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=EPOCHS
        )
        self.criterion = nn.CrossEntropyLoss()

        self.history      = {"train_loss": [], "val_loss": [], "val_acc": []}
        self.best_val_acc = 0.0

    # ── single epoch ──────────────────────────────────────────────────────────
    def train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        for images, labels in tqdm(
            self.train_loader, desc=f"  [{self.model_name}] Train", leave=False
        ):
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
        total_loss = 0.0
        all_preds, all_labels = [], []

        for images, labels in tqdm(
            self.val_loader, desc=f"  [{self.model_name}] Val", leave=False
        ):
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

    # ── full training loop ────────────────────────────────────────────────────
    def train(self) -> dict:
        print(f"\n{'='*60}")
        print(f"  Training {self.model_name.upper()}")
        print(f"{'='*60}")

        for epoch in range(1, EPOCHS + 1):
            train_loss         = self.train_epoch()
            val_loss, val_acc  = self.validate()
            self.scheduler.step()

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)

            print(
                f"  Epoch {epoch:02d}/{EPOCHS} — "
                f"Train Loss: {train_loss:.4f},  "
                f"Val Loss: {val_loss:.4f},  "
                f"Val Acc: {val_acc:.4f}"
            )

            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.save_checkpoint()
                print(f"  ** New best: {val_acc:.4f} **")

        print(f"\n  Best Val Accuracy [{self.model_name}]: {self.best_val_acc:.4f}")
        return self.history

    def save_checkpoint(self):
        path = CHECKPOINT_DIR / f"{self.model_name}_best.pth"
        torch.save(
            {
                "model_name":       self.model_name,
                "head_state_dict":  self.model.head.state_dict(),
                "best_val_acc":     self.best_val_acc,
            },
            path,
        )
        print(f"  Checkpoint saved → {path}")

    # ── final evaluation ──────────────────────────────────────────────────────
    @torch.no_grad()
    def evaluate(self) -> dict:
        self.model.eval()
        all_preds, all_labels = [], []

        for images, labels in tqdm(
            self.val_loader, desc=f"  [{self.model_name}] Evaluate"
        ):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = self.model(images)
            preds   = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        accuracy  = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average="macro", zero_division=0
        )
        per_cls_p, per_cls_r, per_cls_f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, zero_division=0
        )
        cm = confusion_matrix(all_labels, all_preds)

        results = {
            "model":            self.model_name,
            "accuracy":         round(accuracy,  4),
            "precision_macro":  round(precision, 4),
            "recall_macro":     round(recall,    4),
            "f1_macro":         round(f1,        4),
            "per_class": {
                CLASS_NAMES[i]: {
                    "precision": round(per_cls_p[i],  4),
                    "recall":    round(per_cls_r[i],  4),
                    "f1":        round(per_cls_f1[i], 4),
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

# Consistent colours and markers for each model throughout all plots
MODEL_COLORS  = {"deit": "#378ADD", "dinov2": "#1D9E75", "siglip2": "#D85A30"}
MODEL_MARKERS = {"deit": "o",       "dinov2": "s",       "siglip2": "^"}
MODEL_LABELS  = {"deit": "DeiT-Tiny", "dinov2": "DINOv2 ViT-S/14", "siglip2": "SigLIP 2 ViT-B/16"}


def _style_ax(ax, title: str, xlabel: str, ylabel: str):
    """Apply consistent styling to a single Axes."""
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.tick_params(labelsize=10)
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)


# ── 1. Training curves (loss + accuracy) ──────────────────────────────────────
def plot_training_curves(all_histories: dict):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, EPOCHS + 1)

    for name, hist in all_histories.items():
        c = MODEL_COLORS[name]
        m = MODEL_MARKERS[name]
        lbl = MODEL_LABELS[name]
        axes[0].plot(epochs, hist["train_loss"], color=c, marker=m,
                     markersize=5, linewidth=2,   label=f"{lbl} – train")
        axes[0].plot(epochs, hist["val_loss"],   color=c, marker=m,
                     markersize=5, linewidth=1.5, label=f"{lbl} – val",
                     linestyle="--", alpha=0.7)
        axes[1].plot(epochs, hist["val_acc"],    color=c, marker=m,
                     markersize=5, linewidth=2,   label=lbl)

    _style_ax(axes[0], "Training & Validation Loss",     "Epoch", "Cross-Entropy Loss")
    _style_ax(axes[1], "Validation Accuracy over Epochs","Epoch", "Accuracy")
    axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v*100:.1f}%"))

    for ax in axes:
        ax.legend(fontsize=9, framealpha=0.6)
        ax.set_xticks(list(epochs))

    plt.suptitle("Backbone Training Curves — Intel Image Classification",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    out = RESULTS_DIR / "01_training_curves.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out}")


# ── 2. Confusion matrices ──────────────────────────────────────────────────────
def plot_confusion_matrices(all_results: list):
    n   = len(all_results)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    cmaps = {"deit": "Blues", "dinov2": "Greens", "siglip2": "Oranges"}

    for ax, result in zip(axes, all_results):
        name = result["model"]
        cm   = np.array(result["confusion_matrix"])
        im   = ax.imshow(cm, interpolation="nearest", cmap=cmaps.get(name, "Blues"))
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        acc = result["accuracy"]
        ax.set_title(f"{MODEL_LABELS.get(name, name)}\nAcc: {acc*100:.2f}%",
                     fontsize=11, fontweight="bold")
        ax.set_xlabel("Predicted label", fontsize=10)
        ax.set_ylabel("True label", fontsize=10)

        ticks = np.arange(NUM_CLASSES)
        ax.set_xticks(ticks); ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right", fontsize=9)
        ax.set_yticks(ticks); ax.set_yticklabels(CLASS_NAMES, fontsize=9)

        thresh = cm.max() / 2.0
        for i in range(NUM_CLASSES):
            for j in range(NUM_CLASSES):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=9,
                        color="white" if cm[i, j] > thresh else "black")

    plt.suptitle("Confusion Matrices — Intel Image Classification",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    out = RESULTS_DIR / "02_confusion_matrices.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out}")


# ── 3. Metric comparison bar chart ────────────────────────────────────────────
def plot_metric_comparison(all_results: list):
    metrics      = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
    metric_names = ["Accuracy", "Precision (macro)", "Recall (macro)", "F1 (macro)"]

    n_metrics = len(metrics)
    n_models  = len(all_results)
    x         = np.arange(n_metrics)
    width     = 0.22
    offsets   = np.linspace(-(n_models - 1) / 2, (n_models - 1) / 2, n_models) * width

    fig, ax = plt.subplots(figsize=(11, 5))

    for result, offset in zip(all_results, offsets):
        name   = result["model"]
        values = [result[m] for m in metrics]
        bars   = ax.bar(x + offset, values, width, label=MODEL_LABELS.get(name, name),
                        color=MODEL_COLORS[name], edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                    f"{val*100:.1f}%", ha="center", va="bottom", fontsize=8.5)

    ax.set_ylim(0, 1.12)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v*100:.0f}%"))
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, fontsize=11)
    _style_ax(ax, "Overall Metric Comparison", "", "Score")
    ax.legend(fontsize=10, framealpha=0.6)

    plt.tight_layout()
    out = RESULTS_DIR / "03_metric_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out}")


# ── 4. Per-class F1 bar chart ─────────────────────────────────────────────────
def plot_per_class_f1(all_results: list):
    n_classes = NUM_CLASSES
    n_models  = len(all_results)
    x         = np.arange(n_classes)
    width     = 0.22
    offsets   = np.linspace(-(n_models - 1) / 2, (n_models - 1) / 2, n_models) * width

    fig, ax = plt.subplots(figsize=(13, 5))

    for result, offset in zip(all_results, offsets):
        name   = result["model"]
        f1s    = [result["per_class"][c]["f1"] for c in CLASS_NAMES]
        bars   = ax.bar(x + offset, f1s, width, label=MODEL_LABELS.get(name, name),
                        color=MODEL_COLORS[name], edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, f1s):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val*100:.0f}", ha="center", va="bottom", fontsize=8)

    ax.set_ylim(0, 1.12)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v*100:.0f}%"))
    ax.set_xticks(x)
    ax.set_xticklabels([c.capitalize() for c in CLASS_NAMES], fontsize=11)
    _style_ax(ax, "Per-Class F1 Score Comparison", "Class", "F1 Score")
    ax.legend(fontsize=10, framealpha=0.6)

    plt.tight_layout()
    out = RESULTS_DIR / "04_per_class_f1.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out}")


# ── 5. Radar chart ────────────────────────────────────────────────────────────
def plot_radar(all_results: list):
    """
    Radar comparing each model across Accuracy, Precision, Recall, F1,
    and best-epoch val accuracy (from history stored in results if available,
    else falls back to the final val accuracy).
    """
    categories = ["Accuracy", "Precision", "Recall", "F1-Score"]
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]                       # close the polygon

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

    for result in all_results:
        name   = result["model"]
        values = [
            result["accuracy"],
            result["precision_macro"],
            result["recall_macro"],
            result["f1_macro"],
        ]
        values += values[:1]                   # close the polygon
        ax.plot(angles, values,
                color=MODEL_COLORS[name], linewidth=2,
                marker=MODEL_MARKERS[name], markersize=6,
                label=MODEL_LABELS.get(name, name))
        ax.fill(angles, values, color=MODEL_COLORS[name], alpha=0.12)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), categories, fontsize=11)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v*100:.0f}%"))
    ax.set_title("Backbone Comparison Radar\n(all metrics from real evaluation)",
                 fontsize=13, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = RESULTS_DIR / "05_radar_chart.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out}")


# ── 6. Training accuracy GIF ──────────────────────────────────────────────────
def plot_training_gif(all_histories: dict):
    """
    Animated GIF showing validation accuracy building up epoch by epoch.
    Requires Pillow (pip install Pillow).
    """
    try:
        from PIL import Image as PILImage
        import io
    except ImportError:
        print("Pillow not installed — skipping GIF. Run: pip install Pillow")
        return

    frames = []
    epochs = list(range(1, EPOCHS + 1))
    y_min  = min(
        v
        for hist in all_histories.values()
        for v in hist["val_acc"]
    ) - 0.05

    for ep_end in epochs:
        fig, ax = plt.subplots(figsize=(9, 5))

        for name, hist in all_histories.items():
            c   = MODEL_COLORS[name]
            m   = MODEL_MARKERS[name]
            lbl = MODEL_LABELS[name]
            xs  = epochs[:ep_end]
            ys  = hist["val_acc"][:ep_end]
            ax.plot(xs, ys, color=c, marker=m, markersize=6,
                    linewidth=2.2, label=lbl)
            # annotate last point
            ax.annotate(f"{ys[-1]*100:.1f}%",
                        xy=(xs[-1], ys[-1]),
                        xytext=(4, 4), textcoords="offset points",
                        fontsize=9, color=c, fontweight="bold")

        ax.set_xlim(0.5, EPOCHS + 0.5)
        ax.set_ylim(max(0, y_min), 1.02)
        ax.set_xticks(epochs)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v*100:.0f}%"))
        _style_ax(ax, f"Validation Accuracy — Epoch {ep_end}/{EPOCHS}",
                  "Epoch", "Accuracy")
        ax.legend(fontsize=10, framealpha=0.7, loc="lower right")

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        plt.close()
        buf.seek(0)
        frames.append(PILImage.open(buf).copy())
        buf.close()

    # last frame shown longer
    frames += [frames[-1]] * 8

    out = RESULTS_DIR / "06_training_accuracy.gif"
    frames[0].save(
        out, save_all=True, append_images=frames[1:],
        duration=300, loop=0,
    )
    print(f"Saved → {out}")


def print_comparison_table(all_results: list):
    print(f"\n{'='*70}")
    print(f"  BACKBONE COMPARISON SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Model':<12} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print(f"  {'-'*60}")
    for r in all_results:
        print(
            f"  {r['model']:<12} {r['accuracy']:<12.4f} "
            f"{r['precision_macro']:<12.4f} {r['recall_macro']:<12.4f} "
            f"{r['f1_macro']:<12.4f}"
        )
    print(f"{'='*70}\n")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Backbone Comparison: DeiT vs DINOv2 vs SigLIP2"
    )
    parser.add_argument(
        "--model", type=str,
        choices=list(BACKBONE_REGISTRY.keys()),
        help="Run only one backbone (default: all three)",
    )
    parser.add_argument(
        "--download-only", action="store_true",
        help="Only download the dataset then exit",
    )
    parser.add_argument(
        "--dataset-path", type=str, default=None,
        help="Path to an already-downloaded dataset",
    )
    # parse_known_args ignores unrecognised flags injected by
    # IPython / Jupyter / Modal so the script works in all environments.
    args, _ = parser.parse_known_args()

    if args.download_only:
        download_dataset()
        return

    dataset_path = args.dataset_path or download_dataset()
    train_loader, val_loader = create_dataloaders(dataset_path)

    models_to_run = [args.model] if args.model else list(BACKBONE_REGISTRY.keys())

    all_histories: dict = {}
    all_results:   list = []

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
            "epochs":        EPOCHS,
            "batch_size":    BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "image_size":    IMG_SIZE,
            "optimizer":     "AdamW",
            "scheduler":     "CosineAnnealingLR",
        },
    }

    results_path = RESULTS_DIR / "comparison_results.json"
    with open(results_path, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"Comparison results saved → {results_path}")

    # ── Visualizations (always saved to results/ after training) ──────────────
    print("\nGenerating visualizations from real results...")
    plot_training_curves(all_histories)
    plot_confusion_matrices(all_results)
    if len(all_results) > 1:
        plot_metric_comparison(all_results)
        plot_per_class_f1(all_results)
        plot_radar(all_results)
    plot_training_gif(all_histories)
    print(f"\nAll plots saved to: {RESULTS_DIR.resolve()}")


if __name__ == "__main__":
    main()
