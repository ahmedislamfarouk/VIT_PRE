#!/usr/bin/env python3
"""
complete_enhanced_training.py — Finish the AMTD final training
"""
import os
import json
import pickle
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

BASE_DIR = Path("/home/skyvision/ammarbigass5")
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
RESULTS_DIR = BASE_DIR / "results"
CACHE_DIR = BASE_DIR / "cache"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATASET_PATH = "/home/skyvision/.cache/kagglehub/datasets/puneet6060/intel-image-classification/versions/2"
IMG_SIZE = 224
NUM_CLASSES = 6
CLASS_NAMES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]
IMAGENET_STATS = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}


def load_teacher_logits():
    cache_file = CACHE_DIR / "teacher_logits.pkl"
    with open(cache_file, "rb") as f:
        data = pickle.load(f)
    ensemble = (data["deit"] + data["dinov2"] + data["siglip2"]) / 3.0
    return ensemble, data["labels"]


class DistillDataset(Dataset):
    def __init__(self, base_dataset, teacher_logits, transform):
        self.base = base_dataset
        self.logits = teacher_logits
        self.transform = transform
    def __len__(self):
        return len(self.base)
    def __getitem__(self, idx):
        img, label = self.base[idx]
        img = self.transform(img)
        return img, label, self.logits[idx]


class StudentModel(nn.Module):
    def __init__(self, model_name="mobilevit_s", dropout=0.3):
        super().__init__()
        import timm
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0)
        feature_dim = self.backbone.num_features
        self.head = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(feature_dim, 256), nn.ReLU(),
            nn.Dropout(dropout / 2), nn.Linear(256, NUM_CLASSES),
        )
    def forward(self, x):
        return self.head(self.backbone(x))
    def count_params(self):
        return sum(p.numel() for p in self.parameters())


def create_loaders(batch_size, use_augment=True):
    if use_augment:
        train_transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(0.2, 0.2, 0.2),
            transforms.ToTensor(),
            transforms.Normalize(**IMAGENET_STATS),
            transforms.RandomErasing(p=0.25),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(**IMAGENET_STATS),
        ])
    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(**IMAGENET_STATS),
    ])
    train_base = datasets.ImageFolder(os.path.join(DATASET_PATH, "seg_train", "seg_train"))
    val_base = datasets.ImageFolder(os.path.join(DATASET_PATH, "seg_test", "seg_test"))
    teacher_logits, _ = load_teacher_logits()
    train_ds = DistillDataset(train_base, teacher_logits, train_transform)
    val_ds = DistillDataset(val_base, torch.zeros(len(val_base), NUM_CLASSES), val_transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, val_loader


def distillation_loss(s_logits, t_logits, labels, T, alpha, ls):
    hard = F.cross_entropy(s_logits, labels, label_smoothing=ls)
    soft = F.kl_div(F.log_softmax(s_logits / T, dim=1), F.softmax(t_logits / T, dim=1), reduction="batchmean") * (T ** 2)
    return alpha * hard + (1 - alpha) * soft


def train_epoch(model, loader, optimizer, scheduler, T, alpha, ls):
    model.train()
    total_loss = 0.0
    for images, labels, t_logits in loader:
        images, labels, t_logits = images.to(DEVICE), labels.to(DEVICE), t_logits.to(DEVICE)
        optimizer.zero_grad()
        s_logits = model(images)
        loss = distillation_loss(s_logits, t_logits, labels, T, alpha, ls)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
    if scheduler is not None and not isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
        scheduler.step()
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    for images, labels, _ in loader:
        images = images.to(DEVICE)
        logits = model(images)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())
    acc = accuracy_score(all_labels, all_preds)
    p, r, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="macro", zero_division=0)
    return acc, {"accuracy": acc, "precision_macro": p, "recall_macro": r, "f1_macro": f1,
                 "confusion_matrix": confusion_matrix(all_labels, all_preds).tolist()}


def plot_curves(history, out_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    epochs = range(1, len(history["val_acc"]) + 1)
    ax.plot(epochs, [a * 100 for a in history["val_acc"]], marker="o", label="Val Acc", color="green")
    ax.axhline(94.93, color="red", linestyle="--", label="DINOv2 baseline (94.93%)")
    ax.set_ylim(85, 100)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Enhanced AMTD — Validation Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    hparams = {
        "lr": 0.0004624774340441029,
        "weight_decay": 6.992760789566003e-06,
        "dropout": 0.4063734801220985,
        "temperature": 2.4020114399899226,
        "alpha": 0.7535702930160021,
        "label_smoothing": 0.10604367314302202,
        "batch_size": 128,
        "use_augment": True,
        "scheduler": "step",
    }

    print("=" * 70)
    print("Completing AMTD Final Training")
    print("=" * 70)

    train_loader, val_loader = create_loaders(hparams["batch_size"], hparams["use_augment"])
    model = StudentModel(dropout=hparams["dropout"]).to(DEVICE)
    print(f"Student params: {model.count_params() / 1e6:.2f}M")

    optimizer = optim.AdamW(model.parameters(), lr=hparams["lr"], weight_decay=hparams["weight_decay"])
    if hparams["scheduler"] == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    elif hparams["scheduler"] == "step":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=4)

    history = {"train_loss": [], "val_acc": []}
    best_acc = 0.0
    best_state = None
    patience = 12
    patience_counter = 0

    for epoch in range(1, 51):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler,
                                  hparams["temperature"], hparams["alpha"], hparams["label_smoothing"])
        val_acc, val_metrics = evaluate(model, val_loader)
        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_acc)

        history["train_loss"].append(train_loss)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch:02d}/50 — Train Loss: {train_loss:.4f}, Val Acc: {val_acc*100:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            print(f"  *** New best: {best_acc*100:.2f}% ***")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    if best_state:
        model.load_state_dict(best_state)

    final_acc, final_metrics = evaluate(model, val_loader)
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"Best Val Accuracy: {best_acc*100:.2f}%")
    print(f"Final Val Accuracy: {final_acc*100:.2f}%")
    print(f"F1 (macro): {final_metrics['f1_macro']*100:.2f}%")

    baselines = {"DeiT-Tiny": 91.27, "DINOv2-S/14": 94.93, "SigLIP2-B/16": 88.17, "AMTD (Ours)": best_acc * 100}
    print("\nComparison:")
    for name, acc in baselines.items():
        marker = "+" if acc == max(baselines.values()) else "  "
        print(f"  {marker} {name:<18} {acc:.2f}%")

    ckpt_path = CHECKPOINT_DIR / "enhanced_amtd_best.pth"
    torch.save({"model_state_dict": model.state_dict(), "best_val_acc": best_acc,
                "hyperparameters": hparams, "history": history, "final_metrics": final_metrics}, ckpt_path)
    print(f"\n[Checkpoint] Saved -> {ckpt_path}")

    results = {
        "model": "AMTD (Adaptive Multi-Teacher Distillation)",
        "student_backbone": "mobilevit_s",
        "student_params_M": round(model.count_params() / 1e6, 2),
        "best_val_acc": round(best_acc, 4),
        "final_metrics": {k: round(v, 4) if isinstance(v, float) else v for k, v in final_metrics.items()},
        "hyperparameters": hparams,
        "baselines": {"deit": 0.9127, "dinov2": 0.9493, "siglip2": 0.8817},
    }
    with open(RESULTS_DIR / "enhanced_amtd_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"[Results] Saved -> {RESULTS_DIR / 'enhanced_amtd_results.json'}")

    plot_curves(history, RESULTS_DIR / "enhanced_amtd_curves.png")


if __name__ == "__main__":
    main()
