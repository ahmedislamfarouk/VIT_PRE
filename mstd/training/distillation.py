"""
Distillation training and evaluation loops.

Provides the core train_epoch_distill and evaluate_distill functions
used by the AMTD (Adaptive Multi-Teacher Distillation) experiment.

These functions expect DataLoaders that yield (images, labels, teacher_logits)
tuples, as produced by DistillDataset or LogitsDataset.
"""

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from mstd.training.loss import distillation_loss


def train_epoch_distill(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    scheduler,
    temperature: float,
    alpha: float,
    label_smoothing: float,
    device: torch.device,
) -> float:
    """
    Train the student model for one epoch using distillation loss.

    Each batch yields (images, labels, teacher_logits). The student logits
    are computed and the distillation loss is back-propagated.

    Args:
        model: The student model.
        loader: DataLoader yielding (image, label, teacher_logits) tuples.
        optimizer: Optimizer for student parameters.
        scheduler: LR scheduler (skipped if ReduceLROnPlateau).
        temperature: Distillation temperature.
        alpha: CE vs. KL loss weight.
        label_smoothing: Label smoothing factor.
        device: Device to run on.

    Returns:
        Average loss over the entire dataset.
    """
    model.train()
    total_loss = 0.0
    for images, labels, t_logits in loader:
        images = images.to(device)
        labels = labels.to(device)
        t_logits = t_logits.to(device)

        optimizer.zero_grad()
        s_logits = model(images)
        loss = distillation_loss(
            s_logits, t_logits, labels, temperature, alpha, label_smoothing
        )
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)

    if scheduler is not None and not isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
        scheduler.step()
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate_distill(model: nn.Module, loader: DataLoader, device: torch.device):
    """
    Evaluate the student model on a validation set.

    Args:
        model: The student model.
        loader: DataLoader yielding (image, label, _) tuples.
        device: Device to run on.

    Returns:
        Tuple of (accuracy, metrics_dict) where metrics_dict contains
        accuracy, precision_macro, recall_macro, f1_macro, and confusion_matrix.
    """
    model.eval()
    all_preds, all_labels = [], []
    for images, labels, _ in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro", zero_division=0
    )
    return acc, {
        "accuracy": acc,
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
        "confusion_matrix": confusion_matrix(all_labels, all_preds).tolist(),
    }
