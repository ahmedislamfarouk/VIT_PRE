import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from .loss import distillation_loss


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
