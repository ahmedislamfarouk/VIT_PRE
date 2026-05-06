"""
From-scratch training loop for data-efficiency experiments.

Trains a model from randomly initialized weights (no pretraining) using
standard cross-entropy loss. Used to measure how each architecture performs
when trained on limited data (5%, 10%, 25%, 50%, 100%).

This is the programmatic version of the inline loop in scripts/run_scratch.py,
useful when you want to call the training from other Python code.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support


def train_scratch(model, train_loader, val_loader, epochs, lr, device):
    """
    Train a model from scratch with cosine annealing LR schedule.

    Tracks the best validation accuracy and restores the best
    state dict at the end.

    Args:
        model: PyTorch module (randomly initialized).
        train_loader: DataLoader for training.
        val_loader: DataLoader for validation.
        epochs: Number of training epochs.
        lr: Peak learning rate.
        device: Device to run on.

    Returns:
        Dict with accuracy, f1_macro, precision, recall, best_acc, best_state.
    """
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                preds = model(images.to(device)).argmax(1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.numpy())
        acc = accuracy_score(all_labels, all_preds)
        if acc > best_acc:
            best_acc = acc
            best_state = model.state_dict().copy()

    model.load_state_dict(best_state)
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            preds = model(images.to(device)).argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    p, r, _, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro')

    return {
        "accuracy": acc,
        "f1_macro": f1,
        "precision": p,
        "recall": r,
        "best_acc": best_acc,
        "best_state": best_state,
    }
