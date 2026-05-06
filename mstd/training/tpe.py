"""
Tree-structured Parzen Estimator (TPE) hyperparameter optimization for AMTD.

Uses Optuna to search over the distillation hyperparameter space:
  - Learning rate, weight decay, dropout
  - Distillation temperature and alpha
  - Label smoothing
  - Batch size and LR scheduler type

Each trial trains a StudentModel for a small number of epochs and
reports the best validation accuracy. The study uses MedianPruner
to stop unpromising trials early.
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import optuna

from mstd.config import IMG_SIZE, IMAGENET_STATS, NUM_CLASSES, DEVICE
from mstd.data.dataset import DistillDataset
from mstd.models.student import StudentModel
from mstd.training.distillation import train_epoch_distill, evaluate_distill


def create_tpe_loaders(dataset_path, teacher_logits, batch_size, use_augment):
    """
    DataLoaders for TPE trials.

    Same structure as create_distill_loaders but accepts dataset_path
    directly and is defined here to keep the TPE module self-contained.
    """
    if use_augment:
        train_transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
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

    import os
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


def build_objective(dataset_path, teacher_logits, epochs_per_trial=8):
    """
    Build the Optuna objective function for TPE hyperparameter search.

    The returned function samples hyperparameters, trains briefly, and
    returns the best validation accuracy achieved. The study uses this
    to guide the search toward better hyperparameter regions.

    Args:
        dataset_path: Path to Intel dataset.
        teacher_logits: Pre-computed ensemble teacher logits.
        epochs_per_trial: Number of epochs for each trial (short = fast search).

    Returns:
        Callable objective(dict) suitable for study.optimize().
    """
    def objective(trial):
        lr = trial.suggest_float("lr", 1e-5, 5e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
        dropout = trial.suggest_float("dropout", 0.0, 0.5)
        temperature = trial.suggest_float("temperature", 1.0, 10.0, log=True)
        alpha = trial.suggest_float("alpha", 0.1, 0.9)
        label_smoothing = trial.suggest_float("label_smoothing", 0.0, 0.2)
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
        use_augment = trial.suggest_categorical("use_augment", [True])
        scheduler_name = trial.suggest_categorical("scheduler", ["cosine", "step", "plateau"])

        train_loader, val_loader = create_tpe_loaders(
            dataset_path, teacher_logits, batch_size, use_augment
        )

        model = StudentModel(dropout=dropout).to(DEVICE)
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        if scheduler_name == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs_per_trial)
        elif scheduler_name == "step":
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=epochs_per_trial // 3, gamma=0.5)
        else:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max", factor=0.5, patience=2
            )

        best_val_acc = 0.0
        for epoch in range(epochs_per_trial):
            train_loss = train_epoch_distill(
                model, train_loader, optimizer, scheduler,
                temperature, alpha, label_smoothing, DEVICE,
            )
            val_acc, _ = evaluate_distill(model, val_loader, DEVICE)
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_acc)
            best_val_acc = max(best_val_acc, val_acc)
            trial.report(best_val_acc, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return best_val_acc

    return objective
