#!/usr/bin/env python3
"""
Fixed version with proper model configs for SigLIP and DINOv2
"""
import os, sys, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms, datasets
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedShuffleSplit
import timm

from mstd.models.student import (
    MobileViTScratch, MobileViTHard, DeiTScratch, DINOv2Scratch,
    SigLIPScratch, SimpleCNN, AMTDStudentScratch,
)
from mstd.config import DEVICE as mstd_device

def main():
    SUBSET_RATIO = float(sys.argv[1]) if len(sys.argv) > 1 else 0.05
    MODEL_NAME = sys.argv[2] if len(sys.argv) > 2 else 'amtd'
    GPU_ID = int(sys.argv[3]) if len(sys.argv) > 3 else 0

    torch.cuda.set_device(GPU_ID)
    DEVICE = torch.device(f'cuda:{GPU_ID}')

    print(f'=== {MODEL_NAME} on {int(SUBSET_RATIO*100)}% data, GPU {GPU_ID} ===')
    print(f'Training from SCRATCH')

    np.random.seed(42)
    torch.manual_seed(42)

IMG_SIZE = 224
NUM_CLASSES = 6
SEED = 42
IMAGENET_STATS = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}


def get_subset_indices(dataset, ratio, seed):
    targets = np.array([dataset.targets[i] for i in range(len(dataset))])
    if ratio >= 1.0:
        return np.arange(len(targets))
    sss = StratifiedShuffleSplit(n_splits=1, train_size=ratio, random_state=seed)
    indices, _ = next(sss.split(np.zeros(len(targets)), targets))
    return indices


class SimpleDataset(Dataset):
    def __init__(self, base_ds, transform):
        self.base = base_ds
        self.transform = transform
    def __len__(self):
        return len(self.base)
    def __getitem__(self, idx):
        img, label = self.base[idx]
        return self.transform(img), label


def make_loaders(subset_ratio):
    train_t = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.RandomHorizontalFlip(), transforms.RandomRotation(15), transforms.ColorJitter(0.2, 0.2, 0.2), transforms.ToTensor(), transforms.Normalize(**IMAGENET_STATS)])
    val_t = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor(), transforms.Normalize(**IMAGENET_STATS)])

    train_base = datasets.ImageFolder('/home/skyvision/.cache/kagglehub/datasets/puneet6060/intel-image-classification/versions/2/seg_train/seg_train')
    val_base = datasets.ImageFolder('/home/skyvision/.cache/kagglehub/datasets/puneet6060/intel-image-classification/versions/2/seg_test/seg_test')

    indices = get_subset_indices(train_base, subset_ratio, SEED)
    train_ds = SimpleDataset(Subset(train_base, indices), train_t)
    val_ds = SimpleDataset(val_base, val_t)

    return (DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=4, pin_memory=True), DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=4, pin_memory=True), len(indices))


    train_loader, val_loader, n_samples = make_loaders(SUBSET_RATIO)
    print(f'Samples: {n_samples}')

    models = {
        'mobilevit_s': MobileViTScratch,
        'mobilevit_hard': MobileViTHard,
        'deit': DeiTScratch,
        'dinov2': DINOv2Scratch,
        'siglip': SigLIPScratch,
        'cnn': SimpleCNN,
        'amtd': AMTDStudentScratch,
    }
    ModelClass = models[MODEL_NAME]
    model = ModelClass(NUM_CLASSES).to(DEVICE)

    if SUBSET_RATIO <= 0.05:
        epochs = 50
    elif SUBSET_RATIO <= 0.10:
        epochs = 40
    elif SUBSET_RATIO <= 0.50:
        epochs = 25
    else:
        epochs = 15

    print(f'Epochs: {epochs}')

    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                preds = model(images.to(DEVICE)).argmax(1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.numpy())
        acc = accuracy_score(all_labels, all_preds)
        if acc > best_acc:
            best_acc = acc
            best_state = model.state_dict().copy()

        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            print(f'Epoch {epoch+1}/{epochs}: {acc*100:.2f}%')

    model.load_state_dict(best_state)
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            preds = model(images.to(DEVICE)).argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    p, r, _, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro')

    print(f'=== {MODEL_NAME} on {int(SUBSET_RATIO*100)}%: Acc={acc*100:.2f}%, F1={f1*100:.2f}% ===')

    checkpoint_path = f'/home/skyvision/ammarbigass5/checkpoints/scratch_{MODEL_NAME}_{int(SUBSET_RATIO*100)}pct_e{epochs}.pth'
    torch.save({'model': MODEL_NAME, 'subset': SUBSET_RATIO, 'epochs': epochs, 'accuracy': acc, 'f1': f1, 'state_dict': best_state}, checkpoint_path)
    print(f'Checkpoint: {checkpoint_path}')

    out_file = f'/home/skyvision/ammarbigass5/results/scratch_{MODEL_NAME}_{int(SUBSET_RATIO*100)}.json'
    result = {'model': MODEL_NAME, 'subset': SUBSET_RATIO, 'n_samples': n_samples, 'epochs': epochs, 'accuracy': acc, 'f1_macro': f1, 'precision': p, 'recall': r}
    with open(out_file, 'w') as f:
        json.dump(result, f, indent=2)
    print(f'Result: {out_file}')
    print('=== COMPLETE ===')


if __name__ == "__main__":
    main()
