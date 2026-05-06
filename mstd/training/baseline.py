import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from tqdm import tqdm

from ..config import DEVICE, CHECKPOINT_DIR, CLASS_NAMES, NUM_CLASSES


class ModelTrainer:
    def __init__(self, model_name: str, train_loader: DataLoader, val_loader: DataLoader,
                 epochs: int = None, lr: float = None):
        from ..models.backbone import BackboneClassifier
        self.model_name = model_name
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.is_yolo = model_name == "yolov8world"
        self.epochs = epochs or 10
        self.lr = lr or 1e-4

        self.model = BackboneClassifier(model_name).to(DEVICE)
        self.optimizer = optim.AdamW(
            self.model.head.parameters(), lr=self.lr, weight_decay=1e-4
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.epochs
        )
        self.criterion = nn.CrossEntropyLoss()

        self.history = {"train_loss": [], "val_loss": [], "val_acc": []}
        self.best_val_acc = 0.0

    def _forward(self, images: torch.Tensor) -> torch.Tensor:
        if self.is_yolo:
            chunks = []
            for i in range(0, images.size(0), 4):
                chunks.append(self.model(images[i:i + 4]))
            return torch.cat(chunks, dim=0)
        return self.model(images)

    def train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        for images, labels in tqdm(
            self.train_loader, desc=f"  [{self.model_name}] Train", leave=False
        ):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            self.optimizer.zero_grad()
            outputs = self._forward(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * images.size(0)
        return total_loss / len(self.train_loader.dataset)

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_loss = 0.0
        all_preds, all_labels = [], []

        for images, labels in tqdm(
            self.val_loader, desc=f"  [{self.model_name}] Val", leave=False
        ):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = self._forward(images)
            loss = self.criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(self.val_loader.dataset)
        accuracy = accuracy_score(all_labels, all_preds)
        return avg_loss, accuracy

    def train(self) -> dict:
        for epoch in range(1, self.epochs + 1):
            train_loss = self.train_epoch()
            val_loss, val_acc = self.validate()
            self.scheduler.step()

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)

            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.save_checkpoint()

        return self.history

    def save_checkpoint(self):
        path = CHECKPOINT_DIR / f"{self.model_name}_best.pth"
        torch.save(
            {
                "model_name": self.model_name,
                "head_state_dict": self.model.head.state_dict(),
                "best_val_acc": self.best_val_acc,
            },
            path,
        )

    @torch.no_grad()
    def evaluate(self) -> dict:
        self.model.eval()
        all_preds, all_labels = [], []

        for images, labels in tqdm(
            self.val_loader, desc=f"  [{self.model_name}] Evaluate"
        ):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = self._forward(images)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average="macro", zero_division=0
        )
        per_cls_p, per_cls_r, per_cls_f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, zero_division=0
        )
        cm = confusion_matrix(all_labels, all_preds)

        return {
            "model": self.model_name,
            "accuracy": round(accuracy, 4),
            "precision_macro": round(precision, 4),
            "recall_macro": round(recall, 4),
            "f1_macro": round(f1, 4),
            "per_class": {
                CLASS_NAMES[i]: {
                    "precision": round(per_cls_p[i], 4),
                    "recall": round(per_cls_r[i], 4),
                    "f1": round(per_cls_f1[i], 4),
                }
                for i in range(NUM_CLASSES)
            },
            "confusion_matrix": cm.tolist(),
        }
