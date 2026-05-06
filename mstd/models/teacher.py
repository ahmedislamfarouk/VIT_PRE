"""
Teacher models for knowledge distillation.

Provides two teacher implementations:
  - TeacherModel: Single frozen backbone + trained classification head.
  - TeacherEnsemble: Ensemble of all three teacher models (DeiT, DINOv2,
    SigLIP2) whose logits are averaged.

These teachers are used to generate soft targets for the student model
during the AMTD (Adaptive Multi-Teacher Distillation) process.
"""

import torch
import torch.nn as nn

from mstd.config import BACKBONE_REGISTRY, TEACHER_CHECKPOINTS, NUM_CLASSES, DEVICE


class TeacherModel(nn.Module):
    """
    Single frozen teacher: a pretrained timm backbone + a fine-tuned head.

    The backbone and head are both frozen (requires_grad = False).
    The head state_dict is loaded from a checkpoint saved by ModelTrainer.
    """

    def __init__(self, name: str):
        super().__init__()
        import timm
        self.name = name
        timm_name, feature_dim, _ = BACKBONE_REGISTRY[name]
        self.backbone = timm.create_model(
            timm_name, pretrained=True, num_classes=0, dynamic_img_size=True
        )
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False

        head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, NUM_CLASSES),
        )
        ckpt_path = TEACHER_CHECKPOINTS[name]
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            state = ckpt["head_state_dict"]
            stripped = {k.replace("head.", "", 1): v for k, v in state.items()}
            head.load_state_dict(stripped)
        self.head = head

        self.head.eval()
        for p in self.head.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features with backbone, then produce logits with head."""
        feats = self.backbone(x)
        return self.head(feats)


class TeacherEnsemble(nn.Module):
    """
    Ensemble of all three teacher models (DeiT, DINOv2, SigLIP2).

    Each teacher is a frozen backbone + frozen head. Forward pass returns
    the averaged logits across all teachers, providing a richer soft target
    for the student than any single teacher could.
    """

    def __init__(self):
        super().__init__()
        import timm
        self.teachers = nn.ModuleDict()
        for name, (timm_name, feat_dim) in BACKBONE_REGISTRY.items():
            backbone = timm.create_model(timm_name, pretrained=True, num_classes=0, dynamic_img_size=True)
            backbone.eval()
            for p in backbone.parameters():
                p.requires_grad = False
            head = nn.Sequential(
                nn.Linear(feat_dim, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, NUM_CLASSES)
            )
            ckpt = torch.load(TEACHER_CHECKPOINTS[name], map_location="cpu", weights_only=False)
            state = ckpt["head_state_dict"]
            stripped = {k.replace("head.", "", 1): v for k, v in state.items()}
            head.load_state_dict(stripped)
            head.eval()
            for p in head.parameters():
                p.requires_grad = False
            self.teachers[name] = nn.Sequential(backbone, head)

    @torch.no_grad()
    def forward(self, x):
        """Return mean logits across all three teachers."""
        logits = [teacher(x) for teacher in self.teachers.values()]
        return torch.stack(logits, dim=0).mean(dim=0)
