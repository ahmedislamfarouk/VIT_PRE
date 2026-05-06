import torch
import torch.nn as nn

from ..config import BACKBONE_REGISTRY, TEACHER_CHECKPOINTS, NUM_CLASSES, DEVICE


class TeacherModel(nn.Module):
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
        feats = self.backbone(x)
        return self.head(feats)


class TeacherEnsemble(nn.Module):
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
        logits = [teacher(x) for teacher in self.teachers.values()]
        return torch.stack(logits, dim=0).mean(dim=0)
