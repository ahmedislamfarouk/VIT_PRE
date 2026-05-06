import torch
import torch.nn as nn

from ..config import NUM_CLASSES


class StudentModel(nn.Module):
    def __init__(self, dropout: float = 0.3):
        super().__init__()
        import timm
        self.backbone = timm.create_model(
            "mobilevit_s", pretrained=True, num_classes=0
        )
        feature_dim = self.backbone.num_features
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(256, NUM_CLASSES),
        )

    def forward(self, x):
        feats = self.backbone(x)
        return self.head(feats)

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class MobileViTScratch(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        import timm
        self.backbone = timm.create_model('mobilevit_s', pretrained=False, num_classes=0)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(self.backbone.num_features, 256), nn.ReLU(), nn.Dropout(0.15), nn.Linear(256, num_classes))
    def forward(self, x):
        return self.head(self.backbone(x))


class MobileViTHard(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        import timm
        self.backbone = timm.create_model('mobilevit_s', pretrained=False, num_classes=0)
        self.head = nn.Sequential(nn.Dropout(0.1), nn.Linear(self.backbone.num_features, 256), nn.ReLU(), nn.Linear(256, num_classes))
    def forward(self, x):
        return self.head(self.backbone(x))


class DeiTScratch(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        import timm
        self.backbone = timm.create_model('deit_tiny_patch16_224', pretrained=False, num_classes=0)
        self.head = nn.Linear(self.backbone.num_features, num_classes)
    def forward(self, x):
        return self.head(self.backbone(x))


class DINOv2Scratch(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        import timm
        self.backbone = timm.create_model('vit_small_patch16_224', pretrained=False, num_classes=0)
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(self.backbone.num_features, 256), nn.ReLU(), nn.Linear(256, num_classes))
    def forward(self, x):
        return self.head(self.backbone(x))


class SigLIPScratch(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        import timm
        self.backbone = timm.create_model('vit_base_patch16_siglip_224.v2_webli', pretrained=False, num_classes=0)
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(self.backbone.num_features, 256), nn.ReLU(), nn.Linear(256, num_classes))
    def forward(self, x):
        return self.head(self.backbone(x))


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.AdaptiveAvgPool2d(1))
        self.classifier = nn.Linear(256, num_classes)
    def forward(self, x):
        return self.classifier(self.features(x).view(x.size(0), -1))


class AMTDStudentScratch(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        import timm
        self.backbone = timm.create_model('mobilevit_xxs', pretrained=False, num_classes=0)
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(self.backbone.num_features, 128), nn.ReLU(), nn.Dropout(0.1), nn.Linear(128, num_classes))
    def forward(self, x):
        return self.head(self.backbone(x))
