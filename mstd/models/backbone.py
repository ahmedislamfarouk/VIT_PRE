"""
Backbone model loading utilities.

This module provides functions and classes for loading frozen pretrained
backbones from timm (DeiT, DINOv2, SigLIP2) and Ultralytics (YOLOv8-World)
and attaching a lightweight trainable classification head.

Classes:
    ClassificationHead: Simple MLP head with dropout.
    BackboneClassifier: Frozen backbone + trainable head combined model.
    YOLOv8WorldBackbone: YOLOv8 feature extractor via forward hooks.

Functions:
    load_backbone: Load a named backbone by looking up the registry.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from mstd.config import BACKBONE_REGISTRY, NUM_CLASSES, YOLOV8_WORLD_PATH


class ClassificationHead(nn.Module):
    """
    Lightweight classification head: Linear -> ReLU -> Dropout -> Linear.

    Designed to sit on top of a frozen vision backbone.
    """

    def __init__(self, in_dim: int, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Produce class logits from input features."""
        return self.head(x)


class BackboneClassifier(nn.Module):
    """
    Combines a frozen pretrained backbone with a trainable ClassificationHead.

    The backbone is kept in eval mode and its parameters are frozen
    so only the head receives gradient updates during training.
    """

    def __init__(self, backbone_name: str):
        super().__init__()
        self.backbone_name = backbone_name
        self.backbone, feature_dim = load_backbone(backbone_name)
        self.head = ClassificationHead(feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features with frozen backbone, classify with trainable head."""
        with torch.no_grad():
            features = self.backbone(x)
        return self.head(features)


def load_backbone(name: str):
    """
    Load a frozen pretrained backbone by name.

    Args:
        name: Short name from BACKBONE_REGISTRY (e.g. 'deit', 'dinov2',
              'siglip2', 'yolov8world').

    Returns:
        Tuple of (backbone_module, feature_dimension).
    """
    if name not in BACKBONE_REGISTRY:
        raise ValueError(f"Unknown backbone '{name}'. Choose from: {list(BACKBONE_REGISTRY.keys())}")

    if name == "yolov8world":
        backbone = YOLOv8WorldBackbone(YOLOV8_WORLD_PATH)
        backbone.eval()
        actual_dim = backbone.feature_dim
        return backbone, actual_dim

    import timm
    timm_name, feature_dim, description = BACKBONE_REGISTRY[name]
    backbone = timm.create_model(
        timm_name, pretrained=True, num_classes=0, dynamic_img_size=True
    )
    backbone.eval()
    for param in backbone.parameters():
        param.requires_grad = False
    return backbone, feature_dim


class YOLOv8WorldBackbone(nn.Module):
    """
    YOLOv8l-WorldV2 frozen backbone for feature extraction.

    Uses a forward hook attached to the SPPF layer (or a fallback C2f/C3
    layer) to capture intermediate feature maps, then pools to a 1D vector.

    Attributes:
        SPPF_INPUT_SIZE: YOLOv8 native input resolution (640).
        feature_dim: Dimensionality of pooled features.
    """

    SPPF_INPUT_SIZE = 640

    def __init__(self, model_path: str):
        super().__init__()
        from ultralytics import YOLO
        self.yolo = YOLO(model_path)
        self.inner = self.yolo.model.model
        self.yolo.model.eval()
        for p in self.parameters():
            p.requires_grad = False

        self._feat = None
        hook_layer = None
        for m in self.inner:
            if type(m).__name__ == "SPPF":
                hook_layer = m
                break
        if hook_layer is None:
            for m in reversed(list(self.inner)):
                if type(m).__name__ in ("C2f", "C2fAttn", "C3"):
                    hook_layer = m
                    break
        if hook_layer is None:
            raise RuntimeError("Could not find SPPF or C2f layer in YOLO model")
        self._hook_handle = hook_layer.register_forward_hook(self._hook_fn)
        self.pool = nn.AdaptiveAvgPool2d(1)

        with torch.no_grad():
            dummy = torch.randn(1, 3, self.SPPF_INPUT_SIZE, self.SPPF_INPUT_SIZE)
            feat = self._extract_features(dummy)
        self._feature_dim = feat.shape[-1]

    def _hook_fn(self, module, input, output):
        """Forward hook that stores intermediate feature maps."""
        self._feat = output

    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run a forward pass through YOLO and capture the hooked features.

        The try/except is needed because YOLO may error on the final
        detection head — we only care about the backbone features.
        """
        self._feat = None
        with torch.no_grad():
            self.yolo.model.eval()
            try:
                self.yolo.model(x)
            except Exception:
                pass
        feat = self._feat
        if isinstance(feat, (list, tuple)):
            feat = feat[-1]
        return self.pool(feat).flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Resize input to 640px if needed, then extract pooled features.
        """
        if x.shape[-1] != self.SPPF_INPUT_SIZE or x.shape[-2] != self.SPPF_INPUT_SIZE:
            x = F.interpolate(x, size=(self.SPPF_INPUT_SIZE, self.SPPF_INPUT_SIZE),
                               mode="bilinear", align_corners=False)
        return self._extract_features(x)

    @property
    def feature_dim(self) -> int:
        """Return the feature dimensionality determined at init."""
        return self._feature_dim

    def train(self, mode: bool = True):
        """
        Override train() to keep backbone frozen while allowing the
        pooling layer to be toggled.
        """
        self.training = mode
        self.inner.eval()
        for p in self.parameters():
            p.requires_grad = False
        self.pool.train(mode)
        return self
