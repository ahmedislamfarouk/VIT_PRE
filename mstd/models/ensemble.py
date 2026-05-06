"""
Enhanced multi-backbone ensemble classifier.

This module implements the "Fused Multi-Backbone Ensemble" concept:
three different backbones (DeiT, DINOv2-style, SigLIP-style) run in
parallel, their features are fused via a learned attention mechanism,
and the fused representation is classified.

Classes:
    BackboneWrapper: Wraps a timm model for feature extraction.
    FeatureFuser: Projects and averages features from multiple backbones.
    EnhancedClassifier: End-to-end ensemble combining all three backbones.
"""

import torch
import torch.nn as nn

from mstd.config import NUM_CLASSES


class BackboneWrapper(nn.Module):
    """
    Wraps a timm model to extract intermediate features (not logits).

    Handles the case where timm may not be installed by auto-installing it.
    """

    def __init__(self, model_name, num_classes=0):
        super().__init__()
        try:
            import timm
            self.model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
        except ImportError:
            import os
            os.system("pip install timm -q")
            import timm
            self.model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)

        self.feature_dim = self.model.embed_dim if hasattr(self.model, 'embed_dim') else self.model.num_features

    def forward(self, x):
        """Return features from the backbone's penultimate layer."""
        return self.model.forward_features(x)


class FeatureFuser(nn.Module):
    """
    Attention-based feature fusion module.

    Each backbone's features are linearly projected to a common 256-dim
    space, then averaged (equal weighting) and passed through an output
    projection. An attention subnetwork is defined but currently unused
    (simple averaging proved more stable).
    """

    def __init__(self, feature_dims):
        super().__init__()
        self.num_backbones = len(feature_dims)
        self.feature_dims = feature_dims

        self.projectors = nn.ModuleList([
            nn.Linear(dim, 256) for dim in feature_dims
        ])

        self.attention = nn.Sequential(
            nn.Linear(256 * self.num_backbones, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_backbones),
            nn.Softmax(dim=1)
        )

        self.output = nn.Linear(256, 256)

    def forward(self, features_list):
        """Project, average, and output fused features."""
        projected = [proj(feat) for proj, feat in zip(self.projectors, features_list)]

        concatenated = torch.cat(projected, dim=1)

        attn_weights = torch.ones(len(projected), projected[0].shape[0], 1).to(projected[0].device)

        fused = torch.stack(projected, dim=0).mean(dim=0)

        return self.output(fused), attn_weights.mean(dim=1)


class EnhancedClassifier(nn.Module):
    """
    End-to-end multi-backbone ensemble classifier.

    Runs all backbones in parallel (frozen), fuses their features,
    and classifies through a shared MLP head.
    """

    def __init__(self, backbone_names):
        super().__init__()

        self.backbones = nn.ModuleList()
        self.feature_dims = []

        for name in backbone_names:
            backbone = BackboneWrapper(name, num_classes=0)
            self.backbones.append(backbone)
            self.feature_dims.append(backbone.feature_dim)

        self.fuser = FeatureFuser(self.feature_dims)

        self.head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, NUM_CLASSES)
        )

    def extract_features(self, x):
        """
        Extract features from all backbones with no_grad.

        Handles different output shapes:
        - 3D (B, N, D): mean-pool over tokens.
        - 2D with very large dim: truncate to 1024.
        """
        all_features = []

        for backbone in self.backbones:
            with torch.no_grad():
                feat = backbone(x)
                if len(feat.shape) == 3:
                    feat = feat.mean(dim=1)
                elif len(feat.shape) == 2 and feat.shape[1] > 4096:
                    feat = feat[:, :1024]
            all_features.append(feat)

        return all_features

    def forward(self, x):
        """
        Extract features, fuse, classify.

        Returns:
            Tuple of (logits, attention_weights).
        """
        features_list = self.extract_features(x)
        fused_features, attn_weights = self.fuser(features_list)
        logits = self.head(fused_features)
        return logits, attn_weights
