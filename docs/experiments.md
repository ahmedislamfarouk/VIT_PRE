# Experiments

## 1. Frozen Backbone Comparison (`compare.py`)

Compares classification heads trained on top of frozen backbones.

| Backbone | Head | Epochs | LR |
|---|---|---|---|
| DeiT-Tiny | Linear(192→6) | 20 | 1e-3 |
| DINOv2-S/14 | Linear(384→6) | 20 | 1e-3 |
| SigLIP2-B/16 | Linear(768→6) | 20 | 1e-3 |
| YOLOv8-World-v2 | Linear(512→6) | 20 | 1e-3 |

**Outputs:** Training curves, confusion matrices, metric bar chart, per-class F1, radar chart, accuracy GIF.

## 2. AMTD Enhanced (`enhanced_distilled.py`)

Full AMTD pipeline with hyperparameter optimization.

**Phase 1 — Precompute Teacher Logits:**
- Run all 3 teachers on full training set
- Save to `cache/teacher_logits.pkl`

**Phase 2 — Hyperparameter Optimization (Optuna TPE):**
- 25 trials × 8 epochs each
- Search space: lr [1e-5, 1e-3], weight_decay [1e-6, 1e-3], dropout [0.1, 0.5], temperature [1.0, 4.0], alpha [0.3, 0.9], label_smoothing [0.0, 0.2], batch_size [32, 128], scheduler [cosine, step]

**Phase 3 — Final Training:**
- 35 epochs with best HPO params
- MobileViT-S student, pretrained on ImageNet

## 3. From-Scratch Data Hunger (`run_scratch_fixed.py` / `evaluate_data_hunger.py`)

Trains models entirely from random initialization across subset ratios.

**Supported Models:**
| Model | Params | Description |
|---|---|---|
| `cnn` | ~2.8M | SimpleCNN (Conv → Pool → FC) |
| `deit` | 5.7M | DeiT-Tiny |
| `dinov2` | 22M | DINOv2-S/14 |
| `siglip` | 86M | SigLIP2-B/16 |
| `mobilevit_s` | 5.1M | MobileViT-S |
| `mobilevit_hard` | 5.1M | MobileViT-S (CE only, no KD) |
| `amtd` | 1.5M | MobileViT-XXS + Teacher KD |

## 4. Fused Multi-Backbone Ensemble (`enhanced_ensemble.py`)

Experimental: combines features from all 3 backbones via learned attention fusion. Not the main AMTD approach.

## 5. Quick Tests

| Script | Description |
|---|---|
| `run_quick.py` | 3-model comparison (CNN, MobileViT, ResNet18) on one GPU |
| `test5.py` | Quick 5% subset test with SimpleCNN (with BatchNorm) |
| `run_simple.py` | Minimal 7-model trainer for quick prototyping |
