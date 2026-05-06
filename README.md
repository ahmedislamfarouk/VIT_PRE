# MSTD — Multi-Strategy Transfer Distillation

**Comparing vision backbones (DeiT, DINOv2, SigLIP2, YOLOv8-World) and enhancing them with knowledge distillation and ensemble fusion on the Intel Image Classification dataset.**

---

## Project Overview

This project investigates **ViT data-hungriness** and proposes **AMTD (Adaptive Multi-Teacher Distillation)** as a solution. The key insight: large ViTs need lots of data to perform well from scratch, but by distilling knowledge from an ensemble of frozen teachers into a lightweight student, we achieve **higher accuracy with far fewer parameters and less training data**.

### Key Results

| Model | Accuracy | Params |
|-------|----------|--------|
| DeiT-Tiny | 91.27% | 5.7M |
| DINOv2-S/14 | 94.93% | 22M |
| SigLIP2-B/16 | 88.17% | 86M |
| **AMTD (Ours)** | **≥95.5%** | **~5.6M** |

AMTD matches or exceeds DINOv2's accuracy with **15x fewer parameters** and significantly less training data.

---

## Directory Structure

```
MSTD/
├── mstd/                          # Core Python package
│   ├── config.py                  # Shared configuration (constants, paths, model registry)
│   ├── models/                    # Model architectures
│   │   ├── backbone.py            # Frozen backbone loader, ClassificationHead, YOLOv8 hook
│   │   ├── teacher.py             # TeacherModel, TeacherEnsemble (frozen teachers)
│   │   ├── student.py             # StudentModel (MobileViT) + all scratch models
│   │   └── ensemble.py            # Multi-backbone EnhancedClassifier + FeatureFuser
│   ├── data/                      # Dataset & data loading
│   │   ├── dataset.py             # Transforms, DistillDataset, LogitsDataset, subset utilities
│   │   ├── loaders.py             # DataLoader builders for each experiment type
│   │   └── precompute.py          # Teacher logit pre-computation & caching
│   ├── training/                  # Training algorithms
│   │   ├── loss.py                # Knowledge distillation loss
│   │   ├── distillation.py        # Distillation train/eval loops
│   │   ├── baseline.py            # ModelTrainer (frozen backbone + head training)
│   │   ├── scratch.py             # From-scratch training loop
│   │   └── tpe.py                 # Optuna TPE hyperparameter optimization
│   ├── evaluation/                # Evaluation utilities
│   │   └── metrics.py             # evaluate_model, comparison table printer
│   └── visualization/             # Plotting & animations
│       ├── plots.py               # All static plots (curves, confusion matrices, radar, bars)
│       └── gif.py                 # Animated GIFs (training progression, radar animation)
├── scripts/                       # CLI entry points (one per experiment)
│   ├── compare.py                 # Backbone comparison (DeiT vs DINOv2 vs SigLIP2 vs YOLO)
│   ├── enhanced_distilled.py      # AMTD — Adaptive Multi-Teacher Distillation
│   ├── enhanced_ensemble.py       # Fused Multi-Backbone Ensemble
│   ├── evaluate_data_hunger.py    # Data-efficiency evaluation across subset sizes
│   ├── run_scratch.py             # From-scratch training for all architectures
│   ├── complete_enhanced_training.py  # Final AMTD run with best TPE hyperparameters
│   └── radar.py                   # Radar chart animation
├── compare.py (wrapper)           # Backward-compatible root wrappers
├── enhanced_distilled.py (wrapper)
├── enhanced_ensemble.py (wrapper)
├── evaluate_data_hunger.py (wrapper)
├── run_scratch_fixed.py (wrapper)
├── complete_enhanced_training.py (wrapper)
├── radar.py (wrapper)
├── gifs/                          # GIF generation scripts (visualization helpers)
├── docs/                          # Documentation files
├── checkpoints/                   # Saved model checkpoints
├── results/                       # Output plots, JSON results
├── cache/                         # Pre-computed teacher logits
├── logs/                          # Training logs
├── requirements.txt               # Python dependencies
├── launch_full.sh                 # Script to launch all scratch experiments in parallel
└── README.md                      # This file
```

---

## Module Details

### `mstd/config.py` — Shared Configuration

Single source of truth for all constants: dataset info, paths, model registry, device selection, plot colors, and default hyperparameters. Every other module imports from here.

### `mstd/models/` — Model Architectures

| File | Contents | Purpose |
|------|----------|---------|
| `backbone.py` | `ClassificationHead`, `BackboneClassifier`, `load_backbone()`, `YOLOv8WorldBackbone` | Load frozen timm backbones or YOLOv8 via hooks; attach trainable head |
| `teacher.py` | `TeacherModel`, `TeacherEnsemble` | Frozen teacher(s) with trained heads for distillation |
| `student.py` | `StudentModel`, `MobileViTScratch`, `DeiTScratch`, `DINOv2Scratch`, `SigLIPScratch`, `SimpleCNN`, `AMTDStudentScratch` | Student for distillation + all scratch-training architectures |
| `ensemble.py` | `BackboneWrapper`, `FeatureFuser`, `EnhancedClassifier` | Multi-backbone fusion classifier |

### `mstd/data/` — Data Pipeline

| File | Contents | Purpose |
|------|----------|---------|
| `dataset.py` | `get_dataset_path()`, transforms, `DistillDataset`, `LogitsDataset`, `SimpleDataset`, `get_subset_indices()` | Dataset access, image transforms, stratified subsets |
| `loaders.py` | `create_dataloaders()`, `create_distill_loaders()`, `create_subset_loaders()`, `create_scratch_loaders()` | Standardized DataLoader factories for each experiment |
| `precompute.py` | `precompute_teacher_logits()` | One-time forward pass of all teachers; cache to disk |

### `mstd/training/` — Training Algorithms

| File | Contents | Purpose |
|------|----------|---------|
| `loss.py` | `distillation_loss()` | Hinton KD loss: CE + KL with temperature |
| `distillation.py` | `train_epoch_distill()`, `evaluate_distill()` | Distillation training loop |
| `baseline.py` | `ModelTrainer` | Full train/validate/evaluate for frozen backbones |
| `scratch.py` | `train_scratch()` | From-scratch training with cosine annealing |
| `tpe.py` | `build_objective()` | Optuna TPE hyperparameter search |

### `mstd/evaluation/metrics.py`

Quick model evaluation and formatted comparison tables.

### `mstd/visualization/` — Plots & Animation

| File | Contents | Purpose |
|------|----------|---------|
| `plots.py` | `plot_training_curves()`, `plot_confusion_matrices()`, `plot_metric_comparison()`, `plot_per_class_f1()`, `plot_radar()`, `plot_amtd_curves()`, `plot_data_hunger_curves()`, `plot_amtd_val_curve()` | All static plots saved to `results/` |
| `gif.py` | `plot_training_gif()`, `create_radar_animation()` | Animated visualizations |

---

## Experiments

### 1. Backbone Comparison (`scripts/compare.py`)

Trains a frozen backbone + trainable head for each model and compares:
- DeiT-Tiny (ViT, 5.7M params)
- DINOv2-S/14 (self-supervised ViT, 22M params)
- SigLIP2-B/16 (CLIP-style ViT, 86M params)
- YOLOv8l-WorldV2 (CNN-based detector backbone)

Generates: training curves, confusion matrices, bar charts, radar chart, accuracy GIF.

```
python3 compare.py
python3 compare.py --model deit        # run single model
python3 compare.py --download-only     # just download dataset
```

### 2. AMTD — Adaptive Multi-Teacher Distillation (`scripts/enhanced_distilled.py`)

**The main contribution.** A lightweight MobileViT-S student (~5.6M params) learns from an ensemble of 3 frozen teachers (DeiT, DINOv2, SigLIP2) via knowledge distillation. Includes TPE hyperparameter search with Optuna.

Pipeline:
1. Pre-compute teacher logits (cached to disk)
2. TPE search over 9 hyperparameters (25 trials x 8 epochs each)
3. Final training with best hyperparameters (35 epochs)
4. Save checkpoint + results JSON + training curves plot

```
python3 enhanced_distilled.py
python3 enhanced_distilled.py --skip-tpe --final-epochs 20
python3 enhanced_distilled.py --tpe-trials 50 --force-cache
```

### 3. Enhanced Ensemble (`scripts/enhanced_ensemble.py`)

An alternative approach: run 3 different backbones in parallel, fuse their features via learned attention, and classify the fused representation. Memory-intensive (requires ~30GB GPU).

```
python3 enhanced_ensemble.py --epochs 15
```

### 4. Data-Efficiency Evaluation (`scripts/evaluate_data_hunger.py`)

Demonstrates AMTD solves the ViT "data-hungry" problem. Trains all methods on 5%, 10%, 25%, 50%, 100% of training data and compares accuracy curves. AMTD should have the flattest curve (retains performance even with little data).

```
python3 evaluate_data_hunger.py
```

### 5. From-Scratch Training (`scripts/run_scratch.py`)

Trains each architecture from randomly initialized weights on data subsets. Used for data-hunger baselines. Accepts CLI args for batch launching.

```
python3 run_scratch.py 0.05 amtd 0     # 5% data, AMTD model, GPU 0
python3 run_scratch.py 0.10 cnn 1      # 10% data, SimpleCNN, GPU 1
```

### 6. Complete AMTD Final Training (`scripts/complete_enhanced_training.py`)

Self-contained script with hardcoded best TPE hyperparameters. Trains the AMTD student for up to 50 epochs with early stopping.

```
python3 complete_enhanced_training.py
```

### 7. Radar Animation (`scripts/radar.py`)

Generates a sequential radar chart GIF showing DeiT, DINOv2, and SigLIP2 metrics appearing one-by-one.

```
python3 radar.py [output_path]
```

---

## How the Code Fits Together

```
                          ┌─────────────────────┐
                          │    mstd/config.py    │
                          │  (all constants)     │
                          └────────┬────────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              ▼                    ▼                    ▼
     ┌────────────────┐   ┌──────────────┐   ┌──────────────────┐
     │ mstd/models/   │   │ mstd/data/   │   │ mstd/visualization│
     │ - backbone.py  │   │ - dataset.py │   │ - plots.py       │
     │ - teacher.py   │   │ - loaders.py │   │ - gif.py         │
     │ - student.py   │   │ - precompute │   └──────────────────┘
     │ - ensemble.py  │   └──────┬───────┘
     └────────────────┘          │
              │                  │
              ▼                  ▼
     ┌──────────────────────────────────────┐
     │         mstd/training/               │
     │  - loss.py (distillation)            │
     │  - distillation.py (train/eval loop) │
     │  - baseline.py (ModelTrainer)        │
     │  - scratch.py (from-scratch)         │
     │  - tpe.py (Optuna search)            │
     └────────────────┬─────────────────────┘
                      │
                      ▼
     ┌──────────────────────────────────────┐
     │       scripts/*.py (CLI entry)       │
     │  compare.py | enhanced_distilled.py  │
     │  evaluate_data_hunger.py | scratch   │
     └──────────────────────────────────────┘
```

Each `scripts/` entry point:
1. Loads configuration from `mstd/config.py`
2. Builds models from `mstd/models/`
3. Loads data via `mstd/data/`
4. Trains using algorithms from `mstd/training/`
5. Evaluates via `mstd/evaluation/`
6. Generates plots via `mstd/visualization/`

---

## Requirements

```
timm
torch
torchvision
kagglehub
scikit-learn
matplotlib
tqdm
Pillow
numpy
optuna
ultralytics (for YOLOv8-World)
```

Install: `pip install -r requirements.txt`

---

## Dataset

Intel Image Classification ([puneet6060/intel-image-classification](https://www.kaggle.com/datasets/puneet6060/intel-image-classification))

- 6 classes: buildings, forest, glacier, mountain, sea, street
- ~14k training images, ~3k test images
- 150x150 pixels (resized to 224x224)

---

## Citation

If you use this code in your research, please cite the associated paper.
