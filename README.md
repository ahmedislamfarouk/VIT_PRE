# AMTD (Adaptive Multi-Teacher Distillation) - Enhanced Version

## Overview

This project implements an enhanced knowledge distillation approach called **AMTD (Adaptive Multi-Teacher Distillation)** to address the **data hunger problem** in computer vision. The key insight is that large Vision Transformers (ViTs) require massive amounts of data to train, but we can transfer their knowledge to smaller, more efficient models using multiple teacher models.

## The Data Hunger Problem

- Vision Transformers (ViTs) like DeiT, DINOv2, and SigLIP require huge datasets (1M+ images) for effective training
- With limited data (5-10% of training set), these models suffer from severe performance degradation
- **Our solution**: Use knowledge distillation from multiple pre-trained teachers to train a lightweight student from scratch

## Architecture

### Student Model (Our Enhanced AMTD)

- **Backbone**: `MobileViT-XXS` (trained from scratch, random initialization)
  - Parameters: ~1.5M (much smaller than teacher models)
  - Designed for efficiency and fast inference
  
- **Classification Head**:
  ```
  Input Features → Dropout(0.2) → Linear(in_features, 128) → ReLU → 
  Dropout(0.1) → Linear(128, num_classes)
  ```

### Teacher Models (for comparison)

| Model | Type | Parameters | Source |
|-------|------|------------|--------|
| DeiT-Tiny | ViT | ~5M | ImageNet pretrained |
| DINOv2-S/14 | ViT | ~22M | ImageNet pretrained |
| SigLIP2-B/16 | ViT | ~86M | ImageNet pretrained |

## Enhancements Made

### 1. Multi-Teacher Ensemble
Instead of using a single teacher, we combine knowledge from 3 different pre-trained Vision Transformers:
- **DeiT**: Data-efficient training knowledge
- **DINOv2**: Self-supervised visual representations
- **SigLIP**: CLIP-style semantic understanding

### 2. Adaptive Weighting
- Teachers are combined using **soft probability averaging**
- Loss = α × Hard_Label_Loss + (1-α) × Soft_Distillation_Loss
- α = 0.75 (75% hard labels, 25% teacher knowledge)

### 3. Temperature Scaling
- Temperature T = 2.4 for softening probability distributions
- Helps transfer dark knowledge from teachers

### 4. Label Smoothing
- 10% label smoothing to prevent overconfidence

### 5. Optimizer Configuration
- Learning Rate: 4e-4
- Weight Decay: 1e-5
- Scheduler: Cosine Annealing
- Batch Size: 64

## Training Configuration

| Subset | Samples | Epochs | Learning Rate |
|--------|---------|--------|---------------|
| 5%     | 701     | 50     | 1e-3          |
| 10%    | 1,403   | 40     | 1e-3          |
| 50%    | 7,017   | 25     | 1e-3          |
| 100%   | 14,034  | 15     | 1e-3          |

## Key Results (From Scratch Training)

Training ENTIRE model from random initialization - NO pretrained weights:

| Model | 5% (701) | 10% (1.4K) | 50% (7K) | 100% (14K) |
|-------|-----------|------------|-----------|------------|
| **AMTD (Ours)** | **76.9%** | **81.0%** | **87.4%** | **89.3%** |
| CNN (baseline) | 65.9% | 71.6% | 79.2% | 81.0% |
| DeiT | 64.1% | 69.0% | 77.6% | 79.7% |

## Key Findings

1. **AMTD outperforms all baselines** even when training from scratch
2. With only **5% of data** (701 images), AMTD achieves 76.9% accuracy vs 65.9% for CNN
3. **+11% improvement** over CNN baseline with limited data
4. Scales well to full dataset (89.3% with 100% data)

## Files Structure

```
ammarbigass5/
├── run_scratch_fixed.py     # Main training script (from scratch)
├── run_scratch_single.py    # Single model training
├── checkpoints/
│   └── scratch_amtd_5pct_e50.pth    # AMTD checkpoint (5%, 50 epochs)
│   └── scratch_amtd_10pct_e40.pth   # AMTD checkpoint (10%, 40 epochs)
│   └── ...
├── results/
│   └── scratch_amtd_*.json  # Results with accuracy, F1, precision, recall
└── README.md
```

## How to Run

```bash
# Train AMTD on 5% data
python3.10 run_scratch_fixed.py 0.05 amtd

# Train on 10% data
python3.10 run_scratch_fixed.py 0.10 amtd

# Train on 50% data
python3.10 run_scratch_fixed.py 0.50 amtd

# Train on 100% data
python3.10 run_scratch_fixed.py 1.00 amtd
```

## Checkpoint Naming Convention

Format: `scratch_{model}_{percentage}pct_e{epochs}.pth`

Example: `scratch_amtd_5pct_e50.pth` = AMTD trained on 5% data for 50 epochs

## Citation

If you use this code, please cite:

```
AMTD: Adaptive Multi-Teacher Distillation for Data-Efficient Vision Transformers
```

## License

MIT License