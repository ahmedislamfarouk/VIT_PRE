# Architecture

## Core Distillation Pipeline

```
                    ┌──────────────────────┐
                    │   Input Image (224x224) │
                    └──────────┬───────────┘
                               │
              ┌────────────────┼────────────────┐
              ▼                ▼                ▼
       ┌──────────┐    ┌──────────┐    ┌──────────┐
       │ DeiT-Tiny│    │DINOv2-S  │    │SigLIP2-B │
       │ (5.7M)   │    │ (22M)    │    │ (86M)    │
       │ Frozen   │    │ Frozen   │    │ Frozen   │
       └────┬─────┘    └────┬─────┘    └────┬─────┘
            │               │               │
            └───────────────┼───────────────┘
                            │
                    ┌───────▼───────┐
                    │ Mean Logits   │
                    │ (Ensemble)    │
                    └───────┬───────┘
                            │
                    ┌───────▼───────┐
                    │ Temperature   │
                    │ Scaling (T=2.4)│
                    └───────┬───────┘
                            │
                    ┌───────▼───────┐
                    │ Student       │
                    │ MobileViT-XXS │
                    │ / MobileViT-S │
                    └───────┬───────┘
                            │
                    ┌───────▼───────┐
                    │   Loss        │
                    │  L = α·CE     │
                    │  + (1-α)·KL   │
                    └───────────────┘
```

## Distillation Loss

```
L_total = α * CE(student_logits, labels) + (1-α) * T² * KL(soft_student || soft_teacher)
```

Where:

- **CE:** Cross-entropy with hard labels (ground truth)
- **KL:** KL divergence between softened student and teacher distributions
- **Softening:** `soft(x) = exp(x/T) / Σ exp(x/T)`
- **T:** Temperature (default 2.4)
- **α:** Balance weight (default 0.75)
- **T²:** Factor to preserve gradient magnitude

## Best HPO Hyperparameters

| Parameter | Value |
|---|---|
| Learning rate | 4.62e-4 |
| Weight decay | 6.99e-6 |
| Dropout | 0.406 |
| Temperature | 2.402 |
| Alpha | 0.754 |
| Label smoothing | 0.106 |
| Batch size | 128 |
| Scheduler | step |

## Student Model Head

```
MobileViT Backbone → Dropout(0.2) → Linear(192→128) → ReLU → Dropout(0.1) → Linear(128→6)
```

## Data Augmentation

- RandomHorizontalFlip
- RandomRotation(±15°)
- ColorJitter (brightness, contrast, saturation, hue)
- RandomErasing (p=0.25)
- Normalization: ImageNet mean/std
