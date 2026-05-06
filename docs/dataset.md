# Dataset: Intel Image Classification

Source: Kaggle — [puneet6060/intel-image-classification](https://kaggle.com/datasets/puneet6060/intel-image-classification)

## Classes

| Class | Train | Test |
|---|---|---|
| Buildings | 2,192 | 437 |
| Forest | 2,281 | 474 |
| Glacier | 2,404 | 553 |
| Mountain | 2,512 | 525 |
| Sea | 2,274 | 510 |
| Street | 2,371 | 501 |
| **Total** | **14,034** | **3,000** |

## Data Hunger Subsets

Experiments use random stratified subsets of the training data:

| Subset | Samples | Epochs | LR |
|---|---|---|---|
| 5% | 701 | 50 | 1e-3 |
| 10% | 1,403 | 40 | 1e-3 |
| 25% | 3,508 | 30 | 1e-3 |
| 50% | 7,017 | 25 | 1e-3 |
| 100% | 14,034 | 15 | 1e-3 |

## Preprocessing

- **Resize:** 256×256
- **Center crop:** 224×224
- **Augmentation:** Horizontal flip, rotation (±15°), color jitter, random erasing (p=0.25)
- **Normalization:** ImageNet mean (0.485, 0.456, 0.406) and std (0.229, 0.224, 0.225)

## Data Loading

The dataset is downloaded automatically via `kagglehub` on first run. Cached at `~/.cache/kagglehub/datasets/puneet6060/intel-image-classification`.
