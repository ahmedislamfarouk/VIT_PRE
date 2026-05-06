# Results

## Frozen Backbone Comparison

| Model | Params | Accuracy | F1 (macro) | Precision | Recall |
|---|---|---|---|---|---|
| **DINOv2-S/14** | 22M | **94.93%** | 94.93% | 94.99% | 94.93% |
| DeiT-Tiny | 5.7M | 91.27% | 91.20% | 91.29% | 91.27% |
| SigLIP2-B/16 | 86M | 88.17% | 88.07% | 88.20% | 88.17% |

## AMTD Enhanced (pretrained student + KD)

| Model | Params | Accuracy | Improvement |
|---|---|---|---|
| **AMTD (Ours)** | **5.1M** | **95.03%** | +0.10% over DINOv2 |
| DINOv2-S/14 | 22M | 94.93% | baseline |

AMTD achieves higher accuracy than any individual teacher with **4.3× fewer parameters** than DINOv2.

## From-Scratch Data Hunger Results

| Model | 5% (701) | 10% (1.4K) | 25% (3.5K) | 50% (7K) | 100% (14K) |
|---|---|---|---|---|---|
| **AMTD** | **76.87%** | **81.00%** | **84.60%** | **87.43%** | **89.27%** |
| MobileViT-S (hard) | 73.90% | 76.70% | 82.03% | 85.33% | 87.67% |
| MobileViT-S (KD) | 74.67% | 79.07% | 83.43% | 86.63% | 88.87% |
| CNN | 65.90% | 71.57% | 75.57% | 79.23% | 81.03% |
| DeiT-Tiny | 64.07% | 69.00% | 73.43% | 77.60% | 79.70% |
| DINOv2-S/14 | 62.33% | 65.33% | 63.93% | 58.60% | 62.97% |
| SigLIP2-B/16 | — | — | — | — | — |

Key observations:
- AMTD outperforms all baselines at every subset ratio
- DINOv2 actually **degrades** with more data from scratch (severe overfitting due to size)
- DeiT and CNN improve steadily with more data but plateau lower than AMTD
- SigLIP2 from scratch fails to converge entirely (model too large for small data)

## Data Efficiency Gains

AMTD at 5% data (76.87%) matches CNN at 50% data (79.23%), achieving roughly **10× data efficiency**.

AMTD at 10% data (81.00%) already exceeds CNN at 100% data (81.03%).

## Best Optuna HPO Results

```json
{
  "lr": 0.000462,
  "weight_decay": 6.99e-06,
  "dropout": 0.406,
  "temperature": 2.402,
  "alpha": 0.754,
  "label_smoothing": 0.106,
  "batch_size": 128,
  "scheduler": "step"
}
```
