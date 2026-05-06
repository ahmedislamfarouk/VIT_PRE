# Reproducibility

## Step-by-Step Reproduction

### 1. Frozen Backbone Comparison

```bash
python compare.py
```

This runs all 4 frozen backbones (DeiT, DINOv2, SigLIP2, YOLOv8-World) sequentially on the full dataset. Results appear in `results/`.

### 2. AMTD Enhanced (Best Result: 95.03%)

```bash
python enhanced_distilled.py
```

Pipeline:
1. Downloads dataset via kagglehub
2. Caches teacher logits to `cache/teacher_logits.pkl` (~30 min on GPU)
3. Runs 25 Optuna TPE trials (8 epochs each)
4. Final training with best params (35 epochs)

### 3. From-Scratch Data Hunger

**Single experiment:**
```bash
python run_scratch_fixed.py --model amtd --subset 0.05 --gpu 0
```

**Full sweep (all models × all subsets):**
```bash
bash launch_full.sh
```

Or for specific subsets:
```bash
bash run_parallel_data_hunger.sh   # 5 subsets on 5 GPUs
```

**Full evaluation with plots:**
```bash
python evaluate_data_hunger.py
```

### 4. Checkpoint Evaluation

```bash
python compare_all_checkpoints.py
```

This evaluates all `.pth` files in `checkpoints/` and generates comparison tables, confusion matrices, and data hunger analysis plots.

## Data Requirements

- **Internet:** Required for first run (downloads dataset and model weights via kagglehub/timm)
- **Disk:** ~5GB for dataset, model weights, checkpoints, and results
- **RAM:** ~16GB recommended
- **GPU:** 8GB+ VRAM recommended (can run on CPU but slow)

## Expected Run Times (NVIDIA A100)

| Experiment | Time |
|---|---|
| Frozen backbone comparison | ~20 min |
| AMTD enhanced (HPO + train) | ~2 hours |
| From-scratch single run | ~10-30 min (varies by subset and model) |
| Full data hunger sweep (28 experiments) | ~6-8 hours |
| Checkpoint evaluation | ~15 min |

## Random Seeds

The scripts use implicit PyTorch seeding. For exact reproducibility, set explicit seeds:

```python
import torch
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

## Cache

Precomputed teacher logits are stored in `cache/teacher_logits.pkl`. Delete this file to force recomputation:
```bash
rm cache/teacher_logits.pkl
```
