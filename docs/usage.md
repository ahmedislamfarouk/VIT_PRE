# Usage

## Installation

```bash
pip install -r requirements.txt
```

Dependencies: `timm`, `torch`, `torchvision`, `kagglehub`, `scikit-learn`, `matplotlib`, `tqdm`, `Pillow`, `numpy`, `optuna`.

## Script Reference

### Core Training

| Script | Command |
|---|---|
| Frozen backbone comparison | `python compare.py` |
| AMTD enhanced (HPO + train) | `python enhanced_distilled.py` |
| Resume AMTD training | `python complete_enhanced_training.py` |
| Fused multi-backbone ensemble | `python enhanced_ensemble.py` |

### From-Scratch Training

| Script | Command |
|---|---|
| Main from-scratch trainer | `python run_scratch_fixed.py --model amtd --subset 0.05` |
| Single-model launcher | `python run_scratch_single.py --model deit --subset 0.1` |
| Minimal trainer | `python run_simple.py --model cnn --subset 0.05` |
| Quick 3-model test | `python run_quick.py` |
| Quick 5% CNN test | `python test5.py` |

**Arguments:**
- `--model` : `cnn`, `deit`, `dinov2`, `siglip`, `mobilevit_s`, `mobilevit_hard`, `amtd`
- `--subset` : data fraction (default varies by script)
- `--gpu` : GPU ID (default 0)
- `--epochs` : override default epochs

### Data Hunger Evaluation

| Script | Command |
|---|---|
| Full evaluation (all models × all subsets) | `python evaluate_data_hunger.py` |
| Per-subset evaluation | `python run_subset.py --subset 0.05` |

### Visualization

| Script | Command |
|---|---|
| Radar chart (animated) | `python radar.py` |
| Checkpoint comparison | `python compare_all_checkpoints.py` |
| All GIF generation | `python generate_final_gifs.py` |

### Parallel Launch Scripts

| Script | Description |
|---|---|
| `launch_full.sh` | 28 experiments (7 models × 4 subsets) on 8 GPUs |
| `launch_all_parallel.sh` | 14 jobs (7 models × 5%, 10%) on 8 GPUs |
| `launch_scratch.sh` | 4 from-scratch experiments on 4 GPUs |
| `launch_parallel.sh` | 5 subset evaluations on 5 GPUs |
| `run_4_parallel.sh` | 4 MobileViT-S experiments on 4 GPUs |
| `run_one.sh GPU_ID subset model log_suffix` | Single experiment wrapper |
| `run_parallel_data_hunger.sh` | 5 subset evaluations with inline Python |
| `run_siglip_scratch.sh` | 4 SigLIP scratch jobs on 4 GPUs |
| `launch_siglip_remainder.py` | Remaining SigLIP (50%, 100%) experiments |

## Training Configuration

Default training settings used across experiments:

| Parameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 1e-3 |
| Weight decay | 1e-4 |
| Batch size | 64 |
| Scheduler | CosineAnnealingLR |
| Label smoothing | 0.1 |
| Augmentation | Flip, Rotate(15), ColorJitter, RandomErasing(0.25) |
| Image size | 224×224 |
