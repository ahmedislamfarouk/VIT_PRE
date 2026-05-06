# Scripts Reference

## Python Scripts

### Core Training & Comparison

| Script | Description |
|--------|-------------|
| `run_scratch_fixed.py` | **Main from-scratch training script.** Trains any model (CNN, DeiT, DINOv2, SigLIP, MobileViT-S, MobileViT-Hard, AMTD) on a data subset with random initialization (no pretrained weights). Usage: `python run_scratch_fixed.py <subset_ratio> <model_name> <gpu_id>` |
| `compare.py` | **Backbone comparison (frozen teachers).** Trains classification heads on frozen pretrained backbones (DeiT, DINOv2, SigLIP2, YOLOv8). Establishes teacher baselines. Generates confusion matrices, metric plots, and radar charts. |
| `enhanced_distilled.py` | **AMTD with TPE hyperparameter optimization.** Full pipeline: precomputes teacher logits, runs Optuna TPE to find best distillation hyperparams (lr, temperature, alpha, dropout, etc.), then trains final student model. |
| `enhanced_ensemble.py` | **Multi-backbone feature fusion (experimental).** Fuses features from all 3 backbones (DeiT, DINOv2, SigLIP2) via learned attention. Trains a fused classifier head. |
| `complete_enhanced_training.py` | **Resume/finalize AMTD training.** Uses cached teacher logits + best TPE hyperparameters to complete training. Saves final checkpoint and results. |
| `evaluate_data_hunger.py` | **Full data-hunger evaluation.** Trains baselines (frozen backbone + head) and AMTD student across all data fractions (5%-100%). Generates data-efficiency curves and comparison plots. |

### Visualization

| Script | Description |
|--------|-------------|
| `radar.py` | **Radar chart animation generator.** Creates an animated radar chart comparing DeiT, DINOv2, and SigLIP2 metrics (accuracy, precision, recall, F1). |

### GIF Animation Scripts (`gifs/`)

| Script | Description |
|--------|-------------|
| `generate_final_gifs.py` | Professional animated GIFs with sequential per-model animation, cubic-ease-out, and high DPI. |
| `generate_gifs.py` | High-quality smooth animated GIFs for data hunger analysis. |
| `generate_all_animations.py` | Generates all animation types at once. |
| `generate_comparison_gifs.py` | Model comparison animated GIFs. |
| `generate_diverging_gifs.py` | Diverging bar chart animations. |
| `generate_final_comparison.py` | Final comparison animations. |
| `generate_objective_comparison.py` | Objective metrics comparison GIFs. |
| `generate_tweaked_gifs.py` | Tweaked/adjusted versions of animations. |
| `animate_radars.py` | Radar chart animation. |
| `animate_radars_sequential.py` | Sequential radar chart animation (one model at a time). |
| `create_radar_charts.py` | Static radar chart generation. |
| `create_radar_charts_v2.py` | Updated radar chart generation. |
| `create_tables_xlsx.py` | Creates Excel tables from results data. |

---

## Shell Scripts

| Script | Description |
|--------|-------------|
| `launch_full.sh` | **Master launcher.** Runs all 28 experiments (7 models x 4 subsets: 5%, 10%, 50%, 100%) across 8 GPUs in parallel. Requires GPU cluster. |

---

## Other Files

| File | Description |
|------|-------------|
| `requirements.txt` | Python dependencies (timm, torch, torchvision, kagglehub, sklearn, matplotlib, tqdm, Pillow, numpy). |
| `Dockerfile` | Container definition for reproducible environment. |
| `README.md` | Project overview, architecture, results, and usage guide. |
| `docs/AssignmentREQ.txt` | Original assignment requirements document. |
