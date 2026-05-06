# AMTD — Adaptive Multi-Teacher Distillation

A research framework for data-efficient Vision Transformer (ViT) training using multi-teacher knowledge distillation. Trains lightweight student models from scratch with as little as 5% of training data.

## Documentation

| Document | Description |
|---|---|
| [Project Overview](project-overview.md) | Purpose, problem statement, relationship to CADU-OVOD |
| [Architecture](architecture.md) | Core distillation pipeline, loss functions, teacher ensemble |
| [Models](models.md) | Teacher models (DeiT-Tiny, DINOv2-S/14, SigLIP2-B/16) and student (MobileViT) |
| [Dataset](dataset.md) | Intel Image Classification dataset — classes, splits, statistics |
| [Experiments](experiments.md) | All experiments: frozen backbone, AMTD, data hunger, HPO |
| [Usage](usage.md) | Installation, running scripts, command reference |
| [Results](results.md) | Key results, tables, comparisons, data efficiency curves |
| [Visualizations](visualizations.md) | All visualization scripts and generated outputs (GIFs, charts, radar plots) |
| [Docker](docker.md) | Container setup, building, and running |
| [Reproducibility](reproducibility.md) | How to reproduce all results step by step |
| [Checkpoints](checkpoints.md) | Checkpoint naming conventions, format, loading API |

## Quick Links

- **Best accuracy:** 95.03% (AMTD with MobileViT-S, 5.1M params)
- **From scratch at 5% data:** 76.87% (AMTD) vs 65.90% (CNN baseline)
- **Paper:** [amtd_paper.pdf](../amtd_paper.pdf)
- **Docker:** [Dockerfile](../Dockerfile)
