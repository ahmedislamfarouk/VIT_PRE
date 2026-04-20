# VIT_PRE Visualizations

This folder contains generated visual artifacts derived from model-comparison outputs.

## Files
- `vit_category_confidence.png`: Average prompt score per category for each ViT model.
- `vit_top1_non_adversarial_rate.png`: Percentage of images where the top-ranked prompt category is non-adversarial.
- `visualization_summary.json`: Numeric aggregates used to create the charts.

## Regeneration
Run from repository root:

```bash
python generate_visualizations.py --input results_real_clip_test.json --out-dir visualizations
```
