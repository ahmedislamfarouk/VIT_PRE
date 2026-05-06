# Visualizations

## Generated Output Directories

### `results/` — Static Charts

| File | Description |
|---|---|
| `comparison_results.json` | Full frozen backbone metrics |
| `enhanced_amtd_results.json` | AMTD enhanced results |
| `data_hunger_results.csv` | Data hunger results across subsets |
| `data_hunger_curves.png` | Data efficiency line plot |
| `data_efficiency_5_vs_100.png` | 5% vs 100% comparison bar chart |
| `f1_vs_accuracy_scatter.png` | F1 vs accuracy scatter plot |
| `bar_accuracy_by_pct.png` | Accuracy bar chart by subset |
| `bar_f1_by_pct.png` | F1 bar chart by subset |
| `bar_precision_by_pct.png` | Precision bar chart by subset |
| `bar_recall_by_pct.png` | Recall bar chart by subset |
| `grouped_accuracy_by_model.png` | Grouped bar by model |
| `grouped_f1_by_model.png` | Grouped F1 by model |
| `grouped_precision_by_model.png` | Grouped precision by model |
| `radar_best_per_family.png` | Radar chart per model family |
| `confusion_matrix_*.png` | Per-model confusion matrices |
| `enhanced_amtd_curves.png` | AMTD training/validation curves |

### `gifs/` — Animations

| File | Description |
|---|---|
| `accuracy_progression.gif` | Accuracy over training epochs |
| `radar_f1_comparison.gif` | Animated radar chart |
| `training_accuracy.gif` | Training accuracy animation |
| `f1_accuracy_curves.gif` | F1 and accuracy curves |
| `efficiency_gap.gif` | Data efficiency gap visualization |

### `ahmedgifs/` — Per-Class Metrics

| File | Description |
|---|---|
| `accuracy_comparison.gif` | Accuracy comparison across models |
| `f1_macro_comparison.gif` | Macro F1 comparison |
| `precision_macro_comparison.gif` | Macro precision comparison |
| `recall_macro_comparison.gif` | Macro recall comparison |
| `radar_f1_comparison.gif` | Radar F1 comparison |
| `training_accuracy.gif` | Training accuracy over time |
| `overall_metrics_comparison.gif` | Combined metrics overview |
| `{class}_metrics.gif` | Per-class metrics (6 classes) |
| `confusion_matrices.png` | Combined confusion matrices |

### `Resultsold/` — Earlier Run Outputs

| File | Description |
|---|---|
| `01_training_curves.png` | Training and validation curves |
| `02_confusion_matrices.png` | Confusion matrices |
| `03_metric_comparison.png` | Metric comparison bar chart |
| `04_per_class_f1.png` | Per-class F1 scores |
| `05_radar_chart.png` | Static radar chart |
| `06_training_accuracy.gif` | Training accuracy GIF |
| `radar_chart_animation.gif` | Animated radar |
| `smooth_model_comparison.gif` | Smooth comparison animation |
| `smooth_sequential_comparison.gif` | Sequential animation |

## Visualization Scripts

### In `gifs/` Directory

| Script | Purpose |
|---|---|
| `animate_radars.py` | Animated radar charts |
| `animate_radars_sequential.py` | Sequential radar animations |
| `create_radar_charts.py` / `v2` | Static radar charts |
| `create_tables_xlsx.py` | Export results to Excel |
| `generate_all_animations.py` | Master animation runner |
| `generate_comparison_gifs.py` | Model comparison GIFs |
| `generate_diverging_gifs.py` | Diverging bar chart GIFs |
| `generate_final_comparison.py` | Final comparison plots |
| `generate_final_gifs.py` | Professional GIFs (ease-out, blit, high-DPI) |
| `generate_objective_comparison.py` | Objective comparison plots |
| `generate_tweaked_gifs.py` | Refined/tweaked GIFs |

## Key Visualizations

### Data Efficiency Curves
`results/data_hunger_curves.png` shows accuracy vs training subset size for all models, demonstrating AMTD's consistent advantage across all data regimes.

### Radar Charts
Compare model performance across per-class F1 scores, showing the complementary strengths of different architectures.

### Confusion Matrices
Per-model confusion matrices showing class-level misclassification patterns (buildings vs street, glacier vs mountain are common confusions).
