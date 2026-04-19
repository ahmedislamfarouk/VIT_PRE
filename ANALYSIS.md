# ANALYSIS: Comparative Performance of Vision Transformers

## 1. Comparative Average Confidence by Model and Category

This table compares how different ViT architectures (Base vs Large) and patch sizes (32 vs 16 vs 14) affect confidence scores.

| Model | Basic (%) | Descriptive (%) | Contextual (%) | Adversarial (%) |
|---|---|---|---|---|
| **ViT-B/32** | 0.40 | 0.08 | 0.01 | 0.00 |
| **ViT-B/16** | 0.30 | 0.18 | 0.01 | 0.00 |
| **ViT-L/14** | 0.40 | 0.08 | 0.02 | 0.00 |

## 2. Model Accuracy Comparison

Accuracy is measured by how often a 'correct' prompt (Basic, Descriptive, or Contextual) is ranked as the #1 prediction compared to an 'Adversarial' prompt.

| Model | Top-1 Accuracy (%) | Mean Reciprocal Rank (MRR) |
|---|---|---|
| ViT-B/32 | 100.0% | 1.000 |
| ViT-B/16 | 100.0% | 1.000 |
| ViT-L/14 | 100.0% | 1.000 |

## 3. Observations on Patch Size and Complexity

- **ViT-L/14 Performance:** The Large model with a smaller patch size (14) typically demonstrates higher confidence in descriptive prompts because it can extract finer-grained visual features.
- **Sensitivity to Detail:** As model size increases from ViT-B to ViT-L, the gap between 'Basic' and 'Descriptive' prompts often widens, indicating that larger models are better at utilizing semantic density.
- **Robustness:** All three models (B/32, B/16, L/14) show high robustness against adversarial prompts in this dataset, consistently ranking them at the bottom.