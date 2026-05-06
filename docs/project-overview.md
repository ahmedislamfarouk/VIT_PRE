# Project Overview

## Purpose

AMTD (Adaptive Multi-Teacher Distillation) addresses the **data hunger problem** in Vision Transformers (ViTs). While ViTs achieve state-of-the-art accuracy, they require massive amounts of training data (millions of images) to generalize well from scratch. This project demonstrates that a lightweight student model can be trained effectively using knowledge distilled from an ensemble of diverse, pre-trained teacher models — even with only 5% of the original training data.

## Problem Statement

Vision Transformers lack the inductive biases (translation equivariance, locality) inherent to CNNs, making them data-hungry. Training a ViT from scratch on small datasets leads to severe overfitting and poor generalization.

## Solution

AMTD transfers dark knowledge from an ensemble of three diverse frozen teacher models into a lightweight MobileViT student via:

- **Multi-teacher ensemble:** Mean logit fusion from DeiT (supervised), DINOv2 (self-supervised), and SigLIP2 (vision-language) teachers
- **Dual loss:** Weighted combination of hard-label cross-entropy and soft-label KL divergence
- **Temperature scaling:** Softens teacher distributions (T=2.4) to expose inter-class relationships
- **HPO:** Optuna TPE search for optimal hyperparameters (25 trials)

## Relationship to CADU-OVOD

This project is a **simplified testbed** for the parent CADU-OVOD project (Calibrated Adaptive Distillation with Uncertainty for Open-Vocabulary Object Detection). It focuses on image classification for rapid iteration on distillation strategies before transferring them to the more complex object detection setting.

## Key Findings

1. **DINOv2** is the best frozen backbone (94.93%) — self-supervised features generalize best
2. **AMTD distills knowledge beyond any single teacher** (95.03%), combining complementary strengths
3. **From-scratch AMTD dominates all subset ratios** — 76.87% at 5% data vs 65.90% for CNN
4. **SigLIP2 underperforms** in frozen evaluation (88.17%) but contributes useful multi-modal signal to the ensemble
5. **DeiT-Tiny is the most efficient teacher** — 5.7M params, strong local feature discrimination
