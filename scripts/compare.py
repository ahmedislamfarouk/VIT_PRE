#!/usr/bin/env python3
"""
compare_backbones.py — Backbone Comparison for Intel Image Classification
"""
import argparse
import json
import os
import sys

from pathlib import Path

from mstd.config import BACKBONE_REGISTRY, RESULTS_DIR, DEVICE, EPOCHS, BATCH_SIZE
from mstd.data.loaders import create_dataloaders
from mstd.data.dataset import download_dataset
from mstd.training.baseline import ModelTrainer
from mstd.evaluation.metrics import print_comparison_table
from mstd.visualization.plots import (
    plot_training_curves,
    plot_confusion_matrices,
    plot_metric_comparison,
    plot_per_class_f1,
    plot_radar,
)
from mstd.visualization.gif import plot_training_gif


def main():
    parser = argparse.ArgumentParser(
        description="Backbone Comparison: DeiT vs DINOv2 vs SigLIP2 vs YOLOv8-World"
    )
    parser.add_argument(
        "--model", type=str,
        choices=list(BACKBONE_REGISTRY.keys()),
        help="Run only one backbone (default: all)",
    )
    parser.add_argument(
        "--download-only", action="store_true",
        help="Only download the dataset then exit",
    )
    parser.add_argument(
        "--dataset-path", type=str, default=None,
        help="Path to an already-downloaded dataset",
    )
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args, _ = parser.parse_known_args()

    if args.download_only:
        download_dataset()
        return

    dataset_path = args.dataset_path or download_dataset()
    train_loader, val_loader = create_dataloaders(dataset_path, args.batch_size)

    models_to_run = [args.model] if args.model else list(BACKBONE_REGISTRY.keys())

    all_histories = {}
    all_results = []

    for model_name in models_to_run:
        trainer = ModelTrainer(model_name, train_loader, val_loader, epochs=args.epochs)
        history = trainer.train()
        all_histories[model_name] = history
        result = trainer.evaluate()
        all_results.append(result)

    if len(all_results) > 1:
        print_comparison_table(all_results)

    comparison = {
        "models": all_results,
        "hyperparameters": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": 1e-4,
            "image_size": 224,
            "optimizer": "AdamW",
            "scheduler": "CosineAnnealingLR",
        },
    }

    results_path = RESULTS_DIR / "comparison_results.json"
    with open(results_path, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"Comparison results saved -> {results_path}")

    print("\nGenerating visualizations from real results...")
    plot_training_curves(all_histories, epochs=args.epochs)
    plot_confusion_matrices(all_results)
    if len(all_results) > 1:
        plot_metric_comparison(all_results)
        plot_per_class_f1(all_results)
        plot_radar(all_results)
    plot_training_gif(all_histories, epochs=args.epochs)
    print(f"\nAll plots saved to: {RESULTS_DIR.resolve()}")


if __name__ == "__main__":
    main()
