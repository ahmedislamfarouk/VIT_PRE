#!/usr/bin/env python3
"""
enhanced_distilled.py — Adaptive Multi-Teacher Distillation (AMTD)
"""
import argparse
import json

import torch
import torch.optim as optim
import optuna

from mstd.config import (
    CHECKPOINT_DIR, RESULTS_DIR, DEVICE,
)
from mstd.data.dataset import get_dataset_path
from mstd.data.precompute import precompute_teacher_logits
from mstd.data.loaders import create_distill_loaders
from mstd.models.student import StudentModel
from mstd.training.distillation import train_epoch_distill, evaluate_distill
from mstd.training.tpe import build_objective
from mstd.visualization.plots import plot_amtd_curves


def main():
    parser = argparse.ArgumentParser(description="AMTD — Adaptive Multi-Teacher Distillation")
    parser.add_argument("--tpe-trials", type=int, default=25, help="Number of TPE trials")
    parser.add_argument("--tpe-epochs", type=int, default=8, help="Epochs per TPE trial")
    parser.add_argument("--final-epochs", type=int, default=35, help="Epochs for final model")
    parser.add_argument("--skip-tpe", action="store_true", help="Skip TPE and use default hparams")
    parser.add_argument("--force-cache", action="store_true", help="Re-generate teacher logits")
    args = parser.parse_args()

    print("=" * 70)
    print("Adaptive Multi-Teacher Distillation (AMTD)")
    print("=" * 70)
    print(f"Device: {DEVICE}")

    dataset_path = get_dataset_path()
    print(f"Dataset: {dataset_path}")

    t_deit, t_dinov2, t_siglip2, t_labels = precompute_teacher_logits(
        dataset_path, force=args.force_cache
    )
    teacher_ensemble = (t_deit + t_dinov2 + t_siglip2) / 3.0
    print(f"Teacher logits shape: {teacher_ensemble.shape}")

    if not args.skip_tpe:
        print("\n" + "=" * 70)
        print(f"Starting Optuna TPE — {args.tpe_trials} trials x {args.tpe_epochs} epochs")
        print("=" * 70)
        objective = build_objective(dataset_path, teacher_ensemble, args.tpe_epochs)
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(n_startup_trials=5),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=2),
        )
        study.optimize(objective, n_trials=args.tpe_trials, show_progress_bar=True)

        best = study.best_trial
        print(f"\nBest trial #{best.number}: val_acc={best.value:.4f}")
        print("Best hyperparameters:")
        for k, v in best.params.items():
            print(f"  {k}: {v}")

        with open(RESULTS_DIR / "tpe_study.json", "w") as f:
            json.dump(
                {
                    "best_trial": best.number,
                    "best_value": best.value,
                    "best_params": best.params,
                },
                f,
                indent=2,
            )
        hparams = best.params
    else:
        print("\n[Skip TPE] Using default hyperparameters")
        hparams = {
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "dropout": 0.3,
            "temperature": 4.0,
            "alpha": 0.3,
            "label_smoothing": 0.1,
            "batch_size": 64,
            "use_augment": True,
            "scheduler": "cosine",
        }

    print("\n" + "=" * 70)
    print("Final Training — Best Hyperparameters")
    print("=" * 70)
    train_loader, val_loader = create_distill_loaders(
        dataset_path,
        teacher_ensemble,
        batch_size=hparams["batch_size"],
        use_augment=hparams["use_augment"],
    )

    model = StudentModel(dropout=hparams["dropout"]).to(DEVICE)
    print(f"Student params: {model.count_params() / 1e6:.2f}M")

    optimizer = optim.AdamW(
        model.parameters(), lr=hparams["lr"], weight_decay=hparams["weight_decay"]
    )
    if hparams["scheduler"] == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.final_epochs)
    elif hparams["scheduler"] == "step":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.final_epochs // 3, gamma=0.5)
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    best_state = None

    for epoch in range(1, args.final_epochs + 1):
        train_loss = train_epoch_distill(
            model, train_loader, optimizer, scheduler,
            hparams["temperature"], hparams["alpha"], hparams["label_smoothing"], DEVICE,
        )
        val_acc, val_metrics = evaluate_distill(model, val_loader, DEVICE)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(0.0)
        history["val_acc"].append(val_acc)

        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_acc)

        print(
            f"Epoch {epoch:02d}/{args.final_epochs} — "
            f"Train Loss: {train_loss:.4f}, Val Acc: {val_acc*100:.2f}%"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict()
            print(f"  *** New best: {best_val_acc*100:.2f}% ***")

    if best_state:
        model.load_state_dict(best_state)

    final_acc, final_metrics = evaluate_distill(model, val_loader, DEVICE)
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"Best Val Accuracy: {best_val_acc*100:.2f}%")
    print(f"Final Val Accuracy: {final_acc*100:.2f}%")
    print(f"F1 (macro): {final_metrics['f1_macro']*100:.2f}%")
    print(f"Precision (macro): {final_metrics['precision_macro']*100:.2f}%")
    print(f"Recall (macro): {final_metrics['recall_macro']*100:.2f}%")

    baselines = {
        "DeiT-Tiny": 91.27,
        "DINOv2-S/14": 94.93,
        "SigLIP2-B/16": 88.17,
        "AMTD (Ours)": best_val_acc * 100,
    }
    print("\nComparison:")
    for name, acc in baselines.items():
        marker = "+" if acc == max(baselines.values()) else "  "
        print(f"  {marker} {name:<18} {acc:.2f}%")

    ckpt_path = CHECKPOINT_DIR / "enhanced_amtd_best.pth"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "best_val_acc": best_val_acc,
            "hyperparameters": hparams,
            "history": history,
            "final_metrics": final_metrics,
        },
        ckpt_path,
    )
    print(f"\n[Checkpoint] Saved -> {ckpt_path}")

    results = {
        "model": "AMTD (Adaptive Multi-Teacher Distillation)",
        "student_backbone": "mobilevit_s",
        "student_params_M": round(model.count_params() / 1e6, 2),
        "best_val_acc": round(best_val_acc, 4),
        "final_metrics": {k: round(v, 4) if isinstance(v, float) else v for k, v in final_metrics.items()},
        "hyperparameters": hparams,
        "baselines": {"deit": 0.9127, "dinov2": 0.9493, "siglip2": 0.8817},
    }
    with open(RESULTS_DIR / "enhanced_amtd_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"[Results] Saved -> {RESULTS_DIR / 'enhanced_amtd_results.json'}")

    plot_amtd_curves(history, RESULTS_DIR / "enhanced_amtd_curves.png")
    return results


if __name__ == "__main__":
    main()
