import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patheffects

from ..config import MODEL_COLORS, MODEL_MARKERS, MODEL_LABELS, MODEL_CMAPS, CLASS_NAMES, NUM_CLASSES, RESULTS_DIR


def _style_ax(ax, title: str, xlabel: str, ylabel: str):
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.tick_params(labelsize=10)
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)


def plot_training_curves(all_histories: dict, epochs: int = 10):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    epoch_range = range(1, epochs + 1)

    for name, hist in all_histories.items():
        c = MODEL_COLORS.get(name, "#333333")
        m = MODEL_MARKERS.get(name, "o")
        lbl = MODEL_LABELS.get(name, name)
        axes[0].plot(epoch_range, hist["train_loss"], color=c, marker=m,
                     markersize=5, linewidth=2, label=f"{lbl} – train")
        axes[0].plot(epoch_range, hist["val_loss"], color=c, marker=m,
                     markersize=5, linewidth=1.5, label=f"{lbl} – val",
                     linestyle="--", alpha=0.7)
        axes[1].plot(epoch_range, hist["val_acc"], color=c, marker=m,
                     markersize=5, linewidth=2, label=lbl)

    _style_ax(axes[0], "Training & Validation Loss", "Epoch", "Cross-Entropy Loss")
    _style_ax(axes[1], "Validation Accuracy over Epochs", "Epoch", "Accuracy")
    axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v*100:.1f}%"))

    for ax in axes:
        ax.legend(fontsize=9, framealpha=0.6)
        ax.set_xticks(list(epoch_range))

    plt.suptitle("Backbone Training Curves — Intel Image Classification",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    out = RESULTS_DIR / "01_training_curves.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out}")


def plot_confusion_matrices(all_results: list):
    n = len(all_results)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, result in zip(axes, all_results):
        name = result["model"]
        cm = np.array(result["confusion_matrix"])
        im = ax.imshow(cm, interpolation="nearest", cmap=MODEL_CMAPS.get(name, "Blues"))
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        acc = result["accuracy"]
        ax.set_title(f"{MODEL_LABELS.get(name, name)}\nAcc: {acc*100:.2f}%",
                     fontsize=11, fontweight="bold")
        ax.set_xlabel("Predicted label", fontsize=10)
        ax.set_ylabel("True label", fontsize=10)

        ticks = np.arange(NUM_CLASSES)
        ax.set_xticks(ticks)
        ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right", fontsize=9)
        ax.set_yticks(ticks)
        ax.set_yticklabels(CLASS_NAMES, fontsize=9)

        thresh = cm.max() / 2.0
        for i in range(NUM_CLASSES):
            for j in range(NUM_CLASSES):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=9,
                        color="white" if cm[i, j] > thresh else "black")

    plt.suptitle("Confusion Matrices — Intel Image Classification",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    out = RESULTS_DIR / "02_confusion_matrices.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out}")


def plot_metric_comparison(all_results: list):
    metrics = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
    metric_names = ["Accuracy", "Precision (macro)", "Recall (macro)", "F1 (macro)"]

    n_metrics = len(metrics)
    n_models = len(all_results)
    x = np.arange(n_metrics)
    width = 0.22
    offsets = np.linspace(-(n_models - 1) / 2, (n_models - 1) / 2, n_models) * width

    fig, ax = plt.subplots(figsize=(11, 5))

    for result, offset in zip(all_results, offsets):
        name = result["model"]
        values = [result[m] for m in metrics]
        bars = ax.bar(x + offset, values, width, label=MODEL_LABELS.get(name, name),
                      color=MODEL_COLORS.get(name, "#333"), edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                    f"{val*100:.1f}%", ha="center", va="bottom", fontsize=8.5)

    ax.set_ylim(0, 1.12)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v*100:.0f}%"))
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, fontsize=11)
    _style_ax(ax, "Overall Metric Comparison", "", "Score")
    ax.legend(fontsize=10, framealpha=0.6)

    plt.tight_layout()
    out = RESULTS_DIR / "03_metric_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out}")


def plot_per_class_f1(all_results: list):
    n_classes = NUM_CLASSES
    n_models = len(all_results)
    x = np.arange(n_classes)
    width = 0.22
    offsets = np.linspace(-(n_models - 1) / 2, (n_models - 1) / 2, n_models) * width

    fig, ax = plt.subplots(figsize=(13, 5))

    for result, offset in zip(all_results, offsets):
        name = result["model"]
        f1s = [result["per_class"][c]["f1"] for c in CLASS_NAMES]
        bars = ax.bar(x + offset, f1s, width, label=MODEL_LABELS.get(name, name),
                      color=MODEL_COLORS.get(name, "#333"), edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, f1s):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val*100:.0f}", ha="center", va="bottom", fontsize=8)

    ax.set_ylim(0, 1.12)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v*100:.0f}%"))
    ax.set_xticks(x)
    ax.set_xticklabels([c.capitalize() for c in CLASS_NAMES], fontsize=11)
    _style_ax(ax, "Per-Class F1 Score Comparison", "Class", "F1 Score")
    ax.legend(fontsize=10, framealpha=0.6)

    plt.tight_layout()
    out = RESULTS_DIR / "04_per_class_f1.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out}")


def plot_radar(all_results: list):
    categories = ["Accuracy", "Precision", "Recall", "F1-Score"]
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

    for result in all_results:
        name = result["model"]
        values = [
            result["accuracy"],
            result["precision_macro"],
            result["recall_macro"],
            result["f1_macro"],
        ]
        values += values[:1]
        ax.plot(angles, values,
                color=MODEL_COLORS.get(name, "#333"), linewidth=2,
                marker=MODEL_MARKERS.get(name, "o"), markersize=6,
                label=MODEL_LABELS.get(name, name))
        ax.fill(angles, values, color=MODEL_COLORS.get(name, "#333"), alpha=0.12)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), categories, fontsize=11)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v*100:.0f}%"))
    ax.set_title("Backbone Comparison Radar\n(all metrics from real evaluation)",
                 fontsize=13, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = RESULTS_DIR / "05_radar_chart.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out}")


def plot_amtd_curves(history: dict, out_path, baseline=94.93):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(history["train_loss"]) + 1)
    axes[0].plot(epochs, history["train_loss"], label="Train Loss", marker="o")
    axes[0].plot(epochs, history["val_loss"], label="Val Loss", marker="s")
    axes[0].set_title("Loss Curves")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, [a * 100 for a in history["val_acc"]], label="Val Acc", marker="o", color="green")
    axes[1].axhline(baseline, color="red", linestyle="--", label=f"DINOv2 baseline ({baseline}%)")
    axes[1].set_title("Validation Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(80, 100)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[Plot] Saved training curves → {out_path}")


def plot_data_hunger_curves(results: dict, subset_ratios, out_path):
    colors = {"deit": "#378ADD", "dinov2": "#1D9E75", "siglip2": "#D85A30", "amtd": "#8E44AD"}
    labels_map = {"deit": "DeiT-Tiny", "dinov2": "DINOv2-S/14", "siglip2": "SigLIP2-B/16", "amtd": "AMTD (Ours)"}
    markers = {"deit": "o", "dinov2": "s", "siglip2": "^", "amtd": "D"}

    fig, ax = plt.subplots(figsize=(10, 6))
    x = [r * 100 for r in subset_ratios]
    for name in results:
        y = [results[name][r]["accuracy"] * 100 for r in subset_ratios]
        ax.plot(x, y, marker=markers[name], color=colors[name], label=labels_map[name], linewidth=2, markersize=8)

    ax.set_xlabel("Training Data Used (%)", fontsize=12)
    ax.set_ylabel("Test Accuracy (%)", fontsize=12)
    ax.set_title("Data-Efficiency Comparison: Accuracy vs. Training Set Size", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, framealpha=0.7)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(70, 100)
    ax.set_xticks(x)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[Plot] Saved → {out_path}")


def plot_amtd_val_curve(history, out_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    epochs = range(1, len(history["val_acc"]) + 1)
    ax.plot(epochs, [a * 100 for a in history["val_acc"]], marker="o", label="Val Acc", color="green")
    ax.axhline(94.93, color="red", linestyle="--", label="DINOv2 baseline (94.93%)")
    ax.set_ylim(85, 100)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Enhanced AMTD — Validation Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
