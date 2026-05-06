import io

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

from ..config import MODEL_COLORS, MODEL_MARKERS, MODEL_LABELS, RESULTS_DIR


def _style_ax(ax, title: str, xlabel: str, ylabel: str):
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.tick_params(labelsize=10)
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)


def plot_training_gif(all_histories: dict, epochs: int = 10):
    try:
        from PIL import Image as PILImage
    except ImportError:
        print("Pillow not installed — skipping GIF. Run: pip install Pillow")
        return

    frames = []
    epoch_range = list(range(1, epochs + 1))
    y_min = min(
        v
        for hist in all_histories.values()
        for v in hist["val_acc"]
    ) - 0.05

    for ep_end in epoch_range:
        fig, ax = plt.subplots(figsize=(9, 5))

        for name, hist in all_histories.items():
            c = MODEL_COLORS.get(name, "#333")
            m = MODEL_MARKERS.get(name, "o")
            lbl = MODEL_LABELS.get(name, name)
            xs = epoch_range[:ep_end]
            ys = hist["val_acc"][:ep_end]
            ax.plot(xs, ys, color=c, marker=m, markersize=6,
                    linewidth=2.2, label=lbl)
            ax.annotate(f"{ys[-1]*100:.1f}%",
                        xy=(xs[-1], ys[-1]),
                        xytext=(4, 4), textcoords="offset points",
                        fontsize=9, color=c, fontweight="bold")

        ax.set_xlim(0.5, epochs + 0.5)
        ax.set_ylim(max(0, y_min), 1.02)
        ax.set_xticks(epoch_range)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v*100:.0f}%"))
        _style_ax(ax, f"Validation Accuracy — Epoch {ep_end}/{epochs}",
                  "Epoch", "Accuracy")
        ax.legend(fontsize=10, framealpha=0.7, loc="lower right")

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        plt.close()
        buf.seek(0)
        frames.append(PILImage.open(buf).copy())
        buf.close()

    frames += [frames[-1]] * 8

    out = RESULTS_DIR / "06_training_accuracy.gif"
    frames[0].save(
        out, save_all=True, append_images=frames[1:],
        duration=300, loop=0,
    )
    print(f"Saved → {out}")


def create_radar_animation(output_path, deit_data, dinov2_data, siglip2_data, labels=None):
    if labels is None:
        labels = ['Accuracy', 'Macro Precision', 'Macro Recall', 'Macro F1']
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    base_val = 0.85
    deit_target = np.array(deit_data)
    dinov2_target = np.array(dinov2_data)
    siglip2_target = np.array(siglip2_data)
    base_array = np.full_like(deit_target, base_val)

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    fig.subplots_adjust(right=0.75, top=0.85)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    total_frames = 150
    duration = 30
    start_deit = 0
    start_dinov2 = 40
    start_siglip2 = 80

    def ease_out_cubic(t):
        return 1 - pow(1 - t, 3)

    def get_progress(frame, start_frame):
        if frame < start_frame:
            return 0.0
        elif frame >= start_frame + duration:
            return 1.0
        else:
            return ease_out_cubic((frame - start_frame) / duration)

    def update(frame):
        ax.clear()
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=12, fontweight='bold')
        ax.set_ylim(0.85, 0.96)
        ax.set_yticks([0.86, 0.88, 0.90, 0.92, 0.94, 0.96])
        ax.set_yticklabels(['0.86', '0.88', '0.90', '0.92', '0.94', '0.96'], color="grey", size=9)
        ax.set_title("Model Performance Comparison", size=16, weight='bold', pad=20)

        prog_deit = get_progress(frame, start_deit)
        prog_dinov2 = get_progress(frame, start_dinov2)
        prog_siglip2 = get_progress(frame, start_siglip2)

        if prog_deit > 0:
            deit_current = base_array + (deit_target - base_array) * prog_deit
            ax.plot(angles, deit_current, color=colors[0], linewidth=2, label='deit')
            ax.fill(angles, deit_current, color=colors[0], alpha=0.25 * prog_deit)

        if prog_dinov2 > 0:
            dinov2_current = base_array + (dinov2_target - base_array) * prog_dinov2
            ax.plot(angles, dinov2_current, color=colors[1], linewidth=2, label='dinov2')
            ax.fill(angles, dinov2_current, color=colors[1], alpha=0.25 * prog_dinov2)

        if prog_siglip2 > 0:
            siglip2_current = base_array + (siglip2_target - base_array) * prog_siglip2
            ax.plot(angles, siglip2_current, color=colors[2], linewidth=2, label='siglip2')
            ax.fill(angles, siglip2_current, color=colors[2], alpha=0.25 * prog_siglip2)

        if prog_deit > 0:
            ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1.05))

    ani = animation.FuncAnimation(fig, update, frames=total_frames, interval=33, repeat=True)
    ani.save(output_path, writer='pillow', fps=30)
    print(f"Smooth sequential animation saved as {output_path}")
