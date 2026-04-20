import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_results(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def aggregate_category_scores(results):
    model_cat_sum = {}
    model_cat_count = {}
    model_top_non_adv = {}
    model_top_total = {}

    for model_name, model_data in results.items():
        model_cat_sum[model_name] = {}
        model_cat_count[model_name] = {}
        model_top_non_adv[model_name] = 0
        model_top_total[model_name] = 0

        for _, ranked_prompts in model_data.items():
            if not ranked_prompts:
                continue
            top_cat = ranked_prompts[0].get("category", "unknown")
            model_top_total[model_name] += 1
            if top_cat != "adversarial":
                model_top_non_adv[model_name] += 1

            for item in ranked_prompts:
                category = item.get("category", "unknown")
                score = float(item.get("score", 0.0))
                model_cat_sum[model_name][category] = model_cat_sum[model_name].get(category, 0.0) + score
                model_cat_count[model_name][category] = model_cat_count[model_name].get(category, 0) + 1

    categories = sorted(
        {cat for model_stats in model_cat_sum.values() for cat in model_stats.keys()}
    )

    model_cat_avg = {}
    for model_name in model_cat_sum:
        model_cat_avg[model_name] = {}
        for cat in categories:
            total = model_cat_sum[model_name].get(cat, 0.0)
            count = model_cat_count[model_name].get(cat, 0)
            model_cat_avg[model_name][cat] = total / count if count else 0.0

    model_top_acc = {}
    for model_name in model_top_total:
        total = model_top_total[model_name]
        ok = model_top_non_adv[model_name]
        model_top_acc[model_name] = (ok / total) * 100.0 if total else 0.0

    return categories, model_cat_avg, model_top_acc


def plot_category_confidence(categories, model_cat_avg, out_path: Path):
    models = list(model_cat_avg.keys())
    x = range(len(categories))
    width = 0.22

    fig, ax = plt.subplots(figsize=(10, 5))
    for idx, model_name in enumerate(models):
        vals = [model_cat_avg[model_name][cat] for cat in categories]
        offsets = [i + (idx - (len(models) - 1) / 2) * width for i in x]
        ax.bar(offsets, vals, width=width, label=model_name)

    ax.set_title("Average Prompt Score by Category and Model")
    ax.set_ylabel("Average score")
    ax.set_xticks(list(x))
    ax.set_xticklabels(categories, rotation=20)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_top1_accuracy(model_top_acc, out_path: Path):
    models = list(model_top_acc.keys())
    vals = [model_top_acc[m] for m in models]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(models, vals)
    ax.set_title("Top-1 Non-Adversarial Selection Rate")
    ax.set_ylabel("Percentage (%)")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.0, f"{v:.1f}%", ha="center", va="bottom")

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Generate visualizations from ViT comparison results JSON.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).resolve().parent / "results_real_clip_test.json",
        help="Path to results JSON",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "visualizations",
        help="Directory where charts will be saved",
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    results = load_results(args.input)
    categories, model_cat_avg, model_top_acc = aggregate_category_scores(results)

    plot_category_confidence(categories, model_cat_avg, args.out_dir / "vit_category_confidence.png")
    plot_top1_accuracy(model_top_acc, args.out_dir / "vit_top1_non_adversarial_rate.png")

    summary_path = args.out_dir / "visualization_summary.json"
    summary = {
        "input": str(args.input),
        "categories": categories,
        "model_category_average": model_cat_avg,
        "top1_non_adversarial_rate_percent": model_top_acc,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Saved: {args.out_dir / 'vit_category_confidence.png'}")
    print(f"Saved: {args.out_dir / 'vit_top1_non_adversarial_rate.png'}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
