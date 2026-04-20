import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image


# Define models to compare
MODELS_TO_COMPARE = ["ViT-B/32", "ViT-B/16", "ViT-L/14"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _import_clip():
    try:
        import clip  # type: ignore

        return clip
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing dependency 'clip'. Install project requirements with: pip install -r requirements.txt"
        ) from exc


def run_inference(prompts_file: Path, data_dir: Path, output_file: Path, models):
    clip = _import_clip()

    with prompts_file.open("r", encoding="utf-8") as f:
        prompts_data = json.load(f)

    all_model_results = {}

    for model_name in models:
        print(f"--- Loading model: {model_name} ---")
        model, preprocess = clip.load(model_name, device=DEVICE)
        results = {}

        for img_name, categories in prompts_data.items():
            img_path = data_dir / img_name
            if not img_path.exists():
                continue

            image = preprocess(Image.open(img_path)).unsqueeze(0).to(DEVICE)

            # Flatten prompts
            all_prompts = []
            prompt_map = []
            for cat, p_list in categories.items():
                for p in p_list:
                    all_prompts.append(p)
                    prompt_map.append({"text": p, "category": cat})

            text = clip.tokenize(all_prompts).to(DEVICE)

            with torch.no_grad():
                image_features = model.encode_image(image)
                text_features = model.encode_text(text)

                # Normalize
                image_features /= image_features.norm(dim=-1, keepdim=True)
                text_features /= text_features.norm(dim=-1, keepdim=True)

                # Similarity
                similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                top_values, top_indices = similarity[0].topk(len(all_prompts))
                values = top_values.tolist()
                indices = top_indices.tolist()

            img_results = []
            for value, index in zip(values, indices):
                idx = int(index)
                img_results.append(
                    {
                        "prompt": prompt_map[idx]["text"],
                        "category": prompt_map[idx]["category"],
                        "score": float(value),
                    }
                )

            results[img_name] = img_results
            print(f"Model {model_name} processed {img_name}")

        all_model_results[model_name] = results

    with output_file.open("w", encoding="utf-8") as f:
        json.dump(all_model_results, f, indent=2)
    print(f"All results saved to {output_file}")


def build_parser():
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Compare multiple CLIP ViT models on prompt-image alignment.")
    parser.add_argument("--prompts", type=Path, default=script_dir / "prompts.json", help="Path to prompts JSON file")
    parser.add_argument("--data-dir", type=Path, default=script_dir / "data", help="Directory containing input images")
    parser.add_argument("--output", type=Path, default=script_dir / "results.json", help="Output JSON path")
    parser.add_argument(
        "--models",
        nargs="+",
        default=MODELS_TO_COMPARE,
        help="Model names to evaluate (default: ViT-B/32 ViT-B/16 ViT-L/14)",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    try:
        run_inference(args.prompts, args.data_dir, args.output, args.models)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
