from pathlib import Path
import torch

DATASET_NAME = "puneet6060/intel-image-classification"
IMG_SIZE = 224
NUM_CLASSES = 6
CLASS_NAMES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]

CHECKPOINT_DIR = Path("checkpoints")
RESULTS_DIR = Path("results")
CACHE_DIR = Path("cache")

CHECKPOINT_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGENET_STATS = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}

BACKBONE_REGISTRY = {
    "deit": ("deit_tiny_patch16_224", 192, "DeiT-Tiny"),
    "dinov2": ("vit_small_patch14_reg4_dinov2.lvd142m", 384, "DINOv2-S/14"),
    "siglip2": ("vit_base_patch16_siglip_224.v2_webli", 768, "SigLIP2-B/16"),
}

TEACHER_CHECKPOINTS = {
    "deit": CHECKPOINT_DIR / "deit_best.pth",
    "dinov2": CHECKPOINT_DIR / "dinov2_best.pth",
    "siglip2": CHECKPOINT_DIR / "siglip2_best.pth",
}

MODEL_COLORS = {"deit": "#378ADD", "dinov2": "#1D9E75", "siglip2": "#D85A30", "yolov8world": "#9B59B6", "amtd": "#8E44AD"}
MODEL_MARKERS = {"deit": "o", "dinov2": "s", "siglip2": "^", "yolov8world": "D", "amtd": "D"}
MODEL_LABELS = {
    "deit": "DeiT-Tiny",
    "dinov2": "DINOv2 ViT-S/14",
    "siglip2": "SigLIP 2 ViT-B/16",
    "yolov8world": "YOLOv8l-WorldV2",
    "amtd": "AMTD (Ours)",
    "mobilevit_s": "MobileViT-S",
    "mobilevit_hard": "MobileViT-Hard",
    "cnn": "SimpleCNN",
}
MODEL_CMAPS = {"deit": "Blues", "dinov2": "Greens", "siglip2": "Oranges", "yolov8world": "Purples"}

YOLOV8_WORLD_PATH = "/home/skyvision/CADU-OVOD/yolov8l-worldv2.pt"

BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-4
