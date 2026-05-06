# Checkpoints

## Naming Convention

All checkpoints follow the pattern:

```
scratch_{model}_{pct}pct_e{epochs}.pth
```

| Component | Description |
|---|---|
| `{model}` | Model name: `amtd`, `cnn`, `deit`, `dinov2`, `siglip`, `mobilevit_s` |
| `{pct}` | Training subset percentage: `5`, `10`, `50`, `100` |
| `{epochs}` | Number of training epochs: `50`, `40`, `30`, `25`, `15` |

### Examples

- `scratch_amtd_5pct_e50.pth` — AMTD, 5% data, 50 epochs
- `scratch_cnn_100pct_e15.pth` — CNN baseline, 100% data, 15 epochs
- `scratch_siglip_50pct_e25.pth` — SigLIP2 from scratch, 50% data, 25 epochs
- `yolov8world_best.pth` — YOLOv8-World best checkpoint

## Checkpoint Format

Checkpoints are PyTorch state dicts saved with `torch.save()`:

```python
torch.save(model.state_dict(), checkpoint_path)
```

## Loading a Checkpoint

```python
import torch
import timm

# Load model (must match architecture)
model = timm.create_model(
    "mobilevit_xxs",
    pretrained=False,
    num_classes=6
)

# Load checkpoint
checkpoint = torch.load("checkpoints/scratch_amtd_5pct_e50.pth")
model.load_state_dict(checkpoint)
model.eval()
```

## Available Checkpoints

| File | Model | Subset | Epochs |
|---|---|---|---|
| `scratch_amtd_5pct_e50.pth` | AMTD | 5% | 50 |
| `scratch_amtd_10pct_e40.pth` | AMTD | 10% | 40 |
| `scratch_amtd_50pct_e25.pth` | AMTD | 50% | 25 |
| `scratch_amtd_100pct_e15.pth` | AMTD | 100% | 15 |
| `scratch_cnn_5pct_e50.pth` | CNN | 5% | 50 |
| `scratch_cnn_10pct_e40.pth` | CNN | 10% | 40 |
| `scratch_cnn_50pct_e25.pth` | CNN | 50% | 25 |
| `scratch_cnn_100pct_e15.pth` | CNN | 100% | 15 |
| `scratch_deit_5pct_e50.pth` | DeiT-Tiny | 5% | 50 |
| `scratch_deit_10pct_e40.pth` | DeiT-Tiny | 10% | 40 |
| `scratch_deit_50pct_e25.pth` | DeiT-Tiny | 50% | 25 |
| `scratch_deit_100pct_e15.pth` | DeiT-Tiny | 100% | 15 |
| `scratch_dinov2_5pct_e50.pth` | DINOv2 | 5% | 50 |
| `scratch_dinov2_10pct_e40.pth` | DINOv2 | 10% | 40 |
| `scratch_dinov2_50pct_e25.pth` | DINOv2 | 50% | 25 |
| `scratch_dinov2_100pct_e15.pth` | DINOv2 | 100% | 15 |
| `scratch_mobilevit_s_5pct_e50.pth` | MobileViT-S | 5% | 50 |
| `scratch_siglip_5pct_e50.pth` | SigLIP2 | 5% | 50 |
| `scratch_siglip_10pct_e40.pth` | SigLIP2 | 10% | 40 |
| `scratch_siglip_50pct_e25.pth` | SigLIP2 | 50% | 25 |
| `scratch_siglip_100pct_e15.pth` | SigLIP2 | 100% | 15 |
| `yolov8world_best.pth` | YOLOv8-World | 100% | 20 |
