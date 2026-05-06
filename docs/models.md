# Models

## Teacher Ensemble (Frozen)

| Model | timm Name | Params | Pretraining | Strength |
|---|---|---|---|---|
| DeiT-Tiny | `deit_tiny_patch16_224` | 5.7M | ImageNet-1K (supervised) | Efficient supervised training, local feature discrimination |
| DINOv2-S/14 | `vit_small_patch14_reg4_dinov2.lvd142m` | 22M | LVD-142M (self-supervised) | Global scene understanding, robust features |
| SigLIP2-B/16 | `vit_base_patch16_siglip_224.v2_webli` | 86M | WebLI (vision-language) | Multi-modal semantic understanding |

### DeiT-Tiny (`deit_tiny_patch16_224`)

- **Architecture:** ViT-Tiny with knowledge distillation token
- **Patch size:** 16×16
- **Embedding dim:** 192
- **Heads:** 3
- **Layers:** 12
- **Pretraining:** ImageNet-1K with distillation from RegNetY-16GF
- **Accuracy in this project:** 91.27%

### DINOv2-S/14 (`vit_small_patch14_reg4_dinov2.lvd142m`)

- **Architecture:** ViT-Small with register tokens
- **Patch size:** 14×14
- **Embedding dim:** 384
- **Heads:** 6
- **Layers:** 12
- **Registers:** 4
- **Pretraining:** LVD-142M (self-supervised, iBOT-style)
- **Accuracy in this project:** 94.93%

### SigLIP2-B/16 (`vit_base_patch16_siglip_224.v2_webli`)

- **Architecture:** ViT-Base with sigmoid loss
- **Patch size:** 16×16
- **Embedding dim:** 768
- **Heads:** 12
- **Layers:** 12
- **Pretraining:** WebLI (vision-language contrastive, 10B images)
- **Accuracy in this project:** 88.17%

## Student Model

### MobileViT-XXS (from scratch, 1.5M params)

Used in the main data hunger experiments. Combines CNN local processing with Transformer global attention.

### MobileViT-S (pretrained KD, 5.1M params)

Used in the enhanced AMTD distillation (95.03% accuracy).

### Architecture Details

The MobileViT architecture consists of:

1. **Initial stem:** 3×3 conv + BatchNorm + SiLU
2. **MobileNetV2 blocks:** Inverted residuals with depthwise separable convolutions
3. **MobileViT blocks:** Local representation via 3×3 conv → unfolding → Transformer → folding → fusion
4. **Head:** Global average pool → classifier

The local + global design provides:
- **CNN-like inductive biases** (locality, translation equivariance) for data efficiency
- **Transformer attention** for long-range dependencies
- **Lightweight footprint** suitable for from-scratch training on small datasets

## YOLOv8-World (frozen baseline)

- **Model:** YOLOv8-World-v2 (60.9M params)
- **Use:** Frozen backbone comparison (not in teacher ensemble)
- **Features:** Object detection backbone with text-conditional embeddings
