---
marp: true
theme: default
class: lead
backgroundColor: #ffffff
---

# Vision Transformers (ViT)
## Architecture, Comparison, and Future Directions

---

## 1. Introduction
- **The Shift:** Moving from Convolutional Neural Networks (CNNs) to pure Transformer architectures for vision.
- **The Core Idea:** Treat images as sequences of patches, analogous to words in a sentence.
- **The Result:** State-of-the-art performance on image classification, provided sufficient data.

---

## 2. Model Architecture
1. **Patch Extraction:** Divide image into non-overlapping grid patches.
2. **Linear Projection:** Flatten and project to a uniform embedding dimension (Patch Embedding).
3. **Class Token:** A learnable `[class]` token aggregates the global image representation.
4. **Positional Encoding:** Added to embeddings to retain 2D spatial relationships.
5. **Transformer Encoder:** Alternating Multi-Head Self-Attention (MSA) and MLP blocks.

---

## 3. Mathematical Foundations
- **Patch Embedding:** $\mathbf{z}_0 = [\mathbf{x}_{\text{class}}; \mathbf{x}^1_p\mathbf{E}; \dots; \mathbf{x}^N_p\mathbf{E}] + \mathbf{E}_{pos}$
- **Self-Attention:** $\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{D_h}}\right) \mathbf{V}$
- **Encoder Layer:** 
  $\mathbf{z}'_\ell = \text{MSA}(\text{LN}(\mathbf{z}_{\ell-1})) + \mathbf{z}_{\ell-1}$
  $\mathbf{z}_\ell = \text{MLP}(\text{LN}(\mathbf{z}'_\ell)) + \mathbf{z}'_\ell$

---

## 4. Comparing 3 ViT Models

| Feature | ViT-Base (ViT-B) | ViT-Large (ViT-L) | ViT-Huge (ViT-H) |
| :--- | :--- | :--- | :--- |
| **Layers** | 12 | 24 | 32 |
| **Hidden Size** | 768 | 1024 | 1280 |
| **Heads** | 12 | 16 | 16 |
| **Parameters** | 86M | 307M | 632M |

*Takeaway:* Model capacity scales effectively, but requires exponentially more data to prevent overfitting.

---

## 5. Advantages of ViTs
- **Global Receptive Field:** Instant context across the entire image in the first layer.
- **Unified Multimodal Architecture:** Identical to NLP transformers, making vision-language alignment (like CLIP) highly effective.
- **Scaling Laws:** Performance continues to grow with larger models and datasets without plateauing as early as CNNs.

---

## 6. Drawbacks of ViTs
- **Lack of Inductive Bias:** No built-in understanding of translation invariance or local structures.
- **Data Hunger:** Requires massive datasets (JFT-300M, LAION) to learn basic visual rules from scratch.
- **Computational Cost:** Attention mechanism scales quadratically $O(N^2)$ with image resolution.

---

## 7. Vision for Enhancement
1. **Hybrid Architectures:** Integrate Convolutional layers early in the network to reintroduce inductive biases and improve data efficiency.
2. **Linear Attention:** Adopt $O(N)$ attention approximations to process high-resolution images natively without massive memory bottlenecks.
3. **Self-Supervised Learning (Masked Autoencoders):** Train by reconstructing heavily masked images, drastically reducing reliance on human-labeled datasets.

---

## 8. Conclusion
Vision Transformers represent the frontier of computer vision. By addressing their data and computational bottlenecks through hybridization and self-supervision, they will continue to drive the future of generative AI and multimodal intelligence.