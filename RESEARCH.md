# RESEARCH: CLIP Architecture & Zero-Shot Capability Deep Dive

## 1. Introduction to CLIP
Contrastive Language-Image Pre-Training (CLIP) is a multimodal model developed by OpenAI that learns visual concepts from natural language supervision. Unlike traditional computer vision models trained on a fixed set of labels, CLIP is trained to understand the relationship between images and their corresponding textual descriptions.

## 2. Dual-Encoder Architecture
CLIP consists of two primary components:
- **Image Encoder:** Typically a Vision Transformer (ViT) or a ResNet (RN). It transforms an input image into a high-dimensional embedding vector.
- **Text Encoder:** A Transformer-based language model that transforms a sequence of text tokens into a high-dimensional embedding vector.

### 2.1 The Shared Embedding Space
The key innovation of CLIP is that both the image and text encoders project their outputs into a **shared multimodal embedding space**. If an image and a text description are semantically related, their embeddings in this shared space will have a high cosine similarity.

## 3. Contrastive Learning Objective
CLIP is trained using a **symmetric contrastive loss** (InfoNCE). Given a batch of $ (image, text) pairs:
- The model aims to maximize the cosine similarity of the $ correct pairs.
- Simultaneously, it minimizes the cosine similarity of the ^2 - N$ incorrect pairs.

### 3.1 Mathematical Formulation
For a normalized image embedding $ and a normalized text embedding $:
362367\text{Similarity}(I_i, T_j) = \exp(\text{logit\_scale} \cdot I_i^T T_j)362367
The loss is computed as the average of the cross-entropy losses across the rows and columns of the similarity matrix.

## 4. Zero-Shot Capability
Zero-shot classification is achieved by:
1. Encoding all candidate labels (e.g., "a photo of a cat", "a photo of a dog") through the text encoder.
2. Encoding the target image through the image encoder.
3. Calculating the cosine similarity between the image embedding and all text embeddings.
4. Applying a softmax to the similarities to get a probability distribution over the labels.

## 5. Model Selection for Lab
For this lab, we will use the **ViT-B/32** variant.
- **ViT-B/32:** A Vision Transformer with "Base" size and 2 \times 32$ input patches.
- **Reasoning:** It offers a good balance between inference speed and accuracy, making it ideal for experimental prompt design analysis on consumer-grade hardware.

## References
- Radford, A., et al. (2021). "Learning Transferable Visual Models From Natural Language Supervision."
