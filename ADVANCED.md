# ADVANCED: Word Sensitivity & Structural Bias Investigation

## 1. Word Sensitivity Analysis
To understand how specific terminology affects CLIP's zero-shot classification performance, we investigated the sensitivity to synonymous introductory phrases. We tested the impact of prepending basic prompts with "photo of", "image of", and "picture of".

### Methodology
For a subset of our dataset, we evaluated variations of a core prompt:
- **Baseline:** "[Subject]"
- **Variation A:** "A photo of a [Subject]"
- **Variation B:** "An image of a [Subject]"
- **Variation C:** "A picture of a [Subject]"

### Findings & Insights
- **"Photo of" Dominance:** Across almost all natural scenes and real-world objects, "a photo of a..." consistently yielded the highest confidence scores among the prefixes. This is a direct artifact of CLIP's pre-training dataset (WebImageText), which predominantly consists of alt-text and captions where "photo of" is the standard convention for describing real-world photography.
- **"Image of" vs "Picture of":** "Image of" generally performed slightly better than "picture of" but was still lower than "photo of." "Image" tends to map to a broader set of visual representations (including illustrations, graphics, and digital art) in the latent space, whereas "photo" tightly constrains the embedding to photographic realism.
- **The "No Prefix" Baseline:** Interestingly, for highly distinct and iconic objects (like "a pizza" or "a car"), simply using the noun phrase without any prefix often matched or slightly exceeded the "photo of" prefix. This suggests the prefix acts as a regularization mechanism, aligning the textual domain with photographic features, which is crucial when the subject is ambiguous but less necessary when the visual signature is overwhelming.

## 2. Structural Bias Investigation
We also evaluated how the grammatical structure and length of the prompt influence the model's confidence ranking.

### Short vs. Long vs. Structured Prompts

1.  **Short Prompts (1-3 words):** (e.g., "a bird")
    - *Behavior:* Often serve as strong baselines because they map directly to the primary semantic concept without any diluting information. However, they lack the specificity to distinguish between two visually similar but contextually different images.
2.  **Long Unstructured Prompts (10+ words):** (e.g., "a bird that is sitting on a branch and it is a very nice day outside in the wild")
    - *Behavior:* These often perform worse than structured prompts of similar length. The inclusion of conversational filler ("that is", "and it is", "very nice") dilutes the semantic density. The text embedding drifts away from the core visual features because the model must account for abstract concepts ("nice day") that lack strong visual correlates in the image.
3.  **Structured Prompts (Noun phrases with stacked modifiers):** (e.g., "a small colorful bird perched on a branch in a forest")
    - *Behavior:* These consistently provide the highest confidence scores. By stacking descriptive adjectives and spatial context without filler words, the text embedding becomes a dense representation of pure visual features. CLIP's contrastive loss heavily rewards this tight semantic alignment.

## 3. Conclusion on Bias
CLIP exhibits a strong structural bias toward **dense, descriptive noun phrases** over conversational or syntactically complex sentences. It also shows a clear lexical bias toward terminology prevalent in internet image captions (e.g., preferring "photo" over "picture" for natural scenes). Understanding these biases is paramount for effective zero-shot prompt engineering.
