---
name: mae-paper
description: "Knowledge base from \"Masked Autoencoders Are Scalable Vision Learners\" by He, Chen, Xie, Li, Dollar, and Girshick. Use when reasoning about MAE pre-training, masked image modeling, ViT self-supervision, high mask ratios, asymmetric encoder-decoder design, reconstruction targets, or transfer behavior."
---

# Masked Autoencoders Are Scalable Vision Learners
**Authors**: Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollar, Ross Girshick | **Pages**: 14 | **Sections**: 3 | **Generated**: 2026-06-20

## How to Use This Skill

- **Without arguments**: load the core MAE design rules and empirical takeaways.
- **With a topic**: ask about `mask ratio`, `decoder design`, `linear probing`, `transfer`, `pixels vs tokens`, or `implementation`.
- **With a chapter**: ask for `ch01`, `ch02`, or `ch03` to load the focused section.

When a topic is not covered below, read the relevant chapter file before answering.

## Core Frameworks & Mental Models

### MAE Design Recipe

Use MAE when you want a scalable self-supervised pre-training method for ViT-style vision models without contrastive pairs, heavy augmentations, or discrete tokenizer pre-training.

The paper's core recipe:

1. Split the image into non-overlapping ViT patches.
2. Randomly remove a high fraction of patches, typically around 75%.
3. Encode only the visible patches with a standard ViT encoder.
4. Add learned mask tokens only after the encoder.
5. Decode the full token sequence with a lightweight decoder.
6. Reconstruct pixel values only for masked patches.
7. Discard the decoder after pre-training and fine-tune the encoder.

### Asymmetric Encoder-Decoder

Prefer an asymmetric design over a symmetric autoencoder. The encoder should be expensive and representation-focused; the decoder should be small and reconstruction-focused. This separation matters because pixel reconstruction is lower-level than recognition. A heavier decoder can absorb reconstruction-specific work and leave the encoder's latent representation more useful for downstream tasks.

### High Mask Ratio as the Real Pretext Task

Use a high random mask ratio, not a BERT-like low ratio. Vision has much more local redundancy than language, so a small mask can be solved by texture and neighborhood interpolation. A high mask ratio around 75% removes enough local evidence to force more holistic image understanding while reducing encoder compute because only visible patches are processed.

### Remove Mask Tokens from the Encoder

Do not feed mask tokens to the encoder. If the encoder sees many mask tokens during pre-training but no mask tokens during downstream inference, there is a train-deploy mismatch. Moving mask tokens to the decoder improves representation quality and reduces compute. The paper reports large linear-probing degradation when mask tokens are kept in the encoder.

### Pixel Reconstruction Is Enough

Prefer normalized pixel reconstruction before adding a discrete tokenizer. MAE works well by reconstructing pixels, especially with per-patch normalization. Token targets such as dVAE tokens can work, but they add another pre-training stage and extra compute without clear advantage over normalized pixels in the paper's transfer results.

### Random Masking Over Structured Masking

Use simple uniform random patch sampling as the default. Block-wise masking can become too hard at high ratios, while grid masking is too easy and can produce sharper reconstructions but weaker representations. Random masking gives the best balance of difficulty, efficiency, and representation quality.

### Linear Probing Is Not the Whole Story

Do not judge MAE only by linear probing. MAE features can be less linearly separable than contrastive features but stronger after fine-tuning. Partial fine-tuning shows that even tuning a small number of transformer blocks can unlock large gains, so representation quality should be assessed with downstream fine-tuning and transfer tasks.

## Chapter Index

| # | Title | Key Topics |
|---|-------|------------|
| [ch01](chapters/ch01-core-method.md) | Core Method | masked autoencoding, asymmetric encoder-decoder, high mask ratio, pixel loss |
| [ch02](chapters/ch02-ablation-findings.md) | Ablation Findings | decoder depth/width, mask tokens, reconstruction target, augmentation, mask sampling |
| [ch03](chapters/ch03-transfer-and-implementation.md) | Transfer and Implementation | ImageNet scaling, COCO, ADE20K, robustness, training settings, limits |

## Topic Index

- **Asymmetric encoder-decoder**: ch01, ch02
- **BEiT comparison**: ch02, ch03
- **COCO transfer**: ch03
- **Data augmentation**: ch02
- **Decoder design**: ch01, ch02
- **Fine-tuning**: ch02, ch03
- **High mask ratio**: ch01, ch02
- **ImageNet scaling**: ch03
- **Linear probing**: ch02, ch03
- **Mask sampling**: ch02
- **Mask tokens**: ch01, ch02
- **Normalized pixels**: ch01, ch02, ch03
- **Partial fine-tuning**: ch03
- **Pixel reconstruction**: ch01, ch02
- **Random masking**: ch01, ch02
- **Transfer learning**: ch03
- **ViT**: ch01, ch03

## Supporting Files

- [glossary.md](glossary.md): key terms and compact definitions.
- [patterns.md](patterns.md): reusable method patterns from the paper.
- [cheatsheet.md](cheatsheet.md): quick design and experiment reference.

## Scope & Limits

This skill covers the paper `literature/MAE.pdf` extracted with `book-to-skill` in technical mode using `pdftotext` fallback because `docling` was not installed. It captures the method, design rules, experiments, and implementation settings. For exact equations, table formatting, or visual details, inspect the original PDF.
