# Chapter 2: Ablation Findings

## Core Idea

The paper's ablations show that MAE works because several simple design choices reinforce each other: high random masking, no mask tokens in the encoder, a lightweight but nontrivial decoder, and pixel reconstruction. These choices are not independent. The high mask ratio makes the task semantically meaningful and cheaper; the asymmetric architecture makes the high mask ratio computationally useful; the decoder absorbs reconstruction-specific details; and random masking acts as a strong source of variation without heavy augmentation.

## Frameworks Introduced

- **Mask-ratio sweep**:
  - When to use: tuning MAE pre-training difficulty.
  - How: evaluate around 40-80% for fine-tuning, with 75% as the paper's strong default.
- **Decoder capacity allocation**:
  - When to use: balancing linear probing and fine-tuning behavior.
  - How: use a decoder deep enough to support reconstruction but narrow enough to keep pre-training cheap.
- **Target choice test**:
  - When to use: deciding between raw pixels, normalized pixels, PCA coefficients, or token targets.
  - How: start with normalized pixels; use tokens only if there is a demonstrated downstream reason.

## Key Concepts

- **Fine-tuning accuracy**: downstream accuracy after updating the encoder weights.
- **Linear probing accuracy**: downstream accuracy with frozen encoder features and a trained linear classifier.
- **Random masking**: uniformly sampled visible patches without replacement.
- **Block-wise masking**: removes contiguous image blocks; can make reconstruction too hard at high ratio.
- **Grid-wise masking**: keeps patches in a regular pattern; can make reconstruction too easy.
- **Decoder depth**: number of transformer blocks in the decoder.
- **Decoder width**: token dimension in the decoder.

## Mental Models

- Use linear probing to inspect separability, but use fine-tuning and transfer to judge practical representation quality.
- A good reconstruction image is not automatically a good representation; sharper outputs can come from easier pretext tasks.
- Decoder depth can protect the encoder from becoming too reconstruction-specific.
- Random masking is a regularizer and a task generator, so MAE needs less color or crop augmentation than contrastive methods.

## Reference Tables

| Design choice | Paper finding | Practical default |
|---|---|---|
| Mask ratio | 75% works well for fine-tuning and linear probing | Start at 75% |
| Encoder mask tokens | Worse accuracy and slower training | Remove from encoder |
| Decoder depth | Helps linear probing; fine-tuning less sensitive | 8 blocks for baseline, 1 block for speed checks |
| Decoder width | Can be narrower than encoder | 512-d for ViT-L baseline |
| Reconstruction target | Normalized pixels improve over raw pixels | Use per-patch normalized pixels |
| Data augmentation | Cropping enough; color jitter hurts | Keep augmentation simple |
| Mask sampling | Random best among tested strategies | Use uniform random sampling |

## Anti-patterns

- **Optimizing only reconstruction quality**: a sharper reconstruction can come from an easier task and weaker representation.
- **Using color jitter by habit**: contrastive-learning augmentation recipes do not transfer directly to MAE.
- **Treating linear probing as final evidence**: MAE can look weaker under linear probing but stronger under fine-tuning.
- **Structured masking without validation**: block or grid masks change task difficulty and can reduce representation quality.

## Key Takeaways

1. The 75% mask ratio is a central empirical finding.
2. Removing mask tokens from the encoder gives both accuracy and speed benefits.
3. Normalized pixel targets are the simplest strong reconstruction target.
4. MAE does not need contrastive-style heavy augmentation.
5. Random masking beats visually intuitive structured masks in the paper's setup.
6. Linear probing and fine-tuning can disagree; report both carefully.

## Connects To

- **ch01**: explains the method components being ablated.
- **ch03**: shows whether the ablation-friendly defaults transfer beyond ImageNet classification.
