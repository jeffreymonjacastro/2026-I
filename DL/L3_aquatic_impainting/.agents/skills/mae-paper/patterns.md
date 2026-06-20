# Patterns

## High-Ratio Masked Pre-training

**When to use**: You need self-supervised pre-training for image encoders and want a pretext task that does not rely on labels or contrastive pairs.

**How**:
1. Patchify each image.
2. Randomly keep about 25% of patches.
3. Train the encoder only on visible patches.
4. Reconstruct the missing 75% through a lightweight decoder.
5. Fine-tune the encoder on downstream tasks.

**Trade-offs**: High masking makes the task harder and cheaper for the encoder, but too much masking can reduce available signal.

## Decoder as Training Scaffold

**When to use**: Reconstruction is needed during pre-training but not during deployment.

**How**:
1. Keep the encoder large.
2. Keep the decoder narrow and shallow relative to the encoder.
3. Add mask tokens only in the decoder.
4. Discard the decoder after pre-training.

**Trade-offs**: A very small decoder can speed training but may reduce linear probing quality.

## Pixel Target Before Token Target

**When to use**: You are choosing a masked image modeling target.

**How**:
1. Start with normalized pixel reconstruction.
2. Compute loss only on masked patches.
3. Compare against token targets only if you already have a strong tokenizer and evidence that it helps.

**Trade-offs**: Pixel targets are simpler and avoid tokenizer pre-training; token targets can encode higher-level semantics but add complexity and dependency on extra data.

## Random Masking as Augmentation

**When to use**: You want MAE pre-training without heavy contrastive augmentation.

**How**:
1. Generate a fresh random mask each iteration.
2. Use simple crop/flip augmentation.
3. Avoid importing contrastive recipes such as strong color jitter by default.

**Trade-offs**: Random masking provides strong task variation, but downstream fine-tuning still benefits from supervised augmentation recipes.

## Evaluate Beyond Linear Probing

**When to use**: MAE looks weaker than contrastive methods under frozen linear evaluation.

**How**:
1. Report linear probing for comparability.
2. Report full fine-tuning for practical performance.
3. Add partial fine-tuning if you need to understand non-linear feature usefulness.
4. Use transfer tasks to test generality.

**Trade-offs**: More evaluations cost time, but linear probing alone can misrepresent MAE.
