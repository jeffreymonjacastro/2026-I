# Chapter 1: Core Method

## Core Idea

Masked Autoencoders (MAE) adapt the masked-language-modeling idea to images by hiding many image patches and training a model to reconstruct the missing pixels. The important change is that images are spatially redundant, so the mask must be much more aggressive than in language. MAE uses an asymmetric encoder-decoder: the encoder processes only visible patches, and a lightweight decoder reconstructs the full image from encoded visible tokens plus mask tokens. This makes pre-training both cheaper and more effective for large ViT models.

## Frameworks Introduced

- **Masked image autoencoding**:
  - When to use: self-supervised pre-training for image encoders, especially ViT backbones.
  - How: remove patches, encode visible patches, decode full patch sequence, reconstruct masked pixels.
- **Asymmetric encoder-decoder**:
  - When to use: reconstruction target is lower-level than downstream recognition.
  - How: make the encoder large and representation-oriented; make the decoder small and reconstruction-oriented.
- **High-ratio random masking**:
  - When to use: input has heavy local redundancy.
  - How: uniformly sample visible patches and remove about 75% of patches.

## Key Concepts

- **Visible patches**: the subset of image patches kept after masking; only these are sent into the encoder.
- **Mask tokens**: learned vectors inserted after the encoder so the decoder knows which positions must be reconstructed.
- **Patch reconstruction**: predicting the pixel vector for each masked image patch.
- **Masked-patch-only loss**: MSE is computed only on removed patches, not visible patches.
- **Per-patch normalized pixels**: reconstruction target where each patch is normalized by its own mean and standard deviation.
- **Train-deploy mismatch**: mismatch caused when the encoder sees mask tokens during pre-training but not during downstream inference.

## Mental Models

- Think of MAE as BERT for ViT, but with much higher masking because images are easier to locally interpolate.
- Use the encoder as the reusable representation learner and the decoder as disposable training scaffolding.
- Treat random masking as both the corruption process and the main augmentation mechanism.
- Move reconstruction burden into the decoder so the encoder is not forced to be a pixel generator.

## Technical Procedure

```text
Input image
  -> split into fixed-size patches
  -> add patch embeddings and positional embeddings
  -> randomly shuffle patch tokens
  -> keep visible subset, remove masked subset
  -> encode visible tokens only
  -> append mask tokens
  -> unshuffle tokens to original patch positions
  -> decode full sequence
  -> predict pixels for masked patches
  -> compute MSE only on masked patches
```

## Anti-patterns

- **Low mask ratio by default**: image patches are redundant; low masking can be solved with local texture cues.
- **Mask tokens in the encoder**: slows training and creates input mismatch between pre-training and fine-tuning.
- **Heavy decoder as final model**: the decoder is only for pre-training and should be discarded.
- **Tokenizer-first complexity**: dVAE-style token targets add a separate pre-training dependency that normalized pixels can avoid.

## Key Takeaways

1. The encoder should see only real visible patches.
2. Mask tokens belong in the decoder, not the encoder.
3. A high mask ratio is not a nuisance; it is the task.
4. Pixel reconstruction is sufficient when paired with the right architecture.
5. The decoder can be small because it is not used after pre-training.

## Connects To

- **ch02**: validates these design choices through ablations.
- **ch03**: shows that the pre-trained encoder transfers to recognition, detection, and segmentation.
