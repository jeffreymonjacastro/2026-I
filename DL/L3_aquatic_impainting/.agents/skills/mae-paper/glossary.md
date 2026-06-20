# Glossary

**Asymmetric encoder-decoder** - MAE design where the encoder processes only visible patches and the lightweight decoder reconstructs all patches from encoded visible tokens plus mask tokens.

**BEiT** - Masked image modeling method that predicts discrete visual tokens from a tokenizer; MAE contrasts with it by reconstructing pixels.

**Block-wise masking** - Masking strategy that removes contiguous regions; in MAE it is weaker than random masking at high ratios.

**Decoder** - Pre-training-only module that reconstructs missing pixels; discarded after pre-training.

**Denoising autoencoder** - Autoencoder trained to reconstruct clean input from corrupted input; MAE is a masked-image version.

**Fine-tuning** - Supervised downstream training that updates the pre-trained encoder.

**Grid-wise masking** - Regular sampling strategy that keeps patches in a grid; easier reconstruction but weaker representation in the paper.

**High mask ratio** - Removing a large share of patches, around 75%, to reduce redundancy and create a hard visual pretext task.

**Linear probing** - Evaluation where the encoder is frozen and only a linear classifier is trained on top.

**Mask token** - Learned vector inserted for missing patch positions before decoding.

**Masked-patch-only loss** - Reconstruction loss computed only over removed patches.

**MAE** - Masked Autoencoder; a self-supervised ViT pre-training method using high-ratio patch masking and pixel reconstruction.

**Normalized pixels** - Reconstruction target where pixel values are normalized within each patch.

**Patch** - Non-overlapping image region treated as a token by ViT.

**Partial fine-tuning** - Updating only the last transformer blocks to bridge linear probing and full fine-tuning.

**Random masking** - Uniformly sampling visible patches without replacement; the default and best masking strategy in the paper.

**Reconstruction target** - The value predicted for masked patches, such as raw pixels, normalized pixels, PCA coefficients, or discrete tokens.

**Train-deploy mismatch** - Difference between the encoder input during pre-training and downstream use; occurs if mask tokens are used in the encoder.

**ViT** - Vision Transformer architecture used as MAE's encoder backbone.
