# MAE Cheatsheet

## Default Design

| Component | Default from paper |
|---|---|
| Backbone | ViT encoder |
| Mask ratio | 75% random patches |
| Encoder input | visible patches only |
| Mask token location | decoder only |
| Decoder | lightweight transformer |
| Reconstruction target | pixels, preferably per-patch normalized |
| Loss | MSE on masked patches only |
| Decoder after pre-training | discard |

## Use This When

- You need scalable image self-supervision.
- Labels are limited or expensive.
- You use ViT-style patch tokens.
- You want simpler machinery than contrastive learning or token prediction.

## Main Experimental Lessons

| Question | Answer |
|---|---|
| Why high masking? | Images are redundant; high masking forces holistic inference and reduces encoder compute. |
| Why asymmetric? | Encoder learns representation; decoder handles pixel reconstruction. |
| Why no mask tokens in encoder? | Avoids train-deploy mismatch and speeds training. |
| Why random masks? | Best balance of task difficulty and representation quality in the paper. |
| Why pixels? | Normalized pixels are simple and match token targets in tested transfer results. |
| Why not only linear probe? | MAE features can be less linearly separable but stronger after fine-tuning. |

## Minimal Pseudocode

```python
patches = patchify(images)
visible, restore_idx = random_keep(patches, keep_ratio=0.25)
z = encoder(visible)
tokens = append_mask_tokens(z, restore_idx)
pred = decoder(tokens)
loss = mse(pred[masked_positions], patches[masked_positions])
```

## Pitfalls

- Do not tune for pretty reconstructions only.
- Do not feed mask tokens into the encoder unless testing that ablation.
- Do not assume contrastive augmentation recipes help MAE.
- Do not compare MAE only with linear probing.
- Do not add a tokenizer unless normalized pixels fail for your target setting.
