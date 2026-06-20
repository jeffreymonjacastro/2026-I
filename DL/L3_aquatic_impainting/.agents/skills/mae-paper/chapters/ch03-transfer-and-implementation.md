# Chapter 3: Transfer and Implementation

## Core Idea

MAE is presented as a scalable pre-training recipe, not just an ImageNet trick. The strongest evidence is that larger ViT models benefit more from MAE pre-training, and the learned encoders transfer to object detection, instance segmentation, semantic segmentation, and classification datasets. The method competes well while using only ImageNet-1K images for pre-training and avoids the additional tokenizer pre-training used by token-prediction methods.

## Frameworks Introduced

- **Scale test for self-supervision**:
  - When to use: evaluating whether a pre-training method benefits from larger models.
  - How: compare ViT-B, ViT-L, and ViT-H fine-tuning after the same style of pre-training.
- **Transfer validation**:
  - When to use: checking whether learned features are task-general.
  - How: fine-tune the encoder on detection, segmentation, and classification transfer tasks.
- **Partial fine-tuning probe**:
  - When to use: when linear probing and full fine-tuning disagree.
  - How: freeze early blocks and fine-tune only the last N transformer blocks.

## Key Concepts

- **ViT-B/L/H**: Base, Large, and Huge Vision Transformer variants used to test scaling.
- **End-to-end fine-tuning**: updating the full pre-trained model on a supervised downstream task.
- **Partial fine-tuning**: updating only a subset of later transformer blocks.
- **COCO transfer**: object detection and instance segmentation evaluation with Mask R-CNN.
- **ADE20K transfer**: semantic segmentation evaluation with UperNet.
- **Robustness variants**: ImageNet corruption, adversarial, rendition, and sketch datasets used in the appendix.

## Implementation Notes

Pre-training uses AdamW, cosine learning-rate decay, warmup, large batches, and random resized crop. Fine-tuning uses AdamW with layer-wise learning-rate decay and standard ViT regularization such as RandAugment, label smoothing, mixup, cutmix, and drop path. Linear probing uses a separate recipe with LARS, large batch size, no weight decay, and feature normalization before the linear classifier.

For implementation, the paper emphasizes that no special sparse operators are needed. Patch tokens can be shuffled, truncated to keep visible tokens, encoded, concatenated with mask tokens, and unshuffled before decoding.

## Reference Tables

| Evaluation | Main claim |
|---|---|
| ImageNet fine-tuning | MAE scales from ViT-B to ViT-H and improves over scratch training |
| ViT-H at high resolution | Strong ImageNet accuracy using only ImageNet-1K pre-training data |
| COCO Mask R-CNN | MAE improves over supervised pre-training, especially for ViT-L |
| ADE20K UperNet | MAE improves semantic segmentation transfer |
| iNaturalist and Places | larger MAE-pretrained models show strong classification transfer |
| Pixels vs tokens | normalized pixel targets match token targets across tested transfers |

## Anti-patterns

- **Claiming MAE is only an inpainting model**: the decoder reconstructs pixels, but the goal is reusable encoder representation.
- **Comparing only against supervised ViT from scratch**: the paper also compares against self-supervised methods and transfer tasks.
- **Ignoring tuning recipe differences**: linear probing, fine-tuning, and supervised-from-scratch training require different optimizers and regularization.
- **Assuming token targets are required**: the transfer tables support normalized pixel targets as the simpler default.

## Key Takeaways

1. MAE's value is strongest when model capacity grows.
2. Transfer results matter more than reconstruction aesthetics.
3. Pixel reconstruction can compete with token-prediction approaches without extra tokenizer data.
4. Partial fine-tuning helps explain why linear probing may understate MAE feature quality.
5. The method's simplicity is part of its contribution: random masking, ViT encoder, lightweight decoder, pixel loss.

## Connects To

- **ch01**: gives the architecture that is reused for transfer.
- **ch02**: explains why the chosen defaults are empirically justified.
