---
pipeline_tag: feature-extraction
tags:
  - computer-vision
  - vision-language
  - representation-learning
  - dinov3
  - safetensors
base_model:
  - facebook/dinov3-vith16plus-pretrain-lvd1689m
  - sentence-transformers/all-roberta-large-v1
---

# TDN: Text-aligned DINO Network

TDN is the DINOv3-only text-aligned model released with the
[TDDN project](https://github.com/adityaSomak/TDDN). It is the controlled
single-encoder counterpart to TDDN: it starts from a frozen DINOv3 ViT-H/16+
visual representation and aligns it with a frozen RoBERTa-large text encoder
using the same lightweight alignment pipeline as TDDN. The full fused
counterpart is [**PuzzleBench/TDDN**](https://huggingface.co/PuzzleBench/TDDN).

The release contains only the trained alignment-head weights in
`model.safetensors`. The DINOv3 and RoBERTa backbones are fetched separately
by the reference implementation; users must have access to the upstream models.

## Architecture

At the reference 336×336 resolution, frozen DINOv3 ViT-H/16+ produces a
21×21 grid of patch tokens together with a global CLS token. Two trainable
self-attention blocks with rotary position embeddings refine these visual
tokens. The final image embedding concatenates the refined CLS token with the
mean-pooled refined patch tokens, preserving both global semantics and dense
spatial features in a 2,560-dimensional representation.

On the text side, frozen RoBERTa-large token features are refined by two
trainable self-attention blocks, masked-mean pooled, and linearly projected to
the same 2,560-dimensional space. TDN therefore uses only DINOv3 features on
the visual path—there is no diffusion branch or feature-fusion module.

![Qualitative patch-feature visualization on chess](assets/pca_chess.png)

## Quick start

Install the [TDDN repository](https://github.com/adityaSomak/TDDN), its
requirements, and the DINOv3 reference package. Authenticate with Hugging Face
to access the gated DINOv3 backbone, then load TDN directly:

```python
from shared_utils.feature_extraction import load_model

model, metadata = load_model("tdn", device="cuda")
```

The same API supports an explicitly downloaded local snapshot:

```python
model, metadata = load_model("tdn", device="cuda", checkpoint="/path/to/TDN")
```

## Alignment design

TDN trains only the visual and text alignment heads; DINOv3 and RoBERTa-large
remain frozen. Training uses approximately 590K image–caption pairs from MS
COCO 2014 and optimizes a symmetric InfoNCE objective together with the
STRUCTURE regularizer, which preserves pretrained representation geometry
during low-data alignment. The complete training provenance is provided in
`training_config.yaml`.

## Evaluation

All results use frozen backbones. Segmentation is zero-shot open-vocabulary
segmentation (mIoU); SPair-71k is keypoint matching (PCK@0.1); retrieval is
recall at rank 1 (R@1).

### Segmentation (mIoU ↑)

TDN substantially improves over CLIP across all five segmentation benchmarks.
The gains are especially clear on structured scenes: its dense DINOv3 patch
features provide more spatially coherent class assignments while retaining the
semantic organization needed for open-vocabulary prediction. TDDN is shown in
the table as the fused counterpart and improves further on every benchmark.

| Model | ADE20K | Cityscapes | COCO-Stuff | PASCAL-Ctx | Puzzle |
|---|---:|---:|---:|---:|---:|
| CLIP ViT-L/14 | 5.20 | 10.05 | 7.35 | 10.44 | 11.04 |
| TDN | 16.51 | 24.29 | 20.67 | 27.55 | 20.92 |
| TDDN | 18.11 | 32.38 | 24.44 | 32.48 | 22.51 |

### Keypoint matching (PCK@0.1 ↑)

TDN exceeds CLIP on SPair-71k, showing that text alignment does not remove the
correspondence information in DINOv3 patch features. TDDN improves further by
adding CleanDIFT's fine-grained spatial detail, yielding the strongest score of
the three models.

| Model | SPair-71k |
|---|---:|
| CLIP ViT-L/14 | 24.89 |
| TDN | 28.29 |
| TDDN | 32.39 |

### Image–text retrieval (R@1 ↑)

Despite using a much smaller alignment corpus than CLIP, TDN and TDDN remain
competitive on retrieval. Both improve over CLIP on COCO image-to-text,
COCO text-to-image, and Flickr30K text-to-image retrieval; CLIP remains ahead
on Flickr30K image-to-text retrieval. The close TDN/TDDN results indicate that
the fused visual encoder preserves the shared text-aligned embedding space.

| Model | Flickr30K I2T | Flickr30K T2I | MS-COCO-14 I2T | MS-COCO-14 T2I |
|---|---:|---:|---:|---:|
| CLIP ViT-L/14 | 87.7 | 66.96 | 34.60 | 18.53 |
| TDN | 85.8 | 72.88 | 37.1 | 24.1 |
| TDDN | 85.3 | 71.24 | 36.1 | 24.0 |
