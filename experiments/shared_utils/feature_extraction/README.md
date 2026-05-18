# feature_extraction

Unified vision-feature extraction shared across the `experiments/` tree.
Loads every supported backbone, normalizes preprocessing, and exposes a
uniform extractor interface.

## Quick start

```bash
export HF_TOKEN=<your_token>     # gated DINOv3 / RoBERTa-large weights

python -c "
from shared_utils.feature_extraction import build_extractor, build_transform, load_image
ex  = build_extractor('dinov3-vith16plus', device='cuda')
tfm = build_transform('dinov3-vith16plus', 512, 'square_resize')
img = tfm(load_image('some.png')).unsqueeze(0)
out = ex.extract(img)
print(out['patch_tokens'].shape)   # (1, 1280, 32, 32)
"
```

## Registered backbones

| name                | family  | dim       | notes                        |
|---|---|---|---|
| `dinov3-vitb16`     | ViT     | 768       | bf16; fp16 forbidden         |
| `dinov3-vith16plus` | ViT     | 1280      | bf16                         |
| `dinov2-vitb14`     | ViT     | 768       | fp32                         |
| `dinov2-vitl14`     | ViT     | 1024      | fp32                         |
| `dinov2-vitg14`     | ViT     | 1536      | fp32                         |
| `clip-vitl14`       | ViT     | 1024      | input res must be ×14        |
| `clip-vitl14-336`   | ViT     | 1024      | 336 fixed pos-emb            |
| `sd`                | UNet    | per-layer | needs `timestep`/`noise_mode`|
| `cleandift`         | UNet    | per-layer | needs `timestep`/`noise_mode`|
| `vith-roberta`      | trained | 2560      | requires checkpoint          |
| `fused-dinov3-cd`   | trained | 2560      | avg of two ckpts; fp32       |

For ViT backbones, `facet={token,query,key,value}` and `block_idx`
select the extracted activations (defaults: `token`, last block).

## Preprocessing strategies

The backbone fixes mean/std and patch-size validation; the caller picks
the spatial strategy:

- `square_resize` — `Resize((res, res))`.
- `aspect_pad` — aspect-preserving resize + center pad (LANCZOS).
- `imagenet_center_crop` — `Resize(res) + CenterCrop(res)`.

## Diffusion recipes

Each caller passes its own `(timestep, noise_mode, hook_position)` to
the diffusion backbones; there is no implicit default. See the
per-experiment configs for the values in use.

## Trained-model checkpoints

`vith-roberta` and `fused-dinov3-cd` load weights from
`checkpoints/<name>/ckpt/<step>/`. The `.distcp` shards are not
committed (~3.9 GB); place them at the expected paths before using
these backbones.
