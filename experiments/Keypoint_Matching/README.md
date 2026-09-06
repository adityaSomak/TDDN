# Keypoint Matching

PCK@α evaluation on SPair-71K. For each backbone, the eval extracts
spatial `(C, H, W)` feature maps for the source/target image, runs the
cosine-NN PCK matcher in [`evaluation/src/pck.py`](evaluation/src/pck.py),
and aggregates per-pair scores into one row per (model, resolution) in
the headline CSV. Two-component fusions (`ddn`, `sd+dinov2-vitb`,
`sd+dinov2-vitg`) are composed at runtime via `fuse_concat` over the
bilinearly-aligned component feature maps. Distances are normalized
by the **longer side of the target bounding box** (the standard
SPair-71K convention).

Headline metric: **PCK@0.1 (bbox)**.

## Setup

```bash
export HF_TOKEN=<your_token>     # gated DINOv3 / RoBERTa weights
```

Place SPair-71K at
`datasets/Existing_Datasets/Keypoint_Matching/SPair-71K/SPair-71k/`
(populated by `python datasets/download_datasets.py --dataset spair`).

Trained-head tags need their checkpoints under
`experiments/shared_utils/feature_extraction/checkpoints/`:

- `tdn` → `TDN/{config.json,model.safetensors}`
- `tddn` → `TDDN/{config.json,model.safetensors}`

## Supported models

11 model tags configured in [`configs/models.yaml`](configs/models.yaml):
`dinov2-vitb`, `dinov2-vitg`, `sd-2.1`, `dinov3`, `cd`, `clip`,
`sd+dinov2-vitb`, `sd+dinov2-vitg`, `ddn`, `tdn`, `tddn`.

## Run

```bash
python run_eval.py --model <tag|all>                           # full SPair test
python run_eval.py --model <tag> --n-per-cat 20                # 20 pairs/cat
python run_eval.py --model cd --layer-ablation                 # CD per-layer
python run_eval.py --model dinov3 --categories aeroplane --limit 5
                                                               # smoke test
python run_eval.py --model tddn --slidingwindow                # sliding-window protocol
```

| Arg | Default | Meaning |
|---|---|---|
| `--model` | _required_ | model tag from `configs/models.yaml` (or the sliding-window set below), or `all` for a sweep |
| `--spair-root` | `datasets/Existing_Datasets/Keypoint_Matching/SPair-71K/SPair-71k` | path to the unpacked SPair-71K dataset |
| `--split` | `test` | `trn` / `val` / `test` |
| `--categories` | all 18 | space-separated list to restrict the eval (e.g. `aeroplane bicycle`) |
| `--n-per-cat` | `None` (all) | deterministically subsample N pairs per category (`seed=0`) for a faster sweep |
| `--limit` | `None` (no cap) | global pair cap (applied after `--n-per-cat`); useful for smoke runs |
| `--layer-ablation` | `False` | only for diffusion backbones (`cd`, `sd-2.1`) — sweep per-up-block outputs and write to `evaluation/results/ablations/` |
| `--slidingwindow` | `False` | switch to the sliding-window protocol (below); only for the CLIP-lineage + tdn/tddn tags |
| `--device` | `cuda` if available, else `cpu` | torch device |

### Sliding-window protocol

The default single-pass forward uses a per-model canvas (840px for
CLIP/DINOv2, 960px for DINOv3/tdn/tddn/diffusion), letterboxed to a square.
That forces CLIP/SigLIP-family models (fixed learned position-embedding
tables, native ~336–384px) through a large (~2.5–4x) interpolation stretch.
`--slidingwindow` instead applies the GroupViT/TCL-style protocol
Vision_Language_Alignment uses for segmentation (crop=448, stride=224)
uniformly to every model: letterbox to an 896×896 canvas (=448×2) → 3×3=9
overlapping windows → per-window patch features → average-overlap stitch
into one (C, 896, 896) feature canvas → the same cosine-NN PCK matching.
Each window stays close to every model's native training resolution (mild
interpolation for CLIP/SigLIP/fgclip2, none at all for tdn/tddn — DINOv3
uses RoPE, exact at any grid size).

Model tags come from `Vision_Language_Alignment/configs/models.yaml`, not
this experiment's own registry: `clip`, `dfn_l14`, `metaclip_l14`,
`openclip_l14`, `siglip2_l16`, `fgclip2_large`, `tdn`, `tddn`.

## Results

- [`evaluation/results/keypoint_matching.csv`](evaluation/results/keypoint_matching.csv) —
  headline table with columns
  `model, resolution, pck@0.1, pck@0.05, pck@0.01` (one row per model).
- `evaluation/results/<model_tag>.json` — per-category PCK breakdown
  (`overall` block + a `per_category` map keyed by SPair category).
- [`evaluation/results/ablations/`](evaluation/results/ablations/) — committed paper artifacts:
  - `cd_vs_sd-2.1_layers.json` — CD-vs-SD per-up-block PCK breakdown
    (reproducible: `python run_eval.py --model cd --layer-ablation` +
    `--model sd-2.1 --layer-ablation`).
  - `dinov3_facets.json` — DINOv3 token / query / key / value facet
    sweep. Committed artifact; the generation script isn't in tree.
- [`evaluation/results/keypoint_matching_slidingwindow.csv`](evaluation/results/keypoint_matching_slidingwindow.csv) —
  same columns as above with `resolution` replaced by `canvas` (always 896),
  one row per `--slidingwindow` model.
- `evaluation/results/slidingwindow/<model_tag>.json` — per-category
  breakdown for the sliding-window protocol.
