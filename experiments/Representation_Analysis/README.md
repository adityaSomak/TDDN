# Representation Analysis

Tools for analyzing vision-feature representations from pretrained and trained
backbones. Two complementary views:

- **Quantitative** — CKA + PWCCA similarity, uniformity and effective-rank
  quality, computed on a fixed 2000-image COCO val2014 subset.
- **Qualitative** — PCA(3) → RGB activation maps per backbone, optionally fused
  across two backbones.

## Layout

```
Representation_Analysis/
├── run.py                          # subcommands: activation-maps, metrics, plots
├── configs/                        # metrics, models, activation-maps, sample IDs
├── metrics/                        # similarity, quality, feature-matrix builders
├── pca_viz/                        # render_one() — single-image activation map
├── qualitative/
│   ├── samples/                    # input images
│   ├── baselines/activation-maps/  # per-baseline outputs
│   ├── tdn/activation-maps/        # trained DINOv3 + RoBERTa
│   ├── tddn/activation-maps/       # trained DINOv3 + CleanDIFT + RoBERTa
│   └── ddn/activation-maps/        # handcraft DINOv3 + CleanDIFT
└── quantitative/{global,patch}/
    ├── results/                    # *.csv (committed)
    └── plots/                      # *.{pdf,png} (committed)
```

## Setup

The repo-wide [`requirements.txt`](../../requirements.txt) at the
top-level supplies every dependency this experiment needs. Set the
HuggingFace token before the first run for gated weights:

```bash
export HF_TOKEN=<your_token>         # gated DINOv3 / RoBERTa weights
```

Paths resolve via `shared_utils.paths` (`DATASETS_ROOT`,
`CHECKPOINTS_ROOT`, `FEATURES_ROOT`) and respect `EXPERIMENTS_*` env
overrides.

## Usage

```bash
# Regenerate the 5 plots from the committed CSVs (no GPU, instant)
python run.py plots

# Render one activation map
python run.py activation-maps --image qualitative/samples/maze.png --model dinov3

# Full sweep per configs/activation_maps.yaml
python run.py activation-maps --image all --model all

# Run the quantitative pipeline (GPU + trained checkpoints required)
python run.py metrics --global --patch --similarity --quality
```

## Models

| Tag           | Group     | Backbone(s)                       |
|---|---|---|
| `sd-2.1`      | baselines | Stable Diffusion v2.1             |
| `cd`          | baselines | CleanDIFT (SD-v1.5)               |
| `dinov3`      | baselines | DINOv3 ViT-H/16+                  |
| `dinov2-vitb` | baselines | DINOv2 ViT-B/14                   |
| `dinov2-vitg` | baselines | DINOv2 ViT-G/14                   |
| `clip`        | baselines | CLIP ViT-L/14                     |
| `tdn`         | trained   | DINOv3 + RoBERTa (alignment head) |
| `tddn`        | trained   | DINOv3 + CleanDIFT + RoBERTa      |
| `ddn-cd`      | ddn       | DINOv3 + CleanDIFT (handcraft)    |

See `configs/models.yaml` for extractor and preprocessing kwargs.

## Trained checkpoints

`tdn` and `tddn` need their checkpoints at:

```
shared_utils/feature_extraction/checkpoints/
├── vith_roberta_v3_coco_ft/ckpt/99/
└── fused_dinov3_cleandift_coco_ft/ckpt/{149,199}/
```

The `.distcp` shards are gitignored (~3.9 GB); place them yourself before
running `tdn` / `tddn`. Baselines and `ddn-cd` require no checkpoint download.

## Quantitative results

CSV schemas in `quantitative/{global,patch}/results/`:

- `*_quality.csv`: `representation, uniformity, effective_rank`
- `*_similarity.csv`: `pair, linear_cka, pwcca`

The CSVs are committed with the published numbers; editing them and re-running
`python run.py plots` refreshes the figures.
