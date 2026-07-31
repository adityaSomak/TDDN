# Segmentation

Linear-probe semantic segmentation on the Puzzle-Perception 30-class
unified label space (see
[`datasets/Puzzle_Perception/Segmentation/classes.yaml`](../../datasets/Puzzle_Perception/Segmentation/classes.yaml)).
A small head (1×1 conv by default) is trained on top of **frozen**
backbone features; the backbone never receives gradients. Headline
metric: **weighted mIoU** on the test split.

## Setup

```bash
export HF_TOKEN=<your_token>     # gated DINOv3 / RoBERTa weights
```

Place the dataset at
`datasets/Puzzle_Perception/Segmentation/data/` — populated by
`python datasets/download_datasets.py --dataset puzzle_perception`
(see [the datasets README](../../datasets/README.md)).

Trained-head tags need their alignment checkpoints under
`experiments/shared_utils/feature_extraction/checkpoints/`:

- `tdn` → `vith_roberta_v3_coco_ft/ckpt/tdn/`
- `tddn` → `fused_dinov3_cleandift_coco_ft/ckpt/tddn/`

## Supported models

11 model tags configured in [`configs/models.yaml`](configs/models.yaml):
`dinov2-vitb`, `dinov2-vitg`, `sd-2.1`, `dinov3`, `cd`, `clip`,
`sd+dinov2-vitb`, `sd+dinov2-vitg`, `ddn`, `tdn`, `tddn`.

## Train

```bash
python run_train.py --model ddn                                # full run
python run_train.py --model ddn --max-epochs 1                 # smoke (~1 min)
python run_train.py --model ddn --out-root /tmp/smoke          # sandbox outputs
```

| Arg | Default | Meaning |
|---|---|---|
| `--model` | _required_ | model tag from `configs/models.yaml` |
| `--config` | `configs/training.yaml` | training-hyperparameter YAML (loss / optimizer / schedule / precision; uniform across all tags) |
| `--seg-root` | `datasets/Puzzle_Perception/Segmentation/data` | path to the unpacked Puzzle-Perception dataset |
| `--out-root` | `experiments/Segmentation/training/` | parent dir for `checkpoints/<tag>/best.ckpt` and `logs/<tag>/` |
| `--max-epochs` | from `training.yaml:optim.epochs` | override the epoch budget — set `1` for a fast smoke run |
| `--device` | `cuda` if available, else `cpu` | torch device |
| `--devices` | `1` | Lightning `Trainer(devices=...)`. Pass `4` to DDP across 4 GPUs, or `auto` to use everything CUDA-visible. |

Outputs land at:
- `<out-root>/checkpoints/<tag>/best.ckpt` — Lightning checkpoint of the best-val-mIoU head.
- `<out-root>/checkpoints/<tag>/pca.pt` — fitted per-layer PCA bases (only for backbones that use diffusion features; see below).
- `<out-root>/logs/<tag>/` — Lightning's CSV + scalar logs.

### PCA for diffusion features

Backbones whose features come from diffusion UNet hooks (`cd`,
`sd-2.1`, and the fusions `ddn`, `sd+dinov2-vitb`, `sd+dinov2-vitg`
that include them) fit a per-layer Global PCA on a balanced training
subsample before the first training step. The fitted bases are saved
as `pca.pt` alongside the checkpoint and reloaded automatically by
`run_eval.py`. No per-backbone code changes needed — the PCA fit is
driven by the `pca:` block on `cd` / `sd-2.1` in
[`configs/models.yaml`](configs/models.yaml).

## Eval

```bash
python run_eval.py --model ddn                                 # uses training/checkpoints/ddn/best.ckpt
python run_eval.py --model all                                 # sweep tags whose ckpts exist
python run_eval.py --model ddn --split val --limit 5           # smoke test
```

| Arg | Default | Meaning |
|---|---|---|
| `--model` | _required_ | model tag, or `all` to sweep every tag whose ckpt exists |
| `--checkpoint` | `<out-root>/checkpoints/<tag>/best.ckpt` | Lightning `.ckpt` to evaluate; only needed when you want a non-default path |
| `--seg-root` | `datasets/Puzzle_Perception/Segmentation/data` | path to the dataset (must match training) |
| `--split` | `test` | `train` / `val` / `test` |
| `--limit` | _none_ | cap dataset size for smoke runs |
| `--device` | `cuda` if available, else `cpu` | torch device |

## Results

- [`evaluation/results/segmentation.csv`](evaluation/results/segmentation.csv) —
  headline table, one row per model. Columns: `model, mIoU, pixel_acc`.
  Only `mIoU` is currently populated in the committed CSV; `pixel_acc`
  is reserved in the schema but blank.
- `evaluation/results/<model_tag>.json` — per-class IoU + timings,
  written by `run_eval.py` (no per-model JSON ships in the repo;
  produced on demand).
