# ImageNet Classification

k-NN classification on ImageNet-1K from frozen image-encoder features.
For each model tag, the eval extracts one pooled, L2-normalized vector
per image, then runs cosine k-NN (k=20) against a balanced
**100-images-per-class gallery (100K train)** and scores **top-1 / top-5
on the 50K validation split**. Headline metric: top-1 accuracy.

## Setup

```bash
export HF_TOKEN=<your_token>     # gated DINOv3 / RoBERTa weights
```

Place the ImageNet-1K HuggingFace Arrow cache at
`datasets/Existing_Datasets/Classification/ImageNet-1K/imagenet_hf/`
(populated by `python datasets/download_datasets.py --dataset imagenet`).

Trained-head tags need their checkpoints under
`experiments/shared_utils/feature_extraction/checkpoints/`:

- `tdn` → `vith_roberta_v3_coco_ft/ckpt/99/`
- `tddn` → `fused_dinov3_cleandift_coco_ft/ckpt/{149, 199}/` (averaged)

## Supported models

11 model tags configured in [`configs/models.yaml`](configs/models.yaml):
`dinov2-vitb`, `dinov2-vitg`, `sd-2.1`, `dinov3`, `cd`, `clip`,
`sd+dinov2-vitb`, `sd+dinov2-vitg`, `ddn`, `tdn`, `tddn`.

## Run

```bash
python run_eval.py --model <tag|all>                          # full eval
python run_eval.py --model dinov3 --val-subset 200 --per-class-train 10
                                                              # smoke test (~30s)
```

| Arg | Default | Meaning |
|---|---|---|
| `--model` | _required_ | model tag from `configs/models.yaml`, or `all` for a sweep |
| `--k` | `20` (from config) | k-NN neighbour count |
| `--per-class-train` | `100` (from config) | per-class gallery size; 100×1000 → 100K train |
| `--val-subset` | `None` (full 50K) | cap validation samples for smoke runs |
| `--batch-size` / `--num-workers` | `64` / `4` | DataLoader knobs |
| `--cache-features` / `--use-cached` | `False` | persist pooled features to `features/` and reuse on next run |
| `--device` | `cuda` if available, else `cpu` | torch device |

## Results

- [`evaluation/results/imagenet_classification.csv`](evaluation/results/imagenet_classification.csv) —
  one row per model with columns
  `model, top1, top5, dim, k, n_train, n_val`. `top1` is the headline
  metric; `top5` is reported in the same row.
