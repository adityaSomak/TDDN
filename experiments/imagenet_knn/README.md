# imagenet_knn

k-NN classification on ImageNet-1K using image-encoder features.
Headline metric: **top-1 accuracy**.

## Setup

```bash
export HF_TOKEN=<your_token>     # gated DINOv3 / RoBERTa weights
```

Place the ImageNet-1K HuggingFace Arrow cache at
`datasets/Existing_Datasets/Classification/ImageNet-1K/imagenet_hf/`.

Trained-head tags (`tdn`, `tddn`) additionally need the checkpoints
under `experiments/shared_utils/feature_extraction/checkpoints/`.

## Supported models

11 model tags configured in [`configs/models.yaml`](configs/models.yaml):
`dinov2-vitb`, `dinov2-vitg`, `sd-2.1`, `dinov3`, `cd`, `clip`,
`sd+dinov2-vitb`, `sd+dinov2-vitg`, `ddn`, `tdn`, `tddn`.

## Run

```bash
python run_eval.py --model <tag|all>                          # full eval
python run_eval.py --model dinov3 --val-subset 200 --per-class-train 10
                                                              # smoke test
```

Key args: `--model`, `--per-class-train`, `--val-subset`, `--k`,
`--use-cached`, `--cache-features`, `--batch-size`, `--device`.

## Results

- [`evaluation/results/imagenet_knn.csv`](evaluation/results/imagenet_knn.csv) —
  headline table with columns
  `model, top1, top5, dim, k, n_train, n_val` (one row per model).
