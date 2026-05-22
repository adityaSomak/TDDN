# Keypoint_Matching

PCK@α evaluation on SPair-71K. Headline metric: **PCK@0.1 (bbox)**.

## Supported models

11 model tags configured in [`configs/models.yaml`](configs/models.yaml):
`dinov2-vitb`, `dinov2-vitg`, `sd-2.1`, `dinov3`, `cd`, `clip`,
`sd+dinov2-vitb`, `sd+dinov2-vitg`, `ddn`, `tdn`, `tddn`.

## Run

```bash
python run_eval.py --model <tag|all>            # full SPair test
python run_eval.py --model <tag> --n-per-cat 20 # subsample 20 pairs/cat
python run_eval.py --model cd --layer-ablation  # per-layer breakdown
```

Key args: `--model`, `--n-per-cat`, `--categories`, `--limit`,
`--layer-ablation`, `--spair-root`, `--device`.

## Results

- [`evaluation/results/keypoint_matching.csv`](evaluation/results/keypoint_matching.csv) —
  headline table (one row per model).
- `evaluation/results/<model_tag>.json` — per-category PCK breakdown.
- `evaluation/results/layers/*.json` — CD/SD per-layer and DINOv3 facet
  ablations.
