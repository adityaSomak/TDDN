# Segmentation

Linear-probe semantic segmentation on the puzzle-perception 30-class
unified label space (see
[`datasets/Puzzle_Perception/Segmentation/classes.yaml`](../../datasets/Puzzle_Perception/Segmentation/classes.yaml)).
A small head is trained on top of *frozen* backbone features. Headline
metric: **weighted mIoU**.

## Supported models

11 model tags configured in [`configs/models.yaml`](configs/models.yaml):
`dinov2-vitb`, `dinov2-vitg`, `sd-2.1`, `dinov3`, `cd`, `clip`,
`sd+dinov2-vitb`, `sd+dinov2-vitg`, `ddn`, `tdn`, `tddn`.

## Setup

```bash
export HF_TOKEN=<your_token>     # gated DINOv3 / RoBERTa weights
```

Place the puzzle-perception segmentation tree (with
`{train,val,test}/{images,masks}/` and `manifest.csv`) at
`datasets/Puzzle_Perception/Segmentation/data/`. Trained-head tags
(`tdn`, `tddn`) additionally need the alignment checkpoints under
`experiments/shared_utils/feature_extraction/checkpoints/`.

## Run

```bash
python run_train.py --model <tag>                              # train one head
python run_train.py --model <tag> --out-root /tmp/smoke        # sandbox run
python run_eval.py --model <tag|all> --checkpoint <path>       # evaluate
python run_eval.py --model ddn --split val --limit 5           # smoke test
```

Training hyperparameters (loss, optimizer, schedule, precision) live
in [`configs/training.yaml`](configs/training.yaml) and apply
uniformly across all tags. Only the head is trained — backbones stay
frozen.

For backbones with a `pca:` block in `configs/models.yaml`
(currently `cd`, `sd-2.1`, and the fusions that include them), a
per-layer PCA basis is fit on a balanced training subsample at the
start of `run_train.py` and persisted alongside the checkpoint as
`pca.pt`. `run_eval.py` loads the same basis automatically.

## Results

- [`evaluation/results/segmentation.csv`](evaluation/results/segmentation.csv) —
  headline table (one row per model: `model, mIoU, pixel_acc`).
- `evaluation/results/<model_tag>.json` — per-class IoU + timings when
  produced.
