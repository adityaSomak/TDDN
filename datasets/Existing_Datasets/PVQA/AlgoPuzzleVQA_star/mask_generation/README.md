# Mask generation

Scripts that regenerate the segmentation overlays shipped under
`AlgoPuzzleVQA_star/<task>/seg_data/`. Two overlay families are produced:

- `oracle_mask/` — deterministic, derived from the ground-truth grid in the
  task's CSV. No model required.
- `tddn_mask/` — predicted overlays produced by the TDDN tip-adapter
  pipeline; requires a trained vision/text alignment model.

## Layout

```
mask_generation/
├── build_oracle_masks.sh    # entry point: oracle overlays
├── build_tddn_mask.sh       # entry point: TDDN overlays
└── src/
    └── overlays/
        ├── mask_utils.py    # parse_grid, wall_mask_from_grey, dilated_outline
        ├── oracle_mask.py   # deterministic overlays from CSV ground truth
        ├── tddn_mask.py     # tip-adapter prediction + overlay
        ├── tddn_loader.py   # loads the trained alignment model
        └── tddn_prompts.py  # per-class natural-language prompts
```

Outputs land in `../<task>/seg_data/oracle_mask/` and `../<task>/seg_data/tddn_mask/`.

## Usage

```bash
# Oracle overlays for maze + nqueens (no GPU, no model needed)
./build_oracle_masks.sh

# Limit to a few IDs while iterating
./build_oracle_masks.sh --limit 5 --ids 0029 0056

# TDDN overlays (requires GPU + trained alignment model on PYTHONPATH)
./build_tddn_mask.sh --alpha 2.0 --beta 5.5

# One puzzle at a time
PUZZLES="nqueens" ./build_oracle_masks.sh
```

The `DATASET_ROOT` env var overrides the implicit
`AlgoPuzzleVQA_star/` root if you want to point the scripts at a different
copy of the data:

```bash
DATASET_ROOT=/path/to/AlgoPuzzleVQA_star ./build_oracle_masks.sh
```

## TDDN dependencies

`tddn_mask.py` depends on a trained vision/text alignment model that is
**not** shipped in this repository. The loader (`tddn_loader.py`) reads
three required environment variables:

| Variable           | Purpose                                                                |
|--------------------|------------------------------------------------------------------------|
| `PUZZLEBENCH_ROOT` | Path to `PuzzleBench/text_alignment/` (source for `core.*`, `eval.*`)  |
| `DINOV3_ROOT`      | Path to the `dinov3/` source tree (vision backbone)                    |
| `ALIGNMENT_CKPT`   | Training-output dir containing `config.yaml` and `ckpt/<step>/` folders|

Optional overrides:

| Variable                | Default     | Purpose                                                |
|-------------------------|-------------|--------------------------------------------------------|
| `ALIGNMENT_CKPT_STEPS`  | `tddn`      | Checkpoint name under `ckpt/` to load, or two comma-separated names to weight-average |
| `HF_HOME`               | system default | HuggingFace cache directory                         |

Example:

```bash
export PUZZLEBENCH_ROOT=/path/to/PuzzleBench/text_alignment
export DINOV3_ROOT=/path/to/dinov3
export ALIGNMENT_CKPT=/path/to/output/<run_name>
./build_tddn_mask.sh --alpha 2.0 --beta 5.5
```

See the PuzzleBench project's text-alignment pipeline for training the
alignment model and producing the checkpoint directory.

## Chess

The original pipeline also produces chess overlays, but the chess data is
not part of `AlgoPuzzleVQA_star/` — and it is not committed anywhere in the
repo either (see [`datasets/_local/README.md`](../../../../_local/README.md)).
Supply it locally, then point `CHESS_DATASET` at that root (it must contain
`images/`, `masks/`, `text_repr.json`) and add `chess` to `PUZZLES`:

```bash
CHESS_DATASET=../../../../_local/chess_seg269/data PUZZLES="chess" ./build_oracle_masks.sh
```
