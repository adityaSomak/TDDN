# AlgoPuzzleVQA

VLM evaluation pipeline for six algorithmic puzzles, with a three-mode
segmentation-overlay experiment (`raw` baseline plus `oracle_mask` and
`tddn_mask` overlays) on three of them.

See [TASKS.md](TASKS.md) for the puzzle descriptions and the full Q1..QN
question banks each evaluation uses.

## Layout

```
AlgoPuzzleVQA/
├── checker_move/   maze_solve/   nqueens/   wood_slide/    Q1..QN full evals
├── water_jugs/     tower_of_hanoi/                          data only
├── seg_data/{maze,nqueens,chess}/{oracle_mask,tddn_mask}/   generated overlays
├── seg_eval_results/                                        seg-eval JSON outputs
├── src/
│   ├── overlays/    mask_utils.py, oracle_mask.py, tddn_loader.py, tddn_mask.py
│   ├── eval/        backends.py, prompts/<puzzle>.py,
│   │                run_seg_eval.py, run_full_eval.py
│   └── analysis/    seg_delta.py, full_eval_report.py
└── scripts/         launch_vllm.sh, build_oracle_masks.sh, build_tddn_mask.sh
```

## Environment

Use `/data/shanmukha/puzzlebench_venv/.venv` (has scipy, omegaconf, aiohttp,
openai, and the editable `dinov3` install). The `vllm_env` is only for
serving open-source models.

```bash
PY=/data/shanmukha/puzzlebench_venv/.venv/bin/python
```

For OpenAI runs, export `OPENAI_API_KEY` first.

### Prerequisite for TDDN

`src/overlays/tddn_mask.py` (and any seg-eval run with `--mode tddn_mask`)
requires the **`dinov3` package installed in editable mode** before first
use. `puzzlebench_venv` already has it. If recreating the venv:

```bash
pip install -e /data/shanmukha/dinov3
```

Without this, `tddn_mask.py` fails on import with `ModuleNotFoundError:
dinov3` because `AlignmentModel`'s image encoder loads a DINOv3 ViT backbone.

## Usage

### Generate overlays

```bash
$PY -m src.overlays.oracle_mask --puzzle maze
$PY -m src.overlays.tddn_mask   --puzzle nqueens

scripts/build_oracle_masks.sh    # all three puzzles
scripts/build_tddn_mask.sh
```

### Run a seg eval

Three modes: `raw`, `oracle_mask`, `tddn_mask`. The two non-raw modes also
run a raw pass so `seg_delta.py` can form the Img baseline.

`--tasks` choices: `maze`, `nqueens`, `chess_count`, `chess_grid`. Pass any
subset.

```bash
# OpenAI
$PY -m src.eval.run_seg_eval --backend openai \
    --model gpt-4.1-2025-04-14 --mode oracle_mask --tasks maze nqueens

# vLLM: launch first, then wait for "Application startup complete" in
# /tmp/vllm_logs/vllm_<port>.log before running the eval.
scripts/launch_vllm.sh google/gemma-3-27b-it 4 1
$PY -m src.eval.run_seg_eval --backend vllm \
    --model google/gemma-3-27b-it --ports 8001 8002 8003 8004 \
    --mode tddn_mask --tasks chess_grid --concurrency 32
```

Output: `seg_eval_results/<model>_<mode>.json`. Use `--limit N` for a quick
smoke run.

### Run a full eval

`--task` choices: `checker_move`, `maze_solve`, `nqueens`, `wood_slide`
(one task per invocation).
`--reasoning {low,medium,high}` only applies to gpt-5.x reasoning models;
omit for everything else.

```bash
# OpenAI (reasoning model)
$PY -m src.eval.run_full_eval --task checker_move \
    --backend openai --model gpt-5.4-2026-03-05 --reasoning low

# vLLM (wait for the server to be ready first; see seg-eval note above).
$PY -m src.eval.run_full_eval --task wood_slide \
    --backend vllm --model google/gemma-3-27b-it \
    --ports 8001 8002 8003 8004 --concurrency 50
```

Output: `<task>/eval_results_low_detail/<model>/<reasoning>/{results.jsonl, report.txt}`.
Re-running skips records already in `results.jsonl`. Use `--limit N` for a
quick smoke run.

### Analysis

```bash
$PY -m src.analysis.seg_delta
$PY -m src.analysis.seg_delta --models gpt-5 gemma --latex

$PY -m src.analysis.full_eval_report
$PY -m src.analysis.full_eval_report --tasks maze_solve --by-question
```

## Delta definitions

```
Img    = (raw_M + raw_PM) / 2
dM     =  seg_M  - Img             # oracle-mask gain
dPM    =  seg_PM - Img             # TDDN-mask gain
dPMM   =  dPM - dM    ==  seg_PM - seg_M
```

M and PM denote the `_oracle_mask.json` and `_tddn_mask.json` result files.
Per-task metric: `cell` for maze, `f1` for nqueens, mean of black+white
piece F1 for chess.

## Notes

- `tddn_mask.py` requires checkpoints at
  `output/fused_dinov3_cleandift_coco_ft/ckpt/{149,199}` and the editable
  `dinov3` install at `/data/shanmukha/dinov3` (see Prerequisite above).
- Class prompts in each `prompts/<puzzle>.py` are single descriptive
  sentences per class. Existing pre-generated TDDN overlays were built with
  averaged CuPL prompts; running `scripts/build_tddn_mask.sh` overwrites
  them with the single-prompt variant.
- `water_jugs/` and `tower_of_hanoi/` ship as data only.
