# Puzzle Understanding — VLM evaluation

Vision-language model evaluations on the PVQA datasets under
[`datasets/`](../../datasets/). Two eval families:

1. **Full eval** — Q1..QN structured questions over four AlgoPuzzleVQA tasks
   (`checker_move`, `maze_solve`, `nqueens`, `wood_slide`).
2. **Seg eval** — "reconstruct the board" prompts over `maze`, `nqueens`,
   and `chess`, with three image-input modes: `raw`, `oracle_mask`,
   `tddn_mask`.

## Layout

```
Puzzle_Understanding/
├── README.md
├── run_full_eval.py                # Q1..QN runner + --analyze reporter
├── run_seg_eval.py                 # raw/oracle/tddn runner + --analyze reporter
├── utils.py                        # OpenAI + vLLM async dispatch, image encoding
├── prompts/                        # per-puzzle prompt registry + DATASETS_ROOT
├── scripts/
│   └── launch_vllm.sh              # spawn N vLLM servers for open-source models
└── results/                        # populated; --analyze reads from here
    ├── full_eval/<task>/<model>/<reasoning>/{results.jsonl, report.txt}
    └── seg_eval/<model>[_<reasoning>]_<mode>.json
```

## Environments

Two separate venvs — the eval runner uses the repo-wide
[`requirements.txt`](../../requirements.txt); the vLLM server has a
heavy CUDA dependency tree and uses
[`requirements_vllm.txt`](../../requirements_vllm.txt).

```bash
# eval runner (OpenAI calls, parsing, analysis)
python -m venv .eval_env
source .eval_env/bin/activate
pip install -r ../../requirements.txt

# vLLM server (only needed for open-source models)
python -m venv .vllm_env
source .vllm_env/bin/activate
pip install -r ../../requirements_vllm.txt
```

For OpenAI runs, export the API key:

```bash
export OPENAI_API_KEY=sk-...
```

The eval scripts find datasets at `../../datasets/` by default. Override
with `EXPERIMENTS_DATASETS_ROOT=/path/to/datasets` if your data lives
elsewhere.

## Quick start

### Full eval — OpenAI

```bash
python run_full_eval.py --task maze_solve --backend openai --model gpt-4.1-2025-04-14
```

Reasoning models accept an effort flag:

```bash
python run_full_eval.py --task maze_solve --backend openai \
    --model gpt-5.1-2025-11-13 --reasoning low
```

### Seg eval — OpenAI

```bash
python run_seg_eval.py --backend openai --model gpt-4.1-2025-04-14 \
    --mode oracle_mask --tasks maze nqueens chess_count chess_grid
```

Modes:
- `raw` — baseline, raw image only
- `oracle_mask` — raw + image with ground-truth halo overlay
- `tddn_mask` — raw + image with predicted (TDDN) overlay

The `oracle_mask` and `tddn_mask` modes both fire a raw run first so the
analyzer can compute the `Img` baseline.

### Full eval — open-source via vLLM

```bash
# 1. Launch 4 vLLM servers on GPUs 0..3 (default: ports 8001..8004)
scripts/launch_vllm.sh OpenGVLab/InternVL3-8B 4 1

# 2. Wait for "Application startup complete" to appear in the logs
tail -f /tmp/vllm_logs/vllm_8001.log

# 3. Run the eval against them
python run_full_eval.py --task maze_solve --backend vllm --model InternVL3-8B
```

### Analyze (no API calls)

```bash
# Full-eval accuracy across all runs (all tasks, all models)
python run_full_eval.py --analyze

# Filter to one task or one model
python run_full_eval.py --analyze --task maze_solve
python run_full_eval.py --analyze --model-filter gpt-5

# Seg-eval Img/dM/dPM/dPMM delta table
python run_seg_eval.py --analyze

# Optionally render LaTeX rows for paper-style tables
python run_seg_eval.py --analyze --latex
```

## What gets evaluated

| Task | Eval type | Questions | Records | Source |
|---|---|---:|---:|---|
| Checker Move | Full | Q1–Q13 | 4,700  | `datasets/Existing_Datasets/PVQA/AlgoPuzzleVQA/checker_move/` |
| Maze Solve   | Full | Q1–Q15 | 1,500  | `datasets/Existing_Datasets/PVQA/AlgoPuzzleVQA/maze_solve/`   |
| N-Queens     | Full | Q1–Q7  |   700  | `datasets/Existing_Datasets/PVQA/AlgoPuzzleVQA/nqueens/`      |
| Wood Slide   | Full | Q1–Q5  | 2,200  | `datasets/Existing_Datasets/PVQA/AlgoPuzzleVQA/wood_slide/`   |
| Maze         | Seg  | grid reconstruction | 100  | `datasets/Existing_Datasets/PVQA/AlgoPuzzleVQA_star/maze/`    |
| N-Queens     | Seg  | grid reconstruction | 100  | `datasets/Existing_Datasets/PVQA/AlgoPuzzleVQA_star/nqueens/` |
| Chess        | Seg  | piece-count + grid  | 269  | `datasets/Puzzle_Perception/PVQA/test/chess/`                |

Question banks are in the `prompts/<task>.py` modules; the seg-eval
overlays live alongside each task's `seg_data/` directory under
`datasets/`.

## Output layout

Both runners are resume-safe. Existing rows in `results.jsonl` (full eval)
or the per-(model, mode) JSON (seg eval) are kept; only missing
records get re-queried.

```
results/full_eval/<task>/<model>/<reasoning>/
    results.jsonl    # one JSON line per (puzzle_id, question_id)
    report.txt       # overall + per-question accuracy summary

results/seg_eval/
    <model>_raw.json
    <model>_oracle_mask.json
    <model>_tddn_mask.json
    # with optional <reasoning> infix for GPT-5.x models:
    gpt-5.1-2025-11-13_low_oracle_mask.json
```

## Models known to work

| Backend | Models |
|---|---|
| `openai` | `gpt-4.1-2025-04-14`, `gpt-5.1-2025-11-13`, `gpt-5.4-2026-03-05` |
| `vllm`   | `OpenGVLab/InternVL3-{8B, 78B}`, `Qwen/Qwen2.5-VL-{7B, 72B}-Instruct`, `Qwen/Qwen3.6-{27B, 35B-A3B}`, `google/gemma-3-27b-it`, `google/gemma-4-{26B-A4B, 31B}-it`, `llava-hf/llava-v1.6-mistral-7b-hf`, `mistralai/Mistral-Small-3.1-24B-Instruct-2503` |

Any chat-completions-compatible model should work; the special-case
branches in `utils.py` (Qwen3.6 "thinking" disable, GPT-5.x
`reasoning_effort`) are the only model-specific quirks.
