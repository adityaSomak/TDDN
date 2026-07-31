# CRG — Contrastive Region Guidance

Tests whether **TDDN**-predicted regions can replace ground-truth regions as the
negative stream of a training-free, decode-side intervention on a *frozen* VLM. A
multiple-choice perception question is answered in one forward pass: the prompt ends
with a format instruction, so the model's first answer token is the decision, and we
read the next-token logits at that position restricted to the option tokens. CRG runs
two streams — the full image (*positive*) and the same image with the queried region
blacked out (*negative*) — and combines them as
`logits = (1 + α)·logit_pos − α·logit_neg` (α = 1.0). Three arms: `raw` (no negative),
`oracle` (GT cells), `tddn` (predicted cells, the deployable one). Two puzzle tasks,
8 frozen VLMs. **Headline metric: Δ question accuracy vs. the raw-image baseline,
macro over questions (`Δ_o` oracle, `Δ_t` TDDN); `Δ_t ≈ Δ_o` is the claim.**

The gain is governed by headroom: CRG helps where the raw model is not already
saturated, and is flat or slightly negative at the ceiling — which is what the
horizontal rule in the paper table separates.

## Setup

```bash
export HF_TOKEN=<your_token>     # gated Gemma / Qwen VLM weights
pip install -r ../../requirements.txt
```

One environment covers everything. Evaluation needs only a VLM and the committed
dataset — no segmentation stack, no DINOv3, no checkpoints — because the TDDN
detections are shipped with the data.

### Data

Committed under [`datasets/Puzzle_Perception/PVQA/`](../../datasets/Puzzle_Perception/PVQA/),
nothing to download:

| Task | Boards | Questions | Rows |
|---|---|---|---|
| `nqueens` | 100 (8×8–11×11) | 4 coordinate-free probes | 400 |
| `chess` | 800 generated (8×100) | 8 piece / relation probes | 800 |

Each task folder holds `images/`, `questions.yaml`, `answers.csv` and
`tddn_detections.json`. See its [dataset card](../../datasets/Puzzle_Perception/PVQA/README.md)
for the schemas.

### Optional: regenerating regions or boards

Only `--redetect`, `--validate-tddn` and `run_generate.py` need more than the above.
They require `DINOV3_ROOT=/path/to/dinov3` (or `pip install -e <path>/dinov3`), the
TDDN checkpoints under `experiments/shared_utils/feature_extraction/checkpoints/`:

- `tddn` → `fused_dinov3_cleandift_coco_ft/ckpt/tddn/`

and, for chess, the locally-supplied 269-board segmentation set described in
[`datasets/_local/README.md`](../../datasets/_local/README.md).

## Supported models

Eight frozen VLMs, registered in [`configs/models.yaml`](configs/models.yaml) with
their HF id and the launch flags each needed on 2×48 GB:

| Tag | HF id | Launch notes |
|---|---|---|
| `qwen2.5-vl-7b` | `Qwen/Qwen2.5-VL-7B-Instruct` | single GPU, bs 16/8 |
| `internvl3-8b` | `OpenGVLab/InternVL3-8B-hf` | `no_think`, 2-GPU split |
| `gemma-4-12b` | `google/gemma-4-12b-it` | `no_think`, 2-GPU split |
| `gemma-4-26b-a4b` | `google/gemma-4-26B-A4B-it` | `no_think`, 2-GPU split |
| `gemma-3-27b` | `google/gemma-3-27b-it` | `no_think`, 2-GPU split |
| `qwen3.6-27b` | `Qwen/Qwen3.6-27B` | `no_think`, 2-GPU split |
| `gemma-4-31b` | `google/gemma-4-31B-it` | `no_think`, 2-GPU split |
| `qwen3.6-35b-a3b` | `Qwen/Qwen3.6-35B-A3B` | `no_think`, 2-GPU split |

Reasoning-default models need `no_think` so the *first* generated token is the answer
— the single-forward decode has no generation loop to skip a `<think>` block with.
Per-model defaults come from the registry; CLI flags override them.

## Run

```bash
python run_eval.py --task nqueens --model qwen2.5-vl-7b            # 3 arms, 100 boards
python run_eval.py --task chess   --model qwen3.6-27b --arm tddn   # deployable arm only
python run_eval.py --task chess   --model all --publish            # the 8-model paper sweep
python run_eval.py --task nqueens --model gemma-4-12b --limit 8    # smoke test
python run_eval.py --aggregate                                     # crg.csv + crg_table.tex
```

| Arg | Default | Meaning |
|---|---|---|
| `--task` | _required_ (except `--aggregate`) | `nqueens` or `chess` |
| `--model` | _required_ | a registry tag, or `all` for the 8-model paper sweep |
| `--arm` | all three | repeatable; `raw` is always included as the baseline |
| `--alpha` | `1.0` | CRG guidance strength |
| `--batch-size` | per-model | overrides the registry |
| `--limit` | all rows | truncate the row list (thins every question) |
| `--seeds` | `[0]` | N-Queens determinism check |
| `--no-think` / `--load-4bit` / `--max-memory` | per-model | override the registry |
| `--publish` | off | write the committed result file instead of `_live/` |
| `--aggregate` | — | rebuild `crg.csv` + `crg_table.tex` and exit |
| `--validate-dataset` | — | check dataset integrity and exit |
| `--redetect` | off | re-run TDDN instead of reading cached detections (needs DINOv3) |
| `--validate-tddn` | — | (chess) score TDDN piece detection vs GT and exit |

Runs go to `evaluation/results/<task>/_live/` unless `--publish` is given, so a smoke
test cannot overwrite a paper number. Every run validates the dataset first — a
drifted dataset invalidates any number measured against it.

## Negatives and detections

Negatives are **never stored**. They are a deterministic `black(image, cells)` built in
memory per item ([`evaluation/src/negatives.py`](evaluation/src/negatives.py)), where
the cells come from `answers.csv` (oracle) or from the cached TDDN prediction map
(tddn). Two geometries, deliberately different: cell blackouts keep a 5 px frame so
the grid lines survive and only the cell's *contents* are removed, while N-Queens TDDN
boxes are filled exactly, since the detector already padded them.

Only the *detections* are cached, in `tddn_detections.json` (~257 KB for both tasks
versus ~230 MB for the images they imply). This is what removes DINOv3 from the eval
path. `--redetect` recomputes them; `--redetect --save-detections` overwrites the
committed file.

## Results

Committed paper artifacts:

- [`evaluation/results/crg_table.tex`](evaluation/results/crg_table.tex) — the paper
  table, **derived**: `--aggregate` regenerates it byte-identically from the per-model
  JSONs below, so it needs no archive and no GPU.
- [`evaluation/results/crg.csv`](evaluation/results/crg.csv) — headline table, one row
  per (model, task). Columns: `model, display, task, n_boards, n_questions, raw_acc,
  d_oracle, d_tddn, d_diff, recovery_pct`.
- `evaluation/results/chess/<tag>.json` — keys `model, alpha, arms, per_question{qid →
  {cat, options, raw|oracle|tddn: [[image_id, pred, label], …]}}`. **Per-board
  records**, so accuracy and bootstrap CIs are re-derivable from scratch.
- `evaluation/results/nqueens/<tag>.json` — keys `model, alpha, no_think, n_boards,
  seeds, arms, per_question{qid → {binary, raw{auroc,acc,±ci}, oracle|tddn{auroc, acc,
  d_*, ±ci}}}, combined, seed_jitter_combined`. AUROC for the three binary probes,
  accuracy for the 3-way `q4`; CIs are 1000× board-level bootstrap on the paired delta.
- `evaluation/results/figures/` — 18 PNG + 18 PDF qualitative panels, six cases ×
  {original, gt, tddn}: `s1-relation`, `s2-adjacent`, `s3-queenquad`, `s4-exists`
  (successes) and `f1-tactical`, `f2-bishopquad` (failures). Their source boards are
  `relv_wK_bN_001`, `adj_wK_bK_006`, `quad_w_queen_009`, `exists_wQ_018`,
  `queen_line_011`, `quad_w_bishop_030`. Note the `gt` panels black the full 64 px
  cell while the `tddn` panels keep the eval's 5 px frame — they were produced by
  different code paths and are kept as published.
- `evaluation/results/chess/tddn_validation.json` — detector diagnostic from
  `--validate-tddn`: `n_boards, presence{P,R,F1}, localized_n, type_acc, color_acc,
  exact_acc, perclass_f1{}` over the real boards excluded from the support cache.

Never committed: `evaluation/results/<task>/_live/` (runs without `--publish`).

Two provenance notes. The eight paper JSONs were migrated from the pre-restructure
schema (each carries an `imported_from` key); a fresh sweep on different hardware may
differ in the last decimal. And the N-Queens files hold per-*question* metrics rather
than per-board records, because the old writer never emitted them — so their CIs are
frozen values, unlike chess where `--aggregate` recomputes them. A `--publish` re-run
fixes that asymmetry.

Two non-paper results are also kept. `llava-v1.6-vicuna-7b.json` (N-Queens only) is a
full 100-board, 3-seed CRG run on a model left out of the table: both `Δ_o` and `Δ_t`
are ~0 on accuracy, the "nothing to exploit and nothing to localize" end of the
headroom story. It is registered in `models.yaml` without a `paper_row`, so it appears
in `crg.csv` but not in the table, and is re-runnable like any other tag.

`lavida-llada.json` in both task folders records a one-off LaViDa diffusion-LM
baseline. It is **not reproducible from this repo**: LaViDa's `llava` package pins
`transformers==4.50.3`/`torch==2.6.0`, which cannot coexist with the `transformers>=5`
that Gemma-4 and Qwen3.6 need, so its engine was removed rather than maintain a second
environment. `--aggregate` skips it (no `arms` key).

## Files

```
CRG/
├── README.md                     (this file)
├── run_eval.py                   eval + --aggregate / --validate-* / --redetect
├── run_generate.py               mint a NEW chess board set (rare, destructive)
├── configs/models.yaml           8 VLM tags + defaults (alpha, arms, TDDN tuning)
└── evaluation/
    ├── src/
    │   ├── config.py             paths (committed vs locally-supplied), prompts
    │   ├── data.py               questions/answers/detections/boards + NQ rule registry
    │   ├── negatives.py          on-the-fly blackout builders
    │   ├── decode_engine.py      the CRG two-stream single-forward decode
    │   ├── metrics.py            AUROC, accuracy, board-level bootstrap CIs
    │   ├── nqueens_task.py       N-Queens eval (AUROC + CIs + seed jitter)
    │   ├── chess_task.py         chess eval (per-board records)
    │   ├── aggregate.py          crg.csv + crg_table.tex
    │   ├── tddn.py               detectors + diagnostic (only --redetect/--validate-tddn)
    │   └── chess_generate.py     renderer + board generators (only run_generate.py)
    └── results/                  see above
```

Statistics: board-level bootstrap (1000×) 95% CIs on the paired `raw`-vs-CRG deltas.
Each arm is a single deterministic greedy forward, so the only sampling variation is
over the finite board set; the seed loop measures VLM forward jitter alone, since the
region a negative blacks is fixed by the dataset.
