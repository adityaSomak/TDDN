# PVQA — CRG perception-probe dataset

Multiple-choice visual perception probes over puzzle boards, built for the
[CRG experiment](../../../experiments/CRG/README.md) (Contrastive Region Guidance):
each question references a piece *without naming its coordinates*, so a model must
locate the piece before it can answer. That is what makes region guidance able to help,
and what makes the queried region well defined enough to black out.

Fully committed — no download step.

```
PVQA/
├── chess/
│   ├── questions.yaml            8 probes (eval spec + generation spec)
│   ├── answers.csv               800 rows
│   ├── tddn_detections.json      cached TDDN 8x8 class maps, 800 boards (~187 KB)
│   └── images/<image_id>.png     800 generated boards, 512x512 (~72 MB)
└── nqueens/
    ├── questions.yaml            4 probes
    ├── answers.csv               400 rows (100 boards x 4 questions)
    ├── tddn_detections.json      cached TDDN queen boxes, 100 boards (~71 KB)
    └── images/<image_id>.jpg     100 source boards, 1600x1600 (~12 MB)
```

| Task | Boards | Board size | Questions | Rows | Origin |
|---|---|---|---|---|---|
| `chess` | 800 | 8×8, 512 px | 8 (`q1`–`q8`), 100 boards each | 800 | synthesised for this benchmark |
| `nqueens` | 100 | 8×8 – 11×11 | 4 (`q1`–`q4`), all boards each | 400 | AlgoPuzzleVQA* |

Board images are committed **unconverted**, which is why the extension differs by task
(generated PNG vs source JPEG). N-Queens boards are downscaled to 512 px at load time,
not on disk. Consumers should resolve a path as `images/<image_id>` plus the per-task
extension — `evaluation/src/config.py:BOARD_EXT` is the reference.

## `answers.csv`

One long-format schema for both tasks, keyed by `(image_id, qid)`:

| Column | Type | Meaning |
|---|---|---|
| `image_id` | str | stem of the board under `images/` |
| `qid` | str | key into `questions.yaml` |
| `answer` | int | **0-based index** into that question's `options` list |
| `ablate_cells` | JSON | `[[row, col], …]` cells the oracle negative blacks; may be `[]` |
| `board_rows`, `board_cols` | int | grid dimensions |
| `board` | str | ground-truth grid: rows `/`-joined, cells `\|`-joined |

Two things worth knowing before consuming it:

- **`board_rows`/`board_cols` are load-bearing, not decoration.** Chess is always 8×8,
  but N-Queens runs 8×8 through 11×11 (5 / 18 / 30 / 47 boards) — anything assuming
  8×8 is wrong for 95 of the 100 boards.
- **`ablate_cells` can legitimately be empty.** 50 chess rows (all `q8`, "is there a
  white queen?", answer *No*) have no queen to ablate, so the oracle negative equals
  the positive and CRG reduces to the raw logits there. That is the correct behaviour
  for an existence question, not missing data.

Chess cell tokens are the 14-token vocabulary in
`experiments/CRG/evaluation/src/config.py:CHESS_TOKEN2ID` (`white_sq`, `black_sq`,
`w_pawn` … `b_king`); N-Queens cells are `Q` or `0`.

The `/` + `|` grid encoding is used because it is the only CSV-safe option — it embeds
in a single field with no quoted newlines, unlike the newline-delimited encodings the
older PVQA files use.

## `questions.yaml`

A flat map of `qid → spec`. **Every top-level key is treated as a question id**, so the
file carries no metadata block; version notes live in comments.

Shared fields are `text`, `options` (the `answer` column indexes into this) and
`category`. Beyond that:

- **chess** adds `ablate_pieces` (the classes the TDDN negative blacks), an optional
  `extreme: [kind, piece]` for superlative questions (black only the single extreme
  instance), and the generation spec `slug` / `generator` / `params`. `slug` is
  load-bearing: board ids are `{slug}_{NNN}`, and both `answers.csv` and every archived
  per-board result record key on those exact ids.
- **nqueens** adds `rule` (`half` / `third` / `rel`), `role` or `roles`, and `axis`.
  These are the derivation that produced `answer` and `ablate_cells`; the CSV is
  authoritative at eval time and the rules are re-run only as a self-check.

## `tddn_detections.json`

Cached TDDN segmenter output, keyed by `image_id`:

- **chess** — an 8×8 predicted class-id map per board.
- **nqueens** — a list of `{cy, cx, box: [fy0, fy1, fx0, fx1]}` fractional queen
  detections per board. Fractional, so they apply at any render scale.

These are the exact detections behind the published `Δ_t` numbers. Shipping them
(~257 KB) instead of the negative images they imply (~230 MB) is what lets the
deployable TDDN arm be reproduced with no GPU, no DINOv3 and no checkpoints.

**Negative images are not shipped.** They are rebuilt in memory per item from the board
plus either `ablate_cells` (oracle) or these detections (tddn) — see
`experiments/CRG/evaluation/src/negatives.py`.

## Validation

```bash
python experiments/CRG/run_eval.py --validate-dataset
```

Checks that every row resolves to a board and a detection entry, that there are no
orphan images, that answers index inside their option list and ablate cells inside the
declared dimensions, that each `board` parses to those dimensions and agrees across the
rows sharing an `image_id`, that N-Queens answers and ablate cells re-derive from the
board via the YAML rule fields, and that each chess ablate cell actually holds one of
its question's `ablate_pieces`. Every eval run does this first.

## Provenance

Chess boards are synthesised by `experiments/CRG/run_generate.py`, which composites
piece sprites and board colour themes extracted from the 269-board chess segmentation
set. That set is **not committed** (see [`../../_local/README.md`](../../_local/README.md)),
so regeneration needs it supplied locally — evaluation does not.

N-Queens boards and their ground-truth grids come from AlgoPuzzleVQA*
(`datasets/Existing_Datasets/PVQA/AlgoPuzzleVQA_star/nqueens/`); the 100-board subset
and its CRG question set were selected for this benchmark and are self-contained here,
so nothing in the CRG path reads back into `Existing_Datasets/`.

## License

The N-Queens boards inherit AlgoPuzzleVQA's license. The chess boards, both question
sets, the answer CSVs and the cached detections are released under the repository's
root license.
