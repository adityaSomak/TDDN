# Puzzle Tasks and Evaluation Questions

This document describes the six puzzles, what visual reasoning each one tests, and the exact
questions asked in each evaluation.

---

## Overview

| Puzzle | Eval type | Questions | Records | Seg-eval eligible |
|---|---|---|---|---|
| Checker Move | Full eval | Q1–Q13 | 4,700 | No |
| Maze Solve | Full eval + Seg eval | Q1–Q15 | 1,500 | Yes |
| N-Queens | Full eval + Seg eval | Q1–Q7 | 700 | Yes |
| Wood Slide | Full eval | Q1–Q5 | 2,200 | No |
| Chess | Seg eval only | — | — | Yes |
| Water Jugs | Data only | — | — | No |
| Tower of Hanoi | Data only | — | — | No |

**Full eval** — the model is shown a raw puzzle image and asked a sequence of structured questions
(Q1..QN). Each question is answered independently with its own image + question prompt.

**Seg eval** — the model is shown the same image overlaid with segmentation masks and asked to
reconstruct the full board/grid as a structured string. Three modes: `raw` (no overlay), `oracle_mask`
(ground-truth outlines), `tddn_mask` (model-predicted masks). Used to measure how much the overlay
helps or hurts perception.

---

## Full Eval Puzzles

### Checker Move

**What it is.** A horizontal row of cells containing red checkers, green checkers, and one empty
cell. The puzzle represents a single board state from the checker-move (Frog Puzzle) game, where
red and green pieces swap sides by jumping over each other.

**What it tests.** Counting by colour, positional reasoning (left/right of the empty cell),
ordinal position retrieval, and comparison between groups.

**Each puzzle_id has two images:** `checker_move_start_v2.jpg` (initial state) and
`checker_move_end_v2.jpg` (state after the legal move). Questions that reference position use
0-based indexing from the left.

| Q | Answer type | Question |
|---|---|---|
| Q1 | MCQ | How many checkers are there in total? |
| Q2 | MCQ | How many red / green checkers are there? |
| Q3 | boolean | Do the red checkers outnumber the green checkers? |
| Q4 | color | Which color has the higher number of checkers? |
| Q5 | number | How many red checkers are to the left of the empty cell? |
| Q6 | boolean | Is the number of checkers to the left of the empty cell greater than to the right? |
| Q7 | boolean | Are there an equal number of red and green checkers? |
| Q8 | number | How many cells are to the left of the empty cell? |
| Q9 | number | How many green checkers are between every pair of adjacent red checkers? |
| Q10 | color_list | Which color(s) are adjacent to the empty cell? |
| Q11 | position | Determine the position of the empty cell. |
| Q12 | position | What is the position of the first green checker? |
| Q13 | position_list | Determine the positions of the green checkers. |

Q1–Q2 are multiple-choice (4 options). Q3, Q6, Q7 expect `True` or `False`. All others are open-ended.
Questions Q5, Q8–Q13 also have start/end variants (applied to both images).

---

### Maze Solve

**What it is.** A grid-based maze of arbitrary size. Black cells are walls; white cells are
passable path. A green arrow marks the entry cell; a blue arrow marks the exit cell. The grid
always has an outer border of wall cells, which are counted in all dimensions and coordinates.

**What it tests.** Grid dimension estimation, coordinate localisation (entry/exit), global cell
counts, local neighbourhood reasoning (adjacency), and full cell enumeration. Questions progress
from local/easy (entry location) to global/hard (list of all wall cells).

Coordinates are 0-based. Row 0 is the topmost row; the entry cell is typically at `(1, 0)` because
row 0 is the top border wall.

| Q | Answer type | Question |
|---|---|---|
| Q1 | coordinate | Locate the entry cell (green arrow). |
| Q2 | coordinate | Locate the exit cell (blue arrow). |
| Q3 | dimensions | Determine the maze dimensions (rows × columns). |
| Q4 | number | Total number of empty white cells (including entry and exit). |
| Q5 | number | Total number of black wall cells. |
| Q6 | number | Number of empty white cells adjacent to the entry cell. |
| Q7 | number | Number of black wall cells adjacent to the entry cell. |
| Q8 | number | Number of empty white cells adjacent to the exit cell. |
| Q9 | number | Number of black wall cells adjacent to the exit cell. |
| Q10 | coordinate_list | Coordinates of empty white cells adjacent to the entry cell. |
| Q11 | coordinate_list | Coordinates of black wall cells adjacent to the entry cell. |
| Q12 | coordinate_list | Coordinates of empty white cells adjacent to the exit cell. |
| Q13 | coordinate_list | Coordinates of black wall cells adjacent to the exit cell. |
| Q14 | coordinate_list_long | Coordinates of all empty white cells. |
| Q15 | coordinate_list_long | Coordinates of all black wall cells. |

**Seg eval task.** The model reconstructs the full maze as a grid of semicolon-separated values
(`1` = wall, `0` = path, `S` = entry, `E` = exit). Accuracy is measured cell-by-cell (`cell` metric).

---

### N-Queens

**What it is.** A chess-style board of variable size (8×8 to 11×11) with queens placed as a valid
N-Queens solution. Each queen is annotated with a coloured circle and a numeric ID to aid
localisation.

**What it tests.** Board size estimation, queen localisation, row/column occupancy reasoning, and
full board enumeration presented in two different orderings (Q6 vs Q7 test whether answer order
affects accuracy).

Coordinates are 0-based. Row 0 is the topmost row.

| Q | Answer type | Question |
|---|---|---|
| Q1 | integer_list_rows | Rows without a queen. |
| Q2 | integer_list_cols | Columns without a queen. |
| Q3 | coordinate_list | Cells occupied by queens. |
| Q4 | integer_list_rows | Which rows are safe for insertion of a queen? *(same answer as Q1)* |
| Q5 | integer_list_cols | Which columns are safe for insertion of a queen? *(same answer as Q2)* |
| Q6 | cell_dict_empty_first | Location of empty cells, then occupied cells. |
| Q7 | cell_dict_occupied_first | Location of occupied cells, then empty cells. |

Q4/Q5 are semantic variants of Q1/Q2 — identical answer, different phrasing — included to measure
prompt-sensitivity. Q6/Q7 are full board enumerations in opposite key orders.

**Seg eval task.** The model reconstructs the full board as a grid (`Q` = queen, `0` = empty).
Accuracy is measured by piece F1 score (`f1` metric).

---

### Wood Slide

**What it is.** A Klotski-style 5×4 sliding block puzzle. The board contains wooden blocks of
four sizes (1×1, 1×2, 2×1, 2×2) and two empty cells. The puzzle always has a fixed block
composition: one 2×2, four 1×2, two 2×1, and two 1×1.

**What it tests.** Block detection, dimension classification (1×2 vs 2×1 is the hardest visual
distinction), spatial adjacency reasoning, and counting by block type.

Dimensions are written HEIGHT×WIDTH. Coordinates are 0-based `(row, col)`. Each puzzle_id has
two images (start and end board state); questions are asked against each independently.

| Q | Answer type | Question |
|---|---|---|
| Q1 | coordinate_list | Locations of empty cells. |
| Q2 | number | Count of 1×1 blocks. |
| Q3 | dimension_list | Dimensions of blocks adjacent to the empty cell(s). |
| Q4 | number | Count of blocks adjacent to the empty cell(s). |
| Q5 | boolean | Is any 1×1 block adjacent to an empty cell? |

---

## Seg Eval Puzzles

### Chess

**What it is.** An 8×8 chess board with a mid-game piece configuration. Pieces are standard
Staunton style; board colours are light pink and dark blue (non-standard, to test colour-invariant
piece recognition).

**What it tests.** Piece detection, colour discrimination (white vs black pieces), piece type
classification (six types per colour), and full board localisation.

Seg eval is the only eval type for chess — there is no full-eval JSONL.

| Sub-task | Output format | What it measures |
|---|---|---|
| `pieces` | `ANSWER: <integer>` | Total piece count (all colours combined). |
| `empty` | `ANSWER: <integer>` | Count of empty squares. |
| `grid_black` | 8×8 semicolon-separated grid | Per-square black piece type (1–6) or `-` (white piece) or `0` (empty). |
| `grid_white` | 8×8 semicolon-separated grid | Per-square white piece type (1–6) or `-` (black piece) or `0` (empty). |

Piece encoding: pawn=1, knight=2, bishop=3, rook=4, queen=5, king=6.
Accuracy for grid sub-tasks is measured by F1 score per piece type, averaged over black and white.

---

## Seg Eval Modes

All three seg-eligible puzzles (maze, nqueens, chess) are evaluated in three modes:

| Mode | Image | Annotation trust |
|---|---|---|
| `raw` | Original puzzle image, no overlay | Baseline — model perception only |
| `oracle_mask` | Image + ground-truth outline overlay | Annotations are accurate; model should trust them |
| `tddn_mask` | Image + TDDN model-predicted overlay | Annotations may be noisy; model should critically evaluate |

The oracle mask overlays are generated from the puzzle's ground-truth data via
`src/overlays/oracle_mask.py`. The TDDN masks are generated by the text-driven DINOv3 segmentation
model via `src/overlays/tddn_mask.py`.

### Delta metrics

```
Img   = (raw_M  + raw_PM)  / 2    # baseline: mean of the two raw-image runs
dM    =  seg_M  - Img             # oracle-mask gain over baseline
dPM   =  seg_PM - Img             # TDDN-mask gain over baseline
dPMM  =  dPM   - dM               # TDDN gain relative to oracle (= seg_PM - seg_M)
```

M refers to the oracle_mask run; PM refers to the tddn_mask run.
Per-task canonical metric: `cell` for maze, `f1` for nqueens, mean black+white piece F1 for chess.
