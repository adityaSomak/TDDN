"""Dataset read layer: questions, answers, cached detections, board images.

Deliberately torch-free (stdlib + yaml + numpy/PIL only), so ``--validate-dataset``
and any result inspection stay cheap and cannot fail on a GPU-stack import.

Also holds the N-Queens rule registry. Those rules are how ``answer`` and
``ablate_cells`` in answers.csv were derived; at eval time the CSV is authoritative
and the rules are only re-run by ``validate_dataset`` as a self-check.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import yaml
from PIL import Image

from . import config

# axis name -> index into a (row, col) cell
_AXIS = {"row": 0, "col": 1}


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def load_questions(task: str) -> dict[str, dict]:
    """{qid: spec} from the task's questions.yaml.

    Every top-level key is a question id — the file carries no metadata block.
    """
    with open(config.questions_path(task)) as f:
        return yaml.safe_load(f)


def load_rows(task: str, limit: int | None = None) -> list[dict]:
    """answers.csv as typed row dicts, ordered by (qid, image_id).

    Columns: image_id, qid, answer, ablate_cells, board_rows, board_cols, board.
    """
    out = []
    with open(config.answers_path(task), newline="") as f:
        for r in csv.DictReader(f):
            out.append({
                "image_id": r["image_id"],
                "qid": r["qid"],
                "answer": int(r["answer"]),
                "ablate_cells": [tuple(c) for c in json.loads(r["ablate_cells"])],
                "board_rows": int(r["board_rows"]),
                "board_cols": int(r["board_cols"]),
                "board": r["board"],
            })
    return out[:limit] if limit else out


def load_detections(task: str) -> dict:
    """Cached TDDN predictions keyed by image_id.

    chess    -> an 8x8 predicted class-id map per board
    nqueens  -> a list of {cy, cx, box} fractional queen detections per board

    Committed alongside the boards so the tddn arm needs no GPU, no DINOv3 and no
    checkpoints. ``--redetect`` recomputes these instead of reading them.
    """
    with open(config.detections_path(task)) as f:
        return json.load(f)


def parse_grid(board: str) -> list[list[str]]:
    """The committed encoding: rows '/'-joined, cells '|'-joined."""
    return [row.split("|") for row in board.split("/")]


def load_board_image(task: str, image_id: str) -> Image.Image:
    """Board image at the size the VLM is fed.

    N-Queens sources are 1600x1600, so they are thumbnailed to INPUT_SIZE exactly as
    the pre-restructure loader did; chess boards are already 512 and pass through.
    """
    img = Image.open(config.board_path(task, image_id)).convert("RGB")
    if max(img.size) > INPUT_SIZE:
        img.thumbnail((INPUT_SIZE, INPUT_SIZE), Image.Resampling.BILINEAR)
    return img


INPUT_SIZE = 512   # board side fed to the VLM, both streams


# ---------------------------------------------------------------------------
# Item building (shared by the CRG and LaViDa eval paths)
# ---------------------------------------------------------------------------
def build_items(task: str, limit: int | None = None) -> tuple[dict, dict[str, list[dict]]]:
    """(qspecs, {qid: [item, ...]}) ready for either decode engine.

    Each item carries everything a decode needs: the board id, the prompt, the option
    list, the GT answer, and the oracle cells to black. ``limit`` truncates the row
    list, so it thins every question rather than dropping whole questions.
    """
    qspecs = load_questions(task)
    system = config.SYSTEM_PROMPT[task]
    items: dict[str, list[dict]] = {q: [] for q in qspecs}
    for r in load_rows(task, limit):
        spec = qspecs[r["qid"]]
        items[r["qid"]].append({
            "image_id": r["image_id"],
            "qid": r["qid"],
            "label": r["answer"],
            "options": spec["options"],
            "prompt": config.build_prompt(system, spec["text"], spec["options"]),
            "ablate_cells": r["ablate_cells"],
            "board_rows": r["board_rows"],
            "board_cols": r["board_cols"],
            "board": r["board"],
        })
    return qspecs, {q: v for q, v in items.items() if v}


# ---------------------------------------------------------------------------
# N-Queens rule registry (build-time derivation; a self-check at eval time)
# ---------------------------------------------------------------------------
def queen_cells(board: str) -> tuple[list[tuple[int, int]], int, int]:
    """(queen cells, rows, cols) from a committed board grid."""
    grid = parse_grid(board)
    R, C = len(grid), len(grid[0])
    return ([(r, c) for r in range(R) for c in range(C) if grid[r][c].upper() == "Q"], R, C)


def pick_by_role(positions: list[tuple[float, float]], role: str) -> list[int]:
    """Indices of the extreme position(s) named by ``role``.

    positions: comparable (a, b) coordinates — cell (r, c) or fractional (y, x).
    role: '+'-joined tokens from {top, bot, left, right, center, tl}.
    """
    out: list[int] = []
    rng = range(len(positions))
    for token in role.split("+"):
        if token == "top":
            out.append(min(rng, key=lambda i: (positions[i][0], positions[i][1])))
        elif token == "bot":
            out.append(min(rng, key=lambda i: (-positions[i][0], positions[i][1])))
        elif token == "left":
            out.append(min(rng, key=lambda i: (positions[i][1], positions[i][0])))
        elif token == "right":
            out.append(min(rng, key=lambda i: (-positions[i][1], positions[i][0])))
        elif token == "center":
            out.append(min(rng, key=lambda i: (positions[i][0] - 0.5) ** 2
                                               + (positions[i][1] - 0.5) ** 2))
        elif token == "tl":
            out += [i for i in rng if positions[i][0] < 0.5 and positions[i][1] < 0.5]
    return out


def nqueens_ablate_role(spec: dict) -> str:
    """The '+'-joined role string naming the queen cell(s) the negative blacks."""
    return spec["role"] if "role" in spec else "+".join(spec["roles"])


def nqueens_ablate_cells(board: str, spec: dict) -> list[tuple[int, int]]:
    """Re-derive the oracle cells for one (board, question).

    Selection runs over FRACTIONAL centroids, matching the expression the in-memory
    oracle negative uses, so this reproduces the committed ablate_cells exactly.
    """
    cells, R, C = queen_cells(board)
    frac = [((r + 0.5) / R, (c + 0.5) / C) for r, c in cells]
    return [cells[i] for i in pick_by_role(frac, nqueens_ablate_role(spec))]


def nqueens_label(board: str, spec: dict) -> int:
    """Re-derive the GT option index for one (board, question).

    Rules, declared by the YAML ``rule`` field:
      half  -> is the <role> queen's <axis> coord in the first or second half?
      third -> ... in the first / middle / last third? (3-way)
      rel   -> is roles[0]'s <axis> coord before or after roles[1]'s?
    """
    cells, R, C = queen_cells(board)
    axis = _AXIS[spec["axis"]]
    n = R if axis == 0 else C
    extreme = lambda role: cells[pick_by_role(cells, role)[0]][axis]
    if spec["rule"] == "rel":
        return 0 if extreme(spec["roles"][0]) < extreme(spec["roles"][1]) else 1
    coord = extreme(spec["role"])
    if spec["rule"] == "third":
        return 0 if coord < n / 3 else (1 if coord < 2 * n / 3 else 2)
    return 0 if coord < n / 2 else 1        # half


# ---------------------------------------------------------------------------
# Dataset validation
# ---------------------------------------------------------------------------
def validate_dataset(task: str) -> list[str]:
    """Structural + semantic checks. Returns a list of problems ([] == clean).

    Cheap enough (a few thousand stat calls, no image decode) to run at the start of
    every eval, which is where it catches a dataset that has drifted from its CSV.
    """
    problems: list[str] = []
    qspecs = load_questions(task)
    rows = load_rows(task)
    det = load_detections(task)
    ext = config.BOARD_EXT[task]
    on_disk = {p.stem for p in (config.task_dir(task) / "images").glob(f"*{ext}")}
    row_ids = {r["image_id"] for r in rows}

    if missing := row_ids - on_disk:
        problems.append(f"{len(missing)} rows have no board image (e.g. {sorted(missing)[:3]})")
    if orphan := on_disk - row_ids:
        problems.append(f"{len(orphan)} board images have no row (e.g. {sorted(orphan)[:3]})")
    if no_det := row_ids - set(det):
        problems.append(f"{len(no_det)} image_ids missing from tddn_detections.json")
    if unused_q := set(qspecs) - {r["qid"] for r in rows}:
        problems.append(f"questions with no rows: {sorted(unused_q)}")

    boards_by_id: dict[str, set] = {}
    for r in rows:
        qid, iid = r["qid"], r["image_id"]
        if qid not in qspecs:
            problems.append(f"{iid}/{qid}: qid not in questions.yaml")
            continue
        n_opts = len(qspecs[qid]["options"])
        if not 0 <= r["answer"] < n_opts:
            problems.append(f"{iid}/{qid}: answer {r['answer']} outside 0..{n_opts - 1}")
        R, C = r["board_rows"], r["board_cols"]
        for cr, cc in r["ablate_cells"]:
            if not (0 <= cr < R and 0 <= cc < C):
                problems.append(f"{iid}/{qid}: ablate cell ({cr},{cc}) outside {R}x{C}")
        grid = parse_grid(r["board"])
        if len(grid) != R or any(len(g) != C for g in grid):
            problems.append(f"{iid}/{qid}: board grid does not match declared {R}x{C}")
        boards_by_id.setdefault(iid, set()).add((r["board"], R, C))

    if inconsistent := [i for i, v in boards_by_id.items() if len(v) > 1]:
        problems.append(f"{len(inconsistent)} image_ids carry inconsistent board/dims "
                        f"across their rows (e.g. {inconsistent[:3]})")

    # Semantic re-derivation: N-Queens answers and oracle cells must fall out of the
    # board grid plus the YAML rule fields. This is the check that would catch a CSV
    # edited by hand or a board set swapped underneath it.
    if task == "nqueens":
        bad_ans = bad_cells = 0
        for r in rows:
            spec = qspecs.get(r["qid"])
            if not spec:
                continue
            if nqueens_label(r["board"], spec) != r["answer"]:
                bad_ans += 1
            if [tuple(c) for c in nqueens_ablate_cells(r["board"], spec)] != r["ablate_cells"]:
                bad_cells += 1
        if bad_ans:
            problems.append(f"{bad_ans}/{len(rows)} answers do not re-derive from the board")
        if bad_cells:
            problems.append(f"{bad_cells}/{len(rows)} ablate_cells do not re-derive")

    # Chess: each oracle cell must actually hold one of the question's ablate_pieces.
    if task == "chess":
        bad = 0
        for r in rows:
            spec = qspecs.get(r["qid"])
            if not spec:
                continue
            grid = parse_grid(r["board"])
            want = set(spec.get("ablate_pieces", []))
            if any(grid[cr][cc] not in want for cr, cc in r["ablate_cells"]):
                bad += 1
        if bad:
            problems.append(f"{bad}/{len(rows)} rows ablate a cell not holding an ablate_piece")

    return problems
