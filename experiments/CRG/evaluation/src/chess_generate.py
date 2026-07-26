"""Mint a NEW synthetic chess board set: renderer + per-question generators.

Reached only from ``run_generate.py``. **Not needed to evaluate** — the committed
board set is the artifact; this exists so the set can be rebuilt or extended.

Requires the locally-supplied 269-board segmentation set (see
``datasets/_local/README.md``): piece sprites and board colour themes are extracted
from those real boards and their per-pixel GT masks.

Regenerating replaces the dataset. Board ids must stay ``{slug}_{NNN}`` using each
question's ``slug`` from questions.yaml: the committed answers.csv and every archived
per-board result record key on those exact ids, so a differently-named set silently
desynchronises them.
"""
from __future__ import annotations

import csv
import json
import random

import numpy as np
from PIL import Image, ImageFilter

from . import config

_CELL = config.CHESS_CELL_PX
_N = config.CHESS_BOARD_N
_SZ = _CELL * _N
_PIECES = config.CHESS_PIECES
_DEFAULT_THEME = ((211, 210, 210), (133, 133, 133))   # neutral gray, good contrast
# A GT-mask area this large is a clean enough sprite to stop scanning further boards.
_MIN_SPRITE_AREA_PX = 1200
_SPRITES: dict[str, np.ndarray] | None = None


# ---------------------------------------------------------------------------
# Renderer (sprites + themes come from the real boards)
# ---------------------------------------------------------------------------
def _colon_grid(text_repr: str) -> list[list[str]]:
    """The legacy text_repr.json encoding: rows newline-separated, cells ':'-separated."""
    return [row.split(":") for row in text_repr.strip().split("\n")]


def _real_boards() -> dict[str, str]:
    config.require_legacy_chess()
    entries = json.loads(config.LEGACY_CHESS_TEXT_REPR.read_text())
    return {e["filename"].replace(".png", ""): e["text_representation"] for e in entries}


def build_sprites() -> dict[str, np.ndarray]:
    """Per piece type, the largest-mask-area instance as a 64x64 RGBA sprite."""
    global _SPRITES
    if _SPRITES is not None:
        return _SPRITES
    boards = _real_boards()
    best: dict[str, tuple[int, np.ndarray | None]] = {p: (-1, None) for p in _PIECES}
    for pid in sorted(boards):
        if not any(best[p][1] is None or best[p][0] < _MIN_SPRITE_AREA_PX for p in _PIECES):
            break
        g = _colon_grid(boards[pid])
        im = np.array(Image.open(config.LEGACY_CHESS_IMAGES / f"{pid}.png").convert("RGB"))
        m = np.array(Image.open(config.LEGACY_CHESS_MASKS / f"{pid}.png"))
        for r in range(_N):
            for c in range(_N):
                tok = g[r][c]
                if tok not in _PIECES:
                    continue
                cell_m = (m[r * _CELL:(r + 1) * _CELL, c * _CELL:(c + 1) * _CELL]
                          == config.CHESS_TOKEN2ID[tok])
                area = int(cell_m.sum())
                if area > best[tok][0]:
                    rgb = im[r * _CELL:(r + 1) * _CELL, c * _CELL:(c + 1) * _CELL]
                    best[tok] = (area, np.dstack([rgb, (cell_m * 255).astype(np.uint8)]))
    missing = [p for p in _PIECES if best[p][1] is None]
    if missing:
        raise SystemExit(f"no sprite found for {missing} in the legacy board set")
    _SPRITES = {p: best[p][1] for p in _PIECES}
    return _SPRITES


def sample_themes() -> list[tuple]:
    """All (light, dark) RGB square-colour pairs read from the real boards."""
    boards = _real_boards()
    themes = []
    for pid in sorted(boards):
        g = _colon_grid(boards[pid])
        im = np.array(Image.open(config.LEGACY_CHESS_IMAGES / f"{pid}.png").convert("RGB"))
        light = dark = None
        for r in range(_N):
            for c in range(_N):
                patch = im[r * _CELL + 22:r * _CELL + 42, c * _CELL + 22:c * _CELL + 42]
                mean = tuple(patch.reshape(-1, 3).mean(0).round().astype(int))
                if g[r][c] == "white_sq" and light is None:
                    light = mean
                elif g[r][c] == "black_sq" and dark is None:
                    dark = mean
        if light and dark:
            themes.append((light, dark))
    return themes


def render(grid: list[list[str]], theme: tuple | None = None) -> Image.Image:
    """Render an 8x8 token grid to a 512x512 RGB board."""
    light, dark = theme or _DEFAULT_THEME
    spr = build_sprites()
    out = np.zeros((_SZ, _SZ, 3), np.uint8)
    for r in range(_N):
        for c in range(_N):
            out[r * _CELL:(r + 1) * _CELL, c * _CELL:(c + 1) * _CELL] = (
                light if (r + c) % 2 == 0 else dark)
    img = Image.fromarray(out).convert("RGBA")
    for r in range(_N):
        for c in range(_N):
            tok = grid[r][c]
            if tok in _PIECES:
                sprite = spr[tok].copy()
                alpha = Image.fromarray(sprite[:, :, 3]).filter(ImageFilter.GaussianBlur(0.7))
                sprite[:, :, 3] = np.array(alpha)
                img.alpha_composite(Image.fromarray(sprite, "RGBA"), (c * _CELL, r * _CELL))
    return img.convert("RGB")


# ---------------------------------------------------------------------------
# Per-question board generators
# ---------------------------------------------------------------------------
def _sq(r: int, c: int) -> str:
    return "white_sq" if (r + c) % 2 == 0 else "black_sq"


def _find(g, tok):
    return [(r, c) for r in range(_N) for c in range(_N) if g[r][c] == tok]


def _clear(g, tok):
    for r, c in _find(g, tok):
        g[r][c] = _sq(r, c)


def _empty(g):
    return [(r, c) for r in range(_N) for c in range(_N)
            if g[r][c] in ("white_sq", "black_sq")]


def _choice(rng, seq):
    if not seq:
        raise IndexError("no candidate cell")
    return rng.choice(seq)


def _online(a, b) -> bool:
    return a[0] == b[0] or a[1] == b[1] or abs(a[0] - b[0]) == abs(a[1] - b[1])


# Each generator places pieces for answer class ``k`` and returns the GT cells the
# oracle negative blacks (the queried piece location(s)).
def _gen_align(g, k, rng, *, A, B, **_):
    _clear(g, A); _clear(g, B)
    b = _choice(rng, _empty(g)); g[b[0]][b[1]] = B
    e = _empty(g)
    cand = ([c for c in e if c[0] == b[0] or c[1] == b[1]] if k == 0
            else [c for c in e if c[0] != b[0] and c[1] != b[1]])
    a = _choice(rng, cand); g[a[0]][a[1]] = A
    return [[a[0], a[1]], [b[0], b[1]]]


def _gen_adjacent(g, k, rng, *, A, B, **_):
    _clear(g, A); _clear(g, B)
    b = _choice(rng, [c for c in _empty(g) if 1 <= c[0] <= 6 and 1 <= c[1] <= 6])
    g[b[0]][b[1]] = B
    nb = {(b[0] + dr, b[1] + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)
          if (dr, dc) != (0, 0)}
    e = _empty(g)
    cand = [c for c in e if c in nb] if k == 0 else [c for c in e if c not in nb]
    a = _choice(rng, cand); g[a[0]][a[1]] = A
    return [[a[0], a[1]], [b[0], b[1]]]


def _gen_quad(g, k, rng, *, piece, **_):
    _clear(g, piece)
    rows = range(0, 4) if k in (0, 1) else range(4, 8)
    cols = range(0, 4) if k in (0, 2) else range(4, 8)
    c = _choice(rng, [x for x in _empty(g) if x[0] in rows and x[1] in cols])
    g[c[0]][c[1]] = piece
    return [[c[0], c[1]]]


def _gen_relv(g, k, rng, *, A, B, **_):
    _clear(g, A); _clear(g, B)
    b = _choice(rng, [c for c in _empty(g) if 1 <= c[0] <= 6]); g[b[0]][b[1]] = B
    a = _choice(rng, [c for c in _empty(g) if (c[0] < b[0]) == (k == 0) and c[0] != b[0]])
    g[a[0]][a[1]] = A
    return [[a[0], a[1]], [b[0], b[1]]]


def _gen_queen_line(g, k, rng, **_):
    _clear(g, "b_king"); _clear(g, "w_queen")
    bk = _choice(rng, _empty(g)); g[bk[0]][bk[1]] = "b_king"
    q = _choice(rng, [c for c in _empty(g) if c != bk and _online(c, bk) == (k == 0)])
    g[q[0]][q[1]] = "w_queen"
    return [[q[0], q[1]]]


def _gen_exists(g, k, rng, *, piece, **_):
    _clear(g, piece)
    if k == 0:
        c = _choice(rng, _empty(g)); g[c[0]][c[1]] = piece
        return [[c[0], c[1]]]
    return []                     # answer "No": nothing to ablate, by construction


def _gen_psup(g, k, rng, *, color="w", **_):
    pawn = f"{color}_pawn"
    _clear(g, pawn)
    lc = rng.randint(0, 5)
    lrow = rng.choice(list(range(0, 4) if k == 0 else range(4, 8)))
    if g[lrow][lc] not in ("white_sq", "black_sq"):
        raise IndexError("leftmost cell occupied")
    g[lrow][lc] = pawn
    extra = [x for x in _empty(g) if x[1] > lc]
    for c in rng.sample(extra, min(4, len(extra))):
        g[c[0]][c[1]] = pawn
    return [[lrow, lc]]


_GENERATORS = {"align": _gen_align, "adjacent": _gen_adjacent, "quad": _gen_quad,
               "relv": _gen_relv, "queen_line": _gen_queen_line, "exists": _gen_exists,
               "psup": _gen_psup}


def generate(per: int = 100, seed: int = 101) -> None:
    """Render ``per`` boards per question, balanced across answer classes.

    Overwrites images/ and answers.csv for the chess task.
    """
    from . import data

    qspecs = data.load_questions("chess")
    rng = random.Random(seed)
    base = [_colon_grid(t) for t in _real_boards().values()]
    themes = sample_themes()
    build_sprites()
    img_dir = config.task_dir("chess") / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for qid, spec in qspecs.items():
        gen = _GENERATORS[spec["generator"]]
        n_classes = len(spec["options"])
        plan = [c for c in range(n_classes) for _ in range(per // n_classes)]
        rng.shuffle(plan)
        for i, k in enumerate(plan):
            for _ in range(80):
                g = [row[:] for row in rng.choice(base)]
                try:
                    cells = gen(g, k, rng, **spec.get("params", {}))
                    break
                except IndexError:
                    continue
            else:
                continue
            image_id = f"{spec['slug']}_{i:03d}"
            render(g, theme=rng.choice(themes)).save(img_dir / f"{image_id}.png")
            rows.append({"image_id": image_id, "qid": qid, "answer": k,
                         "ablate_cells": json.dumps(cells),
                         "board_rows": _N, "board_cols": _N,
                         "board": "/".join("|".join(r) for r in g)})

    rows.sort(key=lambda d: (int(d["qid"][1:]), d["image_id"]))
    with open(config.answers_path("chess"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image_id", "qid", "answer", "ablate_cells",
                                          "board_rows", "board_cols", "board"])
        w.writeheader()
        w.writerows(rows)

    from collections import Counter
    cc = Counter(r["qid"] for r in rows)
    print(f"generated {len(rows)} boards -> {img_dir}")
    for q in qspecs:
        print(f"  {q}: {cc[q]}")
    print(f"answers -> {config.answers_path('chess')}")
    print("NOTE: the cached tddn_detections.json no longer matches these boards — "
          "re-run `run_eval.py --task chess --redetect --save-detections`.")
