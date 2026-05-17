"""Generate oracle-mask overlays for maze, nqueens, and chess.

A thin coloured halo is painted just outside each silhouette using the
ground-truth board layout; the silhouette itself is preserved so the model
still sees the original symbol.

Output: <DATASET_ROOT>/<puzzle>/seg_data/oracle_mask/<pid>_overlay.jpg
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation

from src.overlays.mask_utils import parse_grid, dilated_outline, wall_mask_from_grey


# Paths resolve against the AlgoPuzzleVQA_star/ tree this script ships in:
#     AlgoPuzzleVQA_star/<task>/data/<task>_v2.csv         (inputs)
#     AlgoPuzzleVQA_star/<task>/data/images/<ID>/...       (inputs)
#     AlgoPuzzleVQA_star/<task>/seg_data/oracle_mask/...   (outputs)
# Override with DATASET_ROOT to point at a different layout.
DATASET_ROOT = Path(os.environ.get("DATASET_ROOT", Path(__file__).resolve().parents[3]))
MAZE_DIR = DATASET_ROOT / "maze" / "data"
NQUEENS_DIR = DATASET_ROOT / "nqueens" / "data"
MAZE_CSV = MAZE_DIR / "maze_solve_v2.csv"
NQUEENS_CSV = NQUEENS_DIR / "nqueens_v2.csv"
CHESS_DATASET = Path(os.environ["CHESS_DATASET"]) if os.environ.get("CHESS_DATASET") else None


# Maze: 8 px wall outline, 4 px arrow outlines.
MAZE_PALETTE = {
    "wall": np.array([220, 130, 40], np.uint8),
    "S":    np.array([0, 200, 0], np.uint8),
    "E":    np.array([200, 0, 0], np.uint8),
}
MAZE_WALL_THICK = 8
MAZE_ARROW_THICK = 4


def build_maze(pid: str, text_rep: str, image_path: Path, out_dir: Path) -> Path:
    img = np.array(Image.open(image_path).convert("RGB"))
    H, W = img.shape[:2]
    grid = parse_grid(text_rep)
    R, C = len(grid), len(grid[0])

    wall_mask = wall_mask_from_grey(img)
    s_arrow = np.zeros_like(wall_mask, dtype=bool)
    e_arrow = np.zeros_like(wall_mask, dtype=bool)
    for r in range(R):
        for c in range(C):
            lab = grid[r][c].upper()
            if lab not in ("S", "E"):
                continue
            y0, y1 = round(r * H / R), round((r + 1) * H / R)
            x0, x1 = round(c * W / C), round((c + 1) * W / C)
            cell = img[y0:y1, x0:x1]
            Rch, Gch, Bch = cell[..., 0].astype(int), cell[..., 1].astype(int), cell[..., 2].astype(int)
            if lab == "S":
                s_arrow[y0:y1, x0:x1] = (Gch > 100) & (Gch > Rch + 30) & (Gch > Bch + 30)
            else:
                e_arrow[y0:y1, x0:x1] = (Bch > 100) & (Bch > Rch + 30) & (Bch > Gch + 30)

    out = img.copy()
    out[dilated_outline(wall_mask, MAZE_WALL_THICK)] = MAZE_PALETTE["wall"]
    out[dilated_outline(s_arrow, MAZE_ARROW_THICK)] = MAZE_PALETTE["S"]
    out[dilated_outline(e_arrow, MAZE_ARROW_THICK)] = MAZE_PALETTE["E"]

    out_path = out_dir / f"{pid}_overlay.jpg"
    Image.fromarray(out).save(out_path, quality=95)
    return out_path


# NQueens: 5 px red halo around each queen sprite.
NQUEENS_THICK = 5
NQUEENS_MARGIN = 12   # skip cell-edge AA pixels
NQUEENS_COLOUR = np.array([200, 0, 0], np.uint8)


def build_nqueens(pid: str, text_rep: str, image_path: Path, out_dir: Path) -> Path:
    img = np.array(Image.open(image_path).convert("RGB"))
    H, W = img.shape[:2]
    grid = parse_grid(text_rep)
    R, C = len(grid), len(grid[0])

    sprite = np.zeros((H, W), dtype=bool)
    for r in range(R):
        for c in range(C):
            if grid[r][c].upper() != "Q":
                continue
            y0, y1 = round(r * H / R), round((r + 1) * H / R)
            x0, x1 = round(c * W / C), round((c + 1) * W / C)
            yy0, yy1 = y0 + NQUEENS_MARGIN, y1 - NQUEENS_MARGIN
            xx0, xx1 = x0 + NQUEENS_MARGIN, x1 - NQUEENS_MARGIN
            cell = img[yy0:yy1, xx0:xx1]
            Rch, Gch, Bch = cell[..., 0].astype(int), cell[..., 1].astype(int), cell[..., 2].astype(int)
            is_green = (Gch > 150) & (Rch < 120) & (Bch < 120)
            is_pink  = (Rch > 200) & (Gch > 130) & (Gch < 220) & (Bch > 150)
            sprite[yy0:yy1, xx0:xx1] = ~(is_green | is_pink)

    out = img.copy()
    out[binary_dilation(sprite, iterations=NQUEENS_THICK) & ~sprite] = NQUEENS_COLOUR

    out_path = out_dir / f"{pid}_overlay.jpg"
    Image.fromarray(out).save(out_path, quality=95)
    return out_path


# Chess: 4 px halo per piece class (3..14).
CHESS_THICK = 4
CHESS_PIECE_LO, CHESS_PIECE_HI = 3, 15
CHESS_CMAP = np.array([
    [0,   0,   0],   [0,   0,   0],   [0,   0,   0],
    [140, 0,   0],   [0,   100, 0],   [0,   0,   140],
    [140, 140, 0],   [120, 0,   120], [0,   100, 120],
    [255, 140, 140], [140, 255, 140], [140, 200, 255],
    [255, 255, 140], [255, 150, 220], [150, 230, 230],
], dtype=np.uint8)


def build_chess(pid: str, image_path: Path, mask_path: Path, out_dir: Path) -> Path:
    img = np.array(Image.open(image_path).convert("RGB").resize((512, 512), Image.BILINEAR))
    mask = np.array(Image.open(mask_path))
    out = img.copy()
    for cls in range(CHESS_PIECE_LO, CHESS_PIECE_HI):
        cls_mask = (mask == cls)
        if not cls_mask.any():
            continue
        out[binary_dilation(cls_mask, iterations=CHESS_THICK) & ~cls_mask] = CHESS_CMAP[cls]
    out_path = out_dir / f"{pid}_overlay.jpg"
    Image.fromarray(out).save(out_path, quality=95)
    return out_path


def _select_pids(all_pids: list[str], ids: list[str] | None, limit: int | None) -> list[str]:
    pids = ids if ids else sorted(all_pids)
    return pids[:limit] if limit else pids


def _run_csv(args, csv_path: Path, text_col: str, build_fn, root: Path,
             label: str, out_dir: Path) -> None:
    """CSV-driven loop shared by maze and nqueens."""
    by_pid = {Path(r["image_path"]).parts[-2]: r
              for r in csv.DictReader(open(csv_path))}
    pids = _select_pids(list(by_pid), args.ids, args.limit)
    print(f"[{label}] {len(pids)} overlays -> {out_dir}")
    for i, pid in enumerate(pids):
        rec = by_pid[pid]
        build_fn(pid, rec[text_col], root / rec["image_path"], out_dir)
        if (i + 1) % 25 == 0 or i == len(pids) - 1:
            print(f"  [{i+1}/{len(pids)}]")


def _run_chess(args, out_dir: Path) -> None:
    import json
    images_dir = CHESS_DATASET / "images"
    masks_dir = CHESS_DATASET / "masks"
    eval_pids = sorted(Path(e["filename"]).stem for e in
                       json.loads((CHESS_DATASET / "text_repr.json").read_text()))
    pids = _select_pids(eval_pids, args.ids, args.limit)
    print(f"[chess] {len(pids)} overlays -> {out_dir}")
    for i, pid in enumerate(pids):
        build_chess(pid, images_dir / f"{pid}.png", masks_dir / f"{pid}.png", out_dir)
        if (i + 1) % 25 == 0 or i == len(pids) - 1:
            print(f"  [{i+1}/{len(pids)}]")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--puzzle", choices=["maze", "nqueens", "chess"], required=True)
    ap.add_argument("--ids", nargs="*")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    out_dir = DATASET_ROOT / args.puzzle / "seg_data" / "oracle_mask"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.puzzle == "maze":
        _run_csv(args, MAZE_CSV, "text_representation_start-position",
                 build_maze, MAZE_DIR, "maze", out_dir)
    elif args.puzzle == "nqueens":
        _run_csv(args, NQUEENS_CSV, "text-representation_start-position",
                 build_nqueens, NQUEENS_DIR, "nqueens", out_dir)
    else:  # chess
        if CHESS_DATASET is None:
            ap.error("chess: set CHESS_DATASET env var to the chess dataset root")
        _run_chess(args, out_dir)
    print("Done.")


if __name__ == "__main__":
    main()
