"""CRG negatives: black out the queried region of a board image.

Negatives are **never persisted**. They are a deterministic ``black(image, cells)``
built in memory for each item at eval time, where the cells come from ground truth
(oracle arm) or from the TDDN detections (tddn arm). Only the *detections* are cached
in the dataset — the images they imply are cheap to reconstruct and would multiply
the committed board count by three.

Two blackout geometries, deliberately different:

* **cell blackout** (both oracle arms, and the chess tddn arm) fills the interior of a
  whole grid cell but keeps a ``_MARGIN`` frame, so the board's grid lines survive and
  the model still sees the cell's position and colour — only its contents are removed.
* **box blackout** (the N-Queens tddn arm) fills a TDDN-detected bounding box exactly.
  No margin is kept because the box is already a tight fit around the detected piece
  and has been padded by the detector; adding a second inset would leave piece edges
  visible.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

from . import config, data

_MARGIN = 5  # px frame kept around a blacked cell (so grid lines survive)


# ---------------------------------------------------------------------------
# Cell geometry
# ---------------------------------------------------------------------------
def cell_box(r: int, c: int, R: int, C: int, H: int, W: int) -> tuple[int, int, int, int]:
    """Pixel bounds (y0, y1, x0, x1) of grid cell (r, c) on an H x W image."""
    return (round(r * H / R), round((r + 1) * H / R),
            round(c * W / C), round((c + 1) * W / C))


def black_cells(board: Image.Image, cells, R: int, C: int) -> Image.Image:
    """Black the interior of each (r, c) cell, keeping a margin frame."""
    if not cells:
        return board
    arr = np.array(board).copy()
    H, W = arr.shape[:2]
    for r, c in cells:
        y0, y1, x0, x1 = cell_box(r, c, R, C, H, W)
        arr[y0 + _MARGIN:y1 - _MARGIN, x0 + _MARGIN:x1 - _MARGIN] = 0
    return Image.fromarray(arr)


# ---------------------------------------------------------------------------
# Deriving cells from cached TDDN detections
# ---------------------------------------------------------------------------
def chess_tddn_cells(pred_map, ablate_pieces: list[str],
                     extreme: list | None = None) -> list[tuple[int, int]]:
    """Cells to black from a TDDN 8x8 prediction map.

    Normal case: every cell whose predicted class is one of ``ablate_pieces``.
    Superlative case (``extreme=[kind, piece]``): only the single extreme detected
    instance, mirroring the oracle's one-cell ablation.
    """
    tok2id = config.CHESS_TOKEN2ID
    N = config.CHESS_BOARD_N
    if extreme:
        kind, piece = extreme
        pid = tok2id[piece]
        cands = [(r, c) for r in range(N) for c in range(N) if pred_map[r][c] == pid]
        if not cands:
            return []
        key = {"leftmost":   lambda x: (x[1], x[0]),
               "rightmost":  lambda x: (-x[1], x[0]),
               "topmost":    lambda x: (x[0], x[1]),
               "bottommost": lambda x: (-x[0], x[1])}[kind]
        return [min(cands, key=key)]
    ids = {tok2id[p] for p in ablate_pieces}
    return [(r, c) for r in range(N) for c in range(N) if pred_map[r][c] in ids]


def nqueens_tddn_boxes(detections: list[dict], role: str) -> list[list[float]]:
    """Fractional boxes of the TDDN-detected queen(s) named by ``role``.

    detections: [{'cy', 'cx', 'box': [fy0, fy1, fx0, fx1]}, ...] as cached.
    Empty when the detector found nothing on this board, in which case the negative
    ends up identical to the positive and CRG degenerates to the raw logits.
    """
    if not detections:
        return []
    centroids = [(d["cy"], d["cx"]) for d in detections]
    return [detections[i]["box"] for i in data.pick_by_role(centroids, role)]


def black_boxes(board: Image.Image, boxes: list[list[float]]) -> Image.Image:
    """Black each fractional [fy0, fy1, fx0, fx1] box outright (no margin)."""
    if not boxes:
        return board
    arr = np.array(board).copy()
    H, W = arr.shape[:2]
    for fy0, fy1, fx0, fx1 in boxes:
        arr[int(fy0 * H):int(fy1 * H), int(fx0 * W):int(fx1 * W)] = 0
    return Image.fromarray(arr)


# ---------------------------------------------------------------------------
# The one entry point the eval loops use
# ---------------------------------------------------------------------------
def build(task: str, arm: str, item: dict, board: Image.Image,
          spec: dict, detections: dict) -> Image.Image:
    """The negative image for one item, or the unmodified board for arm 'raw'.

    ``detections`` is the cached (or freshly re-detected) map keyed by image_id; it is
    only consulted for arm 'tddn'.
    """
    if arm == "raw":
        return board
    R, C = item["board_rows"], item["board_cols"]

    if arm == "oracle":
        return black_cells(board, item["ablate_cells"], R, C)

    if arm != "tddn":
        raise ValueError(f"unknown arm: {arm!r}")

    det = detections.get(item["image_id"])
    if det is None:
        return board
    if task == "chess":
        cells = chess_tddn_cells(det, spec["ablate_pieces"], spec.get("extreme"))
        return black_cells(board, cells, R, C)
    boxes = nqueens_tddn_boxes(det, data.nqueens_ablate_role(spec))
    return black_boxes(board, boxes)
