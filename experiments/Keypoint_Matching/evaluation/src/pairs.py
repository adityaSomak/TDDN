"""SPair-71K pair loader and keypoint-coordinate transform.

The SPair-71K dataset ships per-pair JSON metadata listing source and
target image paths, 2D keypoint coordinates, and tight bounding boxes.
This module reads those files and produces ``PairMeta`` records ready
for the matching / scoring code.

Mirrored pairs (``meta["mirror"] != 0``) are skipped because keypoint
indices flip under reflection and are not directly comparable.

Public API
----------
    load_pairs(spair_root, split, category)   list of PairMeta
    transform_keypoints(kps, w, h, canvas)    raw image → padded canvas
    categories_per_split(pairs)               group pairs by category
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch


SPLITS = ("trn", "val", "test")


@dataclass(frozen=True)
class PairMeta:
    """One SPair-71K image pair.

    Attributes:
        pair_id:      unique identifier of the pair.
        category:     SPair object category (e.g. ``aeroplane``).
        src_path:     source image path on disk.
        tgt_path:     target image path on disk.
        src_kps:      ``(K, 3)`` source keypoints (x, y, visibility).
        tgt_kps:      ``(K, 3)`` target keypoints (x, y, visibility).
        src_size:     source image (width, height) in pixels.
        tgt_size:     target image (width, height) in pixels.
        tgt_bbox_max: longer side of the target bounding box, in raw
                      image pixels. PCK@α thresholds use
                      ``α * tgt_bbox_max``; callers must rescale this
                      from raw pixels into the padded canvas before
                      comparing against canvas-space prediction errors.
    """

    pair_id: str
    category: str
    src_path: Path
    tgt_path: Path
    src_kps: torch.Tensor
    tgt_kps: torch.Tensor
    src_size: tuple[int, int]
    tgt_size: tuple[int, int]
    tgt_bbox_max: float


def load_pairs(
    spair_root: Path,
    split: str = "test",
    category: str | None = None,
) -> list[PairMeta]:
    """Read SPair pair-list JSONs and return ``PairMeta`` objects.

    Args:
        spair_root: path to the unpacked SPair-71K dataset (containing
                    ``Layout/``, ``PairAnnotation/`` and ``JPEGImages/``).
        split:      ``trn``, ``val`` or ``test``.
        category:   restrict to one category; ``None`` = all 18.
    """
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    layout = spair_root / "Layout" / "large" / f"{split}.txt"
    if not layout.exists():
        raise FileNotFoundError(f"missing layout file {layout}")
    pair_files = [line.strip() for line in layout.read_text().splitlines() if line.strip()]

    pairs: list[PairMeta] = []
    for rel in pair_files:
        meta = json.loads((spair_root / "PairAnnotation" / split / f"{rel}.json").read_text())
        if category is not None and meta["category"] != category:
            continue
        # Mirrored pairs: keypoint indices flip under reflection so the
        # raw coordinates aren't directly comparable; standard PCK
        # evaluations skip them.
        if meta.get("mirror", 0) != 0:
            continue
        # SPair keypoints are 2D `[x, y]`; both src and trg lists hold
        # the same K visible keypoints in matched order. Append a
        # visibility column of 1s so downstream code can use the
        # canonical (x, y, vis) layout.
        src_raw = torch.tensor(meta["src_kps"], dtype=torch.float32)
        tgt_raw = torch.tensor(meta["trg_kps"], dtype=torch.float32)
        src_kps = torch.cat([src_raw, torch.ones(src_raw.shape[0], 1)], dim=1)
        tgt_kps = torch.cat([tgt_raw, torch.ones(tgt_raw.shape[0], 1)], dim=1)
        src_w, src_h = meta["src_imsize"][:2]
        tgt_w, tgt_h = meta["trg_imsize"][:2]
        tgt_bbox_max = float(max(
            meta["trg_bndbox"][2] - meta["trg_bndbox"][0],
            meta["trg_bndbox"][3] - meta["trg_bndbox"][1],
        ))
        pairs.append(PairMeta(
            pair_id=rel,
            category=meta["category"],
            src_path=spair_root / "JPEGImages" / meta["category"] / f"{meta['src_imname']}",
            tgt_path=spair_root / "JPEGImages" / meta["category"] / f"{meta['trg_imname']}",
            src_kps=src_kps, tgt_kps=tgt_kps,
            src_size=(src_w, src_h), tgt_size=(tgt_w, tgt_h),
            tgt_bbox_max=tgt_bbox_max,
        ))
    return pairs


def transform_keypoints(kps: torch.Tensor, img_w: int, img_h: int, canvas: int) -> torch.Tensor:
    """Map ``(x, y, vis)`` keypoints from raw image space to a square pad canvas.

    Aspect-preserving padding scales the longer side to ``canvas`` pixels
    and centres the shorter side with equal padding on both edges.
    """
    scale = canvas / max(img_w, img_h)
    pad_x = (canvas - img_w * scale) / 2.0
    pad_y = (canvas - img_h * scale) / 2.0
    out = kps.clone()
    out[:, 0] = kps[:, 0] * scale + pad_x
    out[:, 1] = kps[:, 1] * scale + pad_y
    return out


def categories_per_split(pairs: Iterable[PairMeta]) -> dict[str, list[PairMeta]]:
    """Group a flat ``PairMeta`` list by ``category`` field."""
    by_cat: dict[str, list[PairMeta]] = {}
    for p in pairs:
        by_cat.setdefault(p.category, []).append(p)
    return by_cat
