"""Per-layer PCK ablation for diffusion backbones.

When the configured extractor returns features from several internal
U-Net layers, this module evaluates each layer in isolation so we can
see which layer best supports keypoint matching. Used for the
CleanDIFT / Stable-Diffusion per-layer comparison plots.

The orchestrator supplies an ``extract_per_layer`` callable that runs
the extractor on a single image and returns a dict mapping
``layer_idx -> (C, H, W)`` tensors.

Public API
----------
    subsample_pairs(pairs, n_per_cat)        balanced N-per-category sample.
    run_layer_ablation(extract_per_layer, pairs, canvas)
                                             {layer_idx: {α: PCK_fraction}}.
"""
from __future__ import annotations

import random
from typing import Sequence

import torch

from .pairs import PairMeta, categories_per_split
from .pck import match_keypoints, pck_at_alpha


def subsample_pairs(
    pairs: Sequence[PairMeta], n_per_cat: int, seed: int = 0,
) -> list[PairMeta]:
    """Take up to ``n_per_cat`` pairs from each category, deterministically."""
    rng = random.Random(seed)
    sampled: list[PairMeta] = []
    for _, group in categories_per_split(pairs).items():
        if len(group) <= n_per_cat:
            sampled.extend(group)
            continue
        order = list(range(len(group)))
        rng.shuffle(order)
        sampled.extend(group[i] for i in sorted(order[:n_per_cat]))
    return sampled


@torch.no_grad()
def run_layer_ablation(
    extract_per_layer,
    pairs: Sequence[PairMeta],
    canvas: int,
    *,
    alphas: Sequence[float] = (0.1, 0.05, 0.01),
) -> dict[int, dict[float, float]]:
    """Evaluate PCK separately for each available layer.

    Args:
        extract_per_layer: callable ``(image_path) → {layer_idx: (C, H, W)}``.
        pairs:             ``PairMeta`` list to evaluate.
        canvas:            side length of the square input canvas.
        alphas:            PCK thresholds as fractions of bbox_max.

    Returns:
        ``{layer_idx: {α: PCK_fraction}}`` aggregated across the pairs.
    """
    per_layer_scores: dict[int, list[dict[float, tuple[int, int]]]] = {}
    for pair in pairs:
        src_layers = extract_per_layer(pair.src_path)
        tgt_layers = extract_per_layer(pair.tgt_path)
        visible = pair.src_kps[:, 2] * pair.tgt_kps[:, 2] > 0
        if not visible.any():
            continue
        # bbox_max stored in raw pixels — rescale into the padded canvas
        # to match the units of predicted keypoint errors.
        tgt_scale = canvas / max(pair.tgt_size)
        bbox_max = pair.tgt_bbox_max * tgt_scale
        for layer_idx in sorted(src_layers):
            pred = match_keypoints(
                src_layers[layer_idx], tgt_layers[layer_idx],
                pair.src_kps[visible], canvas,
            )
            row = pck_at_alpha(pred, pair.tgt_kps[visible, :2], bbox_max, alphas)
            per_layer_scores.setdefault(layer_idx, []).append(row)

    out: dict[int, dict[float, float]] = {}
    for layer_idx, rows in per_layer_scores.items():
        agg: dict[float, float] = {}
        for a in alphas:
            correct = sum(r[a][0] for r in rows)
            total = sum(r[a][1] for r in rows)
            agg[float(a)] = float(correct / total) if total > 0 else float("nan")
        out[layer_idx] = agg
    return out
