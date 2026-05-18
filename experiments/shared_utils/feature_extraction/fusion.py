"""Runtime fusion — bilinear-align + L2-normalize + weighted concat.

The standard recipe across SPair, imagenet_knn, and pca_viz when combining
features from two backbones. The trained `fused-dinov3-cd` model is a
separate path (not runtime fusion).
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F


def fuse_concat(
    feats: Sequence[torch.Tensor],
    weights: Sequence[float],
    target_grid: tuple[int, int] | None = None,
    normalize: bool = True,
) -> torch.Tensor:
    """Bilinear-resample each (B, C, H, W) to `target_grid`, weight + concat.

    If `target_grid` is None, uses the largest spatial size found across
    the input list. If `normalize`, per-feature L2-norm along the channel
    axis before applying the weight.
    """
    if len(feats) != len(weights):
        raise ValueError(f"len(feats)={len(feats)} != len(weights)={len(weights)}")
    if not feats:
        raise ValueError("Need at least one feature map.")

    if target_grid is None:
        target_h = max(f.shape[-2] for f in feats)
        target_w = max(f.shape[-1] for f in feats)
        target_grid = (target_h, target_w)

    aligned = []
    for f, w in zip(feats, weights):
        if f.shape[-2:] != target_grid:
            f = F.interpolate(f, size=target_grid, mode="bilinear", align_corners=False)
        if normalize:
            f = F.normalize(f, dim=1)
        aligned.append(f * w)

    return torch.cat(aligned, dim=1)


def fuse_concat_global(
    feats: Sequence[torch.Tensor],
    weights: Sequence[float],
    normalize: bool = True,
    final_normalize: bool = True,
) -> torch.Tensor:
    """Weighted concat of per-image vectors (B, D_i). Used by imagenet_knn.

    Each input is L2-normalized along its feature dim before being weighted;
    the concatenated output is L2-normalized once more (imagenet_knn convention).
    """
    if len(feats) != len(weights):
        raise ValueError(f"len(feats)={len(feats)} != len(weights)={len(weights)}")
    parts = []
    for f, w in zip(feats, weights):
        if normalize:
            f = F.normalize(f, dim=-1)
        parts.append(f * w)
    out = torch.cat(parts, dim=-1)
    if final_normalize:
        out = F.normalize(out, dim=-1)
    return out
