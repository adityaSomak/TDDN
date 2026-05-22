"""PCK@α scoring for SPair-71K-style keypoint matching.

Given paired source / target feature maps, the predicted target location
for each source keypoint is the patch with maximum cosine similarity.
A prediction is correct when its pixel distance to the ground truth is
below ``α * bbox_max_side``, where ``bbox_max_side`` is the longer side
of the target bounding box.

Feature maps come in as ``(C, H, W)`` tensors; this module is agnostic
to which backbone produced them.

Public API
----------
    match_keypoints(src_feat, tgt_feat, src_kps, canvas)
        Cosine-NN match → predicted (x, y) per source keypoint.
    pck_at_alpha(pred_xy, gt_xy, bbox_max, alphas)
        Per-pair correct / total counts at each α threshold.
    aggregate(rows)
        Sum counts across pairs → fractional PCK per α.
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F


def match_keypoints(
    src_feat: torch.Tensor,
    tgt_feat: torch.Tensor,
    src_kps: torch.Tensor,
    canvas: int,
) -> torch.Tensor:
    """Cosine-NN match each source keypoint to a target patch.

    Args:
        src_feat: ``(C, H, W)`` source feature map.
        tgt_feat: ``(C, H, W)`` target feature map (same grid).
        src_kps:  ``(K, 3)`` source keypoints in canvas pixel space;
                  column 2 is visibility (already filtered by caller).
        canvas:   side length of the square input canvas.

    Returns:
        ``(K, 2)`` predicted (x, y) in canvas pixel space.
    """
    C, H, W = src_feat.shape
    src_flat = F.normalize(src_feat.reshape(C, H * W).T, dim=-1)
    tgt_flat = F.normalize(tgt_feat.reshape(C, H * W).T, dim=-1)

    stride_x = canvas / W
    stride_y = canvas / H

    src_x = (src_kps[:, 0] / stride_x).floor().long().clamp(0, W - 1)
    src_y = (src_kps[:, 1] / stride_y).floor().long().clamp(0, H - 1)
    src_idx = src_y * W + src_x

    sim = src_flat[src_idx] @ tgt_flat.T
    nn_idx = sim.argmax(dim=-1)
    nn_y = (nn_idx // W).float()
    nn_x = (nn_idx % W).float()

    pred_x = nn_x * stride_x + stride_x / 2.0 - 0.5
    pred_y = nn_y * stride_y + stride_y / 2.0 - 0.5
    return torch.stack([pred_x, pred_y], dim=1)


def pck_at_alpha(
    pred_xy: torch.Tensor,
    gt_xy: torch.Tensor,
    bbox_max: float,
    alphas: Sequence[float] = (0.1, 0.05, 0.01),
) -> dict[float, tuple[int, int]]:
    """Per-pair correct / total keypoint counts at each α threshold.

    Returns raw counts (not fractions) so multiple pairs can be combined
    with keypoint-micro averaging — pairs with more keypoints carry
    proportionally more weight in the final PCK.

    Args:
        pred_xy:  ``(K, 2)`` predicted target keypoints.
        gt_xy:    ``(K, 2)`` ground-truth target keypoints.
        bbox_max: longer side of the target bounding box (canvas pixels).
        alphas:   PCK thresholds as fractions of ``bbox_max``.

    Returns:
        Mapping ``{α: (correct_keypoints, total_keypoints)}``.
    """
    out: dict[float, tuple[int, int]] = {}
    if pred_xy.numel() == 0:
        return {float(a): (0, 0) for a in alphas}
    err = (pred_xy - gt_xy).norm(dim=-1)
    for a in alphas:
        correct = int((err < a * bbox_max).sum().item())
        out[float(a)] = (correct, int(err.shape[0]))
    return out


def aggregate(pck_per_pair: list[dict[float, tuple[int, int]]]) -> dict[float, float]:
    """Sum per-pair counts and divide → keypoint-micro PCK per α."""
    if not pck_per_pair:
        return {}
    alphas = list(pck_per_pair[0])
    out: dict[float, float] = {}
    for a in alphas:
        correct = sum(row[a][0] for row in pck_per_pair)
        total = sum(row[a][1] for row in pck_per_pair)
        out[a] = float(correct / total) if total > 0 else float("nan")
    return out
