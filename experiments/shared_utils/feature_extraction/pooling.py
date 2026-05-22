"""Reduce an extractor's output dict to a single per-image global vector.

Different backbones surface their per-image features in different shapes
(``cls`` token for ViTs, ``patch_mean`` for diffusion U-Nets, ``global``
for trained alignment heads). ``pool_to_vector`` normalizes them into a
single L2-normalized ``(B, D)`` tensor suitable for downstream
classification, retrieval, or k-NN heads.

Public API
----------
    pool_to_vector(extractor_output, rule)        per-image pooling
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


# Common spatial grid that per-layer pools downsample to.
_PATCH_GRID = 21


def pool_to_vector(out: dict, rule: str) -> torch.Tensor:
    """Reduce an extractor's output dict to a single L2-normalized vector.

    Args:
        out:   dict returned by an extractor in this package.
        rule:  one of ``cls``, ``cls_plus_patch_mean``,
               ``per_layer_mean``, ``global``.

    Returns:
        ``(B, D)`` per-image feature tensor, L2-normalized.
    """
    if rule == "cls":
        return F.normalize(out["cls"], dim=-1)
    if rule == "cls_plus_patch_mean":
        cls = F.normalize(out["cls"], dim=-1)
        mean = F.normalize(out["patch_mean"], dim=-1)
        return F.normalize(torch.cat([cls, mean], dim=-1), dim=-1)
    if rule == "per_layer_mean":
        parts = []
        for layer_idx in sorted(out["per_layer"]):
            feat = out["per_layer"][layer_idx]
            if feat.shape[-1] != _PATCH_GRID:
                feat = F.interpolate(
                    feat, size=(_PATCH_GRID, _PATCH_GRID),
                    mode="bilinear", align_corners=False,
                )
            parts.append(F.normalize(feat.mean(dim=(2, 3)), dim=-1))
        return F.normalize(torch.cat(parts, dim=-1), dim=-1)
    if rule == "global":
        return F.normalize(out["global"], dim=-1)
    raise ValueError(f"Unknown pool rule {rule!r}")
