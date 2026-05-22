"""Segmentation metrics: weighted mIoU and pixel accuracy.

A confusion matrix is accumulated across batches via ``update``. At
``compute`` time, intersection-over-union is calculated per class and
averaged with the optional per-class weights (so under-represented
classes can be weighted up against dominant background).

Public API
----------
    ConfusionMatrixMetric(n_classes, *, weights=None, ignore_index=255)
        ``.update(logits, target)`` accumulates; ``.compute()`` returns
        ``{"miou": float, "pixel_acc": float}``.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class ConfusionMatrixMetric(nn.Module):
    """Running confusion matrix with mIoU and pixel-accuracy readout.

    Args:
        n_classes:    number of classes ``C``.
        weights:      optional ``(C,)`` weights for the mean-IoU
                      aggregation. ``None`` -> uniform.
        ignore_index: label value to exclude from accumulation.
    """

    def __init__(
        self,
        n_classes: int,
        *,
        weights: Optional[torch.Tensor] = None,
        ignore_index: int = 255,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.ignore_index = ignore_index
        if weights is None:
            weights = torch.ones(n_classes, dtype=torch.float32)
        self.register_buffer("weights", weights / weights.sum())
        self.register_buffer(
            "conf", torch.zeros(n_classes, n_classes, dtype=torch.int64),
        )

    def reset(self) -> None:
        """Zero the confusion matrix for a new epoch or split."""
        self.conf.zero_()

    @torch.no_grad()
    def update(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        """Accumulate counts from one mini-batch.

        Args:
            logits: ``(B, C, H, W)`` class scores.
            target: ``(B, H, W)`` integer labels in ``[0, C)`` with
                    ``ignore_index`` for don't-care pixels.
        """
        pred = logits.argmax(dim=1)
        valid = target != self.ignore_index
        pred = pred[valid]
        gold = target[valid]
        index = gold * self.n_classes + pred
        counts = torch.bincount(index, minlength=self.n_classes ** 2)
        self.conf += counts.view(self.n_classes, self.n_classes)

    def compute(self) -> dict[str, float]:
        """Return ``{"miou": float, "pixel_acc": float}`` over the running counts."""
        conf = self.conf.float()
        diag = conf.diag()
        union = conf.sum(dim=0) + conf.sum(dim=1) - diag
        iou = torch.where(union > 0, diag / union, torch.zeros_like(diag))
        miou = float((iou * self.weights.to(iou.device)).sum().item())
        pixel_acc = float((diag.sum() / conf.sum().clamp_min(1)).item())
        return {"miou": miou, "pixel_acc": pixel_acc}
