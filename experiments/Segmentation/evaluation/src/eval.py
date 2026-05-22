"""Segmentation evaluation: report mIoU + pixel accuracy on a split.

Iterates a DataLoader, accumulates a confusion matrix via
``ConfusionMatrixMetric``, and returns headline metrics plus the
per-class IoU vector.

Public API
----------
    evaluate(lit_module, dataloader, n_classes, *, miou_weights,
             ignore_index, device)
        Returns ``{"miou": float, "pixel_acc": float, "per_class_iou": [...]}``.
"""
from __future__ import annotations

import torch

from training.src.metrics import ConfusionMatrixMetric


@torch.no_grad()
def evaluate(
    lit_module,
    dataloader,
    n_classes: int,
    *,
    miou_weights: torch.Tensor,
    ignore_index: int = 255,
    device: str = "cuda",
) -> dict:
    """Run a checkpointed Lightning module over ``dataloader`` and report metrics.

    Args:
        lit_module:   a ``SegmentationLitModule`` (loaded from a ``.ckpt``).
        dataloader:   PyTorch DataLoader yielding ``(image, mask, meta)``.
        n_classes:    number of segmentation classes.
        miou_weights: ``(C,)`` per-class weights for the mIoU readout.
        ignore_index: pixel label to exclude from accumulation.
        device:       torch device.

    Returns:
        Dict with ``miou``, ``pixel_acc`` and ``per_class_iou``.
    """
    lit_module.eval().to(device)
    metric = ConfusionMatrixMetric(n_classes, weights=miou_weights,
                                   ignore_index=ignore_index).to(device)
    for images, masks, _ in dataloader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        logits = lit_module(images)
        metric.update(logits, masks)
    headline = metric.compute()
    conf = metric.conf.float()
    diag = conf.diag()
    union = conf.sum(dim=0) + conf.sum(dim=1) - diag
    per_class = torch.where(union > 0, diag / union, torch.zeros_like(diag)).tolist()
    return {**headline, "per_class_iou": per_class}
