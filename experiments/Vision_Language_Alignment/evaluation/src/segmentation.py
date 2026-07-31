"""Zero-shot open-vocabulary segmentation via cosine sim.

Per-pixel labels are produced by:

  1. Extracting per-patch image features ``(B, C, H, W)`` from the
     alignment model's vision encoder.
  2. Building per-class text embeddings ``(K, C)`` via the standard
     ``classifier.build_*`` helpers (zero-shot template or CuPL).
  3. Cosine-similarity between every patch and every class →
     ``(B, K, H, W)``.
  4. Bilinearly upsampling the class-score map to the input resolution.
  5. Argmax → per-pixel class id.

No segmentation head is trained — every learned parameter lives in the
alignment encoders and the text classifier.

Public API
----------
    accumulate_confusion(predictions, targets, n_classes, *, ignore_index=255) -> conf
    miou_from_confusion(conf) -> dict

Predictions are scored at each image's own resolution (no fixed square, no
letterbox), so ``accumulate_confusion`` takes a running confusion-matrix
total across images rather than stacking them into one fixed-size tensor.
"""
from __future__ import annotations

import torch


@torch.no_grad()
def accumulate_confusion(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    n_classes: int,
    *,
    ignore_index: int = 255,
) -> torch.Tensor:
    """Build an ``(n_classes, n_classes)`` confusion matrix for one chunk.

    Args:
        predictions:  predicted class ids, any shape.
        targets:      ground-truth class ids, same shape as ``predictions``.
        n_classes:    total class count.
        ignore_index: pixel label to exclude.

    Returns:
        ``(n_classes, n_classes)`` float confusion matrix (rows = target,
        cols = prediction) — callers accumulate across chunks by summing
        this with a running total before calling ``miou_from_confusion``.
    """
    valid = (targets != ignore_index) & (targets >= 0) & (targets < n_classes) \
        & (predictions >= 0) & (predictions < n_classes)
    pred = predictions[valid]
    tgt = targets[valid]
    index = tgt * n_classes + pred
    return torch.bincount(index, minlength=n_classes ** 2).view(n_classes, n_classes).float()


def miou_from_confusion(conf: torch.Tensor) -> dict:
    """Per-class + mean IoU from an accumulated confusion matrix.

    Returns:
        ``{"miou": float, "per_class_iou": list[float]}``.
    """
    diag = conf.diag()
    union = conf.sum(dim=0) + conf.sum(dim=1) - diag
    iou = torch.where(union > 0, diag / union, torch.zeros_like(diag))
    return {"miou": float(iou.mean().item()),
            "per_class_iou": iou.tolist()}
