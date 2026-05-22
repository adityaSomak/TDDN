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
    zero_shot_predict(patch_features, classifier, output_size) -> mask
    compute_miou(predictions, targets, n_classes, *, ignore_index=255)
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.no_grad()
def zero_shot_predict(
    patch_features: torch.Tensor,
    classifier: torch.Tensor,
    output_size: int,
) -> torch.Tensor:
    """Cosine-sim between patch features and class prototypes → per-pixel label map.

    Args:
        patch_features: ``(B, C, H, W)`` L2-normalized along channels.
        classifier:     ``(K, C)`` L2-normalized class prototypes.
        output_size:    target side length for the upsampled mask.

    Returns:
        ``(B, output_size, output_size)`` integer prediction tensor.
    """
    B, C, H, W = patch_features.shape
    flat = patch_features.permute(0, 2, 3, 1).reshape(-1, C)          # (B*H*W, C)
    logits = (flat @ classifier.T).reshape(B, H, W, classifier.shape[0])
    logits = logits.permute(0, 3, 1, 2)                               # (B, K, H, W)
    logits = F.interpolate(logits, size=(output_size, output_size),
                           mode="bilinear", align_corners=False)
    return logits.argmax(dim=1)


@torch.no_grad()
def compute_miou(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    n_classes: int,
    *,
    ignore_index: int = 255,
) -> dict:
    """Accumulate intersection / union and return per-class + mean IoU.

    Args:
        predictions:  ``(B, H, W)`` predicted class ids.
        targets:      ``(B, H, W)`` ground-truth class ids.
        n_classes:    total class count.
        ignore_index: pixel label to exclude.

    Returns:
        ``{"miou": float, "per_class_iou": list[float]}``.
    """
    valid = (targets != ignore_index) & (targets >= 0) & (targets < n_classes) \
        & (predictions >= 0) & (predictions < n_classes)
    pred = predictions[valid]
    tgt = targets[valid]
    index = tgt * n_classes + pred
    conf = torch.bincount(index, minlength=n_classes ** 2).view(n_classes, n_classes).float()
    diag = conf.diag()
    union = conf.sum(dim=0) + conf.sum(dim=1) - diag
    iou = torch.where(union > 0, diag / union, torch.zeros_like(diag))
    return {"miou": float(iou.mean().item()),
            "per_class_iou": iou.tolist()}
