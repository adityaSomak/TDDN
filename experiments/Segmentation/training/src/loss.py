"""Weighted cross-entropy + soft-Dice loss for semantic segmentation.

The Dice term reduces the dominance of background pixels at the start
of training and improves boundary fidelity once the linear head has
converged on the easy classes. The CE term carries the bulk of the
gradient signal.

Public API
----------
    CombinedSegLoss(class_weights, *, dice_weight=0.3, ignore_index=255)
        Module returning ``ce + dice_weight * soft_dice``.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def soft_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Multi-class soft Dice, averaged over classes.

    Softmaxes the logits, one-hots the target, accumulates intersection
    and cardinality over ``(B, H, W)``, returns ``1 - mean(per-class Dice)``.

    Args:
        logits: ``(B, C, H, W)`` raw class scores.
        target: ``(B, H, W)`` integer labels in ``[0, C)``.

    Returns:
        Scalar in ``[0, 1]``.
    """
    n_classes = logits.shape[1]
    probs = F.softmax(logits, dim=1)
    onehot = F.one_hot(target, num_classes=n_classes).permute(0, 3, 1, 2).float()
    intersection = (probs * onehot).sum(dim=(0, 2, 3))
    cardinality = probs.sum(dim=(0, 2, 3)) + onehot.sum(dim=(0, 2, 3))
    dice = (2 * intersection + eps) / (cardinality + eps)
    return 1.0 - dice.mean()


class CombinedSegLoss(nn.Module):
    """Sum of a weighted-CE term and a soft-Dice term.

    Args:
        class_weights: ``(C,)`` per-class CE weights.
        dice_weight:   weight on the soft-Dice contribution; CE is 1.0.
        ignore_index:  pixel label to exclude from the CE term.
    """

    def __init__(
        self,
        class_weights: torch.Tensor,
        *,
        dice_weight: float = 0.3,
        ignore_index: int = 255,
    ):
        super().__init__()
        self.register_buffer("class_weights", class_weights)
        self.dice_weight = dice_weight
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Return ``CE(logits, target) + dice_weight * SoftDice(logits, target)``."""
        ce = F.cross_entropy(
            logits, target,
            weight=self.class_weights.to(logits.dtype),
            ignore_index=self.ignore_index,
        )
        dice = soft_dice_loss(logits, target)
        return ce + self.dice_weight * dice
