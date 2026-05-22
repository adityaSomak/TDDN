"""Segmentation head modules.

Trained on top of frozen backbone features. The head is responsible
for:

  1. Mapping per-patch feature vectors to per-class logits.
  2. Bilinearly upsampling the logit grid to the target image resolution.

Two head shapes are supported, controlled by ``hidden_dims``:

  - ``hidden_dims=()``       linear probe (single 1x1 ``Conv2d``).
  - ``hidden_dims=(H1, ...)``  shallow MLP (``Conv2d -> ReLU -> ...``).

Public API
----------
    SegmentationHead(in_channels, n_classes, *, hidden_dims=(), output_size=512)
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class SegmentationHead(nn.Module):
    """Linear or shallow MLP segmentation head.

    Args:
        in_channels:  channel dimension of the input feature map.
        n_classes:    number of segmentation classes.
        hidden_dims:  hidden channels for an MLP head; ``()`` for a
                      linear probe.
        output_size:  side length the logit grid is bilinearly
                      upsampled to before the loss is taken.
    """

    def __init__(
        self,
        in_channels: int,
        n_classes: int,
        *,
        hidden_dims: Sequence[int] = (),
        output_size: int = 512,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        ch = in_channels
        for hidden in hidden_dims:
            layers.append(nn.Conv2d(ch, hidden, kernel_size=1))
            layers.append(nn.ReLU(inplace=True))
            ch = hidden
        layers.append(nn.Conv2d(ch, n_classes, kernel_size=1))
        self.net = nn.Sequential(*layers)
        self.output_size = output_size

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Project per-patch features to ``(B, n_classes, S, S)`` logits."""
        logits = self.net(features)
        return F.interpolate(
            logits, size=(self.output_size, self.output_size),
            mode="bilinear", align_corners=False,
        )
