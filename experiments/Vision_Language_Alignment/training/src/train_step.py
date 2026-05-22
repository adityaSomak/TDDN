"""One-iteration training step: accumulate, compute loss, backward.

Combines :func:`accumulate_micro_batches` with the contrastive +
structure losses and does a single backward pass. Returns a small
dict of scalar losses for logging.

Public API
----------
    linear_warmup_cosine_decay(peak_lr, warmup_iters, total_iters, min_lr)
        -> 1-D numpy schedule indexed by iteration.
    apply_learning_rate(optimizer, lr)
    is_no_decay(name, param) -> bool
    train_step(model, data_iter, ...) -> dict[str, float]
"""
from __future__ import annotations

import math
from typing import Iterator, Optional

import numpy as np
import torch
import torch.nn as nn

from .grad_cache import accumulate_micro_batches
from .losses import StructureLoss, clip_gradients, gathered_clip_loss


# ---------------------------------------------------------------------------
# Learning-rate schedule
# ---------------------------------------------------------------------------

def linear_warmup_cosine_decay(
    peak_lr: float,
    warmup_iters: int,
    total_iters: int,
    min_lr: float = 0.0,
) -> np.ndarray:
    """Linear warmup from 0 → peak_lr over ``warmup_iters``, then cosine to ``min_lr``."""
    schedule = np.zeros(total_iters, dtype=np.float64)
    for i in range(total_iters):
        if i < warmup_iters:
            schedule[i] = peak_lr * (i + 1) / max(1, warmup_iters)
        else:
            progress = (i - warmup_iters) / max(1, total_iters - warmup_iters)
            schedule[i] = min_lr + 0.5 * (peak_lr - min_lr) * (
                1.0 + math.cos(math.pi * progress)
            )
    return schedule


def apply_learning_rate(optimizer: torch.optim.Optimizer, lr: float) -> None:
    """Set every parameter group's learning rate to ``lr``."""
    for pg in optimizer.param_groups:
        pg["lr"] = lr


# ---------------------------------------------------------------------------
# Parameter grouping (no weight decay on 1-D params, biases, and logit_scale)
# ---------------------------------------------------------------------------

def is_no_decay(name: str, param: torch.Tensor) -> bool:
    """Return True for parameters that should not have weight decay applied."""
    return (param.ndim < 2
            or "bias" in name
            or "ln" in name
            or "bn" in name
            or "logit_scale" in name)


# ---------------------------------------------------------------------------
# One training iteration
# ---------------------------------------------------------------------------

def train_step(
    *,
    model: nn.Module,
    data_iter: Iterator,
    loader=None,
    optimizer: torch.optim.Optimizer,
    structure_loss: Optional[StructureLoss],
    n_accum: int,
    cur_iter: int,
    lr: float,
    temperature: float,
    label_smoothing: float = 0.0,
    gradient_clip: Optional[float] = None,
    device: torch.device | str = "cuda",
) -> dict:
    """Run one training iteration and return scalar loss components for logging.

    The contrastive temperature is fixed at ``temperature``; the
    structure-loss coefficient is owned by ``structure_loss`` itself
    (warm-up handled inside its ``current_lambda``).
    """
    apply_learning_rate(optimizer, lr)
    optimizer.zero_grad(set_to_none=True)

    pool = accumulate_micro_batches(model, data_iter, n_accum, device,
                                    loader=loader)

    contrastive = gathered_clip_loss(
        pool["image_features"], pool["text_features"],
        temperature=temperature, label_smoothing=label_smoothing,
    )
    total = contrastive

    struct_val = torch.tensor(0.0, device=contrastive.device)
    if structure_loss is not None:
        struct_val, _info = structure_loss(
            pool["image_original"], pool["image_features"],
            pool["text_original"],  pool["text_features"],
            pool["logit_scale"],
        )
        total = total + struct_val
        structure_loss.step()

    if torch.isnan(total):
        raise RuntimeError(f"NaN loss at iteration {cur_iter}")

    total.backward()
    if gradient_clip is not None:
        clip_gradients(model, gradient_clip)
    optimizer.step()

    return {
        "total":       float(total.item()),
        "contrastive": float(contrastive.item()),
        "structure":   float(struct_val.item()),
        "lr":          float(lr),
        "logit_scale": float(pool["logit_scale"].item()),
    }
