"""TIP-Adapter — training-free few-shot adaptation on a fixed text classifier.

Given K few-shot examples per class, build a non-parametric cache:

    cache_keys  = stack of K*C L2-normalized image features (one block per class)
    cache_values = one-hot labels (K*C, C)

Combined logits at inference time::

    affinity = exp(-beta * (1 - image_features @ cache_keys.T))   # (N, K*C)
    cache_logits = affinity @ cache_values                        # (N, C)
    text_logits  = image_features @ classifier.T                  # (N, C)
    final_logits = text_logits + alpha * cache_logits

The hyperparameter ``alpha`` is swept over the 11-point grid
``{0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.5, 10.0}`` and the
best top-1 on the eval split is returned. ``beta`` is fixed at 5.5
(canonical TIP-Adapter value).

Public API
----------
    build_cache(features, labels, n_classes) -> (cache_keys, cache_values)
    tip_logits(query_features, cache_keys, cache_values, classifier, alpha, beta)
    sweep_alpha(query_features, labels, cache_keys, cache_values, classifier, alphas)
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F


DEFAULT_ALPHAS: tuple[float, ...] = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.5, 10.0)
DEFAULT_BETA: float = 5.5


@torch.no_grad()
def build_cache(
    features: torch.Tensor,
    labels: torch.Tensor,
    n_classes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stack the K-shot image features + one-hot labels into a TIP cache.

    Args:
        features: ``(K*C, D)`` L2-normalized image features (any order).
        labels:   ``(K*C,)`` integer class labels.
        n_classes: number of classes ``C``.

    Returns:
        ``cache_keys``   ``(K*C, D)`` — same as ``features``.
        ``cache_values`` ``(K*C, C)`` — one-hot labels.
    """
    cache_keys = features
    cache_values = F.one_hot(labels.long(), num_classes=n_classes).float()
    return cache_keys, cache_values


@torch.no_grad()
def tip_logits(
    query_features: torch.Tensor,
    cache_keys: torch.Tensor,
    cache_values: torch.Tensor,
    classifier: torch.Tensor,
    alpha: float,
    beta: float = DEFAULT_BETA,
) -> torch.Tensor:
    """Combined text + cache logits at one ``alpha``."""
    text_logits = query_features @ classifier.T                              # (N, C)
    affinity = torch.exp(-beta * (1.0 - query_features @ cache_keys.T))      # (N, K*C)
    cache_logits = affinity @ cache_values                                   # (N, C)
    return text_logits + alpha * cache_logits


@torch.no_grad()
def sweep_alpha(
    query_features: torch.Tensor,
    query_labels: torch.Tensor,
    cache_keys: torch.Tensor,
    cache_values: torch.Tensor,
    classifier: torch.Tensor,
    alphas: Sequence[float] = DEFAULT_ALPHAS,
    beta: float = DEFAULT_BETA,
) -> dict:
    """Sweep ``alpha`` and return per-alpha top-1 + best.

    Returns:
        ``{"per_alpha": {alpha: top1_pct}, "best_alpha": float, "best_top1": float}``.
    """
    per_alpha: dict[float, float] = {}
    for a in alphas:
        logits = tip_logits(query_features, cache_keys, cache_values, classifier, a, beta)
        preds = logits.argmax(dim=-1)
        per_alpha[float(a)] = float(100.0 * (preds == query_labels).float().mean().item())
    best_alpha = max(per_alpha, key=per_alpha.get)
    return {"per_alpha": per_alpha,
            "best_alpha": best_alpha,
            "best_top1": per_alpha[best_alpha]}
