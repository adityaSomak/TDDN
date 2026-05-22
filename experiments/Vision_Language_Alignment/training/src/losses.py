"""Contrastive + structure-preservation losses.

Two terms are combined at every training step:

  - **Symmetric CLIP InfoNCE** over L2-normalized image and text
    embeddings, with all-gathered negatives across GPUs. The gather
    is differentiable so gradients flow back to every rank's local
    contribution.

  - **Structure regularizer** (Jensen-Shannon divergence between the
    softmaxed similarity matrix of the *aligned* embeddings and the
    softmaxed similarity matrix of the *frozen reference*
    embeddings, averaged over a configurable number of
    matrix-power "levels"). Encourages the aligned embeddings to
    preserve the intra-modal similarity structure of the frozen
    backbones.

Public API
----------
    gathered_clip_loss(image_emb, text_emb, temperature, label_smoothing) -> scalar
    StructureLoss(base_lambda, warmup_steps, ...)
        forward(image_original, image_aligned, text_original, text_aligned, logit_scale)
            -> (scalar, info_dict)
        step()                # advance the warmup counter
"""
from __future__ import annotations

from enum import Enum
from typing import Optional, Tuple

import torch
import torch.distributed as dist
import torch.distributed.nn as dnn
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def clip_gradients(model: nn.Module, clip: float) -> list[float]:
    """Per-parameter L2 gradient clipping.

    Iterates each parameter independently and scales its gradient by
    ``clip / (param_norm + 1e-6)`` whenever ``param_norm > clip``.
    Differs from ``torch.nn.utils.clip_grad_norm_`` which computes a
    single total norm across all parameters and scales them uniformly.
    """
    norms: list[float] = []
    for _name, p in model.named_parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            norms.append(param_norm.item())
            clip_coef = clip / (param_norm + 1e-6)
            if clip_coef < 1:
                p.grad.data.mul_(clip_coef)
    return norms


def _are_normalized(x: torch.Tensor, eps: float = 1e-6) -> bool:
    """Return True if every row of ``x`` has unit L2 norm to within ``eps``."""
    norms = x.norm(p=2, dim=-1)
    return torch.all((norms - 1.0).abs() < eps).item()


def _safe_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """L2-normalize along the last dimension, no-op if already normalized.

    Skipping the re-normalize when the input is already unit-norm avoids
    extending the autograd graph for a numerically-identical op.
    """
    if _are_normalized(x):
        return x
    return F.normalize(x, p=2, dim=-1, eps=eps)


class Centering(Enum):
    none = "none"
    mean = "mean"
    standard = "standard"


class DistanceFunction(Enum):
    cosine = "cosine"
    rbf = "rbf"


def _center(x: torch.Tensor, ctype: Centering) -> torch.Tensor:
    if ctype == Centering.none:
        return x
    if ctype == Centering.mean:
        return x - x.mean(dim=0, keepdim=True)
    if ctype == Centering.standard:
        return (x - x.mean(dim=0, keepdim=True)) / x.std(dim=0, keepdim=True)
    return x


def _similarity(x: torch.Tensor, dtype: DistanceFunction,
                temperature: float, gamma: float = 1.0) -> torch.Tensor:
    if dtype == DistanceFunction.cosine:
        return (x @ x.T) / temperature
    if dtype == DistanceFunction.rbf:
        d2 = torch.cdist(x, x, p=2).pow(2)
        return torch.exp(-gamma * d2)
    raise ValueError(f"Unknown distance {dtype!r}")


# ---------------------------------------------------------------------------
# Contrastive loss
# ---------------------------------------------------------------------------

def gathered_clip_loss(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    temperature: float = 0.05,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Symmetric InfoNCE with differentiable all-gathered negatives.

    On a single rank or with ``torch.distributed`` uninitialized this
    collapses to the standard in-batch InfoNCE.
    """
    image_features = _safe_normalize(image_features)
    text_features = _safe_normalize(text_features)

    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        all_image = torch.cat(dnn.all_gather(image_features), dim=0)
        all_text = torch.cat(dnn.all_gather(text_features), dim=0)
        rank = dist.get_rank()
        batch = image_features.shape[0]
        labels = (torch.arange(batch, device=image_features.device)
                  + rank * batch)
        logits_i2t = image_features @ all_text.T / temperature
        logits_t2i = text_features @ all_image.T / temperature
    else:
        logits_i2t = image_features @ text_features.T / temperature
        logits_t2i = logits_i2t.T
        labels = torch.arange(image_features.shape[0],
                              device=image_features.device)

    loss_i = F.cross_entropy(logits_i2t, labels, label_smoothing=label_smoothing)
    loss_t = F.cross_entropy(logits_t2i, labels, label_smoothing=label_smoothing)
    return 0.5 * (loss_i + loss_t)


# ---------------------------------------------------------------------------
# Structure regularizer
# ---------------------------------------------------------------------------

def structure_reg(
    original_embeddings: torch.Tensor,
    aligned_embeddings: torch.Tensor,
    levels: int = 3,
    temperature: float = 0.05,
    gamma: float = 1.0,
    margin: float = 0.0,
    eps: float = 1e-12,
    weighting: str = "inverse",
    distance_type: DistanceFunction = DistanceFunction.cosine,
    centering_type: Centering = Centering.mean,
    center_first: bool = False,
) -> torch.Tensor:
    """Multi-level Jensen-Shannon divergence between two similarity matrices.

    For each level ``l`` in ``1..levels`` the (softmaxed) similarity
    matrix is raised to the ``l``-th matrix power; the JS divergence
    between the aligned and reference soft structures is then taken,
    optionally weighted as ``1 / l``. The result is averaged over
    levels.
    """
    with torch.amp.autocast("cuda", enabled=False):
        if center_first:
            original_embeddings = _center(original_embeddings, centering_type)
            aligned_embeddings = _center(aligned_embeddings, centering_type)

        original_norm = _safe_normalize(original_embeddings)
        aligned_norm = _safe_normalize(aligned_embeddings)

        if not center_first:
            original_norm = _center(original_norm, centering_type)
            aligned_norm = _center(aligned_norm, centering_type)

        original_sim = _similarity(original_norm, distance_type, temperature, gamma)
        aligned_sim = _similarity(aligned_norm, distance_type, temperature, gamma)

        total = torch.tensor(0.0, device=aligned_embeddings.device)
        for level in range(1, levels + 1):
            original_struct = torch.matrix_power(F.softmax(original_sim, dim=-1), level)
            aligned_struct = torch.matrix_power(F.softmax(aligned_sim, dim=-1), level)
            m = 0.5 * (original_struct + aligned_struct)
            js = 0.5 * (
                F.kl_div((aligned_struct + eps).log(), m + eps, reduction="batchmean")
                + F.kl_div((original_struct + eps).log(), m + eps, reduction="batchmean")
            )
            js = F.relu(js - margin)
            if weighting == "none":
                total = total + js
            elif weighting == "inverse":
                total = total + js * (1.0 / level)
            else:
                raise ValueError(f"Unknown weighting {weighting!r}")
    return total / levels


class StructureLoss(nn.Module):
    """Joint image+text structure regularizer with internal warmup.

    The effective coefficient is
    ``base_lambda * min(1, train_step / warmup_steps)``; call
    :meth:`step` once per training iteration to advance the schedule.
    """

    def __init__(
        self,
        base_lambda: float = 10.0,
        warmup_steps: int = 1,
        temperature: float = 0.05,
        levels: int = 1,
        weighting: str = "inverse",
        margin: float = 0.0,
        centering: str = "mean",
        distance: str = "cosine",
        center_first: bool = False,
    ):
        super().__init__()
        self.base_lambda = base_lambda
        self.warmup_steps = max(int(warmup_steps), 1)
        self.temperature = temperature
        self.levels = levels
        self.weighting = weighting
        self.margin = margin
        self.centering = Centering(centering)
        self.distance = DistanceFunction(distance)
        self.center_first = center_first
        self.train_step = 0

    @property
    def current_lambda(self) -> float:
        return self.base_lambda * min(1.0, self.train_step / self.warmup_steps)

    def step(self) -> None:
        """Advance the warmup counter by one iteration."""
        self.train_step += 1

    def forward(
        self,
        image_original: torch.Tensor,
        image_aligned: torch.Tensor,
        text_original: torch.Tensor,
        text_aligned: torch.Tensor,
        logit_scale: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        img_loss = structure_reg(
            image_original.float(), image_aligned.float(),
            levels=self.levels, temperature=self.temperature,
            weighting=self.weighting, margin=self.margin,
            distance_type=self.distance, centering_type=self.centering,
            center_first=self.center_first,
        )
        txt_loss = structure_reg(
            text_original.float(), text_aligned.float(),
            levels=self.levels, temperature=self.temperature,
            weighting=self.weighting, margin=self.margin,
            distance_type=self.distance, centering_type=self.centering,
            center_first=self.center_first,
        )
        raw = (img_loss + txt_loss) / 2
        lam = self.current_lambda
        return lam * raw, {"structure_loss_raw": float(raw.item()),
                           "structure_lambda": lam}
