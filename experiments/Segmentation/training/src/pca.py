"""Per-layer Global PCA — fit on a training subset, apply during fwd.

When an extractor returns multi-layer feature maps with very high
channel counts (e.g. diffusion U-Nets), fitting one PCA basis per
layer on a small sample of training images and projecting every patch
through it cuts the head's input dimension dramatically without
hurting the linear probe's accuracy.

Fitting uses ``torch.pca_lowrank`` directly on GPU. Bases are plain
``dict[int, dict[str, Tensor]]`` and round-trip cleanly through
``torch.save`` / ``torch.load`` — no sklearn dependency.

Public API
----------
    fit_layer_pca(extractor, dataset, layers, *, n_samples=200,
                  n_components=512, seed=42, device="cuda")
        Sample ``n_samples`` images, collect per-layer patches, fit
        PCA on the joint pool per layer.
    apply_pca_layers(per_layer, basis)
        Project ``(B, C, H, W)`` feature maps through the saved basis
        per layer.
"""
from __future__ import annotations

import random
from typing import Sequence

import torch
from torch.utils.data import Dataset


@torch.no_grad()
def fit_layer_pca(
    extractor,
    dataset: Dataset,
    layers: Sequence[int],
    *,
    n_samples: int = 200,
    n_components: int = 512,
    seed: int = 42,
    device: str = "cuda",
) -> dict[int, dict[str, torch.Tensor]]:
    """Fit Global PCA per layer on a balanced subsample of the dataset.

    Args:
        extractor:    a built feature extractor whose ``.extract(batch)``
                      returns a dict with a ``per_layer`` field keyed
                      by layer index.
        dataset:      PyTorch Dataset yielding ``(image_tensor, ...)``.
                      Only the first element is consumed.
        layers:       layer indices to fit (must be present in the
                      extractor output's ``per_layer`` dict).
        n_samples:    number of images to draw from ``dataset``.
        n_components: dimensions to keep per layer.
        seed:         seed for the index sampler.
        device:       torch device used for the forward pass and SVD.

    Returns:
        ``{layer_idx: {"mean": (C,) tensor, "components": (k, C) tensor}}``
        on CPU — ready to be passed through ``torch.save``.
    """
    rng = random.Random(seed)
    n_samples = min(n_samples, len(dataset))
    indices = sorted(rng.sample(range(len(dataset)), n_samples))

    per_layer_pool: dict[int, list[torch.Tensor]] = {idx: [] for idx in layers}
    for i in indices:
        sample = dataset[i]
        image = sample[0] if isinstance(sample, (tuple, list)) else sample
        out = extractor.extract(image.unsqueeze(0).to(device))
        for idx in layers:
            feat = out["per_layer"][idx]                       # (1, C, H, W)
            _, C, H, W = feat.shape
            per_layer_pool[idx].append(feat.reshape(C, H * W).T)

    bases: dict[int, dict[str, torch.Tensor]] = {}
    for idx, parts in per_layer_pool.items():
        X = torch.cat(parts, dim=0).float()                    # (N_patches, C)
        mean = X.mean(dim=0)
        centered = X - mean
        k = min(n_components, X.shape[1])
        _, _, V = torch.pca_lowrank(centered, q=k)             # V: (C, k)
        bases[idx] = {"mean": mean.cpu(), "components": V.T.contiguous().cpu()}
        print(f"  layer {idx}: fit on {X.shape[0]} patches "
              f"(C={X.shape[1]} → k={k})")
    return bases


def apply_pca_layers(
    per_layer: dict[int, torch.Tensor],
    basis: dict[int, dict[str, torch.Tensor]],
) -> dict[int, torch.Tensor]:
    """Project per-layer feature maps through their saved PCA basis.

    Args:
        per_layer: ``{layer_idx: (B, C, H, W) tensor}``.
        basis:     output of ``fit_layer_pca``.

    Returns:
        ``{layer_idx: (B, k, H, W) tensor}`` on the same device as the
        input features.
    """
    out: dict[int, torch.Tensor] = {}
    for idx, feat in per_layer.items():
        if idx not in basis:
            out[idx] = feat
            continue
        B, C, H, W = feat.shape
        mean = basis[idx]["mean"].to(feat.device, dtype=feat.dtype)
        comp = basis[idx]["components"].to(feat.device, dtype=feat.dtype)  # (k, C)
        flat = feat.permute(0, 2, 3, 1).reshape(-1, C)         # (B*H*W, C)
        reduced = (flat - mean) @ comp.T                       # (B*H*W, k)
        k = comp.shape[0]
        out[idx] = reduced.reshape(B, H, W, k).permute(0, 3, 1, 2).contiguous()
    return out
