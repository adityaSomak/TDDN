"""Three PCA recipes used by the diffusion eval pipelines.

  - `fit_global_pca / apply_pca` — segmentation: fit one basis per layer
    on a fixed training subset, then apply to all splits. 512 per layer.

  - `hierarchical_co_pca`        — SPair: per-pair joint PCA on [src, tgt]
    per layer, bilinear-upsample each layer to the largest spatial grid,
    concat. 512 per layer.

  - `per_image_pca_rgb`          — pca_viz: PCA(3) on a single image's
    flat patch features → per-channel min-max (or 2-98 percentile) → RGB.
    Used by both `patches` and `interpolated` rendering modes.

ImageNet_Classification imports nothing from this module — it never PCA-reduces the
diffusion features (`run_knn.py:57`).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA


# ============================================================================
# Global PCA — fit once, apply many. Segmentation recipe.
# ============================================================================

@dataclass
class GlobalPCABasis:
    """Per-layer PCA bases (mean + components) for later `apply_pca` calls.

    `per_layer[layer_idx]` is a fitted `sklearn.decomposition.PCA` object.
    Stored as-is so we can call `.transform` directly.
    """
    per_layer: dict[int, PCA]
    n_components_per_layer: int


def fit_global_pca(
    features_by_layer: dict[int, list[torch.Tensor]],
    n_components_per_layer: int = 512,
) -> GlobalPCABasis:
    """Fit one PCA basis per layer over the concatenated training samples.

    `features_by_layer[idx]` is a list of `(C, H, W)` tensors (one per
    training image; H/W may vary by layer but match within a layer).
    """
    bases: dict[int, PCA] = {}
    for layer_idx, feats in features_by_layer.items():
        flat = []
        for f in feats:
            C, H, W = f.shape
            flat.append(f.reshape(C, H * W).T.float().cpu().numpy())   # (H*W, C)
        all_pts = np.concatenate(flat, axis=0)
        n_comp = min(n_components_per_layer, all_pts.shape[1])
        pca = PCA(n_components=n_comp)
        pca.fit(all_pts)
        bases[layer_idx] = pca
    return GlobalPCABasis(per_layer=bases, n_components_per_layer=n_components_per_layer)


def apply_pca(
    features_by_layer: dict[int, torch.Tensor],
    basis: GlobalPCABasis,
    target_grid: tuple[int, int] | None = None,
) -> torch.Tensor:
    """Project + spatial-align + concat per-layer features → (B, C_total, H, W).

    Each layer is independently transformed by its fitted basis, then
    bilinear-resampled to the same target grid (default: max H, W in input).
    """
    if target_grid is None:
        target_h = max(f.shape[-2] for f in features_by_layer.values())
        target_w = max(f.shape[-1] for f in features_by_layer.values())
        target_grid = (target_h, target_w)

    projected = []
    for layer_idx, f in features_by_layer.items():
        B, C, H, W = f.shape
        flat = f.reshape(B, C, H * W).permute(0, 2, 1).cpu().numpy()    # (B, HW, C)
        pca = basis.per_layer[layer_idx]
        # sklearn transform expects (N_samples, N_features); batch the B dim.
        out = np.stack([pca.transform(flat[b]) for b in range(B)], axis=0)  # (B, HW, k)
        out_t = torch.from_numpy(out).to(f.device).float()
        out_t = out_t.permute(0, 2, 1).reshape(B, -1, H, W)
        if (H, W) != target_grid:
            out_t = F.interpolate(out_t, size=target_grid, mode="bilinear", align_corners=False)
        projected.append(out_t)
    return torch.cat(projected, dim=1)


# ============================================================================
# Hierarchical co-PCA — SPair recipe (per-pair).
# ============================================================================

def hierarchical_co_pca(
    src_features_by_layer: dict[int, torch.Tensor],
    tgt_features_by_layer: dict[int, torch.Tensor],
    n_components_per_layer: int = 512,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit PCA jointly on [src, tgt] features per layer; upsample + concat.

    Each `dict[layer_idx]` value is `(1, C, H, W)` or `(B, C, H, W)`. We
    fit a PCA basis on the concatenated [src, tgt] patches per layer, then
    bilinear-upsample to the largest layer's grid and concat.
    """
    if src_features_by_layer.keys() != tgt_features_by_layer.keys():
        raise ValueError("src and tgt must contain identical layer indices.")

    target_h = max(f.shape[-2] for f in src_features_by_layer.values())
    target_w = max(f.shape[-1] for f in src_features_by_layer.values())

    src_parts, tgt_parts = [], []
    for layer_idx in src_features_by_layer:
        sf = src_features_by_layer[layer_idx]
        tf = tgt_features_by_layer[layer_idx]
        B, C, H, W = sf.shape
        sf_flat = sf.reshape(B, C, H * W).permute(0, 2, 1).cpu().numpy()
        tf_flat = tf.reshape(B, C, H * W).permute(0, 2, 1).cpu().numpy()
        joint = np.concatenate([sf_flat.reshape(-1, C), tf_flat.reshape(-1, C)], axis=0)
        n_comp = min(n_components_per_layer, C)
        pca = PCA(n_components=n_comp)
        pca.fit(joint)

        def _project(arr_flat_B_HW_C, B):
            out = np.stack([pca.transform(arr_flat_B_HW_C[b]) for b in range(B)], axis=0)
            return out  # (B, HW, k)

        sp = _project(sf_flat, B)
        tp = _project(tf_flat, B)
        sp_t = torch.from_numpy(sp).float().permute(0, 2, 1).reshape(B, -1, H, W).to(sf.device)
        tp_t = torch.from_numpy(tp).float().permute(0, 2, 1).reshape(B, -1, H, W).to(tf.device)
        if (H, W) != (target_h, target_w):
            sp_t = F.interpolate(sp_t, size=(target_h, target_w), mode="bilinear", align_corners=False)
            tp_t = F.interpolate(tp_t, size=(target_h, target_w), mode="bilinear", align_corners=False)
        src_parts.append(sp_t)
        tgt_parts.append(tp_t)

    return torch.cat(src_parts, dim=1), torch.cat(tgt_parts, dim=1)


# ============================================================================
# Per-image PCA(3) → RGB. pca_viz recipe.
# ============================================================================

def raw_concat_layers(
    layers: dict[int, torch.Tensor],
) -> torch.Tensor:
    """Bilinear-align each per-layer feature to the largest grid, then concat.

    No PCA reduction, no L2-norm between layers. Keeps the underlying
    feature distribution intact so a backbone's standalone palette and
    its fusion-partner palette stay consistent.

    ``layers[layer_idx]`` is ``(1, C_l, H_l, W_l)``. Returns
    ``(1, sum(C_l), H_max, W_max)``.
    """
    target_h = max(f.shape[-2] for f in layers.values())
    target_w = max(f.shape[-1] for f in layers.values())
    out_parts = []
    for _, f in sorted(layers.items()):
        if f.shape[-2:] != (target_h, target_w):
            f = F.interpolate(f, size=(target_h, target_w),
                              mode="bilinear", align_corners=False)
        out_parts.append(f)
    return torch.cat(out_parts, dim=1)


def per_image_pca_layer_reduce(
    layers: dict[int, torch.Tensor],
    n_components_per_layer: int = 512,
    normalize_per_layer: bool = False,
) -> torch.Tensor:
    """Per-image, per-layer PCA reduction with bilinear-aligned concat.

    Order of operations: upsample each layer to the largest layer's grid,
    then fit a per-image PCA. Fitting after upsampling lets the deepest
    diffusion layer (which has few native patches at high input resolution)
    contribute a meaningfully ranked basis.

    ``layers[layer_idx]`` is ``(1, C, H, W)``. Returns ``(1, K_total, H_max, W_max)``
    where each layer contributes ``min(n_components_per_layer, C, H_max*W_max)``
    channels. If ``normalize_per_layer`` is True, each reduced layer is
    L2-normalized along the channel axis before concatenation.
    """
    target_h = max(f.shape[-2] for f in layers.values())
    target_w = max(f.shape[-1] for f in layers.values())
    HW_max = target_h * target_w
    out_parts = []
    for _, f in sorted(layers.items()):
        if f.shape[-2:] != (target_h, target_w):
            f = F.interpolate(f, size=(target_h, target_w),
                              mode="bilinear", align_corners=False)
        B, C, H, W = f.shape
        flat = f.reshape(B, C, H * W).permute(0, 2, 1).cpu().numpy()  # (B, HW, C)
        k = min(n_components_per_layer, C, HW_max)
        pca = PCA(n_components=k)
        reduced = pca.fit_transform(flat[0])                          # (HW, k)
        reduced_t = (torch.from_numpy(reduced).float()
                          .permute(1, 0).reshape(1, k, H, W).to(f.device))
        if normalize_per_layer:
            reduced_t = F.normalize(reduced_t, dim=1)
        out_parts.append(reduced_t)
    return torch.cat(out_parts, dim=1)


def per_image_pca_rgb(
    features: torch.Tensor | np.ndarray,
    patch_h: int,
    patch_w: int,
    target_size: int = 512,
    mode: str = "patches",
    normalize: str = "minmax",
    interp: str = "bilinear",
) -> np.ndarray:
    """PCA(3) on a single image's features → RGB float32 in [0, 1].

    `mode="patches"`       — fit PCA at the native patch resolution, then
                             bilinear-upsample the small RGB to `target_size`.
    `mode="interpolated"`  — bilinear-upsample the *features* to `target_size`
                             first, then fit PCA at the dense resolution.

    `normalize="minmax"`    — per-channel min/max over (H, W) → [0, 1].
    `normalize="percentile"`— per-channel 2-98 percentile clip → [0, 1].
    """
    if isinstance(features, torch.Tensor):
        features = features.detach().float().cpu().numpy()
    feats_2d = features.reshape(patch_h, patch_w, -1)

    def _normalize_rgb(rgb: np.ndarray) -> np.ndarray:
        if normalize == "percentile":
            for c in range(3):
                lo, hi = np.percentile(rgb[..., c], [2, 98])
                rgb[..., c] = np.clip((rgb[..., c] - lo) / (hi - lo + 1e-8), 0, 1)
            return rgb
        lo = rgb.min(axis=(0, 1), keepdims=True)
        hi = rgb.max(axis=(0, 1), keepdims=True)
        return (rgb - lo) / (hi - lo + 1e-8)

    if mode == "patches":
        flat = feats_2d.reshape(-1, feats_2d.shape[-1])
        pca = PCA(n_components=3)
        rgb_small = pca.fit_transform(flat).reshape(patch_h, patch_w, 3).astype(np.float32)
        rgb_small = _normalize_rgb(rgb_small)
        rgb_t = torch.from_numpy(rgb_small).permute(2, 0, 1).unsqueeze(0).float()
        rgb_t = F.interpolate(rgb_t, size=(target_size, target_size),
                              mode=interp, align_corners=False if interp != "nearest" else None)
        return rgb_t.squeeze(0).permute(1, 2, 0).numpy()

    if mode == "interpolated":
        feats_t = torch.from_numpy(feats_2d).permute(2, 0, 1).unsqueeze(0).float()
        feats_t = F.interpolate(feats_t, size=(target_size, target_size),
                                mode=interp, align_corners=False if interp != "nearest" else None)
        feats_interp = feats_t.squeeze(0).permute(1, 2, 0).numpy()
        flat = feats_interp.reshape(-1, feats_interp.shape[-1])
        pca = PCA(n_components=3)
        rgb = pca.fit_transform(flat).reshape(target_size, target_size, 3).astype(np.float32)
        return _normalize_rgb(rgb)

    raise ValueError(f"Unknown mode {mode!r}. Choose 'patches' or 'interpolated'.")
