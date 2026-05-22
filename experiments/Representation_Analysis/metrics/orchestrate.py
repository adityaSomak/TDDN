"""Quantitative-pipeline orchestrator.

Reads the per-image features cached by ``extract.extract_features``,
composes the published representation matrices, and writes the four
result CSVs::

    quantitative/global/results/global_quality.csv
    quantitative/global/results/global_similarity.csv
    quantitative/patch/results/patch_quality.csv
    quantitative/patch/results/patch_similarity.csv

Representation labels and pair lists are pinned to match the committed
paper-canonical CSVs (8 global + 7 patch quality rows; 11 global + 6
patch similarity pairs).

Public API
----------
    compute_quality(scope, features_root, image_stems, ...)   -> DataFrame
    compute_similarity(scope, features_root, image_stems, ...) -> DataFrame
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .feature_utils import (
    build_diffusion_combined_pca,
    build_fused,
    build_global_matrix,
    build_patch_matrix_with_indices,
    make_patch_indices,
)
from .quality import effective_rank, uniformity
from .similarity import linear_cka, pwcca

logger = logging.getLogger("repr_analysis.orchestrate")


# ---------------------------------------------------------------------------
# Representation labels (must match the committed CSVs exactly)
# ---------------------------------------------------------------------------

GLOBAL_QUALITY_ORDER = [
    "dino(cls)", "dino(mean)", "cd(2+5+8)", "sd(2+5+8)",
    "clip(image)", "ddn_g", "fused", "vith",
]
PATCH_QUALITY_ORDER = [
    "dino_p", "cd_p", "fused_p", "clip_p",
    "fused_trained_p", "vith_p", "sd_p",
]

# Pair format follows the committed CSV's "a ↔ b" convention.
# ``DDN_g`` is the upper-case alias the similarity rows use for ``ddn_g``.
GLOBAL_SIMILARITY_PAIRS = [
    ("dino(cls)",  "dino(mean)"),
    ("dino(cls)",  "cd(2+5+8)"),
    ("dino(cls)",  "DDN_g"),
    ("dino(mean)", "cd(2+5+8)"),
    ("dino(mean)", "DDN_g"),
    ("cd(2+5+8)",  "DDN_g"),
    ("fused",      "dino(cls)"),
    ("fused",      "dino(mean)"),
    ("fused",      "cd(2+5+8)"),
    ("vith",       "dino(cls)"),
    ("vith",       "dino(mean)"),
]
PATCH_SIMILARITY_PAIRS = [
    ("dino_p", "fused_p"),
    ("dino_p", "fused_trained_p"),
    ("dino_p", "vith_p"),
    ("cd_p",   "fused_p"),
    ("cd_p",   "fused_trained_p"),
    ("cd_p",   "vith_p"),
]
_PAIR_LABEL_ALIASES = {"DDN_g": "ddn_g"}                  # pair-name -> rep-dict key


# ---------------------------------------------------------------------------
# Representation builders
# ---------------------------------------------------------------------------

def _global_matrix(features_root: Path, layer: str, stems: list[str],
                   target: int) -> np.ndarray:
    """Mean-pooled global matrix from one layer of cached features."""
    return build_global_matrix(str(features_root), layer, stems, target=target)


def build_global_representations(
    features_root: Path,
    image_stems: list[str],
    pca_dim: int,
    target: int = 32,
) -> tuple[dict[str, np.ndarray], dict, dict]:
    """Return ``{label: (N, D) matrix}`` plus fitted PCA reducers.

    The reducers are returned so the patch-side build can re-project
    its diffusion features against the same axes.
    """
    reps: dict[str, np.ndarray] = {}

    reps["dino(cls)"]   = _global_matrix(features_root, "dinov3_cls",     image_stems, target)
    reps["dino(mean)"]  = _global_matrix(features_root, "dinov3_patches", image_stems, target)

    cd2 = _global_matrix(features_root, "cd_layer2", image_stems, target)
    cd5 = _global_matrix(features_root, "cd_layer5", image_stems, target)
    cd8 = _global_matrix(features_root, "cd_layer8", image_stems, target)
    cd_combined, cd_reducers = build_diffusion_combined_pca(
        cd2, cd5, cd8, pca_dim=pca_dim,
        layer_keys=("cd2", "cd5", "cd8"),
    )
    reps["cd(2+5+8)"] = cd_combined

    sd2 = _global_matrix(features_root, "sd-2.1_layer2", image_stems, target)
    sd5 = _global_matrix(features_root, "sd-2.1_layer5", image_stems, target)
    sd8 = _global_matrix(features_root, "sd-2.1_layer8", image_stems, target)
    sd_combined, sd_reducers = build_diffusion_combined_pca(
        sd2, sd5, sd8, pca_dim=pca_dim,
        layer_keys=("sd2", "sd5", "sd8"),
    )
    reps["sd(2+5+8)"] = sd_combined

    reps["clip(image)"] = _global_matrix(features_root, "clip_cls", image_stems, target)

    # Handcraft DDN fusion: 0.5·L2(dino_mean) ⊕ 0.5·L2(cd_combined).
    reps["ddn_g"] = build_fused(reps["dino(mean)"], cd_combined)

    reps["fused"] = _global_matrix(features_root, "tddn_global", image_stems, target)
    reps["vith"]  = _global_matrix(features_root, "tdn_global",  image_stems, target)

    return reps, cd_reducers, sd_reducers


def _patch_matrix(features_root: Path, layer: str, stems: list[str],
                  patch_indices: np.ndarray, target: int) -> np.ndarray:
    """Sub-sampled patch matrix from one layer of cached features."""
    return build_patch_matrix_with_indices(
        str(features_root), layer, stems, patch_indices, target=target,
    )


def build_patch_representations(
    features_root: Path,
    image_stems: list[str],
    n_subsample: int,
    pca_dim: int,
    cd_reducers: dict,
    sd_reducers: dict,
    target: int = 32,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Patch-level reps. PCA bases are reused from the global build."""
    patch_indices = make_patch_indices(
        n_images=len(image_stems),
        n_patches_total=target * target,
        n_subsample=n_subsample,
        seed=seed,
    )
    reps: dict[str, np.ndarray] = {}

    reps["dino_p"] = _patch_matrix(features_root, "dinov3_patches",
                                   image_stems, patch_indices, target)

    cd2_p = _patch_matrix(features_root, "cd_layer2", image_stems, patch_indices, target)
    cd5_p = _patch_matrix(features_root, "cd_layer5", image_stems, patch_indices, target)
    cd8_p = _patch_matrix(features_root, "cd_layer8", image_stems, patch_indices, target)
    cd_p, _ = build_diffusion_combined_pca(
        cd2_p, cd5_p, cd8_p, pca_dim=pca_dim,
        reducers=cd_reducers, layer_keys=("cd2", "cd5", "cd8"),
    )
    reps["cd_p"] = cd_p

    sd2_p = _patch_matrix(features_root, "sd-2.1_layer2", image_stems, patch_indices, target)
    sd5_p = _patch_matrix(features_root, "sd-2.1_layer5", image_stems, patch_indices, target)
    sd8_p = _patch_matrix(features_root, "sd-2.1_layer8", image_stems, patch_indices, target)
    sd_p, _ = build_diffusion_combined_pca(
        sd2_p, sd5_p, sd8_p, pca_dim=pca_dim,
        reducers=sd_reducers, layer_keys=("sd2", "sd5", "sd8"),
    )
    reps["sd_p"] = sd_p

    reps["clip_p"] = _patch_matrix(features_root, "clip_patches",
                                   image_stems, patch_indices, target)

    # fused_p (handcraft DDN at patch level).
    reps["fused_p"] = build_fused(reps["dino_p"], cd_p)

    reps["fused_trained_p"] = _patch_matrix(features_root, "tddn_patches",
                                            image_stems, patch_indices, target)
    reps["vith_p"]          = _patch_matrix(features_root, "tdn_patches",
                                            image_stems, patch_indices, target)
    return reps


# ---------------------------------------------------------------------------
# Metric tables
# ---------------------------------------------------------------------------

def quality_table(reps: dict[str, np.ndarray], order: Sequence[str],
                  uniformity_subsample: int | None) -> pd.DataFrame:
    """One row per representation with ``uniformity`` and ``effective_rank``."""
    rows = []
    for name in order:
        X = reps[name]
        rows.append({
            "representation":  name,
            "uniformity":      uniformity(X, n_subsample=uniformity_subsample),
            "effective_rank":  effective_rank(X),
        })
        logger.info(f"  quality  {name:18s}  uniformity={rows[-1]['uniformity']:.4f}  "
                    f"eff_rank={rows[-1]['effective_rank']:.2f}")
    return pd.DataFrame(rows)


def similarity_table(reps: dict[str, np.ndarray],
                     pairs: Sequence[tuple[str, str]]) -> pd.DataFrame:
    """One row per ``(a, b)`` pair with ``linear_cka`` and ``pwcca``."""
    rows = []
    for a, b in pairs:
        Xa = reps[_PAIR_LABEL_ALIASES.get(a, a)]
        Xb = reps[_PAIR_LABEL_ALIASES.get(b, b)]
        cka = linear_cka(Xa, Xb)
        pwc = pwcca(Xa, Xb)
        rows.append({"pair": f"{a} ↔ {b}", "linear_cka": cka, "pwcca": pwc})
        logger.info(f"  sim  {a:14s} ↔ {b:14s}  cka={cka:.4f}  pwcca={pwc:.4f}")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Top-level entry points
# ---------------------------------------------------------------------------

def compute_global(
    features_root: Path,
    image_stems: list[str],
    *,
    pca_dim: int = 512,
    target: int = 32,
    uniformity_subsample: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
    """Build global reps and return (quality_df, similarity_df, cd_reducers, sd_reducers)."""
    logger.info(f"Building global reps for {len(image_stems)} images")
    reps, cd_reducers, sd_reducers = build_global_representations(
        features_root, image_stems, pca_dim=pca_dim, target=target,
    )
    quality_df = quality_table(reps, GLOBAL_QUALITY_ORDER, uniformity_subsample)
    sim_df = similarity_table(reps, GLOBAL_SIMILARITY_PAIRS)
    return quality_df, sim_df, cd_reducers, sd_reducers


def compute_patch(
    features_root: Path,
    image_stems: list[str],
    cd_reducers: dict,
    sd_reducers: dict,
    *,
    n_subsample: int = 100,
    pca_dim: int = 512,
    target: int = 32,
    uniformity_subsample: int = 10000,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build patch reps and return (quality_df, similarity_df)."""
    logger.info(f"Building patch reps for {len(image_stems)} images "
                f"× {n_subsample} patches/image")
    reps = build_patch_representations(
        features_root, image_stems, n_subsample=n_subsample,
        pca_dim=pca_dim, cd_reducers=cd_reducers, sd_reducers=sd_reducers,
        target=target, seed=seed,
    )
    quality_df = quality_table(reps, PATCH_QUALITY_ORDER, uniformity_subsample)
    sim_df = similarity_table(reps, PATCH_SIMILARITY_PAIRS)
    return quality_df, sim_df
