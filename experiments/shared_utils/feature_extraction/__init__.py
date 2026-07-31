"""Unified feature extraction for vision-language eval pipelines.

Public API:
    from shared_utils.feature_extraction import build_extractor, MODEL_REGISTRY
    from shared_utils.feature_extraction.preprocessing import build_transform, load_image
    from shared_utils.feature_extraction.fusion import fuse_concat, fuse_concat_global
    from shared_utils.feature_extraction.pca_reduction import (
        fit_global_pca, apply_pca,
        hierarchical_co_pca,
        per_image_pca_rgb,
    )

Models are registered by short name (e.g., `dinov3-vith16plus`). See
``registry.py`` for the full list and ``loaders.py`` for the loader
factories. Trained-model code is vendored under ``text_alignment/``.
"""
from .registry import MODEL_REGISTRY, RegistryEntry, build_extractor, loader_kwargs_for
from .preprocessing import build_transform, load_image
from .fusion import fuse_concat, fuse_concat_global
from .pooling import pool_to_vector
from .pca_reduction import (
    fit_global_pca,
    apply_pca,
    hierarchical_co_pca,
    per_image_pca_layer_reduce,
    per_image_pca_rgb,
    raw_concat_layers,
    GlobalPCABasis,
)

__all__ = [
    "MODEL_REGISTRY", "RegistryEntry", "build_extractor", "loader_kwargs_for",
    "build_transform", "load_image",
    "fuse_concat", "fuse_concat_global",
    "pool_to_vector",
    "fit_global_pca", "apply_pca", "hierarchical_co_pca",
    "per_image_pca_layer_reduce", "per_image_pca_rgb",
    "raw_concat_layers",
    "GlobalPCABasis",
]
