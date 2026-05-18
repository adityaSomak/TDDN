"""Similarity and quality metrics, plus the feature-matrix builders.

    similarity.py    linear_cka, svcca, pwcca
    quality.py       uniformity, effective_rank
    feature_utils.py global / patch matrix builders, PCAReducer, fusion helper

SVCCA / PWCCA reference implementations are vendored under ``third_party/``
(Apache-2.0).
"""

from .similarity import linear_cka, svcca, pwcca
from .quality import uniformity, effective_rank, l2_normalize
from .feature_utils import (
    PCAReducer,
    build_global_matrix,
    build_patch_matrix_with_indices,
    build_cd_combined,
    build_cd_combined_pca,
    build_diffusion_combined_pca,
    build_fused,
    make_patch_indices,
)

__all__ = [
    "linear_cka", "svcca", "pwcca",
    "uniformity", "effective_rank", "l2_normalize",
    "PCAReducer",
    "build_global_matrix", "build_patch_matrix_with_indices",
    "build_cd_combined", "build_cd_combined_pca",
    "build_diffusion_combined_pca", "build_fused",
    "make_patch_indices",
]
