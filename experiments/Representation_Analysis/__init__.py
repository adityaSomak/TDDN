"""Representation_Analysis — CKA + PCA analysis of vision-feature backbones.

Public sub-packages:

    metrics    similarity (linear_cka, pwcca, svcca) + quality (uniformity,
               effective_rank) + matrix builders for global / patch analysis.
    pca_viz    Single-function ``render_one`` that turns one image + one (or
               two) backbone(s) into a PCA(3) -> RGB activation map.

Entry point: ``python run.py {activation-maps,metrics,plots}``.
"""
