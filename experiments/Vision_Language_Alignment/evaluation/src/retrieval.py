"""Bidirectional image-text retrieval — Recall@K.

Given paired image / text features (one image, one-or-more captions
per image), compute Image-to-Text (I2T) and Text-to-Image (T2I) Recall
at multiple ``K`` values.

A retrieval is correct when the ground-truth pair (or any caption of
the same image, in the multi-caption case) appears in the top-K
similarity ranking.

Public API
----------
    bidirectional_recall(img_feats, txt_feats, caption_to_image,
                         k_list=(1, 5, 10))
        Returns ``{"i2t_r{k}": float, "t2i_r{k}": float}``.
"""
from __future__ import annotations

from typing import Sequence

import torch


@torch.no_grad()
def bidirectional_recall(
    img_feats: torch.Tensor,
    txt_feats: torch.Tensor,
    caption_to_image: torch.Tensor,
    k_list: Sequence[int] = (1, 5, 10),
) -> dict[str, float]:
    """Bidirectional Recall@K on L2-normalized features.

    Args:
        img_feats:        ``(N, D)`` image features, one per image.
        txt_feats:        ``(M, D)`` caption features. ``M >= N`` since
                          some datasets have multiple captions per image.
        caption_to_image: ``(M,)`` int tensor — for each caption, the
                          index of its image in ``img_feats``.
        k_list:           recall thresholds to report.

    Returns:
        Mapping ``{"i2t_r1": ..., "t2i_r1": ..., "i2t_r5": ..., ...}``
        with values in percent.
    """
    sim_it = img_feats @ txt_feats.T          # (N, M)  image-to-caption
    sim_ti = sim_it.T                         # (M, N)  caption-to-image
    out: dict[str, float] = {}

    # I2T: for each image, gold positives are captions whose c2i == this image.
    n_images = img_feats.shape[0]
    image_to_captions: list[set[int]] = [set() for _ in range(n_images)]
    for cap_idx, img_idx in enumerate(caption_to_image.tolist()):
        image_to_captions[int(img_idx)].add(cap_idx)

    for k in k_list:
        topk_cap = sim_it.topk(min(k, sim_it.shape[-1]), dim=-1).indices.tolist()
        hits = 0
        for img_idx, retrieved in enumerate(topk_cap):
            if image_to_captions[img_idx].intersection(retrieved):
                hits += 1
        out[f"i2t_r{k}"] = 100.0 * hits / n_images

    # T2I: each caption has a single positive image (caption_to_image[i]).
    n_captions = txt_feats.shape[0]
    for k in k_list:
        topk_img = sim_ti.topk(min(k, sim_ti.shape[-1]), dim=-1).indices
        gold = caption_to_image.view(-1, 1).to(topk_img.device)
        hits = (topk_img == gold).any(dim=-1).float().sum().item()
        out[f"t2i_r{k}"] = 100.0 * hits / n_captions

    return out
