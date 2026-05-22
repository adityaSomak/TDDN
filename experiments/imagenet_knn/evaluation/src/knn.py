"""Cosine k-NN classification on pre-pooled image features.

Operates on L2-normalized ``(N, D)`` gallery and query matrices.

Public API
----------
    knn_classify(gallery_x, gallery_y, query_x, query_y, k=20)
        Top-1 / top-5 accuracy via majority vote over k nearest neighbours.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import torch


def knn_classify(
    gallery_x: np.ndarray,
    gallery_y: np.ndarray,
    query_x: np.ndarray,
    query_y: np.ndarray,
    *,
    k: int = 20,
    chunk: int = 1024,
    device: str = "cuda",
) -> dict:
    """Cosine k-NN top-1 / top-5 by majority vote over the k nearest neighbours.

    Top-1 = most-common neighbour label. Top-5 = ground-truth label is
    in the five most-common neighbour labels.

    Args:
        gallery_x: ``(N_train, D)`` L2-normalized gallery features.
        gallery_y: ``(N_train,)`` integer labels.
        query_x:   ``(N_val, D)`` L2-normalized query features.
        query_y:   ``(N_val,)`` integer labels.
        k:         number of neighbours.
        chunk:     query batch size (bounds GPU memory).
        device:    torch device for the similarity computation.

    Returns:
        ``{"top1", "top5", "k", "n_train", "n_val", "dim"}``.
    """
    gx = torch.from_numpy(gallery_x).float().to(device)
    gy = torch.from_numpy(gallery_y).long().to(device)
    n_val = query_x.shape[0]
    top1 = top5 = 0
    for i in range(0, n_val, chunk):
        q = torch.from_numpy(query_x[i : i + chunk]).float().to(device)
        sim = q @ gx.T
        top_idx = sim.topk(k, dim=1).indices
        neighbour_labels = gy[top_idx].cpu().tolist()
        gt = query_y[i : i + chunk].tolist()
        for row, label in zip(neighbour_labels, gt):
            most_common = [lbl for lbl, _ in Counter(row).most_common(5)]
            if most_common and most_common[0] == label:
                top1 += 1
            if label in most_common:
                top5 += 1
    return {
        "top1": 100.0 * top1 / n_val,
        "top5": 100.0 * top5 / n_val,
        "k": k,
        "n_train": int(gallery_x.shape[0]),
        "n_val": int(n_val),
        "dim": int(gallery_x.shape[1]),
    }
