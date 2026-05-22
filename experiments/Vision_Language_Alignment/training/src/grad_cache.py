"""Micro-batch gradient accumulation for the contrastive training step.

The contrastive objective benefits from a large negative pool: at
each iteration we run ``grad_cache_multiplier`` micro-batches, store
their outputs, concatenate the embeddings, and compute one loss
against the concatenated pool (which then gets all-gathered across
ranks inside the loss).

Memory stays bounded because the backbones run under ``no_grad``
(only alignment-head activations are kept around for backward), and
each micro-batch only contributes its trainable-head activations to
the autograd graph.

Public API
----------
    accumulate_micro_batches(model, data_iter, n_accum, device, *, loader=None)
        -> dict of stacked tensors and the final ``logit_scale``.
        ``loader`` is an optional fallback used to re-create
        ``data_iter`` if it runs out (the production sampler is
        infinite, but a one-shot iterator passed in for testing will
        exhaust).
"""
from __future__ import annotations

from typing import Iterator

import torch


@torch.enable_grad()
def accumulate_micro_batches(
    model: torch.nn.Module,
    data_iter: Iterator,
    n_accum: int,
    device: torch.device | str,
    *,
    loader=None,
) -> dict:
    """Run ``n_accum`` micro-batches through the model and stack their outputs.

    The production ``data_iter`` is built from an infinite sampler so
    it never raises ``StopIteration``. If a finite iterator is passed
    (e.g. for testing) and ``loader`` is provided, the iterator is
    re-created from the loader whenever it exhausts.

    Args:
        model:     ``AlignmentModel`` (or FSDP-wrapped equivalent).
        data_iter: iterator yielding ``(images, text_inputs)`` tuples.
            ``text_inputs`` may be either a dict of tokenized tensors
            (for RoBERTa) or a single tokenized tensor (for CLIP).
        n_accum:   number of micro-batches to accumulate.
        device:    target CUDA device.
        loader:    optional DataLoader used to re-create the iterator
            when ``data_iter`` exhausts.

    Returns:
        A dict with the four concatenated tensors needed by the
        loss functions (``image_features``, ``text_features``,
        ``image_original``, ``text_original``) plus the latest
        ``logit_scale`` scalar.
    """
    img_feats:  list[torch.Tensor] = []
    txt_feats:  list[torch.Tensor] = []
    img_orig:   list[torch.Tensor] = []
    txt_orig:   list[torch.Tensor] = []
    logit_scale: torch.Tensor | None = None

    for _ in range(n_accum):
        try:
            images, text_inputs = next(data_iter)
        except StopIteration:
            if loader is None:
                raise
            data_iter = iter(loader)
            images, text_inputs = next(data_iter)
        images = images.to(device=device, non_blocking=True)
        if isinstance(text_inputs, dict):
            text_inputs = {k: v.to(device=device, non_blocking=True)
                           for k, v in text_inputs.items()}
        else:
            text_inputs = text_inputs.to(device=device, non_blocking=True)

        out = model(images, text_inputs)
        img_feats.append(out.image_features)
        txt_feats.append(out.text_features)
        img_orig.append(out.image_original)
        txt_orig.append(out.text_original)
        logit_scale = out.logit_scale

    return {
        "image_features":  torch.cat(img_feats, dim=0),
        "text_features":   torch.cat(txt_feats, dim=0),
        "image_original":  torch.cat(img_orig, dim=0),
        "text_original":   torch.cat(txt_orig, dim=0),
        "logit_scale":     logit_scale,
    }
