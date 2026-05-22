"""Text-classifier builders for zero-shot / CuPL classification.

Three classifier modes:

  - ``zero_shot``     Average L2-normalized text embeddings of every
                       prompt template applied to each class name.
  - ``cupl``           Same averaging, but the "prompts" are
                       per-class LLM-generated descriptions loaded
                       from a JSON file. Falls back to ``zero_shot``
                       if no descriptions are available.
  - (TIP-Adapter)      Few-shot cache + alpha sweep — see ``tip_adapter.py``.

The classifier output is a ``(num_classes, embed_dim)`` matrix of L2-
normalized prototypes. Image features (also L2-normalized) are scored
by cosine similarity → argmax → class prediction.

Public API
----------
    build_zero_shot_classifier(text_encoder, classes, templates) -> (C, D) tensor
    build_cupl_classifier(text_encoder, classes, descriptions, templates)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Sequence

import torch
import torch.nn.functional as F


@torch.no_grad()
def build_zero_shot_classifier(
    text_encoder,
    classes: Sequence[str],
    templates: Sequence[Callable[[str], str]],
    device: str = "cuda",
) -> torch.Tensor:
    """Average prompt-template text embeddings per class.

    Args:
        text_encoder: callable ``(list[str]) -> (N, D)`` returning L2-norm
                      text embeddings.
        classes:      list of class name strings.
        templates:    list of lambdas ``c -> formatted_prompt``.
        device:       torch device.

    Returns:
        ``(len(classes), D)`` L2-normalized class prototypes.
    """
    prototypes = []
    for c in classes:
        prompts = [t(c) for t in templates]
        embs = text_encoder(prompts).to(device)
        emb = F.normalize(embs.mean(dim=0, keepdim=True), dim=-1)
        prototypes.append(emb)
    return torch.cat(prototypes, dim=0)


@torch.no_grad()
def build_cupl_classifier(
    text_encoder,
    classes: Sequence[str],
    descriptions_path: Path,
    templates: Sequence[Callable[[str], str]] | None = None,
    prefix_template: Callable[[str], str] = lambda c: f"a photo of a {c}. ",
    device: str = "cuda",
) -> torch.Tensor:
    """Average LLM-generated description embeddings per class.

    Each class gets ``"a photo of a {class}. " + description`` for every
    description in the JSON. Embeddings are averaged then L2-normalized.
    If a class has no descriptions in the file, falls back to the
    zero-shot template ensemble for that class only.

    Args:
        text_encoder:      callable ``(list[str]) -> (N, D)``.
        classes:           list of class name strings.
        descriptions_path: JSON file mapping ``class_name -> [str, ...]``.
        templates:         used to fall back when a class is missing
                           descriptions; required if any class is missing.
        prefix_template:   prepended to each description.
        device:            torch device.

    Returns:
        ``(len(classes), D)`` L2-normalized class prototypes.
    """
    desc_map: dict[str, list[str]] = json.loads(Path(descriptions_path).read_text())

    prototypes = []
    for c in classes:
        descs = desc_map.get(c)
        if descs:
            prompts = [prefix_template(c) + d for d in descs]
        else:
            if templates is None:
                raise ValueError(f"class {c!r} missing from {descriptions_path} "
                                 "and no fallback templates provided")
            prompts = [t(c) for t in templates]
        embs = text_encoder(prompts).to(device)
        emb = F.normalize(embs.mean(dim=0, keepdim=True), dim=-1)
        prototypes.append(emb)
    return torch.cat(prototypes, dim=0)


@torch.no_grad()
def classify(features: torch.Tensor, classifier: torch.Tensor) -> torch.Tensor:
    """Cosine-similarity classification.

    Args:
        features:   ``(N, D)`` L2-normalized image features.
        classifier: ``(C, D)`` L2-normalized class prototypes.

    Returns:
        ``(N,)`` integer class predictions (argmax).
    """
    logits = features @ classifier.T
    return logits.argmax(dim=-1)


@torch.no_grad()
def top_k_accuracy(
    features: torch.Tensor,
    labels: torch.Tensor,
    classifier: torch.Tensor,
    ks: Sequence[int] = (1, 5),
) -> dict[int, float]:
    """Top-k accuracy from cosine logits.

    Returns a dict mapping ``k -> top-k accuracy (%)``.
    """
    logits = features @ classifier.T
    out: dict[int, float] = {}
    for k in ks:
        topk = logits.topk(min(k, logits.shape[-1]), dim=-1).indices
        hits = (topk == labels.unsqueeze(-1)).any(dim=-1).float()
        out[k] = 100.0 * hits.mean().item()
    return out
