"""Backbone registry — name → (loader, extractor class, default kwargs).

A task picks a backbone by name, instantiates the loader to get
`(model, meta)`, then constructs the extractor. The `resolution` and
diffusion `(timestep, noise_mode)` are *never* set here — those are
task-supplied per the recipe matrix in the plan.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .extractors import (
    CLIPExtractor,
    DINOv2Extractor,
    DINOv3Extractor,
    DiffusionExtractor,
    FusedDINOv3CDExtractor,
    VithRobertaExtractor,
)
from .loaders import (
    load_clip,
    load_diffusion,
    load_dinov2,
    load_dinov3,
    load_fused_dinov3_cd,
    load_vith_roberta,
)


@dataclass
class RegistryEntry:
    loader: Callable
    extractor_cls: type
    loader_kwargs: dict[str, Any] = field(default_factory=dict)


MODEL_REGISTRY: dict[str, RegistryEntry] = {
    "dinov3-vitb16":      RegistryEntry(load_dinov3, DINOv3Extractor, {"variant": "vitb16"}),
    "dinov3-vith16plus":  RegistryEntry(load_dinov3, DINOv3Extractor, {"variant": "vith16plus"}),
    "dinov2-vitb14":      RegistryEntry(load_dinov2, DINOv2Extractor, {"variant": "vitb14"}),
    "dinov2-vitl14":      RegistryEntry(load_dinov2, DINOv2Extractor, {"variant": "vitl14"}),
    "dinov2-vitg14":      RegistryEntry(load_dinov2, DINOv2Extractor, {"variant": "vitg14"}),
    "clip-vitl14":        RegistryEntry(load_clip,   CLIPExtractor,   {"variant": "vit-l-14"}),
    "clip-vitl14-336":    RegistryEntry(load_clip,   CLIPExtractor,   {"variant": "vit-l-14-336"}),
    "sd":                 RegistryEntry(load_diffusion, DiffusionExtractor, {"backbone": "sd21"}),
    "cleandift":          RegistryEntry(load_diffusion, DiffusionExtractor, {"backbone": "cleandift"}),
    "vith-roberta":       RegistryEntry(load_vith_roberta, VithRobertaExtractor, {}),
    "fused-dinov3-cd":    RegistryEntry(load_fused_dinov3_cd, FusedDINOv3CDExtractor, {}),
}


def build_extractor(
    name: str,
    device,
    *,
    extractor_kwargs: dict[str, Any] | None = None,
    loader_kwargs_override: dict[str, Any] | None = None,
):
    """Build an extractor by registry name.

    `extractor_kwargs` are passed through to the extractor class
    (`facet`, `block_idx` for ViTs; `timestep`, `noise_mode`,
    `hook_position` for `DiffusionExtractor`; `return_patches` for
    `FusedDINOv3CDExtractor`).
    """
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model {name!r}. Registered: {sorted(MODEL_REGISTRY)}")
    entry = MODEL_REGISTRY[name]
    lkwargs = dict(entry.loader_kwargs)
    if loader_kwargs_override:
        lkwargs.update(loader_kwargs_override)
    model, meta = entry.loader(device=device, **lkwargs)
    ekwargs = extractor_kwargs or {}
    return entry.extractor_cls(model, meta, **ekwargs)
