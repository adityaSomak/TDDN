"""Per-image feature extraction for the quantitative metrics pipeline.

Loads each model in ``configs/models.yaml`` via
``shared_utils.feature_extraction.build_extractor`` and writes per-image
``(H*W, C)`` float16 ``.npy`` files under
``<features_root>/<layer>/val/<image_stem>.npy``. The on-disk layout is
the one consumed by ``feature_utils.build_global_matrix`` /
``feature_utils.build_patch_matrix_with_indices``.

The "layer" subdirectory name encodes both the model tag and the
specific output view, e.g.:

    dinov3_cls       — dinov3 CLS token (1, 1280) per image
    dinov3_patches   — dinov3 patch grid flattened to (H*W, 1280)
    cd_layer2        — CleanDIFT up-block 2 hook output
    sd-2.1_layer5    — Stable Diffusion v2.1 up-block 5 hook output
    clip_cls         — CLIP last-hidden CLS token
    clip_patches     — CLIP last-hidden patch grid
    tdn_global       — TDN aligned global vector (1, 2560)
    tdn_patches      — TDN trained-head patch tokens (H*W, 1280)
    tddn_global      — TDDN aligned global vector (1, 2560)
    tddn_patches     — TDDN trained-head patch tokens (H*W, 1280)

``ddn-cd`` (handcraft DINOv3 + CleanDIFT fusion) is not extracted
here; the orchestrator composes it from ``dinov3_patches`` +
``cd_layer{2,5,8}``.

Public API
----------
    extract_features(tag, models_cfg, metrics_cfg, image_stems,
                     coco_dir, features_root, device, force=False)
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger("repr_analysis.extract")


# Maps a model tag to the set of (layer-subdir, output-key) pairs we
# persist. ``output-key`` selects which field of the extractor's output
# dict to dump: ``cls``, ``patch_mean``, ``global``, ``patches``, or
# ``per_layer:<idx>``.
OUTPUTS_BY_TAG: dict[str, list[tuple[str, str]]] = {
    "dinov3":      [("dinov3_cls", "cls"), ("dinov3_patches", "patches")],
    "dinov2-vitb": [("dinov2-vitb_cls", "cls"), ("dinov2-vitb_patches", "patches")],
    "dinov2-vitg": [("dinov2-vitg_cls", "cls"), ("dinov2-vitg_patches", "patches")],
    "clip":        [("clip_cls", "cls"), ("clip_patches", "patches")],
    "cd":          [("cd_layer2", "per_layer:2"),
                    ("cd_layer5", "per_layer:5"),
                    ("cd_layer8", "per_layer:8")],
    "sd-2.1":      [("sd-2.1_layer2", "per_layer:2"),
                    ("sd-2.1_layer5", "per_layer:5"),
                    ("sd-2.1_layer8", "per_layer:8")],
    "tdn":         [("tdn_global", "global"), ("tdn_patches", "patches")],
    "tddn":        [("tddn_global", "global"), ("tddn_patches", "patches")],
}


def _grid_to_flat(grid: torch.Tensor) -> torch.Tensor:
    """``(B, C, H, W) -> (B, H*W, C)`` then squeeze the batch dim later."""
    b, c, h, w = grid.shape
    return grid.permute(0, 2, 3, 1).reshape(b, h * w, c)


def _save_per_image(arr: np.ndarray, out_dir: Path, stem: str) -> None:
    """Persist a single image's ``(H*W, C)`` features as float16."""
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"{stem}.npy", arr.astype(np.float16))


def _extractor_view(out: dict, key: str) -> torch.Tensor:
    """Return ``(B, H*W, C)`` from the named view of an extractor output."""
    if key == "cls":
        cls = out["cls"]
        if cls is None:
            raise ValueError("extractor returned no `cls` token")
        return cls.unsqueeze(1)                                    # (B, 1, C)
    if key == "global":
        glb = out["global"]
        if glb is None:
            raise ValueError("extractor returned no `global` vector")
        return glb.unsqueeze(1)                                    # (B, 1, C)
    if key == "patches":
        grid = out["patch_tokens"]
        if grid is None:
            raise ValueError("extractor returned no `patch_tokens` grid")
        return _grid_to_flat(grid)
    if key.startswith("per_layer:"):
        idx = int(key.split(":", 1)[1])
        layer = out["per_layer"][idx]                              # (B, C, H, W)
        return _grid_to_flat(layer)
    raise ValueError(f"unknown output key {key!r}")


def _load_image(coco_dir: Path, stem: str, transform) -> torch.Tensor:
    """Read a COCO val2014 jpg by stem and apply the model's transform."""
    path = coco_dir / f"{stem}.jpg"
    img = Image.open(path).convert("RGB")
    return transform(img).unsqueeze(0)                              # (1, 3, H, W)


def _build_extractor(tag: str, model_entry: dict, device):
    """Build an extractor from the models.yaml entry for ``tag``."""
    from shared_utils.feature_extraction import build_extractor, build_transform

    backbone = model_entry["backbone"]
    tfm_cfg = model_entry.get("transform", {}) or {}
    input_size = int(tfm_cfg.get("input_size", 1024))
    extractor_kwargs = dict(model_entry.get("extractor", {}) or {})
    loader_kwargs: dict = {}

    # The fused TDDN extractor needs ``return_patches=True`` to expose
    # the patch grid, and ``common_grid_override`` so the internal
    # CleanDIFT patch grid matches the DINOv3 grid at the requested
    # input resolution (otherwise the trained-at-336 default of 21
    # collides with the 64-patch DINOv3 grid at 1024px input).
    if backbone == "fused-dinov3-cd":
        extractor_kwargs["return_patches"] = True
        loader_kwargs["common_grid_override"] = max(1, input_size // 16)

    extractor = build_extractor(
        backbone, device,
        extractor_kwargs=extractor_kwargs,
        loader_kwargs_override=loader_kwargs or None,
    )
    transform = build_transform(
        backbone, input_size, tfm_cfg.get("strategy", "square_resize"),
    )
    return extractor, transform


def _already_done(features_root: Path, layer: str, stems: Iterable[str]) -> bool:
    """All stems already on disk for this layer?"""
    out_dir = features_root / layer / "val"
    if not out_dir.is_dir():
        return False
    return all((out_dir / f"{s}.npy").is_file() for s in stems)


def extract_features(
    tag: str,
    model_entry: dict,
    image_stems: list[str],
    coco_dir: Path,
    features_root: Path,
    device: str = "cuda",
    *,
    force: bool = False,
) -> None:
    """Extract and persist every output view for one model tag.

    Args:
        tag:           model tag (``dinov3`` / ``cd`` / ``tdn`` / ...).
        model_entry:   one entry from ``configs/models.yaml`` (the value
                       under ``baselines:<tag>`` or ``trained:<tag>``).
        image_stems:   list of COCO image stems to process.
        coco_dir:      directory containing the source ``.jpg`` files.
        features_root: root under which ``<layer>/val/<stem>.npy`` are
                       written.
        device:        target device (e.g. ``"cuda"``).
        force:         re-extract even when every output is already on
                       disk.
    """
    if tag not in OUTPUTS_BY_TAG:
        raise KeyError(f"no extraction wiring for tag {tag!r}; known: "
                       f"{sorted(OUTPUTS_BY_TAG)}")
    outputs = OUTPUTS_BY_TAG[tag]

    if not force and all(_already_done(features_root, layer, image_stems)
                         for layer, _ in outputs):
        logger.info(f"[{tag}] all outputs already cached; skipping")
        return

    extractor, transform = _build_extractor(tag, model_entry, device)
    logger.info(f"[{tag}] extracting {len(image_stems)} images -> "
                f"{[layer for layer, _ in outputs]}")

    for i, stem in enumerate(image_stems):
        # Skip per-stem when every layer's file for this stem exists.
        if (not force and all(
                (features_root / layer / "val" / f"{stem}.npy").is_file()
                for layer, _ in outputs)):
            continue

        img = _load_image(coco_dir, stem, transform).to(device)
        with torch.no_grad():
            out = extractor.extract(img)

        for layer, key in outputs:
            flat = _extractor_view(out, key)                      # (B, N, C)
            arr = flat[0].cpu().numpy()                           # (N, C)
            _save_per_image(arr, features_root / layer / "val", stem)

        if (i + 1) % 100 == 0:
            logger.info(f"[{tag}] {i + 1}/{len(image_stems)}")

    logger.info(f"[{tag}] done")
