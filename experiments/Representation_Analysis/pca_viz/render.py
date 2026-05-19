"""PCA(3) -> RGB activation-map rendering for any registered backbone.

Render knobs:

  - ``input_size``  image resolution before patching.
  - ``target_size`` output PNG/PDF resolution.
  - ``mode``        ``patches`` (PCA at patch resolution, upsample RGB) or
                    ``interpolated`` (upsample features, PCA at output).

For two-model fusion (e.g. ``["dinov3-vith16plus", "cleandift"]``), the
two spatial feature maps are bilinear-aligned, L2-normalized, weighted
and concatenated by ``fuse_concat`` before the final PCA(3).

Public API
----------
    render_one(image, models, output, ...)   programmatic entry point
    main()                                   CLI wrapper for python -m render
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
from PIL import Image

from shared_utils.feature_extraction import (
    build_extractor,
    build_transform,
    fuse_concat,
    load_image,
    per_image_pca_layer_reduce,
    per_image_pca_rgb,
    raw_concat_layers,
)


# Default diffusion-extraction settings for the pca_viz code path.
PCA_VIZ_DIFFUSION_DEFAULTS = {
    "cleandift": {"timestep": 0,   "noise_mode": "clean", "hook_position": "resnets[2]"},
    "sd":        {"timestep": 261, "noise_mode": "clean", "hook_position": "resnets[2]"},
}

# Content-specific text prompts used when extracting diffusion features.
# CleanDIFT is sensitive to text conditioning even at t=0; matching the
# image domain gives noticeably cleaner activation maps than a generic prompt.
DIFFUSION_TEXT_PROMPTS = {
    "maze":  "a photo of a maze puzzle",
    "chess": "a photo of a chess board",
    "hanoi": "a photo of a tower of hanoi puzzle",
}


def resolve_text_prompt(image_path, override: Optional[str] = None) -> str:
    """Pick a diffusion-extraction prompt by image filename.

    Returns the first keyword match from ``DIFFUSION_TEXT_PROMPTS``, or
    ``"A photo"`` if no keyword matches. An explicit ``override`` short-circuits.
    """
    if override is not None:
        return override
    lo = str(image_path).lower()
    for key, prompt in DIFFUSION_TEXT_PROMPTS.items():
        if key in lo:
            return prompt
    return "A photo"


def _build(model_name: str, device, dino_facet: str, dino_block_idx: int,
           input_size: int):
    """Resolve per-backbone extractor kwargs and build the extractor."""
    ekw: dict = {}
    lkw: dict = {}
    if model_name.startswith(("dinov2-", "dinov3-")):
        ekw = {"facet": dino_facet, "block_idx": dino_block_idx}
    elif model_name == "cleandift":
        ekw = PCA_VIZ_DIFFUSION_DEFAULTS["cleandift"].copy()
    elif model_name == "sd":
        ekw = PCA_VIZ_DIFFUSION_DEFAULTS["sd"].copy()
    elif model_name == "fused-dinov3-cd":
        # Return spatial patches (not the global vector) and resize the
        # encoder's internal common_grid to match the runtime patch grid.
        ekw = {"return_patches": True}
        lkw = {"common_grid_override": input_size // 16}
    return build_extractor(model_name, device=device,
                           extractor_kwargs=ekw,
                           loader_kwargs_override=lkw)


def _to_spatial_grid(model_name: str, out: dict,
                     cd_pca_dim: Optional[int] = None,
                     normalize_per_layer: bool = False) -> torch.Tensor:
    """Reduce an extractor's output to a single ``(1, C, H, W)`` feature map.

    For diffusion backbones with per-layer outputs:

    - ``cd_pca_dim=None`` (default): raw concat of bilinear-aligned layers.
      Keeps the CD feature distribution identical between standalone-CD and
      DDN-fusion renders, so the colour palette stays consistent.
    - ``cd_pca_dim=<int>``: opt-in per-layer PCA reduction. Useful when the
      deepest layer's channel count would otherwise be dwarfed by the
      shallower ones in the final PCA(3) basis.
    """
    if out.get("patch_tokens") is not None:
        return out["patch_tokens"]
    if out.get("per_layer"):
        if cd_pca_dim is None:
            return raw_concat_layers(out["per_layer"])
        return per_image_pca_layer_reduce(
            out["per_layer"],
            n_components_per_layer=cd_pca_dim,
            normalize_per_layer=normalize_per_layer,
        )
    raise RuntimeError(f"Cannot render {model_name!r} -- no spatial features in extractor output.")


def render_one(
    image: Path,
    models: Sequence[str],
    output: Path,
    *,
    weights: Optional[Sequence[float]] = None,
    input_size: int = 512,
    target_size: int = 512,
    mode: str = "patches",
    normalize: str = "percentile",
    interp: str = "bilinear",
    strategy: str = "square_resize",
    device: Optional[str] = None,
    dino_facet: str = "token",
    dino_block_idx: int = -1,
    cd_pca_dim: Optional[int] = None,
    normalize_per_layer: bool = False,
    prompt: Optional[str] = None,
) -> Path:
    """Render a PCA(3) -> RGB activation map for one image and one (or two) models.

    Args:
        image: input image path.
        models: one model name (single render) or two (handcraft fusion).
        output: output PNG path; parent dir auto-created.
        weights: required when ``len(models) == 2``; per-model weights for fusion.
        input_size: image resolution before patching.
        target_size: output PNG resolution.
        mode: ``"patches"`` or ``"interpolated"``.
        normalize: ``"percentile"`` or ``"minmax"`` channel normalization.
        interp: upsampling interpolation (``bilinear`` / ``nearest`` / ``bicubic``).
        strategy: preprocessing strategy passed to ``build_transform``.
        device: ``"cuda"`` / ``"cpu"`` / specific device; auto-detect when None.
        dino_facet, dino_block_idx: ViT facet knobs (ignored for non-ViT models).
        cd_pca_dim: per-layer PCA components for diffusion backbones.
        prompt: explicit text prompt for diffusion extraction. If None, the
            prompt is resolved from the image filename via ``DIFFUSION_TEXT_PROMPTS``
            (maze / chess / hanoi keywords), falling back to "A photo".

    Returns:
        The output path the image was written to (same as ``output``).
    """
    if len(models) > 1 and (weights is None or len(weights) != len(models)):
        raise ValueError("`weights` must match `models` when fusing.")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    extractors, transforms = [], []
    for m in models:
        extractors.append(_build(m, device, dino_facet, dino_block_idx, input_size))
        transforms.append(build_transform(m, input_size, strategy))

    text_prompt = resolve_text_prompt(image, override=prompt)

    t0 = time.time()
    spatial_grids = []
    for model_name, ex, tfm in zip(models, extractors, transforms):
        img_t = tfm(load_image(image)).unsqueeze(0).to(device)
        try:
            out = ex.extract(img_t, prompt=text_prompt)
        except TypeError:
            out = ex.extract(img_t)
        spatial_grids.append(_to_spatial_grid(
            model_name, out,
            cd_pca_dim=cd_pca_dim,
            normalize_per_layer=normalize_per_layer,
        ))

    if len(spatial_grids) == 1:
        feats = spatial_grids[0]
    else:
        target_grid = (max(g.shape[-2] for g in spatial_grids),
                       max(g.shape[-1] for g in spatial_grids))
        feats = fuse_concat(spatial_grids, weights, target_grid=target_grid)

    feats_hwc = feats[0].permute(1, 2, 0).detach().cpu().numpy()
    H, W = feats_hwc.shape[:2]
    rgb = per_image_pca_rgb(
        feats_hwc, patch_h=H, patch_w=W,
        target_size=target_size,
        mode=mode,
        normalize=normalize,
        interp=interp,
    )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray((rgb * 255).clip(0, 255).astype(np.uint8))

    # Write both .png and .pdf at 300 dpi. The `output` argument is treated
    # as a stem; any user-supplied extension is replaced.
    stem = output.with_suffix("")
    png_path = stem.with_suffix(".png")
    pdf_path = stem.with_suffix(".pdf")
    img.save(png_path, dpi=(300, 300))
    img.save(pdf_path, "PDF", resolution=300)
    print(f"Saved: {png_path.name} + .pdf  ({time.time() - t0:.1f}s)  -> {png_path.parent}")
    return png_path


def main() -> None:
    """CLI entry: parse arguments and dispatch to render_one()."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--image", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--models", nargs="+", required=True,
                   help="One model name, or two for fusion.")
    p.add_argument("--weights", nargs="*", type=float, default=None,
                   help="Per-model weights when fusing two models.")
    p.add_argument("--input-size", type=int, default=512)
    p.add_argument("--target-size", type=int, default=512)
    p.add_argument("--mode", default="patches", choices=("patches", "interpolated"))
    p.add_argument("--normalize", default="percentile", choices=("percentile", "minmax"))
    p.add_argument("--interp", default="bilinear", choices=("bilinear", "nearest", "bicubic"))
    p.add_argument("--strategy", default="square_resize",
                   choices=("square_resize", "aspect_pad", "imagenet_center_crop"))
    p.add_argument("--device", default=None)
    p.add_argument("--dino-facet", default="token")
    p.add_argument("--dino-block-idx", type=int, default=-1)
    a = p.parse_args()

    render_one(
        image=a.image, models=a.models, output=a.output,
        weights=a.weights, input_size=a.input_size, target_size=a.target_size,
        mode=a.mode, normalize=a.normalize, interp=a.interp,
        strategy=a.strategy, device=a.device,
        dino_facet=a.dino_facet, dino_block_idx=a.dino_block_idx,
    )


if __name__ == "__main__":
    main()
