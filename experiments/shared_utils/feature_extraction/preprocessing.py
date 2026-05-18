"""PIL → tensor transforms with three spatial strategies.

The four eval tasks each use a materially different spatial layout. Rather
than hide that in per-task helpers, the strategy is an explicit argument:

  - "square_resize"          : Resize((res, res)). Segmentation, pca_viz.
                               Puzzle images are already square; no aspect
                               preservation is needed.

  - "aspect_pad"             : Resize preserving aspect with LANCZOS, then
                               center-pad to (res, res). SPair correspondence
                               needs unwarped pixels — squashing breaks PCK.

  - "imagenet_center_crop"   : Resize(res, BICUBIC) + CenterCrop(res).
                               ImageNet-KNN; matches DINOv2/CLIP eval recipes.

The model determines mean/std and patch-size validation; the strategy
determines layout. These two concerns are independent.
"""
from __future__ import annotations

from PIL import Image
import torch
from torchvision import transforms
from torchvision.transforms.functional import resize as tv_resize

from .constants import NORMALIZATION, PATCH_SIZES

_Strategy = str  # "square_resize" | "aspect_pad" | "imagenet_center_crop"


def _aspect_pad_pil(img: Image.Image, target_res: int) -> Image.Image:
    """Resize preserving aspect ratio (LANCZOS), then center-pad to a square.

    sd-dino's canonical recipe — keypoints stay in the same unwarped pixel
    coordinate frame.
    """
    w, h = img.size
    if w >= h:
        new_w = target_res
        new_h = int(round(target_res * h / w))
    else:
        new_h = target_res
        new_w = int(round(target_res * w / h))
    img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    canvas = Image.new(img.mode, (target_res, target_res), 0)
    canvas.paste(img, ((target_res - new_w) // 2, (target_res - new_h) // 2))
    return canvas


class _AspectPad:
    """torchvision-style callable wrapper around _aspect_pad_pil."""

    def __init__(self, target_res: int):
        self.target_res = target_res

    def __call__(self, img):
        return _aspect_pad_pil(img, self.target_res)


def build_transform(
    model_name: str,
    resolution: int,
    strategy: _Strategy,
    normalize: bool = True,
) -> transforms.Compose:
    """Build the PIL → tensor transform for (model, resolution, strategy).

    Validates that the resolution is compatible with the model's patch size
    before returning.
    """
    if model_name not in PATCH_SIZES:
        raise ValueError(
            f"Unknown model {model_name!r}. Known: {sorted(PATCH_SIZES)}"
        )
    patch = PATCH_SIZES[model_name]
    if resolution % patch != 0:
        raise ValueError(
            f"Resolution {resolution} must be divisible by the {model_name} "
            f"patch size ({patch}). Use a multiple of {patch}."
        )

    steps = []
    if strategy == "square_resize":
        steps.append(transforms.Resize((resolution, resolution), interpolation=transforms.InterpolationMode.BICUBIC))
    elif strategy == "aspect_pad":
        steps.append(_AspectPad(resolution))
    elif strategy == "imagenet_center_crop":
        steps.append(transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BICUBIC))
        steps.append(transforms.CenterCrop(resolution))
    else:
        raise ValueError(f"Unknown strategy {strategy!r}")

    steps.append(transforms.ToTensor())  # PIL → [0, 1] FloatTensor

    if normalize:
        mean, std = NORMALIZATION[model_name]
        steps.append(transforms.Normalize(mean=mean, std=std))

    return transforms.Compose(steps)


def load_image(path) -> Image.Image:
    """Open an image and force-convert to RGB (handles RGBA, L, etc.)."""
    return Image.open(path).convert("RGB")
