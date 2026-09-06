"""Normalization constants, patch sizes, and environment-resolved HF token.

Resolutions are *not* defined here. They are task-dependent — each task
chooses its own input resolution and passes it to `build_transform`.
"""
from __future__ import annotations

import os

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

# SD/CleanDIFT eat images in [-1, 1] — there is no mean/std normalize step.
# The transform produces a [0, 1] tensor; the diffusion extractor handles
# the -1/+1 shift before the VAE.

# Patch sizes per backbone family. Used to validate that the chosen task
# resolution is divisible by the patch size before any forward pass.
PATCH_SIZES = {
    "dinov3-vitb16": 16,
    "dinov3-vith16plus": 16,
    "dinov2-vitb14": 14,
    "dinov2-vitl14": 14,
    "dinov2-vitg14": 14,
    "clip-vitl14": 14,
    "sd": 8,             # SD VAE downsamples by 8
    "cleandift": 8,
    "vith-roberta": 16,  # wraps DINOv3-ViT-H+
    "fused-dinov3-cd": 16,
}

NORMALIZATION = {
    "dinov3-vitb16": (IMAGENET_MEAN, IMAGENET_STD),
    "dinov3-vith16plus": (IMAGENET_MEAN, IMAGENET_STD),
    "dinov2-vitb14": (IMAGENET_MEAN, IMAGENET_STD),
    "dinov2-vitl14": (IMAGENET_MEAN, IMAGENET_STD),
    "dinov2-vitg14": (IMAGENET_MEAN, IMAGENET_STD),
    "clip-vitl14": (CLIP_MEAN, CLIP_STD),
    "sd": (IMAGENET_MEAN, IMAGENET_STD),
    "cleandift": (IMAGENET_MEAN, IMAGENET_STD),
    "vith-roberta": (IMAGENET_MEAN, IMAGENET_STD),
    "fused-dinov3-cd": (IMAGENET_MEAN, IMAGENET_STD),
}

def require_hf_token() -> str:
    token = os.environ.get("HF_TOKEN")
    if not token:
        try:
            from huggingface_hub import get_token
            token = get_token()
        except ImportError:
            token = None
    if not token:
        raise RuntimeError(
            "No Hugging Face token was found. DINOv3 is gated; run `hf auth login` "
            "or export HF_TOKEN before invoking the feature extractor."
        )
    return token
