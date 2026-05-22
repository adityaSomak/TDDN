"""Unified image + text encoder interface for clip / tdn / tddn.

Each ``AlignmentEncoder`` exposes:

  - ``encode_image(images: Tensor) -> (N, D)`` — L2-normalized global vector.
  - ``encode_text(texts: list[str]) -> (N, D)`` — L2-normalized text vector.
  - ``encode_patches(images: Tensor) -> (N, C, H, W)`` — L2-normalized
    patch grid (used by zero-shot segmentation).
  - ``image_transform`` — PIL → tensor preprocessing.

This is the only module that knows about the per-tag wiring; the eval
scripts work against the abstract interface.

Public API
----------
    build_alignment_encoder(tag, device) -> AlignmentEncoder
"""
from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F

from shared_utils.feature_extraction import build_extractor, build_transform


class AlignmentEncoder:
    """Common image + text + patch interface for an alignment model.

    Subclasses fill ``_encode_image_global``, ``_encode_text``,
    ``_encode_patches`` and ``image_transform``.
    """

    image_transform: Callable

    @torch.no_grad()
    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        emb = self._encode_image_global(images).float()
        return F.normalize(emb, dim=-1)

    @torch.no_grad()
    def encode_text(self, texts: list[str]) -> torch.Tensor:
        emb = self._encode_text(texts).float()
        return F.normalize(emb, dim=-1)

    @torch.no_grad()
    def encode_patches(self, images: torch.Tensor) -> torch.Tensor:
        feats = self._encode_patches(images).float()
        return F.normalize(feats, dim=1)

    # Subclass hooks ----------------------------------------------------------
    def _encode_image_global(self, images: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _encode_text(self, texts: list[str]) -> torch.Tensor:
        raise NotImplementedError

    def _encode_patches(self, images: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# CLIP (HuggingFace transformers ViT-L/14 @ 336)
# ---------------------------------------------------------------------------

class _CLIPEncoder(AlignmentEncoder):
    """CLIP ViT-L/14 @ 336 baseline (HuggingFace ``openai/clip-vit-large-patch14-336``)."""

    def __init__(self, device: str, input_size: int = 336):
        from transformers import CLIPModel, CLIPProcessor

        model_id = "openai/clip-vit-large-patch14-336"
        self.device = device
        self.model = CLIPModel.from_pretrained(model_id).to(device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_id)
        self.image_transform = build_transform("clip-vitl14", input_size, "imagenet_center_crop")

    def _encode_image_global(self, images: torch.Tensor) -> torch.Tensor:
        return self.model.get_image_features(pixel_values=images.to(self.device))

    def _encode_text(self, texts: list[str]) -> torch.Tensor:
        tok = self.processor.tokenizer(texts, padding=True, truncation=True,
                                       return_tensors="pt").to(self.device)
        return self.model.get_text_features(**tok)

    def _encode_patches(self, images: torch.Tensor) -> torch.Tensor:
        out = self.model.vision_model(pixel_values=images.to(self.device))
        # Drop the CLS token and project each patch into the shared
        # image-text embedding space (1024 -> 768 for ViT-L/14); without
        # this projection the per-patch features cannot be cosined
        # against the projected text embeddings.
        patches = out.last_hidden_state[:, 1:, :]
        patches = self.model.visual_projection(patches)
        B, N, D = patches.shape
        side = int(round(N ** 0.5))
        return patches.reshape(B, side, side, D).permute(0, 3, 1, 2).contiguous().float()


# ---------------------------------------------------------------------------
# TDN / TDDN (trained alignment heads via shared_utils.feature_extraction)
# ---------------------------------------------------------------------------

class _TrainedAlignmentEncoder(AlignmentEncoder):
    """DINOv3+RoBERTa or DINOv3+CleanDIFT+RoBERTa trained alignment encoder."""

    def __init__(self, backbone: str, device: str, input_size: int = 336):
        # fp32 keeps the image tower numerically stable at single-image
        # inference (the fp16 path can overflow on small batches).
        loader_kwargs: dict = {"dtype": torch.float32}
        extractor_kwargs: dict = {}
        if backbone == "fused-dinov3-cd":
            loader_kwargs["common_grid_override"] = max(1, input_size // 16)
            extractor_kwargs["return_patches"] = True
        self.extractor = build_extractor(
            backbone, device,
            extractor_kwargs=extractor_kwargs,
            loader_kwargs_override=loader_kwargs,
        )
        self.device = device
        self.image_transform = build_transform(backbone, input_size, "imagenet_center_crop")
        self.text_encoder = self.extractor.model.text_encoder
        self.tokenizer = self.text_encoder.tokenizer

    def _encode_image_global(self, images: torch.Tensor) -> torch.Tensor:
        out = self.extractor.extract(images)
        return out["global"]

    def _encode_text(self, texts: list[str]) -> torch.Tensor:
        tok = self.tokenizer(texts, padding=True, truncation=True,
                             return_tensors="pt", max_length=77).to(self.device)
        out = self.text_encoder(input_ids=tok.input_ids, attention_mask=tok.attention_mask)
        return out.aligned

    def _encode_patches(self, images: torch.Tensor) -> torch.Tensor:
        out = self.extractor.extract(images)
        return out["patch_tokens"]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_REGISTRY = {
    "clip": ("clip-vitl14", 336),
    "tdn":  ("vith-roberta", 336),
    "tddn": ("fused-dinov3-cd", 336),
}


def build_alignment_encoder(tag: str, device: str) -> AlignmentEncoder:
    """Build the encoder adapter for one of clip / tdn / tddn."""
    if tag not in _REGISTRY:
        raise ValueError(f"Unknown model_tag {tag!r}; choices: {list(_REGISTRY)}")
    backbone, input_size = _REGISTRY[tag]
    if tag == "clip":
        return _CLIPEncoder(device=device, input_size=input_size)
    return _TrainedAlignmentEncoder(backbone=backbone, device=device, input_size=input_size)
