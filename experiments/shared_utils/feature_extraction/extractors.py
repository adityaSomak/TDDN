"""Extractor classes — uniform interface over heterogeneous backbones.

Every extractor exposes `extract(images)` which takes a batched, normalized
tensor `(B, 3, H_img, W_img)` and returns a dict:

    {
      "patch_tokens": Tensor[B, C, H, W] | None,    # spatial grid
      "cls":          Tensor[B, C] | None,           # CLS token if any
      "patch_mean":   Tensor[B, C] | None,           # mean over spatial
      "global":       Tensor[B, D] | None,           # aligned/projected vec
      "per_layer":    dict[int, Tensor[B, C_l, H, W]] | None,  # diffusion
      "meta":         dict,
    }

Not every backbone produces every output. Tasks pick the keys they need.

Facet support (V/Q/K projections) is only meaningful for the DINOv2/v3
extractors and is parameterized inline as `facet={"token","query","key","value"}`
plus `block_idx` (default -1 = last block). Defaults match the SPair eval
defaults (`token`, last block).
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

import torch
import torch.nn.functional as F

from .loaders import ModelMeta

# Default diffusion layer hook positions per task lineage (see audit).
#   - "resnets[2]"  : ImageNet_Classification / segmentation / pca_viz (3rd ResBlock)
#   - "resnets[1]"  : SPair-correspondence (2nd sub-layer; attentions[1] for
#                     CrossAttn up_blocks)
_DEFAULT_DIFFUSION_HOOK = "resnets[2]"


# ============================================================================
# Helpers
# ============================================================================

def _tokens_to_grid(tokens: torch.Tensor, num_special: int) -> torch.Tensor:
    """(B, N+special, C) -> (B, C, H, W) where H = W = sqrt(N)."""
    patch = tokens[:, num_special:, :]
    B, N, C = patch.shape
    side = int(round(math.sqrt(N)))
    if side * side != N:
        raise RuntimeError(f"Patch count {N} is not a square; cannot infer grid.")
    return patch.reshape(B, side, side, C).permute(0, 3, 1, 2).contiguous()


class _FacetHook:
    """Forward hook on a transformer block's attention to capture Q/K/V.

    Supports both HF DINOv3-H+ layouts (where the attention module exposes
    `.q_proj/.k_proj/.v_proj` directly) and older DINOv2 HF layouts (where
    Q/K/V are nested inside `.attention.{query,key,value}`).
    """

    def __init__(self, attn_module, facet: str):
        if facet not in {"query", "key", "value"}:
            raise ValueError(f"facet must be one of {{query,key,value}}; got {facet!r}")
        if hasattr(attn_module, "v_proj"):
            proj_q, proj_k, proj_v = attn_module.q_proj, attn_module.k_proj, attn_module.v_proj
        elif hasattr(attn_module, "attention"):
            proj_q = attn_module.attention.query
            proj_k = attn_module.attention.key
            proj_v = attn_module.attention.value
        else:
            raise RuntimeError(
                f"Cannot find Q/K/V projections on {type(attn_module).__name__}; "
                "facet hook needs either q_proj/k_proj/v_proj attributes or a "
                "nested .attention.{query,key,value} group."
            )
        self.proj_q, self.proj_k, self.proj_v = proj_q, proj_k, proj_v
        self.facet = facet
        self._captured: list[torch.Tensor] = []
        self._handle = attn_module.register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        x = inputs[0]
        if self.facet == "value":
            cap = self.proj_v(x)
        elif self.facet == "query":
            cap = self.proj_q(x)
        else:
            cap = self.proj_k(x)
        self._captured.append(cap.detach())

    def pop(self) -> torch.Tensor:
        assert len(self._captured) == 1, f"expected one capture, got {len(self._captured)}"
        out = self._captured.pop()
        return out

    def remove(self):
        self._handle.remove()


def _dinov3_blocks(model):
    """List of transformer blocks (DINOv3 HF layout: `model.layer`)."""
    return model.layer


def _dinov2_blocks(model):
    """List of transformer blocks (DINOv2 HF layout: `model.encoder.layer`)."""
    return model.encoder.layer


def _dinov3_attention(model, block_idx: int):
    """Attention module of DINOv3 block `block_idx`. Exposes `q_proj/k_proj/v_proj`."""
    return _dinov3_blocks(model)[block_idx].attention


def _dinov2_attention(model, block_idx: int):
    """Attention module of DINOv2 block `block_idx`. Exposes `attention.{query,key,value}`."""
    return _dinov2_blocks(model)[block_idx].attention


# ============================================================================
# DINOv3 / DINOv2 — shared ViT extractor with facet support
# ============================================================================

class _ViTExtractor:
    """Base for DINOv2/DINOv3. Token / facet selection at a configurable block."""

    def __init__(
        self,
        model,
        meta: ModelMeta,
        facet: str = "token",
        block_idx: int = -1,
        blocks_finder=None,
        attention_finder=None,
    ):
        self.model = model
        self.meta = meta
        self.facet = facet
        self.block_idx = block_idx
        self._blocks_finder = blocks_finder
        self._attention_finder = attention_finder

    @torch.no_grad()
    def extract(self, images: torch.Tensor) -> dict:
        images = images.to(next(self.model.parameters()).device).to(self.meta.extra["dtype"])

        if self.facet == "token":
            outputs = self.model(images, output_hidden_states=True)
            tokens = outputs.hidden_states[self.block_idx]
        else:
            # Install a temporary facet hook on the chosen block's attention.
            n_layers = len(self._blocks_finder(self.model))
            idx = self.block_idx if self.block_idx >= 0 else n_layers + self.block_idx
            attn_mod = self._attention_finder(self.model, idx)
            hook = _FacetHook(attn_mod, self.facet)
            try:
                self.model(images)
                tokens = hook.pop()
            finally:
                hook.remove()

        # tokens: (B, N+special, C). Build all the standard views.
        special = self.meta.num_special_tokens
        cls = tokens[:, 0, :] if special >= 1 else None
        patch_grid = _tokens_to_grid(tokens, special)
        patch_mean = patch_grid.mean(dim=(2, 3))

        return {
            "patch_tokens": patch_grid.float(),
            "cls": cls.float() if cls is not None else None,
            "patch_mean": patch_mean.float(),
            "global": None,
            "per_layer": None,
            "meta": {"model": self.meta.name, "grid": patch_grid.shape[-2:],
                     "dim": self.meta.dim, "facet": self.facet, "block_idx": self.block_idx},
        }


class DINOv3Extractor(_ViTExtractor):
    def __init__(self, model, meta: ModelMeta, facet: str = "token", block_idx: int = -1):
        super().__init__(model, meta, facet, block_idx,
                         blocks_finder=_dinov3_blocks, attention_finder=_dinov3_attention)


class DINOv2Extractor(_ViTExtractor):
    def __init__(self, model, meta: ModelMeta, facet: str = "token", block_idx: int = -1):
        super().__init__(model, meta, facet, block_idx,
                         blocks_finder=_dinov2_blocks, attention_finder=_dinov2_attention)


# ============================================================================
# CLIP — vision tower; CLS dropped via slicing
# ============================================================================

class CLIPExtractor:
    def __init__(self, model, meta: ModelMeta):
        self.model = model
        self.meta = meta

    @torch.no_grad()
    def extract(self, images: torch.Tensor) -> dict:
        images = images.to(next(self.model.parameters()).device).to(self.meta.extra["dtype"])
        vis = self.model.vision_model(images, interpolate_pos_encoding=True)
        tokens = vis.last_hidden_state  # (B, 1 + N, C)
        cls = tokens[:, 0, :]
        patch_grid = _tokens_to_grid(tokens, num_special=1)
        return {
            "patch_tokens": patch_grid.float(),
            "cls": cls.float(),
            "patch_mean": patch_grid.mean(dim=(2, 3)).float(),
            "global": None,
            "per_layer": None,
            "meta": {"model": self.meta.name, "grid": patch_grid.shape[-2:], "dim": self.meta.dim},
        }


# ============================================================================
# Diffusion — SD 2.1 or CleanDIFT with multi-layer hooks
# ============================================================================

_PROMPT_TEMPLATE = "A photo of {}"


class DiffusionExtractor:
    """SD/CleanDIFT multi-layer feature extractor.

    The `timestep` and `noise_mode` knobs are required — there is no
    universally correct default. See the appendix table in the plan for
    the per-task choices.

    `hook_position` selects which sub-module of each `up_blocks[i]` is
    hooked. Three of the four tasks (ImageNet_Classification / segmentation / pca_viz)
    hook the third ResBlock (`resnets[2]`); SPair correspondence hooks
    `resnets[1]` (or `attentions[1]` if the up_block has cross-attention).
    """

    def __init__(
        self,
        pipe,
        meta: ModelMeta,
        timestep: int,
        noise_mode: str,
        layers: Sequence[int] = (2, 5, 8),
        hook_position: str = _DEFAULT_DIFFUSION_HOOK,
        seed: int = 42,
    ):
        if noise_mode not in {"clean", "noisy"}:
            raise ValueError(f"noise_mode must be 'clean' or 'noisy'; got {noise_mode!r}")
        if hook_position not in {"resnets[2]", "resnets[1]"}:
            raise ValueError(f"hook_position must be 'resnets[2]' or 'resnets[1]'.")

        self.pipe = pipe
        self.meta = meta
        self.timestep = timestep
        self.noise_mode = noise_mode
        self.layers = tuple(layers)
        self.hook_position = hook_position
        self.seed = seed

        self._device = pipe.device
        self._vae_scale = pipe.vae.config.scaling_factor
        self._t_tensor = torch.tensor([timestep], device=self._device, dtype=torch.long)

        # Cache text embeddings per prompt to avoid repeat tokenization.
        self._text_cache: dict[str, torch.Tensor] = {}

        # Register persistent hooks once.
        self._captured: dict[int, torch.Tensor] = {}
        self._handles = []
        for layer_idx in self.layers:
            target = self._hook_target(layer_idx)
            self._handles.append(target.register_forward_hook(self._make_hook(layer_idx)))

    def _hook_target(self, layer_idx: int):
        up_idx = layer_idx // 3
        block = self.pipe.unet.up_blocks[up_idx]
        if self.hook_position == "resnets[2]":
            # 3rd ResBlock. Used by ImageNet_Classification / segmentation / pca_viz.
            return block.resnets[2]
        # resnets[1] — SPair correspondence path; falls back to attentions[1]
        # when the up_block has cross-attention.
        if hasattr(block, "attentions") and block.attentions is not None and len(block.attentions) > 1:
            return block.attentions[1]
        return block.resnets[1]

    def _make_hook(self, layer_idx: int):
        captured = self._captured
        def hook(module, inputs, output):
            if isinstance(output, torch.Tensor):
                feat = output
            elif isinstance(output, tuple):
                feat = output[0]
            elif hasattr(output, "sample"):
                feat = output.sample
            else:
                feat = output
            captured[layer_idx] = feat.detach()
        return hook

    def _text_embedding(self, prompt: str) -> torch.Tensor:
        if prompt not in self._text_cache:
            tokens = self.pipe.tokenizer(
                prompt, padding="max_length", max_length=77, return_tensors="pt",
            )
            self._text_cache[prompt] = self.pipe.text_encoder(tokens.input_ids.to(self._device))[0]
        return self._text_cache[prompt]

    def _to_latent_input(self, images: torch.Tensor) -> torch.Tensor:
        """ImageNet-normalized images → VAE-encoded latent in pipeline dtype."""
        # Diffusers VAE expects images in [-1, 1]. The caller normalized with
        # ImageNet mean/std; we undo that.
        from .constants import IMAGENET_MEAN, IMAGENET_STD
        mean = torch.tensor(IMAGENET_MEAN, device=images.device).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, device=images.device).view(1, 3, 1, 1)
        imgs = images * std + mean      # → [0, 1]
        imgs = imgs * 2.0 - 1.0          # → [-1, 1]
        imgs = imgs.to(next(self.pipe.vae.parameters()).dtype)
        with torch.inference_mode():
            latent = self.pipe.vae.encode(imgs).latent_dist.mean
        latent = latent * self._vae_scale
        return latent

    def _apply_noise(self, latent: torch.Tensor) -> torch.Tensor:
        """`scheduler.add_noise` if noise_mode='noisy'; else clean."""
        if self.noise_mode == "clean":
            return latent
        gen = torch.Generator(device=self._device).manual_seed(self.seed)
        noise = torch.randn(latent.shape, generator=gen, device=self._device, dtype=latent.dtype)
        t = self._t_tensor.expand(latent.shape[0])
        return self.pipe.scheduler.add_noise(latent, noise, t)

    @torch.no_grad()
    def extract(self, images: torch.Tensor, prompt: Optional[str] = None) -> dict:
        if prompt is None:
            prompt = "A photo"
        images = images.to(self._device)
        text_emb = self._text_embedding(prompt)
        text_emb = text_emb.expand(images.shape[0], -1, -1)

        latent = self._to_latent_input(images)
        latent = self._apply_noise(latent)

        t = self._t_tensor.expand(latent.shape[0])
        self._captured.clear()
        self.pipe.unet(latent, t, encoder_hidden_states=text_emb)

        per_layer = {idx: self._captured[idx].float() for idx in self.layers}
        return {
            "patch_tokens": None,
            "cls": None,
            "patch_mean": None,
            "global": None,
            "per_layer": per_layer,
            "meta": {"model": self.meta.name, "timestep": self.timestep,
                     "noise_mode": self.noise_mode, "layers": list(self.layers),
                     "hook_position": self.hook_position},
        }

    def close(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()


# ============================================================================
# Text-aligned vith_roberta — DINOv3-H+ backbone + projection heads
# ============================================================================

class VithRobertaExtractor:
    """Returns the aligned global (CLS + mean(patches), projected) and the
    raw patch tokens from the DINOv3 backbone for downstream spatial use."""

    def __init__(self, model, meta: ModelMeta):
        self.model = model
        self.meta = meta

    @torch.no_grad()
    def extract(self, images: torch.Tensor) -> dict:
        images = images.to(next(self.model.parameters()).device).to(self.meta.extra["dtype"])
        # The encoder's forward returns an ImageEncoderOutput NamedTuple with
        # .aligned (B, 2C) and .patch_tokens (B, N, D). Patch tokens are in
        # the *projected head* space (1280-d for ViT-H+), not raw backbone.
        out = self.model.image_encoder(images)
        B, N, D = out.patch_tokens.shape
        side = int(round(N ** 0.5))
        patch_grid = out.patch_tokens.reshape(B, side, side, D).permute(0, 3, 1, 2).contiguous()
        return {
            "patch_tokens": patch_grid.float(),
            "cls": None,                       # CLS lives inside `aligned`
            "patch_mean": patch_grid.mean(dim=(2, 3)).float(),
            "global": out.aligned.float(),
            "per_layer": None,
            "meta": {"model": self.meta.name, "grid": (side, side), "dim": self.meta.dim},
        }


# ============================================================================
# Fused DINOv3+CleanDIFT — trained encoder; returns the aligned global
# ============================================================================

class FusedDINOv3CDExtractor:
    """Trained fused encoder. Returns the aligned 2C-dim global by default.

    The image_encoder accepts only images (it runs both DINOv3 and CleanDIFT
    backbones internally and fuses them). If `return_patches=True`, we also
    expose the DINOv3-side patch grid for downstream spatial use.
    """

    def __init__(self, model, meta: ModelMeta, return_patches: bool = False):
        self.model = model
        self.meta = meta
        self.return_patches = return_patches

    @torch.no_grad()
    def extract(self, images: torch.Tensor) -> dict:
        images = images.to(next(self.model.parameters()).device).to(self.meta.extra["dtype"])
        out = self.model.image_encoder(images)

        patch_grid = None
        patch_mean = None
        if self.return_patches and out.patch_tokens is not None:
            B, N, D = out.patch_tokens.shape
            side = int(round(N ** 0.5))
            patch_grid = out.patch_tokens.reshape(B, side, side, D).permute(0, 3, 1, 2).contiguous().float()
            patch_mean = patch_grid.mean(dim=(2, 3))

        return {
            "patch_tokens": patch_grid,
            "cls": None,
            "patch_mean": patch_mean,
            "global": out.aligned.float(),
            "per_layer": None,
            "meta": {"model": self.meta.name, "dim": self.meta.dim,
                     "grid": patch_grid.shape[-2:] if patch_grid is not None else None},
        }
