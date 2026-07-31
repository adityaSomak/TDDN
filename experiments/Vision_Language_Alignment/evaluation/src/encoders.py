"""Image + text encoders for every registered model tag.

Two builders over one registry (``configs/models.yaml``):

    build_alignment_encoder(tag, device, ...) -> AlignmentEncoder
        One global vector per image, for classification and retrieval.
        ``encode_image`` / ``encode_text``, both L2-normalized.

    build_dense_encoder(tag, device, crop_size=...) -> dense encoder
        A per-patch feature grid, for sliding-window segmentation.
        ``encode_patches_logits(window, classifier)`` / ``encode_text``.

This is the only module that knows per-tag wiring; the eval code works against
the two interfaces. Dense encoders take windows already cropped to
``crop_size`` by ``slide_inference``, so their ``image_transform`` is
ToTensor+Normalize only and must not resize.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F
import yaml
from torchvision import transforms as T

from shared_utils.feature_extraction import build_extractor, build_transform, loader_kwargs_for
from shared_utils.feature_extraction.constants import CLIP_MEAN, CLIP_STD, NORMALIZATION

_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "models.yaml"


@lru_cache(maxsize=1)
def registry() -> dict[str, dict]:
    """``{tag: spec}`` from configs/models.yaml, groups flattened."""
    spec = yaml.safe_load(_CONFIG.read_text())
    flat: dict[str, dict] = {}
    for group in ("baselines", "trained"):
        flat.update(spec.get(group) or {})
    return flat


def _spec(tag: str) -> dict:
    try:
        return registry()[tag]
    except KeyError:
        raise ValueError(
            f"Unknown model tag {tag!r}; choices: {sorted(registry())}") from None


# ---------------------------------------------------------------------------
# Global encoders
# ---------------------------------------------------------------------------

class AlignmentEncoder:
    """Common global image + text interface.

    Subclasses fill ``_encode_image_global``, ``_encode_text`` and
    ``image_transform``.
    """

    image_transform: Callable
    wants_pil: bool = False

    @torch.no_grad()
    def encode_image(self, images) -> torch.Tensor:
        return F.normalize(self._encode_image_global(images).float(), dim=-1)

    @torch.no_grad()
    def encode_text(self, texts: list[str]) -> torch.Tensor:
        return F.normalize(self._encode_text(texts).float(), dim=-1)

    @torch.no_grad()
    def encode_patches(self, images: torch.Tensor) -> torch.Tensor:
        return F.normalize(self._encode_patches(images).float(), dim=1)

    def _encode_image_global(self, images) -> torch.Tensor:
        raise NotImplementedError

    def _encode_text(self, texts: list[str]) -> torch.Tensor:
        raise NotImplementedError

    def _encode_patches(self, images: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


def _clip_joint(out) -> torch.Tensor:
    """Joint-space embedding from HF CLIP's ``get_image_features`` / ``get_text_features``.

    Newer transformers return a model-output object whose ``pooler_output`` is
    already projected into the joint space; older versions return that tensor
    directly. Either way no further projection is applied.
    """
    return out if torch.is_tensor(out) else out.pooler_output


class _CLIPEncoder(AlignmentEncoder):
    """HuggingFace ``openai/clip-vit-large-patch14-336``."""

    def __init__(self, device: str, input_size: int = 336,
                 transform_strategy: str = "imagenet_center_crop"):
        from transformers import CLIPModel, CLIPProcessor

        model_id = "openai/clip-vit-large-patch14-336"
        self.device = device
        self.model = CLIPModel.from_pretrained(model_id).to(device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_id)
        self.image_transform = build_transform("clip-vitl14", input_size, transform_strategy)

    def _encode_image_global(self, images: torch.Tensor) -> torch.Tensor:
        return _clip_joint(self.model.get_image_features(pixel_values=images.to(self.device)))

    def _encode_text(self, texts: list[str]) -> torch.Tensor:
        tok = self.processor.tokenizer(texts, padding=True, truncation=True,
                                       return_tensors="pt").to(self.device)
        return _clip_joint(self.model.get_text_features(**tok))


class _TrainedAlignmentEncoder(AlignmentEncoder):
    """A trained alignment head (tdn / tddn) via ``shared_utils.feature_extraction``."""

    def __init__(self, tag: str, device: str, input_size: int | None = None,
                 transform_strategy: str = "imagenet_center_crop"):
        spec = _spec(tag)
        backbone = spec["backbone"]
        input_size = input_size or spec.get("input_size", 336)
        # fp32 keeps the image tower numerically stable at single-image
        # inference; the fp16 path can overflow on small batches.
        loader_kwargs: dict = {"dtype": torch.float32, **loader_kwargs_for(spec)}
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
        self.backbone = backbone
        self.image_transform = build_transform(backbone, input_size, transform_strategy)
        self.text_encoder = self.extractor.model.text_encoder
        self.tokenizer = self.text_encoder.tokenizer

    def _encode_image_global(self, images: torch.Tensor) -> torch.Tensor:
        return self.extractor.extract(images)["global"]

    def _encode_text(self, texts: list[str]) -> torch.Tensor:
        tok = self.tokenizer(texts, padding=True, truncation=True,
                             return_tensors="pt", max_length=77).to(self.device)
        out = self.text_encoder(input_ids=tok.input_ids, attention_mask=tok.attention_mask)
        return out.aligned

    def _encode_patches(self, images: torch.Tensor) -> torch.Tensor:
        return self.extractor.extract(images)["patch_tokens"]


class _ClsHalfEncoder(AlignmentEncoder):
    """Takes the first half of a trained encoder's image and text vectors.

    Valid only for tags whose embedding is a two-part concatenation; the
    boundary is derived at call time, not hardcoded.
    """

    def __init__(self, inner: _TrainedAlignmentEncoder):
        self.inner = inner
        self.image_transform = inner.image_transform

    def _encode_image_global(self, images: torch.Tensor) -> torch.Tensor:
        full = self.inner._encode_image_global(images)
        return full[:, : full.shape[-1] // 2]

    def _encode_text(self, texts: list[str]) -> torch.Tensor:
        full = self.inner._encode_text(texts)
        return full[:, : full.shape[-1] // 2]


class _OpenCLIPGlobal(AlignmentEncoder):
    """open_clip model at its native pretrained resolution, own ``preprocess_val``."""

    def __init__(self, tag: str, device: str):
        import open_clip

        spec = _spec(tag)
        arch, pretrained = spec["arch"], spec["pretrained"]
        self.model, _, preprocess = open_clip.create_model_and_transforms(
            arch, pretrained=pretrained)
        self.model = self.model.to(device).eval()
        self.tokenizer = open_clip.get_tokenizer(arch)
        self.device = device
        self.image_transform = preprocess

    def _encode_image_global(self, images: torch.Tensor) -> torch.Tensor:
        return self.model.encode_image(images.to(self.device))

    def _encode_text(self, texts: list[str]) -> torch.Tensor:
        return self.model.encode_text(self.tokenizer(texts).to(self.device))


class _FGClip2Global(AlignmentEncoder):
    """FG-CLIP2 via HF ``get_image_features`` / ``get_text_features``.

    A NaFlex model: its processor consumes original-aspect PIL images with a
    per-call ``max_num_patches`` budget, so ``image_transform`` only converts to
    RGB and ``wants_pil`` tells the eval loop to pass a list of PIL images
    instead of a stacked tensor.

    ``patch_mode``:
      ``native``     per-image budget from the image's own patch grid, capped at 1024
      ``fixed576``   576 patches for every image
      ``fixed1024``  1024 patches for every image
    """

    _FIXED = {"fixed576": 576, "fixed1024": 1024}

    def __init__(self, tag: str, device: str, patch_mode: str = "fixed1024"):
        from transformers import AutoImageProcessor, AutoModelForCausalLM, AutoTokenizer

        if patch_mode not in ("native", "fixed576", "fixed1024"):
            raise ValueError(f"Unknown fgclip2 patch_mode {patch_mode!r}")
        model_root = _spec(tag)["hf_id"]
        self.device = device
        self.patch_mode = patch_mode
        self._fixed_patches = self._FIXED.get(patch_mode)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_root, trust_remote_code=True).to(device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_root)
        self.image_processor = AutoImageProcessor.from_pretrained(model_root)
        self.wants_pil = True
        self.image_transform = lambda im: im.convert("RGB")

    @staticmethod
    def _max_num_patches(pil) -> int:
        w, h = pil.size
        mv = (w // 16) * (h // 16)
        if mv > 784:
            return 1024
        if mv > 576:
            return 784
        if mv > 256:
            return 576
        if mv > 128:
            return 256
        return 128

    def _encode_image_global(self, images) -> torch.Tensor:
        # Bucket by patch budget so each processor call stays batched (the NaFlex
        # processor pads variable-size images within a bucket), then reassemble
        # in input order.
        feats: list = [None] * len(images)
        buckets: dict[int, list[int]] = {}
        for i, pil in enumerate(images):
            mnp = (self._fixed_patches if self._fixed_patches is not None
                   else self._max_num_patches(pil))
            buckets.setdefault(mnp, []).append(i)
        for mnp, idxs in buckets.items():
            image_input = self.image_processor(
                images=[images[i] for i in idxs], max_num_patches=mnp,
                return_tensors="pt").to(self.device)
            out = self.model.get_image_features(**image_input)
            for j, i in enumerate(idxs):
                feats[i] = out[j]
        return torch.stack(feats, dim=0)

    def _encode_text(self, texts: list[str]) -> torch.Tensor:
        tok = self.tokenizer(texts, padding="max_length", max_length=64,
                             truncation=True, return_tensors="pt").to(self.device)
        return self.model.get_text_features(**tok, walk_type="short")


def build_alignment_encoder(
    tag: str, device: str, *, transform_strategy: str = "imagenet_center_crop",
    cls_half: bool = False, patch_mode: str | None = None,
) -> AlignmentEncoder:
    """Build the global encoder for ``tag``.

    ``cls_half`` is accepted only for the trained tags. ``patch_mode`` applies
    only to the fgclip2 family; ``None`` uses the registry's value.
    """
    spec = _spec(tag)
    family = spec.get("family", "trained")

    if cls_half and family != "trained":
        raise ValueError(
            f"cls_half is only valid for the trained tags, not {tag!r} ({family})")

    if family == "trained":
        base = _TrainedAlignmentEncoder(tag, device, transform_strategy=transform_strategy)
        return _ClsHalfEncoder(base) if cls_half else base
    if family == "clip":
        return _CLIPEncoder(device, input_size=spec.get("input_size", 336),
                            transform_strategy=transform_strategy)
    if family in ("open_clip", "siglip"):
        return _OpenCLIPGlobal(tag, device)
    if family == "fgclip2":
        return _FGClip2Global(tag, device,
                              patch_mode=patch_mode or spec.get("patch_mode", "fixed1024"))
    raise ValueError(f"Unknown family {family!r} for tag {tag!r}")


# ---------------------------------------------------------------------------
# Dense encoders
# ---------------------------------------------------------------------------

def _interp_pos_embed(pos_embed: torch.Tensor, new_grid: int, num_prefix: int) -> torch.Tensor:
    """Bicubic-resample a flat absolute position-embedding table to a new grid.

    Prefix tokens (e.g. CLS) are left unchanged. Applied once at construction,
    since every window in this pipeline is exactly ``crop_size``.
    """
    squeeze = pos_embed.dim() == 2
    pe = pos_embed.unsqueeze(0) if squeeze else pos_embed
    prefix, patch_pos = pe[:, :num_prefix], pe[:, num_prefix:]
    D = patch_pos.shape[-1]
    old_grid = int(round(patch_pos.shape[1] ** 0.5))
    assert old_grid * old_grid == patch_pos.shape[1], f"non-square pos grid: {patch_pos.shape[1]}"
    patch_pos = patch_pos.reshape(1, old_grid, old_grid, D).permute(0, 3, 1, 2)
    patch_pos = F.interpolate(patch_pos, size=(new_grid, new_grid),
                              mode="bicubic", align_corners=False)
    patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, new_grid * new_grid, D)
    out = torch.cat([prefix, patch_pos], dim=1)
    return out.squeeze(0) if squeeze else out


def _logits_from_patches(patches: torch.Tensor, classifier: torch.Tensor,
                         out_hw) -> torch.Tensor:
    """(B, D, h, w) patches x (K, D) classifier -> (B, K, *out_hw) on CPU."""
    B, D, h, w = patches.shape
    flat = patches.permute(0, 2, 3, 1).reshape(-1, D)
    logits = (flat @ classifier.T).reshape(B, h, w, classifier.shape[0]).permute(0, 3, 1, 2)
    return F.interpolate(logits, size=out_hw, mode="bilinear", align_corners=False).cpu()


class _DenseEncoder:
    """Common dense interface: subclasses fill ``encode_patches_raw`` / ``encode_text``."""

    image_transform: Callable

    @torch.no_grad()
    def encode_patches_logits(self, window: torch.Tensor,
                              classifier: torch.Tensor) -> torch.Tensor:
        return _logits_from_patches(self.encode_patches_raw(window), classifier,
                                    window.shape[-2:])


class _CLIPDense(_DenseEncoder):
    """HF CLIP ViT-L/14@336 fed larger windows via interpolated position embeddings."""

    def __init__(self, device: str, crop_size: int = 448):
        from transformers import CLIPModel, CLIPProcessor

        model_id = "openai/clip-vit-large-patch14-336"
        self.device = device
        self.model = CLIPModel.from_pretrained(model_id).to(device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_id)
        self.image_transform = T.Compose(
            [T.ToTensor(), T.Normalize(mean=CLIP_MEAN, std=CLIP_STD)])

    @torch.no_grad()
    def encode_text(self, texts: list[str]) -> torch.Tensor:
        tok = self.processor.tokenizer(texts, padding=True, truncation=True,
                                       return_tensors="pt").to(self.device)
        out = _clip_joint(self.model.get_text_features(**tok))
        return F.normalize(out.float(), dim=-1)

    @torch.no_grad()
    def encode_patches_raw(self, window: torch.Tensor) -> torch.Tensor:
        out = self.model.vision_model(pixel_values=window.to(self.device),
                                      interpolate_pos_encoding=True,
                                      output_hidden_states=True)
        x = out.hidden_states[-2]
        layer = self.model.vision_model.encoder.layers[-1]
        v = layer.self_attn.v_proj(layer.layer_norm1(x))
        x = x + layer.self_attn.out_proj(v)
        x = x + layer.mlp(layer.layer_norm2(x))
        patches = self.model.visual_projection(x[:, 1:, :])
        B, N, D = patches.shape
        side = int(round(N ** 0.5))
        assert side * side == N, f"non-square patch grid: N={N}"
        patches = patches.reshape(B, side, side, D).permute(0, 3, 1, 2).contiguous().float()
        return F.normalize(patches, dim=1)


def _value_only_attention(self, qkv: torch.Tensor, attn_bias=None, rope=None) -> torch.Tensor:
    """Replacement for ``SelfAttention.compute_attention`` returning the V third.

    ``rope`` is accepted only to match the call signature.
    """
    B, N, _ = qkv.shape
    C = self.qkv.in_features
    v = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads)[:, :, 2]
    return v.reshape(B, N, C)


class _TrainedDense(_DenseEncoder):
    """A trained alignment head (tdn / tddn) producing a patch grid at ``crop_size``.

    The registry's ``dense_bypass`` selects the readout variant per tag. Any
    substitution it implies is installed for the duration of one extract call
    and removed afterwards.
    """

    def __init__(self, tag: str, device: str, crop_size: int = 448):
        import types

        spec = _spec(tag)
        self.encoder = _TrainedAlignmentEncoder(tag, device, input_size=crop_size)
        self.device = device
        mean, std = NORMALIZATION[spec["backbone"]]
        self.image_transform = T.Compose([T.ToTensor(), T.Normalize(mean=mean, std=std)])

        mode = spec.get("dense_bypass", "none")
        blocks = list(self.encoder.extractor.model.image_encoder.blocks)
        if mode == "all":
            targeted = blocks
        elif mode == "last":
            targeted = blocks[-1:]
        elif mode == "none":
            targeted = []
        else:
            raise ValueError(f"Unknown dense_bypass {mode!r} for tag {tag!r}")
        self._attns = [b.attn for b in targeted]
        self._orig = [a.compute_attention for a in self._attns]
        self._patched = [types.MethodType(_value_only_attention, a) for a in self._attns]

    @torch.no_grad()
    def encode_text(self, texts: list[str]) -> torch.Tensor:
        return self.encoder.encode_text(texts)

    @torch.no_grad()
    def encode_patches_raw(self, window: torch.Tensor) -> torch.Tensor:
        for attn, patched in zip(self._attns, self._patched):
            attn.compute_attention = patched
        try:
            return self.encoder.encode_patches(window.to(self.device))
        finally:
            for attn, orig in zip(self._attns, self._orig):
                attn.compute_attention = orig

    @torch.no_grad()
    def encode_patches_logits(self, window: torch.Tensor,
                              classifier: torch.Tensor) -> torch.Tensor:
        patches = self.encode_patches_raw(window)
        patch_dim = patches.shape[1]
        # The trained text vector is a two-part concatenation; the patch grid
        # matches its second half.
        if classifier.shape[-1] == 2 * patch_dim:
            classifier = F.normalize(classifier[:, patch_dim:], dim=-1)
        return _logits_from_patches(patches, classifier, window.shape[-2:])


class _OpenCLIPDense(_DenseEncoder):
    """open_clip ViT, position embeddings resampled to the ``crop_size`` grid."""

    def __init__(self, tag: str, device: str, crop_size: int = 448):
        import open_clip

        spec = _spec(tag)
        self.model, _, _ = open_clip.create_model_and_transforms(
            spec["arch"], pretrained=spec["pretrained"])
        self.model = self.model.to(device).eval()
        self.tokenizer = open_clip.get_tokenizer(spec["arch"])
        self.device = device

        visual = self.model.visual
        patch_size = (visual.patch_size[0] if isinstance(visual.patch_size, tuple)
                      else visual.patch_size)
        new_grid = crop_size // patch_size
        with torch.no_grad():
            new_pe = _interp_pos_embed(visual.positional_embedding.data, new_grid, num_prefix=1)
        visual.positional_embedding = torch.nn.Parameter(new_pe.to(device))
        self.image_transform = T.Compose(
            [T.ToTensor(), T.Normalize(mean=visual.image_mean, std=visual.image_std)])

    @torch.no_grad()
    def encode_text(self, texts: list[str]) -> torch.Tensor:
        emb = self.model.encode_text(self.tokenizer(texts).to(self.device)).float()
        return F.normalize(emb, dim=-1)

    @torch.no_grad()
    def encode_patches_raw(self, window: torch.Tensor) -> torch.Tensor:
        out = self.model.visual.forward_intermediates(window.to(self.device), output_fmt="NLC")
        x = out["image_intermediates"][-2]
        blk = self.model.visual.transformer.resblocks[-1]
        ln1_out = blk.ln_1(x)
        embed_dim = blk.attn.embed_dim
        w_v = blk.attn.in_proj_weight[2 * embed_dim:3 * embed_dim, :]
        b_v = (blk.attn.in_proj_bias[2 * embed_dim:3 * embed_dim]
               if blk.attn.in_proj_bias is not None else None)
        attn_out = blk.attn.out_proj(F.linear(ln1_out, w_v, b_v))
        x = x + blk.ls_1(attn_out)
        x = x + blk.ls_2(blk.mlp(blk.ln_2(x)))
        patches = x @ self.model.visual.proj
        B, N, D = patches.shape
        side = int(round(N ** 0.5))
        assert side * side == N, f"non-square patch grid: N={N}"
        patches = patches.reshape(B, side, side, D).permute(0, 3, 1, 2).contiguous().float()
        return F.normalize(patches, dim=1)


class _SigLIPDense(_DenseEncoder):
    """SigLIP/SigLIP2 via open_clip's timm-backed ViT (no CLS token).

    SigLIP's text-comparable space is reached through ``trunk.attn_pool``, so
    the per-patch path applies that head's value projection and residual MLP
    per patch, after ``trunk.norm``.
    """

    def __init__(self, tag: str, device: str, crop_size: int = 448):
        import open_clip

        spec = _spec(tag)
        self.model, _, _ = open_clip.create_model_and_transforms(
            spec["arch"], pretrained=spec["pretrained"])
        self.model = self.model.to(device).eval()
        self.tokenizer = open_clip.get_tokenizer(spec["arch"])
        self.device = device

        trunk = self.model.visual.trunk
        patch_size = trunk.patch_embed.patch_size[0]
        new_grid = crop_size // patch_size
        with torch.no_grad():
            new_pe = _interp_pos_embed(trunk.pos_embed.data, new_grid,
                                       num_prefix=trunk.num_prefix_tokens)
        trunk.pos_embed = torch.nn.Parameter(new_pe.to(device))
        trunk.patch_embed.strict_img_size = False
        trunk.patch_embed.img_size = (crop_size, crop_size)
        trunk.patch_embed.grid_size = (new_grid, new_grid)
        trunk.patch_embed.num_patches = new_grid * new_grid

        mean = getattr(self.model.visual, "image_mean", None) or (0.5, 0.5, 0.5)
        std = getattr(self.model.visual, "image_std", None) or (0.5, 0.5, 0.5)
        self.image_transform = T.Compose([T.ToTensor(), T.Normalize(mean=mean, std=std)])

    @torch.no_grad()
    def encode_text(self, texts: list[str]) -> torch.Tensor:
        emb = self.model.encode_text(self.tokenizer(texts).to(self.device)).float()
        return F.normalize(emb, dim=-1)

    @torch.no_grad()
    def encode_patches_raw(self, window: torch.Tensor) -> torch.Tensor:
        trunk = self.model.visual.trunk
        blk = trunk.blocks[-1]
        inter = trunk.forward_intermediates(
            window.to(self.device), indices=[len(trunk.blocks) - 2],
            output_fmt="NLC", intermediates_only=True)
        x = inter[0]

        ln1_out = blk.norm1(x)
        B, N, C = ln1_out.shape
        qkv = blk.attn.qkv(ln1_out).reshape(B, N, 3, blk.attn.num_heads, blk.attn.head_dim)
        v = qkv[:, :, 2].reshape(B, N, blk.attn.num_heads * blk.attn.head_dim)
        attn_out = blk.attn.proj(blk.attn.norm(v))
        x = x + blk.drop_path1(blk.ls1(attn_out))
        x = x + blk.drop_path2(blk.ls2(blk.mlp(blk.norm2(x))))
        x = trunk.norm(x)

        pool = trunk.attn_pool
        pkv = pool.kv(x).reshape(B, N, 2, pool.num_heads, pool.head_dim)
        pv = pkv[:, :, 1].reshape(B, N, pool.num_heads * pool.head_dim)
        pout = pool.proj_drop(pool.proj(pv))
        if pool.mlp is not None:
            pout = pout + pool.mlp(pool.norm(pout))

        side = int(round(N ** 0.5))
        assert side * side == N, f"non-square patch grid: N={N}"
        patches = pout.reshape(B, side, side, -1).permute(0, 3, 1, 2).contiguous().float()
        return F.normalize(patches, dim=1)


class _FGClip2Dense(_DenseEncoder):
    """FG-CLIP2 via its built-in ``get_image_dense_feature``.

    Natively variable-resolution (NaFlex patchification via ``max_num_patches``),
    so no position-embedding resampling. ``image_transform`` is ToTensor only —
    the processor normalizes internally — and windows are converted back to PIL
    before the processor call. Class-name text uses ``walk_type="box"``, the
    mode paired with the dense image features.
    """

    def __init__(self, tag: str, device: str, crop_size: int = 448):
        from transformers import AutoImageProcessor, AutoModelForCausalLM, AutoTokenizer

        model_root = _spec(tag)["hf_id"]
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(
            model_root, trust_remote_code=True).to(device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_root)
        self.image_processor = AutoImageProcessor.from_pretrained(model_root)
        self.to_pil = T.ToPILImage()
        self.max_num_patches = (crop_size // 16) ** 2
        self.image_transform = T.ToTensor()

    @torch.no_grad()
    def encode_text(self, texts: list[str]) -> torch.Tensor:
        tok = self.tokenizer(texts, padding="max_length", max_length=64,
                             truncation=True, return_tensors="pt").to(self.device)
        emb = self.model.get_text_features(**tok, walk_type="box").float()
        return F.normalize(emb, dim=-1)

    @torch.no_grad()
    def encode_patches_raw(self, window: torch.Tensor) -> torch.Tensor:
        pils = [self.to_pil(window[i].cpu()) for i in range(window.shape[0])]
        image_input = self.image_processor(
            images=pils, max_num_patches=self.max_num_patches,
            return_tensors="pt").to(self.device)
        dense = self.model.get_image_dense_feature(**image_input).float()
        spatial = image_input["spatial_shapes"]
        assert bool((spatial == spatial[0]).all()), "mixed spatial_shapes within a batch"
        h, w = spatial[0].tolist()
        B, N, D = dense.shape
        patches = dense.reshape(B, h, w, D).permute(0, 3, 1, 2).contiguous()
        return F.normalize(patches, dim=1)


def build_dense_encoder(tag: str, device: str, *, crop_size: int = 448):
    """Build the dense (per-patch) encoder for ``tag``."""
    family = _spec(tag).get("family", "trained")
    if family == "trained":
        return _TrainedDense(tag, device, crop_size=crop_size)
    if family == "clip":
        return _CLIPDense(device, crop_size=crop_size)
    if family == "open_clip":
        return _OpenCLIPDense(tag, device, crop_size=crop_size)
    if family == "siglip":
        return _SigLIPDense(tag, device, crop_size=crop_size)
    if family == "fgclip2":
        return _FGClip2Dense(tag, device, crop_size=crop_size)
    raise ValueError(f"Unknown family {family!r} for tag {tag!r}")
