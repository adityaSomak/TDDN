"""Model factories — one per backbone family.

Every loader returns `(model, meta)` where `model` is a frozen, eval-mode
torch module (or diffusers pipeline) on `device`, and `meta` carries the
small set of facts that downstream extractors need to know about the
backbone (patch size, hidden dim, number of special tokens, etc.) without
re-querying the model.

Most loaders read configs from HuggingFace. The two trained models
(`vith_roberta`, `fused-dinov3-cd`) load weights from this package's
`checkpoints/` directory via the vendored `text_alignment` package.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from omegaconf import OmegaConf

from .constants import require_hf_token

_PKG_ROOT = Path(__file__).resolve().parent
_CHECKPOINTS_ROOT = _PKG_ROOT / "checkpoints"


@dataclass
class ModelMeta:
    """Per-backbone info needed by extractors and tasks.

    `num_special_tokens` is for ViTs (CLS + register tokens at the front of
    the patch sequence). `patch_size` is for the spatial backbones. `dim`
    is the hidden dimension; for diffusion this is unused (set 0) since the
    extractor returns per-layer outputs.
    """
    name: str
    patch_size: int
    dim: int
    num_special_tokens: int = 0
    extra: dict = None


# ============================================================================
# DINOv3 — HuggingFace facebook/dinov3-{vitb16,vith16plus}-pretrain-lvd1689m
# ============================================================================

_DINOV3_IDS = {
    "vitb16": "facebook/dinov3-vitb16-pretrain-lvd1689m",
    "vith16plus": "facebook/dinov3-vith16plus-pretrain-lvd1689m",
}


def load_dinov3(
    variant: str,
    device: torch.device | str,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[torch.nn.Module, ModelMeta]:
    """Load DINOv3 ViT-B/16 or ViT-H/16+ from HuggingFace.

    Default dtype is bfloat16. **Do not use fp16** — DINOv3 is numerically
    unstable in fp16 (matrix products underflow); ImageNet_Classification discovered
    this the hard way. fp32 and bf16 are both fine.
    """
    if variant not in _DINOV3_IDS:
        raise ValueError(f"Unknown DINOv3 variant {variant!r}. Choose from {list(_DINOV3_IDS)}.")
    if dtype == torch.float16:
        raise ValueError("DINOv3 is unstable in fp16. Use bf16 or fp32.")

    from transformers import AutoModel
    model_id = _DINOV3_IDS[variant]
    model = AutoModel.from_pretrained(model_id, token=require_hf_token(), torch_dtype=dtype)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False

    # ViT-B has 1 CLS only; ViT-H+ has CLS + 4 register tokens.
    num_register = getattr(model.config, "num_register_tokens", 0) or 0
    num_special_tokens = 1 + num_register

    meta = ModelMeta(
        name=f"dinov3-{variant}",
        patch_size=16,
        dim=model.config.hidden_size,
        num_special_tokens=num_special_tokens,
        extra={"model_id": model_id, "dtype": dtype},
    )
    return model, meta


# ============================================================================
# DINOv2 — HuggingFace facebook/dinov2-{base,large,giant}
# ============================================================================

_DINOV2_IDS = {
    "vitb14": "facebook/dinov2-base",
    "vitl14": "facebook/dinov2-large",
    "vitg14": "facebook/dinov2-giant",
}


def load_dinov2(
    variant: str,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.nn.Module, ModelMeta]:
    """Load DINOv2 from HuggingFace. No register tokens — only CLS."""
    if variant not in _DINOV2_IDS:
        raise ValueError(f"Unknown DINOv2 variant {variant!r}. Choose from {list(_DINOV2_IDS)}.")

    from transformers import AutoModel
    model_id = _DINOV2_IDS[variant]
    model = AutoModel.from_pretrained(model_id, torch_dtype=dtype)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False

    meta = ModelMeta(
        name=f"dinov2-{variant}",
        patch_size=14,
        dim=model.config.hidden_size,
        num_special_tokens=1,  # CLS only
        extra={"model_id": model_id, "dtype": dtype},
    )
    return model, meta


# ============================================================================
# CLIP — HuggingFace openai/clip-vit-large-patch14[-336]
# ============================================================================

_CLIP_IDS = {
    "vit-l-14": "openai/clip-vit-large-patch14",
    "vit-l-14-336": "openai/clip-vit-large-patch14-336",
}


def load_clip(
    variant: str,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.nn.Module, ModelMeta]:
    """Load CLIP vision tower. Resolution must be a multiple of 14."""
    if variant not in _CLIP_IDS:
        raise ValueError(f"Unknown CLIP variant {variant!r}. Choose from {list(_CLIP_IDS)}.")

    from transformers import CLIPModel
    model_id = _CLIP_IDS[variant]
    model = CLIPModel.from_pretrained(model_id, torch_dtype=dtype)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False

    meta = ModelMeta(
        name=f"clip-{variant}",
        patch_size=14,
        dim=model.config.vision_config.hidden_size,
        num_special_tokens=1,  # CLS
        extra={"model_id": model_id, "dtype": dtype},
    )
    return model, meta


# ============================================================================
# Diffusion — SD 2.1 (vanilla) and CleanDIFT (SD 2.1 + custom UNet weights)
# ============================================================================

_SD_BASE_ID = "Charles-Elena/stable-diffusion-2-1"
_CLEANDIFT_REPO = "CompVis/cleandift"
_CLEANDIFT_FILE = "cleandift_sd21_unet.safetensors"


def load_diffusion(
    backbone: str,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
) -> tuple[object, ModelMeta]:
    """Load the SD 2.1 pipeline; overlay CleanDIFT UNet weights if requested.

    Returns the StableDiffusionPipeline. The extractor wires forward hooks
    onto `pipe.unet.up_blocks[0/1/2].resnets[1]` (or `.attentions[1]` for
    CrossAttn blocks) to capture the layer-2/5/8 outputs at hook-time.
    """
    if backbone not in {"sd21", "cleandift"}:
        raise ValueError(f"Unknown diffusion backbone {backbone!r}.")

    from diffusers import StableDiffusionPipeline
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    pipe = StableDiffusionPipeline.from_pretrained(
        _SD_BASE_ID,
        torch_dtype=dtype,
        safety_checker=None,
    ).to(device)

    if backbone == "cleandift":
        ckpt_path = hf_hub_download(repo_id=_CLEANDIFT_REPO, filename=_CLEANDIFT_FILE)
        state_dict = load_file(ckpt_path)
        pipe.unet.load_state_dict(state_dict, strict=True)

    pipe.unet.eval()
    pipe.vae.eval()
    pipe.text_encoder.eval()
    for m in (pipe.unet, pipe.vae, pipe.text_encoder):
        for p in m.parameters():
            p.requires_grad = False

    meta = ModelMeta(
        name=backbone,
        patch_size=8,  # VAE stride
        dim=0,         # per-layer; downstream knows {1280, 1280, 640}
        num_special_tokens=0,
        extra={"sd_base_id": _SD_BASE_ID, "dtype": dtype},
    )
    return pipe, meta


# ============================================================================
# Text-aligned vith_roberta — DINOv3-H+ backbone + 2 head blocks + RoBERTa text
# ============================================================================

def _build_alignment_model(config_path: Path, device, dtype: torch.dtype):
    """Construct an AlignmentModel from a saved training config."""
    from .text_alignment import AlignConfig, AlignmentModel

    cfg = OmegaConf.load(config_path)
    config = AlignConfig(**OmegaConf.to_container(cfg))
    model = AlignmentModel(config).to(device=device, dtype=dtype).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, config


def _resolve_ckpt_paths(ckpt_dir: Path, spec) -> list[Path]:
    """Normalize a checkpoint selector and check the directories exist.

    Raises before any load so a bad selector reports itself, rather than failing
    deep inside ``dcp.load`` with an opaque error.
    """
    from .text_alignment import resolve_ckpt_steps

    root = Path(ckpt_dir) / "ckpt"
    names = resolve_ckpt_steps(spec)
    paths = [root / n for n in names]
    missing = [p.name for p in paths if not p.is_dir()]
    if missing:
        available = sorted(p.name for p in root.iterdir() if p.is_dir()) if root.is_dir() else []
        raise FileNotFoundError(
            f"checkpoint(s) {missing} not found under {root}. "
            f"Available: {available or '(none)'}")
    return paths


def load_vith_roberta(
    device: torch.device | str,
    ckpt_dir: Optional[Path] = None,
    ckpt_steps="tdn",
    dtype: torch.dtype = torch.float16,
) -> tuple[torch.nn.Module, ModelMeta]:
    """Text-aligned DINOv3-H+ + RoBERTa model.

    ``ckpt_steps`` names one checkpoint directory, or lists two to weight-average.
    """
    from .text_alignment import average_states, load_trainable_state

    if ckpt_dir is None:
        ckpt_dir = _CHECKPOINTS_ROOT / "vith_roberta_v3_coco_ft"
    ckpt_dir = Path(ckpt_dir)

    model, config = _build_alignment_model(ckpt_dir / "config.yaml", device, dtype)
    paths = _resolve_ckpt_paths(ckpt_dir, ckpt_steps)
    states = [load_trainable_state(model, p) for p in paths]
    trainable = average_states(states) if len(states) > 1 else states[0]
    _, unexpected = model.load_state_dict(trainable, strict=False)
    # Only frozen-backbone keys should be "missing" — that's expected.
    if unexpected:
        raise RuntimeError(f"Unexpected keys when loading vith_roberta: {unexpected[:5]}...")

    meta = ModelMeta(
        name="vith-roberta",
        patch_size=16,
        dim=config.embed_dim,  # global = 2*hidden (CLS + mean(patches))
        num_special_tokens=5,   # DINOv3-H+
        extra={"ckpt_steps": [p.name for p in paths], "ckpt_dir": str(ckpt_dir),
               "dtype": dtype, "config": config},
    )
    return model, meta


def load_fused_dinov3_cd(
    device: torch.device | str,
    ckpt_dir: Optional[Path] = None,
    ckpt_steps="tddn",
    dtype: torch.dtype = torch.float32,
    common_grid_override: Optional[int] = None,
) -> tuple[torch.nn.Module, ModelMeta]:
    """Trained DINOv3+CleanDIFT fused encoder.

    ``ckpt_steps`` names one checkpoint directory, or lists two to weight-average.
    """
    from .text_alignment import average_states, load_trainable_state

    if ckpt_dir is None:
        ckpt_dir = _CHECKPOINTS_ROOT / "fused_dinov3_cleandift_coco_ft"
    ckpt_dir = Path(ckpt_dir)

    model, config = _build_alignment_model(ckpt_dir / "config.yaml", device, dtype)

    # Load CleanDIFT backbone weights — these are frozen and not in the DCP
    # checkpoint (which holds only the trainable heads).
    if hasattr(model, "load_cleandift_backbone"):
        model.load_cleandift_backbone(device)

    paths = _resolve_ckpt_paths(ckpt_dir, ckpt_steps)
    states = [load_trainable_state(model, p) for p in paths]
    avg = average_states(states) if len(states) > 1 else states[0]
    _, unexpected = model.load_state_dict(avg, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected keys when loading fused model: {unexpected[:5]}...")

    # Optional: override the trained common_grid (typically 21 @ 336) so we
    # can run a single forward at a larger resolution (e.g., 32 @ 512).
    if common_grid_override is not None and hasattr(model, "image_encoder"):
        if hasattr(model.image_encoder, "common_grid"):
            model.image_encoder.common_grid = common_grid_override

    meta = ModelMeta(
        name="fused-dinov3-cd",
        patch_size=16,
        dim=config.embed_dim,
        num_special_tokens=5,
        extra={"ckpt_steps": [p.name for p in paths], "ckpt_dir": str(ckpt_dir),
               "dtype": dtype, "common_grid": common_grid_override},
    )
    return model, meta
