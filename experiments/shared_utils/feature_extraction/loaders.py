"""Model factories — one per backbone family.

Every loader returns `(model, meta)` where `model` is a frozen, eval-mode
torch module (or diffusers pipeline) on `device`, and `meta` carries the
small set of facts that downstream extractors need to know about the
backbone (patch size, hidden dim, number of special tokens, etc.) without
re-querying the model.

Most loaders read configs from Hugging Face. TDN and TDDN use the generic
``load_model`` API with flat local or Hub-hosted Safetensors releases; explicit
legacy DCP steps remain supported for reproducibility.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Optional

import torch
from omegaconf import OmegaConf

from ..paths import CHECKPOINTS_ROOT
from .constants import require_hf_token

_VARIANTS = {
    "tdn": {
        "release_dir": "TDN",
        "hub_repo": "PuzzleBench/TDN",
        "legacy_dir": "vith_roberta_v3_coco_ft",
        "meta_name": "vith-roberta",
        "default_dtype": torch.bfloat16,
        "fused": False,
    },
    "tddn": {
        "release_dir": "TDDN",
        "hub_repo": "PuzzleBench/TDDN",
        "legacy_dir": "fused_dinov3_cleandift_coco_ft",
        "meta_name": "fused-dinov3-cd",
        "default_dtype": torch.float32,
        "fused": True,
    },
}


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
    dinov3_root = os.environ.get("DINOV3_ROOT")
    if dinov3_root and dinov3_root not in sys.path:
        sys.path.insert(0, dinov3_root)

    from .text_alignment import AlignConfig, AlignmentModel

    if config_path.suffix == ".json":
        payload = json.loads(config_path.read_text())
        allowed = {f.name for f in fields(AlignConfig)}
        cfg = {k: v for k, v in payload.items() if k in allowed}
    else:
        cfg = OmegaConf.to_container(OmegaConf.load(config_path))
    config = AlignConfig(**cfg)
    model = AlignmentModel(config).to(device=device, dtype=dtype).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, config


def _resolve_legacy_paths(ckpt_dir: Path, spec) -> list[Path]:
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


def _download_release_snapshot(repo_id: str) -> Path:
    """Download the minimal release snapshot required by the generic loader."""
    from huggingface_hub import snapshot_download

    try:
        return Path(snapshot_download(
            repo_id=repo_id,
            allow_patterns=["config.json", "model.safetensors"],
            token=require_hf_token(),
        ))
    except Exception as exc:
        raise RuntimeError(f"Could not download checkpoint repository {repo_id!r}: {exc}") from exc


def _resolve_release_dir(checkpoint, variant: dict) -> Path:
    """Resolve a local release directory or a Hugging Face model snapshot."""
    required_names = ("config.json", "model.safetensors")
    if checkpoint is None:
        local_path = CHECKPOINTS_ROOT / variant["release_dir"]
        path = (local_path if all((local_path / name).is_file() for name in required_names)
                else _download_release_snapshot(variant["hub_repo"]))
    elif isinstance(checkpoint, Path):
        path = checkpoint.expanduser()
    else:
        raw = str(checkpoint)
        candidate = Path(raw).expanduser()
        if candidate.is_dir() or candidate.is_absolute() or raw.startswith("."):
            path = candidate
        elif "/" not in raw:
            path = CHECKPOINTS_ROOT / raw
        else:
            path = _download_release_snapshot(raw)

    required = [path / name for name in required_names]
    missing = [p.name for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Incomplete {variant['release_dir']} checkpoint at {path}: missing {missing}. "
            "Expected a Hugging Face-style directory containing config.json and model.safetensors."
        )
    return path


def _load_release_state(model, path: Path) -> dict[str, torch.Tensor]:
    from safetensors.torch import load_file
    from .text_alignment import trainable_keys

    state = load_file(str(path / "model.safetensors"), device="cpu")
    expected = set(trainable_keys(model))
    actual = set(state)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise RuntimeError(
            f"Checkpoint tensor keys do not match the model (missing={missing[:5]}, "
            f"unexpected={unexpected[:5]})."
        )
    return {k: v.float() for k, v in state.items()}


def load_model(
    name: str,
    device: torch.device | str,
    checkpoint: str | Path | None = None,
    checkpoint_steps=None,
    dtype: torch.dtype | None = None,
    common_grid_override: Optional[int] = None,
) -> tuple[torch.nn.Module, ModelMeta]:
    """Load a released TDN/TDDN model or explicit legacy DCP training steps.

    The default and Hub formats are ``config.json`` + ``model.safetensors``.
    Supplying ``checkpoint_steps`` switches to the legacy
    ``<checkpoint>/ckpt/<step>`` DCP layout, with up to two steps averaged.
    """
    from .text_alignment import average_states, load_trainable_state

    key = str(name).lower()
    if key not in _VARIANTS:
        raise ValueError(f"Unknown trained model {name!r}; expected one of {sorted(_VARIANTS)}")
    variant = _VARIANTS[key]
    dtype = dtype or variant["default_dtype"]
    if key == "tdn" and dtype == torch.float16:
        raise ValueError("TDN's DINOv3 backbone is numerically unstable in fp16; use bf16 or fp32.")

    if checkpoint_steps is None:
        ckpt_dir = _resolve_release_dir(checkpoint, variant)
        config_path = ckpt_dir / "config.json"
        release_config = json.loads(config_path.read_text())
        if release_config.get("variant") != key:
            raise ValueError(
                f"Checkpoint {ckpt_dir} declares variant {release_config.get('variant')!r}, "
                f"but load_model was called with {key!r}."
            )
        if release_config.get("format_version") != 1:
            raise ValueError(
                f"Unsupported checkpoint format_version "
                f"{release_config.get('format_version')!r} in {config_path}; expected 1."
            )
    else:
        if checkpoint is None:
            ckpt_dir = CHECKPOINTS_ROOT / variant["legacy_dir"]
        else:
            ckpt_dir = Path(checkpoint).expanduser()
        config_path = ckpt_dir / "config.yaml"
        if not config_path.is_file():
            raise FileNotFoundError(
                f"Legacy checkpoint config not found: {config_path}. "
                "Pass the training-output tree containing config.yaml and ckpt/."
            )

    model, config = _build_alignment_model(config_path, device, dtype)

    if variant["fused"] and hasattr(model, "load_cleandift_backbone"):
        model.load_cleandift_backbone(device)

    if checkpoint_steps is None:
        trainable = _load_release_state(model, ckpt_dir)
        source = str(ckpt_dir)
    else:
        paths = _resolve_legacy_paths(ckpt_dir, checkpoint_steps)
        states = [load_trainable_state(model, p) for p in paths]
        trainable = average_states(states) if len(states) > 1 else states[0]
        source = [str(p) for p in paths]

    _, unexpected = model.load_state_dict(trainable, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected keys when loading {key}: {unexpected[:5]}...")

    if variant["fused"] and common_grid_override is not None and hasattr(model, "image_encoder"):
        if hasattr(model.image_encoder, "common_grid"):
            model.image_encoder.common_grid = common_grid_override

    meta = ModelMeta(
        name=variant["meta_name"],
        patch_size=16,
        dim=config.embed_dim,
        num_special_tokens=5,
        extra={"variant": key, "checkpoint": source, "dtype": dtype,
               "config": config, "common_grid": common_grid_override},
    )
    return model, meta
