"""Load the text-aligned vision/text model used by ``tddn_mask.py``.

The model itself is **not** shipped in this repository — it lives in the
PuzzleBench text-alignment project. Three environment variables tell the
loader where to find it:

    PUZZLEBENCH_ROOT   path to the PuzzleBench/text_alignment/ source tree
                       (provides core.config, core.model, eval._common.ckpt)
    DINOV3_ROOT        path to the dinov3/ source tree (vision backbone)
    ALIGNMENT_CKPT     path to a training-output directory containing
                       config.yaml and ckpt/<step>/ subdirectories

Optionally:

    ALIGNMENT_CKPT_STEPS   comma-separated list of step numbers to weight-
                           average (defaults to "149,199")
    HF_HOME                HuggingFace cache directory
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any


def _require_env(name: str) -> Path:
    val = os.environ.get(name)
    if not val:
        raise RuntimeError(
            f"{name} environment variable is required by tddn_loader. "
            f"See mask_generation/README.md for setup."
        )
    return Path(val).resolve()


def load_alignment_model(
    device: str = "cuda",
    common_grid: int | None = 64,
) -> Any:
    """Build the alignment model, weight-average checkpoints, place on `device`.

    `common_grid` overrides the image encoder's patch grid (IMG_SIZE / PATCH_SIZE).
    Pass None to keep the checkpoint's value.

    Resolves dependency locations from env vars at call time so that simply
    importing this module does not require the trained-model environment.
    """
    puzzlebench_root = _require_env("PUZZLEBENCH_ROOT")
    dinov3_root = _require_env("DINOV3_ROOT")
    alignment_ckpt = _require_env("ALIGNMENT_CKPT")

    sys.path.insert(0, str(puzzlebench_root))
    sys.path.insert(0, str(dinov3_root))

    from omegaconf import OmegaConf
    from core.config import AlignConfig
    from core.model import AlignmentModel
    from eval._common.ckpt import average_states, load_trainable_state

    cfg_path = alignment_ckpt / "config.yaml"
    ckpt_dir = alignment_ckpt / "ckpt"
    ckpt_steps = tuple(
        int(s) for s in os.environ.get("ALIGNMENT_CKPT_STEPS", "149,199").split(",")
    )

    cfg = OmegaConf.load(cfg_path)
    model = AlignmentModel(AlignConfig(**OmegaConf.to_container(cfg, resolve=True))).to(device)
    states = [load_trainable_state(model, ckpt_dir / str(s)) for s in ckpt_steps]
    model.load_state_dict(average_states(states), strict=False)
    model.image_encoder.load_backbone(device)
    model.text_encoder.backbone.to(device).eval()
    model.eval()
    if common_grid is not None and getattr(model.image_encoder, "common_grid", None) != common_grid:
        model.image_encoder.common_grid = common_grid
    return model
