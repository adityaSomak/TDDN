"""Load TDDN for ``tddn_mask.py`` through the shared generic model API.

The default is the flat repository checkpoint. ``ALIGNMENT_CKPT`` may override
it with a local release directory or Hugging Face repo id. For legacy training
trees, ``ALIGNMENT_CKPT_STEPS`` selects an explicit checkpoint step.
"""

from __future__ import annotations

import os
import sys
from typing import Any

from pathlib import Path


def load_alignment_model(
    device: str = "cuda",
    common_grid: int | None = 64,
) -> Any:
    """Build TDDN, optionally from explicit legacy steps, and place on `device`.

    `common_grid` overrides the image encoder's patch grid (IMG_SIZE / PATCH_SIZE).
    Pass None to keep the checkpoint's value.

    Environment variables are resolved at call time so importing this module
    does not initialize any model dependencies.
    """
    repo_root = Path(__file__).resolve().parents[7]
    sys.path.insert(0, str(repo_root / "experiments"))

    from shared_utils.feature_extraction import load_model

    checkpoint = os.environ.get("ALIGNMENT_CKPT") or None
    raw_steps = os.environ.get("ALIGNMENT_CKPT_STEPS")
    checkpoint_steps = None
    if raw_steps:
        checkpoint_steps = [step.strip() for step in raw_steps.split(",") if step.strip()]

    model, _ = load_model(
        "tddn",
        device=device,
        checkpoint=checkpoint,
        checkpoint_steps=checkpoint_steps,
        common_grid_override=common_grid,
    )
    return model
