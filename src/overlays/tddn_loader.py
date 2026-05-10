"""Load the text-aligned DINOv3+CleanDIFT+RoBERTa model used for TDDN masks."""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, "/data/shanmukha/PuzzleBench/text_alignment")
sys.path.insert(0, "/data/shanmukha/dinov3")
os.environ.setdefault("HF_HOME", "/data/shanmukha/.cache/huggingface")

from omegaconf import OmegaConf  # noqa: E402

from core.config import AlignConfig  # noqa: E402
from core.model import AlignmentModel  # noqa: E402
from eval._common.ckpt import average_states, load_trainable_state  # noqa: E402


CFG_PATH = Path("/data/shanmukha/output/fused_dinov3_cleandift_coco_ft/config.yaml")
CKPT_DIR = Path("/data/shanmukha/output/fused_dinov3_cleandift_coco_ft/ckpt")
DEFAULT_STEPS: tuple[int, ...] = (149, 199)


def load_alignment_model(
    device: str = "cuda:0",
    cfg_path: Path = CFG_PATH,
    ckpt_dir: Path = CKPT_DIR,
    ckpt_steps: tuple[int, ...] = DEFAULT_STEPS,
    common_grid: int | None = 64,
) -> AlignmentModel:
    """Build AlignmentModel, weight-average the given checkpoint steps, place on device.

    `common_grid` overrides the image encoder's patch grid (IMG_SIZE / PATCH_SIZE).
    Pass None to keep the checkpoint's value.
    """
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


