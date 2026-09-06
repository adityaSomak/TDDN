"""Trained text-vision alignment model code used by the trained backbones.

Public API
----------
    AlignConfig             hyperparameter dataclass.
    AlignmentModel          end-to-end (image, text) -> similarity logits.
    load_trainable_state    load the trainable head params from a DCP ckpt.
    average_states          element-wise mean of N state dicts.
    resolve_ckpt_steps      normalize a checkpoint selector to a list of names.

The ``dinov3`` package (Meta AI reference implementation) must be available
on the import path for the ViT building blocks used by the vision encoder.
"""

from __future__ import annotations

import torch.distributed.checkpoint as dcp

from .config import AlignConfig
from .model import AlignmentModel


def trainable_keys(model) -> list[str]:
    """Return the alignment-head keys persisted by release checkpoints."""
    return [
        k for k in model.state_dict()
        if not k.startswith("image_encoder.backbone.")
        and not k.startswith("image_encoder.pca_")
        and not k.startswith("text_encoder.backbone.")
    ]


def load_trainable_state(model, ckpt_path):
    """Load an FSDP DCP checkpoint and return only the trainable params.

    Trainable = projection heads + logit scale. Frozen vision/text backbones
    and PCA buffers are excluded so the model can be reconstructed by loading
    the backbone from HuggingFace and overlaying these tensors.

    Args:
        model: AlignmentModel instance whose state_dict provides the key
            list and dtype/shape templates.
        ckpt_path: directory containing the DCP shards (``__N_0.distcp``).

    Returns:
        dict[str, torch.Tensor] of trainable params on CPU as float32.
    """
    full_sd = model.state_dict()
    state_dict = {"model": {k: full_sd[k] for k in trainable_keys(model)}}
    dcp.load(state_dict, checkpoint_id=str(ckpt_path))
    return {k: v.cpu().float() for k, v in state_dict["model"].items()}


def average_states(states):
    """Element-wise mean of a list of state dicts (same keys, same shapes)."""
    avg = {k: states[0][k].clone() for k in states[0]}
    for sd in states[1:]:
        for k in avg:
            avg[k] += sd[k]
    n = len(states)
    for k in avg:
        avg[k] /= n
    return avg


MAX_CKPT_STEPS = 2


def resolve_ckpt_steps(spec) -> list[str]:
    """Normalize a checkpoint selector into a list of subdirectory names.

    Two forms are accepted, and nothing else:

        "tddn" / 99        one checkpoint, loaded as-is
        [99, 149]          two checkpoints, weight-averaged at load

    A scalar must not be iterated: ``tuple("tddn")`` would yield
    ``('t','d','d','n')`` and look for four checkpoints named after single
    letters, so scalars are wrapped rather than expanded.

    Returns:
        list[str] of 1 or 2 names, ready to join onto ``<ckpt_dir>/ckpt/``.
    """
    if spec is None:
        raise ValueError("checkpoint selector is None; expected a name or a list of two")
    if isinstance(spec, (str, int)):
        steps = [spec]
    else:
        try:
            steps = list(spec)
        except TypeError:
            raise TypeError(
                f"cannot interpret checkpoint selector {spec!r} of type "
                f"{type(spec).__name__}") from None
    if not steps:
        raise ValueError("checkpoint selector is empty")
    if len(steps) > MAX_CKPT_STEPS:
        raise ValueError(
            f"{len(steps)} checkpoints requested ({steps}); at most "
            f"{MAX_CKPT_STEPS} are supported (one, or two to average)")
    return [str(s) for s in steps]


__all__ = ["AlignConfig", "AlignmentModel", "trainable_keys", "load_trainable_state",
           "average_states", "resolve_ckpt_steps", "MAX_CKPT_STEPS"]
