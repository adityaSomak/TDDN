"""Trained text-vision alignment model code used by the trained backbones.

Public API
----------
    AlignConfig             hyperparameter dataclass.
    AlignmentModel          end-to-end (image, text) -> similarity logits.
    load_trainable_state    load the trainable head params from a DCP ckpt.
    average_states          element-wise mean of N state dicts.

The ``dinov3`` package (Meta AI reference implementation) must be available
on the import path for the ViT building blocks used by the vision encoder.
"""

from __future__ import annotations

import torch.distributed.checkpoint as dcp

from .config import AlignConfig
from .model import AlignmentModel


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
    trainable_keys = [
        k for k in full_sd
        if not k.startswith("image_encoder.backbone.")
        and not k.startswith("image_encoder.pca_")
        and not k.startswith("text_encoder.backbone.")
    ]
    state_dict = {"model": {k: full_sd[k] for k in trainable_keys}}
    dcp.load(state_dict, checkpoint_id=str(ckpt_path))
    return {k: v.cpu().float() for k, v in state_dict["model"].items()}


def average_states(states):
    """Element-wise mean of a list of state dicts (same keys, same shapes).

    Used to average two checkpoint epochs (e.g. fused-dinov3-cd ckpts 149+199)
    for inference.
    """
    avg = {k: states[0][k].clone() for k in states[0]}
    for sd in states[1:]:
        for k in avg:
            avg[k] += sd[k]
    n = len(states)
    for k in avg:
        avg[k] /= n
    return avg


__all__ = ["AlignConfig", "AlignmentModel", "load_trainable_state", "average_states"]
