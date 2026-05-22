"""Checkpoint save / load for the alignment model.

Writes sharded DCP checkpoints in the same layout the inference
loaders consume::

    <run_dir>/ckpt/<step>/.metadata
    <run_dir>/ckpt/<step>/__<rank>_0.distcp

Backbone parameters are excluded via a state-dict hook installed by
:func:`register_frozen_dont_save_hooks` so only the trainable heads +
``logit_scale`` (plus the optimizer state) land on disk.

Training is expected to be launched under ``torchrun`` — DCP is built
for distributed runs and is not reliable when invoked from a plain
single-process Python interpreter.

Public API
----------
    register_frozen_dont_save_hooks(model)
    save_dcp(ckpt_dir, model, *, iteration, optimizer=None)
    load_dcp(ckpt_dir, model, *, optimizer=None, strict=False) -> int
    find_latest_checkpoint(ckpt_root) -> Path | None
    keep_last_n_checkpoints(ckpt_root, n)
"""
from __future__ import annotations

import logging
import re
import shutil
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import FileSystemReader, FileSystemWriter
from torch.distributed.checkpoint import state_dict as dcpsd

logger = logging.getLogger("vla.training")


_INT_RE = re.compile(r"^\d+$")


def _frozen_param_keys(model: torch.nn.Module) -> list[str]:
    """Names of state-dict entries belonging to frozen backbones."""
    keys: list[str] = []
    for k in model.state_dict():
        if k.startswith("image_encoder.backbone."):
            keys.append(k)
        elif k.startswith("image_encoder.pca_"):
            keys.append(k)
        elif k.startswith("text_encoder.backbone."):
            keys.append(k)
    return keys


def register_frozen_dont_save_hooks(model: torch.nn.Module) -> None:
    """Install state-dict hooks that drop the frozen backbones from save/load.

    The frozen backbones (DINOv3, RoBERTa, optional CleanDIFT PCA
    buffers) are reloaded from HuggingFace at every run, so they
    don't belong in the DCP shards.
    """
    dont_save = set(
        k.replace("_checkpoint_wrapped_module.", "")
        for k in _frozen_param_keys(model)
    )
    if not dont_save:
        return

    def _save_post_hook(_module, state_dict, prefix, _meta):
        for k in list(dont_save):
            full = prefix + k
            if full in state_dict:
                del state_dict[full]

    state = {"prefix": None}

    def _load_pre_hook(_m, _sd, prefix, _meta, _strict, _missing, _unexpected, _errs):
        state["prefix"] = prefix

    def _load_post_hook(_module, incompatible_keys):
        prefix = state["prefix"] or ""
        for key in list(incompatible_keys.missing_keys):
            tail = key.removeprefix(prefix).replace("_checkpoint_wrapped_module.", "")
            if tail in dont_save:
                incompatible_keys.missing_keys.remove(key)
        state["prefix"] = None

    model.register_state_dict_post_hook(_save_post_hook)
    model.register_load_state_dict_pre_hook(_load_pre_hook)
    model.register_load_state_dict_post_hook(_load_post_hook)
    logger.info(f"Registered dont-save hooks for {len(dont_save)} frozen params")


def save_dcp(
    ckpt_dir: str | Path,
    model: torch.nn.Module,
    *,
    iteration: int,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> None:
    """Write trainable-head + optimizer state to ``ckpt_dir`` as DCP shards."""
    if not (dist.is_available() and dist.is_initialized()):
        raise RuntimeError(
            "save_dcp expects torch.distributed to be initialized; "
            "launch the training script under torchrun."
        )

    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.parent.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    payload: dict = {"iteration": iteration,
                     "model": dcpsd.get_model_state_dict(model)}
    if optimizer is not None:
        payload["optimizer"] = dcpsd.get_optimizer_state_dict(model, optimizer)
    dcp.save(payload, storage_writer=FileSystemWriter(ckpt_dir))
    dist.barrier()
    logger.info(f"Saved checkpoint at iter {iteration} -> {ckpt_dir}")


def load_dcp(
    ckpt_dir: str | Path,
    model: torch.nn.Module,
    *,
    optimizer: Optional[torch.optim.Optimizer] = None,
    strict: bool = False,
) -> int:
    """Load DCP shards into ``model`` (and ``optimizer`` if provided)."""
    if not (dist.is_available() and dist.is_initialized()):
        raise RuntimeError(
            "load_dcp expects torch.distributed to be initialized; "
            "launch the training script under torchrun."
        )

    ckpt_dir = Path(ckpt_dir)
    payload: dict = {"iteration": 0,
                     "model": dcpsd.get_model_state_dict(model)}
    if optimizer is not None:
        payload["optimizer"] = dcpsd.get_optimizer_state_dict(model, optimizer)
    planner = dcp.default_planner.DefaultLoadPlanner(allow_partial_load=not strict)
    dcp.load(payload, storage_reader=FileSystemReader(ckpt_dir), planner=planner)
    dcpsd.set_model_state_dict(model, payload["model"])
    if optimizer is not None:
        dcpsd.set_optimizer_state_dict(model, optimizer, payload["optimizer"])
    iteration = int(payload["iteration"])
    logger.info(f"Loaded checkpoint -> iter {iteration} from {ckpt_dir}")
    return iteration


def find_latest_checkpoint(ckpt_root: str | Path) -> Optional[Path]:
    """Return the subdirectory with the largest integer name, or ``None``."""
    ckpt_root = Path(ckpt_root)
    if not ckpt_root.is_dir():
        return None
    ckpts = [p for p in ckpt_root.iterdir() if p.is_dir() and _INT_RE.match(p.name)]
    if not ckpts:
        return None
    return max(ckpts, key=lambda p: int(p.name))


def keep_last_n_checkpoints(ckpt_root: str | Path, n: Optional[int]) -> None:
    """Delete all but the ``n`` highest-numbered checkpoints under ``ckpt_root``."""
    if n is None:
        return
    ckpt_root = Path(ckpt_root)
    if not ckpt_root.is_dir():
        return
    ckpts = sorted(
        (p for p in ckpt_root.iterdir() if p.is_dir() and _INT_RE.match(p.name)),
        key=lambda p: int(p.name),
    )
    for old in ckpts[:-n]:
        try:
            shutil.rmtree(old)
            logger.info(f"Pruned old checkpoint {old}")
        except OSError as e:
            logger.warning(f"Could not prune {old}: {e}")
