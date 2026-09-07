"""Launcher-agnostic RANK/WORLD_SIZE/LOCAL_RANK resolution.

Reads whichever launcher set these env vars -- DeepSpeed, torchrun, srun, or none
(bare `python run_caption.py`, rank 0 of 1) -- without assuming any one of them.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class DistInfo:
    rank: int
    world_size: int
    local_rank: int
    local_size: int


def _first_int(*names: str, default: int) -> int:
    for name in names:
        val = os.environ.get(name)
        if val is not None:
            return int(val)
    return default


def resolve() -> DistInfo:
    rank = _first_int("RANK", "SLURM_PROCID", default=0)
    world_size = _first_int("WORLD_SIZE", "SLURM_NTASKS", default=1)
    local_rank = _first_int("LOCAL_RANK", "SLURM_LOCALID", default=0)
    local_size = _first_int("LOCAL_SIZE", "LOCAL_WORLD_SIZE", "SLURM_NTASKS_PER_NODE",
                             default=world_size)
    return DistInfo(rank=rank, world_size=world_size, local_rank=local_rank,
                     local_size=local_size)


def slice_visible_devices(local_rank: int, tp: int) -> str:
    """Slice CUDA_VISIBLE_DEVICES *by position*, not by range(), since a launcher
    (DeepSpeed in particular) may have already narrowed it to a non-0..N-1 list.
    Must be called before `import vllm` / any torch.cuda touch."""
    import torch  # local import: this must run before vllm, but torch itself is fine here

    inherited = os.environ.get("CUDA_VISIBLE_DEVICES")
    avail = inherited.split(",") if inherited else [str(i) for i in range(torch.cuda.device_count())]
    lo, hi = local_rank * tp, (local_rank + 1) * tp
    if hi > len(avail):
        raise RuntimeError(f"local_rank={local_rank} tp={tp} needs {hi} of {len(avail)} visible GPUs")
    return ",".join(avail[lo:hi])
