"""Dataset wrapper that resizes masks to match the head's output grid.

Wraps the puzzle-perception segmentation loader at
``datasets/Puzzle_Perception/Segmentation/dataset.py`` and additionally
nearest-resizes the mask to ``(mask_size, mask_size)`` so it lines up
with the head's bilinearly-upsampled logits at loss time.

Public API
----------
    PuzzleSegmentationFeatures(root, split, transform, mask_size=512)
        PyTorch Dataset yielding ``(image_tensor, mask_long, meta)``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from shared_utils.paths import REPO_ROOT


def _load_puzzle_seg_module():
    """Import the puzzle-perception loader by file path.

    The local ``datasets/`` directory shares a top-level name with the
    HuggingFace ``datasets`` library, so we load this module explicitly
    instead of putting the repo root on ``sys.path``. The loaded module
    is cached in ``sys.modules`` so dataclasses and pickling can find
    it later.
    """
    import importlib.util
    import sys

    module_name = "puzzle_segmentation_dataset"
    if module_name in sys.modules:
        return sys.modules[module_name]
    path = REPO_ROOT / "datasets" / "Puzzle_Perception" / "Segmentation" / "dataset.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


PuzzleSegmentationDataset = _load_puzzle_seg_module().PuzzleSegmentationDataset


def _resize_mask(arr: np.ndarray, size: int) -> np.ndarray:
    """Nearest-neighbour resize a label mask to ``(size, size)`` integers."""
    img = Image.fromarray(arr.astype(np.uint8), mode="L")
    return np.array(img.resize((size, size), Image.NEAREST), dtype=np.int64)


class PuzzleSegmentationFeatures(Dataset):
    """Puzzle-perception segmentation dataset with mask resizing.

    Args:
        root:      path to the segmentation tree (``{train,val,test}/{images,masks}/``).
        split:     ``train``, ``val`` or ``test``.
        transform: callable applied to each PIL image.
        mask_size: target side length for the resized mask (must match
                   the head's output size).
    """

    def __init__(
        self,
        root: Path,
        split: str,
        transform: Callable,
        mask_size: int = 512,
    ):
        self._inner = PuzzleSegmentationDataset(root=root, split=split, transform=transform)
        self.mask_size = mask_size

    def __len__(self) -> int:
        return len(self._inner)

    def __getitem__(self, idx: int):
        image, mask, meta = self._inner[idx]
        if mask.shape[-1] != self.mask_size:
            mask = torch.from_numpy(_resize_mask(mask.numpy(), self.mask_size))
        return image, mask, meta

    @property
    def classes(self):
        """Expose the underlying class metadata (weights, names, ...)."""
        return self._inner.classes
