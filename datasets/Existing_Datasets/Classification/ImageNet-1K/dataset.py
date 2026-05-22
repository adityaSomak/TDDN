"""ImageNet-1K wrapper with per-class balanced sampling.

Loads an ImageNet-1K split from a local HuggingFace Arrow cache (path
read from the ``IMAGENET_HF_CACHE`` env var, with a sensible default
under ``datasets/Existing_Datasets/Classification/ImageNet-1K/``).

Per-class balanced sampling is the standard k-NN gallery recipe — pass
``per_class=100`` to get a 100K-image (100 per class) balanced subset.

Public API
----------
    HFImageNet(split, transform, *, per_class=None, subset=None, seed=42)
        PyTorch Dataset yielding ``(image_tensor, label, idx)``.
"""
from __future__ import annotations

import os
import random
from typing import Callable, Optional

from datasets import load_dataset
from torch.utils.data import Dataset

from shared_utils.paths import DATASETS_ROOT


DEFAULT_CACHE = (
    DATASETS_ROOT / "Existing_Datasets" / "Classification" / "ImageNet-1K" / "imagenet_hf"
)


def _cache_dir() -> str:
    """Resolve the HuggingFace dataset cache directory."""
    return os.environ.get("IMAGENET_HF_CACHE", str(DEFAULT_CACHE))


class HFImageNet(Dataset):
    """ImageNet-1K split with optional balanced or truncated sampling.

    Args:
        split:     ``"train"`` or ``"validation"``.
        transform: callable applied to each PIL image.
        per_class: keep exactly this many images per class (balanced).
                   Mutually exclusive with ``subset``.
        subset:    truncate the split to its first N examples (smoke tests).
        seed:      seed for the per-class sampler.
    """

    def __init__(
        self,
        split: str,
        transform: Callable,
        *,
        per_class: Optional[int] = None,
        subset: Optional[int] = None,
        seed: int = 42,
    ):
        if split not in ("train", "validation"):
            raise ValueError(f"split must be 'train' or 'validation'; got {split!r}")
        if per_class is not None and subset is not None:
            raise ValueError("`per_class` and `subset` are mutually exclusive.")

        self.ds = load_dataset(
            "ILSVRC/imagenet-1k", cache_dir=_cache_dir(), split=split
        )
        if per_class is not None:
            self.ds = self.ds.select(self._balanced_indices(self.ds["label"], per_class, seed))
        elif subset is not None:
            self.ds = self.ds.select(range(min(subset, len(self.ds))))

        self.transform = transform

    @staticmethod
    def _balanced_indices(labels: list[int], per_class: int, seed: int) -> list[int]:
        """Return up to ``per_class`` shuffled indices for every label."""
        by_class: dict[int, list[int]] = {}
        for idx, lbl in enumerate(labels):
            by_class.setdefault(int(lbl), []).append(idx)
        rng = random.Random(seed)
        chosen: list[int] = []
        for cls in sorted(by_class):
            pool = by_class[cls]
            rng.shuffle(pool)
            chosen.extend(pool[:per_class])
        chosen.sort()
        return chosen

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int):
        ex = self.ds[idx]
        img = ex["image"]
        if img.mode != "RGB":
            img = img.convert("RGB")
        return self.transform(img), int(ex["label"]), idx
