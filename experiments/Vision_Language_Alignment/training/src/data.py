"""COCO captions + LAION image-folder loaders.

Two data sources mixed via ``ConcatDataset`` at the natural ratio
(~14% COCO / 86% LAION at the round-1 sample counts):

  - **COCO 2014 train captions** — 82,783 images × ~5 captions each.
    Each ``__getitem__`` samples one of the captions uniformly at
    random so the same image pairs with different positives across
    epochs.
  - **LAION-style folder** — flat (image, caption) pairs read from
    ``image_dir`` + ``metadata.csv``.

Round 2 (COCO fine-tune) sets ``laion_shards: ""``; the builder
short-circuits to COCO-only in that case.

Public API
----------
    build_train_dataset(coco_root, coco_ann_file, laion_shards, transform)
        Returns a torch ``Dataset`` yielding ``(image_tensor, caption_str)``.
"""
from __future__ import annotations

import csv
import math
import random
from pathlib import Path
from typing import Callable, Iterator, Optional

import torch
import torch.distributed as dist
from PIL import Image
from torch.utils.data import ConcatDataset, Dataset, Sampler


class LaionFolder(Dataset):
    """LAION image folder + flat captions CSV (``metadata.csv``)."""

    def __init__(self, image_dir: Path, csv_path: Path, transform: Callable):
        self.image_dir = Path(image_dir)
        self.transform = transform
        with open(csv_path) as f:
            self.rows = list(csv.DictReader(f))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        path = self.image_dir / row["filename"]
        img = Image.open(path).convert("RGB")
        return self.transform(img), row.get("caption", "")


class CocoCaptions(Dataset):
    """Wraps pycocotools to provide ``(image, caption)`` pairs.

    Each call picks one caption uniformly at random from the ~5
    annotations attached to the image; this gives the contrastive
    objective different positives across epochs.
    """

    def __init__(self, root: str | Path, ann_file: str | Path,
                 transform: Optional[Callable] = None):
        from pycocotools.coco import COCO

        self.root = Path(root)
        self.coco = COCO(str(ann_file))
        self.ids = list(self.coco.imgs.keys())
        self.transform = transform

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int):
        img_id = self.ids[idx]
        img_info = self.coco.imgs[img_id]
        image = Image.open(self.root / img_info["file_name"]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        caption = random.choice(anns)["caption"] if anns else ""
        return image, caption


class InfiniteShardedSampler(Sampler):
    """Infinite, shuffled, rank-sharded sampler for map-style datasets.

    Cycles through a deterministic permutation of the dataset indices,
    re-shuffling with a new RNG seed at the end of each epoch. Indices
    are sharded so disjoint subsets land on each distributed rank.
    """

    def __init__(self, dataset: Dataset, seed: int = 0, shuffle: bool = True):
        self.dataset = dataset
        self.seed = seed
        self.shuffle = shuffle
        self.epoch = 0

    def __iter__(self) -> Iterator[int]:
        rank = dist.get_rank() if dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        n = len(self.dataset)
        while True:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            if self.shuffle:
                indices = torch.randperm(n, generator=g).tolist()
            else:
                indices = list(range(n))
            # Pad to a multiple of world_size so every rank gets the same count.
            padding = math.ceil(n / world_size) * world_size - n
            indices += indices[:padding]
            per_rank = len(indices) // world_size
            start = rank * per_rank
            yield from indices[start : start + per_rank]
            self.epoch += 1

    def __len__(self) -> int:
        n = len(self.dataset)
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        return math.ceil(n / world_size)


def build_train_dataset(
    coco_root: Path,
    coco_ann_file: Path,
    laion_shards: Optional[Path],
    transform: Callable,
) -> Dataset:
    """Compose the training dataset from COCO + optional LAION."""
    coco = CocoCaptions(coco_root, coco_ann_file, transform=transform)
    if not laion_shards:
        return coco
    laion = LaionFolder(
        image_dir=Path(laion_shards) / "images",
        csv_path=Path(laion_shards) / "metadata.csv",
        transform=transform,
    )
    return ConcatDataset([laion, coco])
