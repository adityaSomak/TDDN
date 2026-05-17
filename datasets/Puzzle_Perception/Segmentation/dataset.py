"""PyTorch ``Dataset`` for the unified puzzle-perception segmentation tree.

Layout (fetched by the top-level ``download_datasets.py`` from HuggingFace)::

    <root>/manifest.csv
    <root>/{train,val,test}/images/<task>_<id>.png
    <root>/{train,val,test}/masks/<task>_<id>.png

Mask PNGs are 8-bit grayscale with values in ``[0, 29]`` — already in the
unified 30-class label space (see ``classes.yaml``). No remapping happens at
load time.

Usage::

    from datasets.Puzzle_Perception.Segmentation.dataset import (
        PuzzleSegmentationDataset,
    )

    ds = PuzzleSegmentationDataset(root="path/to/data", split="train")
    image, mask, meta = ds[0]
    # image: float tensor [3, H, W], normalised to [0, 1]
    # mask:  long tensor [H, W], values in [0, 29]
    # meta:  {"unified_id": str, "source_task": str, "source_id": str}
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import Dataset


CLASSES_YAML = Path(__file__).resolve().parent / "classes.yaml"
SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class ClassInfo:
    """One row from ``classes.yaml``."""
    id: int
    name: str
    source_task: str
    weight_ce: float
    weight_miou: float


class Classes:
    """Loaded class metadata. Exposes lookup tables and weight tensors."""

    def __init__(self, yaml_path: Path = CLASSES_YAML):
        spec = yaml.safe_load(yaml_path.read_text())
        self.entries: list[ClassInfo] = [ClassInfo(**e) for e in spec["classes"]]
        self.label_offsets: dict[str, int] = dict(spec["label_offsets"])
        self.num_classes: int = len(self.entries)

    def __len__(self) -> int:
        return self.num_classes

    def names(self) -> list[str]:
        """Return class names indexed by unified id."""
        return [c.name for c in self.entries]

    def source_tasks(self) -> list[str]:
        """Return source task indexed by unified id."""
        return [c.source_task for c in self.entries]

    def ce_weights(self) -> torch.Tensor:
        """Per-class cross-entropy weights, indexed by unified id."""
        return torch.tensor([c.weight_ce for c in self.entries], dtype=torch.float32)

    def miou_weights(self) -> torch.Tensor:
        """Per-class mIoU weights, indexed by unified id."""
        return torch.tensor([c.weight_miou for c in self.entries], dtype=torch.float32)

    def ids_for_task(self, source_task: str) -> list[int]:
        """Return the unified-id list belonging to one source task."""
        return [c.id for c in self.entries if c.source_task == source_task]


class PuzzleSegmentationDataset(Dataset):
    """Image + mask pairs from the puzzle-perception segmentation tree.

    Args:
        root:             path to the ``data/`` directory containing
                          ``{train,val,test}/{images,masks}/``.
        split:            ``"train"``, ``"val"`` or ``"test"``.
        transform:        optional callable applied to the PIL image.
        target_transform: optional callable applied to the mask numpy array
                          before it is converted to a tensor.
    """

    def __init__(
        self,
        root: str | Path,
        split: str,
        transform=None,
        target_transform=None,
    ):
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
        self.root = Path(root).resolve()
        self.split = split
        self.transform = transform
        self.target_transform = target_transform

        images_dir = self.root / split / "images"
        masks_dir = self.root / split / "masks"
        if not images_dir.is_dir() or not masks_dir.is_dir():
            raise FileNotFoundError(
                f"expected {images_dir} and {masks_dir} — has the dataset been "
                f"downloaded yet? Run "
                f"`python download_datasets.py --dataset puzzle_perception`."
            )

        self.image_paths: list[Path] = sorted(images_dir.glob("*.png"))
        self.masks_dir = masks_dir
        self.classes = Classes()

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, dict]:
        img_path = self.image_paths[idx]
        unified_id = img_path.stem
        source_task, _, source_id = unified_id.partition("_")

        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        else:
            image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0

        mask_arr = np.array(Image.open(self.masks_dir / f"{unified_id}.png"))
        if self.target_transform is not None:
            mask_arr = self.target_transform(mask_arr)
        mask = torch.from_numpy(mask_arr).long()

        meta = {
            "unified_id": unified_id,
            "source_task": source_task,
            "source_id": source_id,
        }
        return image, mask, meta
