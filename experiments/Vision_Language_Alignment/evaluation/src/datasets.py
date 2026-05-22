"""Eval-dataset loaders.

Thin wrappers around ``torchvision.datasets.*`` (and one custom ADE20K
+ the repo's ``PuzzleSegmentationDataset``) that yield a uniform
``(image_tensor, target)`` interface per task.

Classification → ``(image, int_label)``.
Retrieval     → ``(image, list[str captions])``.
Segmentation  → ``(image, mask_long)``.

Public API
----------
    build_classification_dataset(name, root, transform) -> (Dataset, list[str])
    build_retrieval_dataset(name, root, transform)      -> Dataset
    build_segmentation_dataset(name, root, transform)   -> (Dataset, list[str], n_classes)

Each ``build_*`` returns the class-name list (or None for retrieval)
alongside the Dataset so the eval scripts don't import the per-dataset
prompt module separately.
"""
from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from shared_utils.paths import REPO_ROOT


# ---------------------------------------------------------------------------
# Helpers — load class lists from the local ``prompts/`` modules.
# ---------------------------------------------------------------------------

def _classes_for(dataset: str) -> list[str]:
    """Return ``*_CLASSES`` from ``evaluation/prompts/<dataset>.py``."""
    here = Path(__file__).resolve().parents[1]
    prompts_dir = here / "prompts"
    if str(prompts_dir) not in sys.path:
        sys.path.insert(0, str(prompts_dir))
    mod = importlib.import_module(dataset)
    # The convention is one ALL-CAPS list per module.
    for name in dir(mod):
        if name.isupper() and isinstance(getattr(mod, name), list):
            return list(getattr(mod, name))
    raise RuntimeError(f"no *_CLASSES list found in prompts/{dataset}.py")


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def build_classification_dataset(
    name: str, root: Path, transform: Callable,
) -> tuple[Dataset, list[str]]:
    """Build a classification eval Dataset + the class-name list."""
    if name == "cifar100":
        from torchvision.datasets import CIFAR100
        ds = CIFAR100(str(root), train=False, download=False, transform=transform)
        classes = _classes_for("cifar100")
    elif name == "caltech101":
        from torchvision.datasets import Caltech101
        ds = Caltech101(str(root), download=False, transform=transform)
        classes = _classes_for("caltech101")
    elif name == "food101":
        from torchvision.datasets import Food101
        ds = Food101(str(root), split="test", download=False, transform=transform)
        classes = _classes_for("food101")
    elif name == "gtsrb":
        from torchvision.datasets import GTSRB
        ds = GTSRB(str(root), split="test", download=False, transform=transform)
        classes = _classes_for("gtsrb")
    elif name == "imagenet1k":
        # The dataset module bundled alongside the data exposes the
        # validation split through HuggingFace ``datasets``.
        spec = importlib.util.spec_from_file_location(
            "imagenet_dataset_loader",
            REPO_ROOT / "datasets" / "Existing_Datasets" / "Classification"
            / "ImageNet-1K" / "dataset.py",
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules["imagenet_dataset_loader"] = module
        spec.loader.exec_module(module)
        ds = module.HFImageNet("validation", transform=transform)
        classes = _classes_for("imagenet1k")
    else:
        raise ValueError(f"Unknown classification dataset {name!r}")
    return ds, classes


# ---------------------------------------------------------------------------
# Retrieval — yields (image, list[caption])
# ---------------------------------------------------------------------------

class _Flickr30k(Dataset):
    """Flickr30K split — one image, 5 captions each.

    Accepts two common on-disk layouts:

    * ``results_20130124.token`` — tab-separated ``imageid.jpg#k<TAB>caption``
      lines, images under ``flickr30k-images/``.
    * ``results.csv`` — header row + ``image_name|comment_number|comment``
      pipe-delimited lines, images alongside the CSV.
    """

    def __init__(self, root: Path, transform: Callable):
        self.transform = transform
        token_path = root / "results_20130124.token"
        csv_candidates = [root / "results.csv", root / "images" / "results.csv"]
        csv_path = next((p for p in csv_candidates if p.is_file()), None)
        groups: dict[str, list[str]] = {}
        if token_path.is_file():
            self.image_dir = root / "flickr30k-images"
            for line in token_path.read_text().splitlines():
                head, cap = line.rstrip("\n").split("\t", 1)
                stem = head.split("#", 1)[0]
                groups.setdefault(stem, []).append(cap)
        elif csv_path is not None:
            self.image_dir = csv_path.parent
            for i, line in enumerate(csv_path.read_text().splitlines()):
                if i == 0 or not line.strip():
                    continue
                parts = [p.strip() for p in line.split("|", 2)]
                if len(parts) < 3:
                    continue
                stem, _, cap = parts
                groups.setdefault(stem, []).append(cap)
        else:
            raise FileNotFoundError(
                f"No Flickr30K captions found under {root} "
                "(expected results_20130124.token or results.csv)"
            )
        self.items = sorted(groups.items())

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        stem, caps = self.items[idx]
        img = Image.open(self.image_dir / stem).convert("RGB")
        return self.transform(img), caps


def build_retrieval_dataset(name: str, root: Path, transform: Callable) -> Dataset:
    """Build a retrieval Dataset yielding ``(image_tensor, list[caption])``."""
    if name == "flickr30k":
        return _Flickr30k(root, transform)
    if name == "coco":
        from torchvision.datasets import CocoCaptions
        # `root` here is .../MS-COCO-2014/val2014 (image dir);
        # the captions JSON lives one level up.
        ann_file = root.parent / "annotations" / "captions_val2014.json"
        return CocoCaptions(str(root), str(ann_file), transform=transform)
    raise ValueError(f"Unknown retrieval dataset {name!r}")


# ---------------------------------------------------------------------------
# Segmentation — yields (image, mask_long)
# ---------------------------------------------------------------------------

class _ADE20kVal(Dataset):
    """ADE20K SceneParseChallenge val split.

    Layout under ``<root>``::

        ADEChallengeData2016/images/validation/ADE_val_*.jpg
        ADEChallengeData2016/annotations/validation/ADE_val_*.png
    """

    def __init__(self, root: Path, transform: Callable, mask_size: int = 512):
        challenge = root / "ADEChallengeData2016"
        if not challenge.is_dir():
            challenge = root  # caller already pointed at the inner dir
        self.image_dir = challenge / "images" / "validation"
        self.mask_dir = challenge / "annotations" / "validation"
        self.transform = transform
        self.mask_size = mask_size
        self.stems = sorted(p.stem for p in self.image_dir.glob("*.jpg"))

    def __len__(self) -> int:
        return len(self.stems)

    def __getitem__(self, idx: int):
        stem = self.stems[idx]
        img = Image.open(self.image_dir / f"{stem}.jpg").convert("RGB")
        mask = Image.open(self.mask_dir / f"{stem}.png").resize(
            (self.mask_size, self.mask_size), Image.NEAREST,
        )
        # SceneParse150 masks use 1..150 with 0 = background/unknown;
        # remap to 0..149 and send 0 → 255 (the metric's ignore index).
        m = np.array(mask, dtype=np.int64)
        m = np.where(m == 0, 255, m - 1)
        return self.transform(img), torch.from_numpy(m)


def build_segmentation_dataset(
    name: str, root: Path, transform: Callable, mask_size: int = 512,
) -> tuple[Dataset, list[str], int]:
    """Build a segmentation Dataset + class list + n_classes."""
    if name == "ade20k":
        ds = _ADE20kVal(root, transform, mask_size=mask_size)
        classes = _classes_for("ade20k")
        return ds, classes, len(classes)
    if name == "puzzle":
        spec = importlib.util.spec_from_file_location(
            "puzzle_segmentation_dataset",
            REPO_ROOT / "datasets" / "Puzzle_Perception" / "Segmentation" / "dataset.py",
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules["puzzle_segmentation_dataset"] = module
        spec.loader.exec_module(module)
        ds = module.PuzzleSegmentationDataset(root, split="test", transform=transform)
        # Use natural-language prompts (better zero-shot text embeddings)
        # rather than the dataset's internal ``snake_case`` field names.
        classes = _classes_for("puzzle")
        assert len(classes) == ds.classes.num_classes, (
            f"puzzle prompt count ({len(classes)}) must match dataset class "
            f"count ({ds.classes.num_classes})"
        )
        return ds, classes, ds.classes.num_classes
    raise ValueError(f"Unknown segmentation dataset {name!r}")
