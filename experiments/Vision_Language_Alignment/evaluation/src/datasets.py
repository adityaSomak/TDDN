"""Eval-dataset loaders.

Wrappers that yield a uniform ``(image_tensor, target)`` interface per task.

Classification → ``(image, int_label)``.
Retrieval      → ``(image, list[str captions])``.
Segmentation   → ``(image, mask_long)`` at each image's own resolution.

Public API
----------
    build_classification_dataset(name, root, transform) -> (Dataset, list[str])
    build_retrieval_dataset(name, root, transform)      -> Dataset
    build_segmentation_dataset(name, root, transform)   -> (Dataset, list[str], n_classes)
    seg_target(name)                                    -> int

Each ``build_*`` returns the class-name list (or None for retrieval) alongside
the Dataset so the eval scripts don't import the per-dataset prompt module
separately.
"""
from __future__ import annotations

import ast
import csv
import importlib
import importlib.util
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .slide_inference import resize_shorter_side


# ---------------------------------------------------------------------------
# Class-name lists, resolved from the local ``prompts/`` modules.
# ---------------------------------------------------------------------------

# dataset key -> (prompts module, symbol). Explicit rather than inferred: the
# key and the module name differ for context59, and a heuristic scan would pick
# whichever list happened to sort first in a module exporting more than one.
_CLASS_SOURCES: dict[str, tuple[str, str]] = {
    "ade20k":            ("ade20k",            "ADE20K_CLASSES"),
    "caltech101":        ("caltech101",        "CALTECH101_CLASSES"),
    "cifar100":          ("cifar100",          "CIFAR100_CLASSES"),
    "cityscapes":        ("cityscapes",        "CITYSCAPES_CLASSES"),
    "coco_stuff":        ("coco_stuff",        "COCO_STUFF_CLASSES"),
    "context59":         ("pascal_context",    "PASCAL_CONTEXT_CLASSES"),
    "food101":           ("food101",           "FOOD101_CLASSES"),
    "gtsrb":             ("gtsrb",             "GTSRB_CLASSES"),
    "puzzle_perception": ("puzzle_perception", "PUZZLE_PERCEPTION_CLASSES"),
}


def _classes_for(dataset: str) -> list[str]:
    """Class-name list for ``dataset``, in label-id order."""
    try:
        module_name, symbol = _CLASS_SOURCES[dataset]
    except KeyError:
        raise ValueError(f"no class list registered for dataset {dataset!r}") from None
    prompts_dir = Path(__file__).resolve().parents[1] / "prompts"
    if str(prompts_dir) not in sys.path:
        sys.path.insert(0, str(prompts_dir))
    mod = importlib.import_module(module_name)
    try:
        return list(getattr(mod, symbol))
    except AttributeError:
        raise RuntimeError(f"prompts/{module_name}.py does not define {symbol}") from None


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
    elif name == "caltech101":
        from torchvision.datasets import Caltech101
        ds = Caltech101(str(root), download=False, transform=transform)
    elif name == "food101":
        from torchvision.datasets import Food101
        ds = Food101(str(root), split="test", download=False, transform=transform)
    elif name == "gtsrb":
        from torchvision.datasets import GTSRB
        ds = GTSRB(str(root), split="test", download=False, transform=transform)
    else:
        raise ValueError(f"Unknown classification dataset {name!r}")
    return ds, _classes_for(name)


# ---------------------------------------------------------------------------
# Retrieval — yields (image, list[caption])
# ---------------------------------------------------------------------------

class _FlickrKarpathyTest(Dataset):
    """Flickr30K, 1000-image Karpathy test split.

    Reads ``flickr_annotations_30k.csv`` (columns: raw, sentids, split,
    filename, img_id), where ``raw`` is a Python-literal list of 5 captions.
    Restricting to the standard test split is what makes Recall@K comparable
    to published numbers: scoring against all ~31k images inflates the
    candidate set 31x and collapses Recall@1.
    """

    def __init__(self, root: Path, transform: Callable, split: str = "test"):
        self.image_dir = root / "flickr30k-images"
        self.transform = transform
        csv_path = root / "flickr_annotations_30k.csv"
        self.items: list[tuple[str, list[str]]] = []
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                if row["split"] != split:
                    continue
                caps = ast.literal_eval(row["raw"])
                self.items.append((row["filename"], [str(c) for c in caps]))
        self.items.sort()
        if not self.items:
            raise RuntimeError(f"no rows with split={split!r} in {csv_path}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        fname, caps = self.items[idx]
        img = Image.open(self.image_dir / fname).convert("RGB")
        return self.transform(img), caps


# Candidate set each retrieval dataset is scored against. Recall@K depends
# directly on this, so it is recorded with every result.
#
# flickr30k  the 1,000-image Karpathy test split — the published protocol.
# coco       all 40,504 val2014 images, NOT the 5,000-image Karpathy test split.
#            The larger pool lowers Recall@K, so these numbers rank the models
#            against each other but are not comparable to figures reported on
#            the 5k split.
_RETRIEVAL_PROTOCOL = {
    "flickr30k": "karpathy_test_1k",
    "coco": "val2014_all_40k",
}


def retrieval_protocol(name: str) -> str:
    """Candidate-set identifier for ``name``."""
    try:
        return _RETRIEVAL_PROTOCOL[name]
    except KeyError:
        raise ValueError(f"Unknown retrieval dataset {name!r}") from None


def build_retrieval_dataset(name: str, root: Path, transform: Callable) -> Dataset:
    """Build a retrieval Dataset yielding ``(image_tensor, list[caption])``."""
    if name == "flickr30k":
        return _FlickrKarpathyTest(root, transform)
    if name == "coco":
        from torchvision.datasets import CocoCaptions
        # ``root`` is .../MS-COCO-2014/val2014 (image dir); captions live one level up.
        ann_file = root.parent / "annotations" / "captions_val2014.json"
        return CocoCaptions(str(root), str(ann_file), transform=transform)
    raise ValueError(f"Unknown retrieval dataset {name!r}")


# ---------------------------------------------------------------------------
# Segmentation — yields (image, mask_long) at native (post-resize) resolution
# ---------------------------------------------------------------------------

class ADE20kNativeVal(Dataset):
    """ADE20K SceneParse150 val split (2,000 images).

    Layout under ``<root>``::

        ADEChallengeData2016/images/validation/ADE_val_*.jpg
        ADEChallengeData2016/annotations/validation/ADE_val_*.png
    """

    def __init__(self, root: Path, target: int = 448, max_long_side: int = 2048,
                 transform: Callable | None = None, split: str = "validation"):
        challenge = root / "ADEChallengeData2016"
        if not challenge.is_dir():
            challenge = root
        self.image_dir = challenge / "images" / split
        self.mask_dir = challenge / "annotations" / split
        self.target = target
        self.max_long_side = max_long_side
        self.transform = transform
        self.stems = sorted(p.stem for p in self.image_dir.glob("*.jpg"))

    def __len__(self) -> int:
        return len(self.stems)

    def __getitem__(self, idx: int):
        stem = self.stems[idx]
        img = Image.open(self.image_dir / f"{stem}.jpg").convert("RGB")
        mask = Image.open(self.mask_dir / f"{stem}.png")

        img_r = resize_shorter_side(img, self.target, self.max_long_side)
        mask_r = mask.resize(img_r.size, Image.NEAREST)

        # SceneParse150 masks use 1..150 with 0 = background/unknown; remap to
        # 0..149 and send 0 -> 255 (the metric's ignore index).
        m = np.array(mask_r, dtype=np.int64)
        m = np.where(m == 0, 255, m - 1)

        image_t = self.transform(img_r) if self.transform is not None else img_r
        return image_t, torch.from_numpy(m)


# Cityscapes raw ``labelIds`` (0..33) -> the 19 evaluation "trainId" classes,
# verbatim from cityscapesScripts' ``cityscapesscripts/helpers/labels.py``.
# Anything absent from this table falls to 255 via the ``np.full`` default.
_CITYSCAPES_ID_TO_TRAINID = {
    0: 255, 1: 255, 2: 255, 3: 255, 4: 255, 5: 255, 6: 255, 7: 0, 8: 1,
    9: 255, 10: 255, 11: 2, 12: 3, 13: 4, 14: 255, 15: 255, 16: 255, 17: 5,
    18: 255, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13,
    27: 14, 28: 15, 29: 255, 30: 255, 31: 16, 32: 17, 33: 18,
}
_CITYSCAPES_LUT = np.full(256, 255, dtype=np.int64)
for _id, _train_id in _CITYSCAPES_ID_TO_TRAINID.items():
    _CITYSCAPES_LUT[_id] = _train_id


class CityscapesNativeVal(Dataset):
    """Cityscapes val split (500 images, 19 classes).

    Layout under ``<root>``::

        leftImg8bit/val/<city>/<stem>_leftImg8bit.png
        gtFine/val/<city>/<stem>_gtFine_labelIds.png
    """

    def __init__(self, root: Path, target: int = 448, max_long_side: int = 2048,
                 transform: Callable | None = None, split: str = "val"):
        self.image_dir = root / "leftImg8bit" / split
        self.mask_dir = root / "gtFine" / split
        self.target = target
        self.max_long_side = max_long_side
        self.transform = transform
        self.stems = sorted(
            p.name[: -len("_leftImg8bit.png")]
            for p in self.image_dir.glob("*/*_leftImg8bit.png")
        )

    def __len__(self) -> int:
        return len(self.stems)

    def __getitem__(self, idx: int):
        stem = self.stems[idx]
        city = stem.split("_")[0]
        img = Image.open(self.image_dir / city / f"{stem}_leftImg8bit.png").convert("RGB")
        mask = Image.open(self.mask_dir / city / f"{stem}_gtFine_labelIds.png")

        img_r = resize_shorter_side(img, self.target, self.max_long_side)
        mask_r = mask.resize(img_r.size, Image.NEAREST)
        m = _CITYSCAPES_LUT[np.array(mask_r, dtype=np.int64)]

        image_t = self.transform(img_r) if self.transform is not None else img_r
        return image_t, torch.from_numpy(m)


# COCO-Stuff raw ``stuffthingmaps`` category ids (0..181, 255=ignore) -> the 171
# contiguous trainId classes, verbatim from mmsegmentation's
# ``tools/dataset_converters/coco_stuff164k.py`` (``clsID_to_trID``).
_COCO_STUFF_CLSID_TO_TRID = {
    0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9,
    10: 10, 12: 11, 13: 12, 14: 13, 15: 14, 16: 15, 17: 16, 18: 17, 19: 18, 20: 19,
    21: 20, 22: 21, 23: 22, 24: 23, 26: 24, 27: 25, 30: 26, 31: 27, 32: 28, 33: 29,
    34: 30, 35: 31, 36: 32, 37: 33, 38: 34, 39: 35, 40: 36, 41: 37, 42: 38, 43: 39,
    45: 40, 46: 41, 47: 42, 48: 43, 49: 44, 50: 45, 51: 46, 52: 47, 53: 48, 54: 49,
    55: 50, 56: 51, 57: 52, 58: 53, 59: 54, 60: 55, 61: 56, 62: 57, 63: 58, 64: 59,
    66: 60, 69: 61, 71: 62, 72: 63, 73: 64, 74: 65, 75: 66, 76: 67, 77: 68, 78: 69,
    79: 70, 80: 71, 81: 72, 83: 73, 84: 74, 85: 75, 86: 76, 87: 77, 88: 78, 89: 79,
    91: 80, 92: 81, 93: 82, 94: 83, 95: 84, 96: 85, 97: 86, 98: 87, 99: 88, 100: 89,
    101: 90, 102: 91, 103: 92, 104: 93, 105: 94, 106: 95, 107: 96, 108: 97, 109: 98, 110: 99,
    111: 100, 112: 101, 113: 102, 114: 103, 115: 104, 116: 105, 117: 106, 118: 107, 119: 108, 120: 109,
    121: 110, 122: 111, 123: 112, 124: 113, 125: 114, 126: 115, 127: 116, 128: 117, 129: 118, 130: 119,
    131: 120, 132: 121, 133: 122, 134: 123, 135: 124, 136: 125, 137: 126, 138: 127, 139: 128, 140: 129,
    141: 130, 142: 131, 143: 132, 144: 133, 145: 134, 146: 135, 147: 136, 148: 137, 149: 138, 150: 139,
    151: 140, 152: 141, 153: 142, 154: 143, 155: 144, 156: 145, 157: 146, 158: 147, 159: 148, 160: 149,
    161: 150, 162: 151, 163: 152, 164: 153, 165: 154, 166: 155, 167: 156, 168: 157, 169: 158, 170: 159,
    171: 160, 172: 161, 173: 162, 174: 163, 175: 164, 176: 165, 177: 166, 178: 167, 179: 168, 180: 169,
    181: 170, 255: 255,
}
_COCO_STUFF_LUT = np.full(256, 255, dtype=np.int64)
for _cls_id, _train_id in _COCO_STUFF_CLSID_TO_TRID.items():
    _COCO_STUFF_LUT[_cls_id] = _train_id


class COCOStuffNativeVal(Dataset):
    """COCO-Stuff val2017 split (5,000 images, 171 classes).

    Layout under ``<root>``::

        images/val2017/*.jpg
        annotations/val2017/*.png     (raw stuffthingmaps ids)
    """

    def __init__(self, root: Path, target: int = 448, max_long_side: int = 2048,
                 transform: Callable | None = None, split: str = "val2017"):
        self.image_dir = root / "images" / split
        self.mask_dir = root / "annotations" / split
        self.target = target
        self.max_long_side = max_long_side
        self.transform = transform
        self.stems = sorted(p.stem for p in self.image_dir.glob("*.jpg"))

    def __len__(self) -> int:
        return len(self.stems)

    def __getitem__(self, idx: int):
        stem = self.stems[idx]
        img = Image.open(self.image_dir / f"{stem}.jpg").convert("RGB")
        mask = Image.open(self.mask_dir / f"{stem}.png")

        img_r = resize_shorter_side(img, self.target, self.max_long_side)
        mask_r = mask.resize(img_r.size, Image.NEAREST)
        m = _COCO_STUFF_LUT[np.array(mask_r, dtype=np.int64)]

        image_t = self.transform(img_r) if self.transform is not None else img_r
        return image_t, torch.from_numpy(m)


class PascalContext59NativeVal(Dataset):
    """PASCAL-Context val split (5,104 images, 59 classes).

    Layout under ``<root>``::

        VOCdevkit/VOC2010/JPEGImages/*.jpg
        VOCdevkit/VOC2010/SegmentationClassContext/*.png
        VOCdevkit/VOC2010/ImageSets/SegmentationContext/val.txt

    Masks are pre-converted with 0 = background and 1..59 the real classes;
    background is dropped from the classifier, so 0 -> ignore and 1..59 shift
    down to 0..58.
    """

    def __init__(self, root: Path, target: int = 448, max_long_side: int = 2048,
                 transform: Callable | None = None, split: str = "val"):
        devkit = root / "VOCdevkit" / "VOC2010"
        if not devkit.is_dir():
            devkit = root
        self.image_dir = devkit / "JPEGImages"
        self.mask_dir = devkit / "SegmentationClassContext"
        self.target = target
        self.max_long_side = max_long_side
        self.transform = transform
        split_file = devkit / "ImageSets" / "SegmentationContext" / f"{split}.txt"
        self.stems = [ln.strip() for ln in split_file.read_text().splitlines() if ln.strip()]

    def __len__(self) -> int:
        return len(self.stems)

    def __getitem__(self, idx: int):
        stem = self.stems[idx]
        img = Image.open(self.image_dir / f"{stem}.jpg").convert("RGB")
        mask = Image.open(self.mask_dir / f"{stem}.png")

        img_r = resize_shorter_side(img, self.target, self.max_long_side)
        mask_r = mask.resize(img_r.size, Image.NEAREST)

        m = np.array(mask_r, dtype=np.int64)
        new_m = m - 1
        new_m[m == 0] = 255

        image_t = self.transform(img_r) if self.transform is not None else img_r
        return image_t, torch.from_numpy(new_m)


# Unified 30-id puzzle mask -> the 11 scored class ids. Composed from the
# per-source remap arrays and the classes.yaml id offsets (maze 0, chess +8,
# hanoi +23) rather than transcribed, so the two stages stay verifiable.
_MAZE_REMAP = [0, 1, 1, 2, 3, 4, 5, 6]
_CHESS_REMAP = [7, 8, 9, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11]
_HANOI_REMAP = [12, 13, 14, 14, 14, 14, 14]
_STAGE1 = np.full(256, 255, dtype=np.int64)
for _uid in range(8):
    _STAGE1[_uid] = _MAZE_REMAP[_uid]
for _uid in range(8, 23):
    _STAGE1[_uid] = _CHESS_REMAP[_uid - 8]
for _uid in range(23, 30):
    _STAGE1[_uid] = _HANOI_REMAP[_uid - 23]

_STAGE2 = [0, 1, 2, 2, 2, 2, 2, 3, 4, 5, 6, 7, 8, 9, 10]
_PUZZLE_LUT = np.full(256, 255, dtype=np.int64)
for _uid in range(30):
    _s1 = int(_STAGE1[_uid])
    _PUZZLE_LUT[_uid] = _STAGE2[_s1] if _s1 != 255 else 255


class PuzzlePerceptionNativeVal(Dataset):
    """Puzzle-Perception test split (1,500 images, 11 classes).

    Layout under ``<root>``::

        <split>/images/<task>_<id>.png
        <split>/masks/<task>_<id>.png

    Wraps the dataset's own ``PuzzleSegmentationDataset`` for split validation
    and path resolution, then loads images directly so the aspect-preserving
    resize this protocol needs is applied.
    """

    def __init__(self, root: Path, target: int = 336, max_long_side: int = 2048,
                 transform: Callable | None = None, split: str = "test"):
        module_path = Path(root).parent / "dataset.py"
        spec = importlib.util.spec_from_file_location(
            "puzzle_segmentation_dataset", module_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules["puzzle_segmentation_dataset"] = module
        spec.loader.exec_module(module)
        base = module.PuzzleSegmentationDataset(root=root, split=split)

        self.image_paths = base.image_paths
        self.masks_dir = base.masks_dir
        self.target = target
        self.max_long_side = max_long_side
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(self.masks_dir / img_path.name)

        img_r = resize_shorter_side(img, self.target, self.max_long_side)
        mask_r = mask.resize(img_r.size, Image.NEAREST)
        m = _PUZZLE_LUT[np.array(mask_r, dtype=np.int64)]

        image_t = self.transform(img_r) if self.transform is not None else img_r
        return image_t, torch.from_numpy(m)


# dataset key -> (Dataset class, shorter-side resize target). The target also
# sets the sliding-window crop size; the stride is half of it.
_SEG_SPECS: dict[str, tuple[type[Dataset], int]] = {
    "ade20k":            (ADE20kNativeVal,           448),
    "cityscapes":        (CityscapesNativeVal,       448),
    "coco_stuff":        (COCOStuffNativeVal,        448),
    "context59":         (PascalContext59NativeVal,  448),
    "puzzle_perception": (PuzzlePerceptionNativeVal, 336),
}


def seg_target(name: str) -> int:
    """Shorter-side resize target (and sliding-window crop size) for ``name``."""
    try:
        return _SEG_SPECS[name][1]
    except KeyError:
        raise ValueError(f"Unknown segmentation dataset {name!r}") from None


def build_segmentation_dataset(
    name: str, root: Path, transform: Callable, target: int | None = None,
) -> tuple[Dataset, list[str], int]:
    """Build a segmentation Dataset + class list + n_classes."""
    try:
        cls, default_target = _SEG_SPECS[name]
    except KeyError:
        raise ValueError(f"Unknown segmentation dataset {name!r}") from None
    ds = cls(root, target=target or default_target, transform=transform)
    classes = _classes_for(name)
    return ds, classes, len(classes)
