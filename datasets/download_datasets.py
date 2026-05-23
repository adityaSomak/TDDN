#!/usr/bin/env python3
"""Download every dataset used by this project, into its canonical location.

Public benchmarks are fetched from their original upstream source
(torchvision mirrors, HuggingFace Hub, or the canonical project URL).
Project-hosted datasets (LAION recaptions, Puzzle-Perception segmentation)
are fetched from the ``PuzzleBench`` organisation on HuggingFace.

Supported datasets
------------------

    cifar100              -> Existing_Datasets/Classification/CIFAR-100/
    caltech101            -> Existing_Datasets/Classification/Caltech-101/
    food101               -> Existing_Datasets/Classification/Food-101/
    gtsrb                 -> Existing_Datasets/Classification/GTSRB/
    imagenet              -> Existing_Datasets/Classification/ImageNet-1K/   [gated]
    spair                 -> Existing_Datasets/Keypoint_Matching/SPair-71K/
    flickr30k             -> Existing_Datasets/Retrieval/Flickr30K/
    ade20k                -> Existing_Datasets/Segmentation/ADE20K/
    coco                  -> Existing_Datasets/Vision_Language_Alignment/MS-COCO-2014/
    recaptioned_laion     -> Existing_Datasets/Vision_Language_Alignment/Recaptioned_LAION/data/
    puzzle_perception     -> Puzzle_Perception/Segmentation/data/

Usage
-----

    # Download a single dataset
    python download_datasets.py --dataset cifar100

    # Download every dataset listed above
    python download_datasets.py --all

    # Overwrite an already-populated target
    python download_datasets.py --dataset cifar100 --force

Gated datasets (ImageNet-1K, sometimes Flickr30K) require a HuggingFace
access token in the ``HF_TOKEN`` environment variable.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent


# ---------- helpers -----------------------------------------------------------

def _log(tag: str, msg: str) -> None:
    print(f"[{tag}] {msg}", flush=True)


def _is_populated(p: Path) -> bool:
    if not p.exists():
        return False
    return any(c.name != ".gitkeep" for c in p.iterdir())


def _reset_target(target: Path) -> None:
    if target.exists():
        if target.is_file() or target.is_symlink():
            target.unlink()
        else:
            shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.mkdir(parents=True, exist_ok=True)


def _run(cmd: list[str]) -> None:
    _log("run", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _curl_extract(target: Path, url: str, archive_name: str, kind: str) -> None:
    """Download via ``curl`` then extract with Python stdlib (no
    ``unzip`` system dependency).
    """
    archive = target / archive_name
    _run(["curl", "-L", "--fail", "-o", str(archive), url])
    if kind == "tar.gz":
        with tarfile.open(archive, "r:gz") as tf:
            tf.extractall(target)
    elif kind == "zip":
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(target)
    else:
        raise ValueError(f"unknown archive kind: {kind}")
    archive.unlink()


# ---------- downloaders -------------------------------------------------------

def _dl_torchvision(target: Path, cls_name: str, kwargs_list: list[dict]) -> None:
    try:
        import torchvision.datasets as tvd  # type: ignore
    except ImportError:
        sys.exit(f"{cls_name}: pip install torchvision")
    cls = getattr(tvd, cls_name)
    for kw in kwargs_list:
        _log("dl", f"{cls_name}({kw})")
        cls(root=str(target), download=True, **kw)


def _dl_cifar100(target: Path) -> None:
    _dl_torchvision(target, "CIFAR100", [{"train": True}, {"train": False}])


def _dl_caltech101(target: Path) -> None:
    _dl_torchvision(target, "Caltech101", [{"target_type": "category"}])


def _dl_food101(target: Path) -> None:
    _dl_torchvision(target, "Food101", [{"split": "train"}, {"split": "test"}])


def _dl_gtsrb(target: Path) -> None:
    _dl_torchvision(target, "GTSRB", [{"split": "train"}, {"split": "test"}])


def _dl_imagenet(target: Path) -> None:
    """Cache ImageNet-1K to ``<target>/imagenet_hf/`` so the consumer
    ``HFImageNet`` (in ``datasets/Existing_Datasets/Classification/ImageNet-1K/dataset.py``)
    finds it at its default location.
    """
    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit(
            "imagenet: ILSVRC/imagenet-1k is gated. Set HF_TOKEN in the "
            "environment after requesting access at "
            "https://huggingface.co/datasets/ILSVRC/imagenet-1k"
        )
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        sys.exit("imagenet: pip install datasets")
    cache_dir = target / "imagenet_hf"
    cache_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "validation"):
        _log("dl", f"HF load_dataset ILSVRC/imagenet-1k split={split}")
        load_dataset(
            "ILSVRC/imagenet-1k",
            split=split,
            cache_dir=str(cache_dir),
            token=token,
        )


def _dl_spair(target: Path) -> None:
    _curl_extract(
        target,
        "https://cvlab.postech.ac.kr/research/SPair-71k/data/SPair-71k.tar.gz",
        "SPair-71k.tar.gz",
        "tar.gz",
    )


def _dl_flickr30k(target: Path) -> None:
    """Fetch the Flickr30K bundle (4.4 GB zip + 30K-row CSV) from
    ``nlphuji/flickr30k``, extract the images into ``flickr30k-images/``,
    and rewrite the annotations into the legacy ``results_20130124.token``
    format the VLA retrieval loader expects.

    Final on-disk layout::

        <target>/
        ├── flickr30k-images/<id>.jpg     # 31,783 flat JPGs
        ├── results_20130124.token        # <id>.jpg#k<TAB><caption>
        ├── flickr_annotations_30k.csv    # original HF annotations (kept for reference)
        └── README.md
    """
    import json

    _hf_snapshot(target, "nlphuji/flickr30k")

    # Extract the image zip into ``flickr30k-images/`` (zip's internal layout).
    zip_path = target / "flickr30k-images.zip"
    if zip_path.is_file():
        print(f"[extract] {zip_path.name}")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(target)
        zip_path.unlink()

    # Convert the HF CSV (one row per image, ``raw`` column with a
    # JSON-encoded 5-caption list) into the legacy token format the
    # loader keys off of (``<image>#k<TAB><caption>`` per row, 5 rows
    # per image).
    src_csv = target / "flickr_annotations_30k.csv"
    token_path = target / "results_20130124.token"
    if src_csv.is_file() and not token_path.is_file():
        with open(src_csv, newline="") as fi, open(token_path, "w") as fo:
            for row in csv.DictReader(fi):
                for i, cap in enumerate(json.loads(row["raw"])):
                    cap = cap.replace("\t", " ").replace("\n", " ")
                    fo.write(f"{row['filename']}#{i}\t{cap}\n")
        print(f"[token] wrote {token_path}")

    # Skip the unused HF loader script.
    py_loader = target / "flickr30k.py"
    if py_loader.is_file():
        py_loader.unlink()


def _dl_ade20k(target: Path) -> None:
    _curl_extract(
        target,
        "http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip",
        "ADEChallengeData2016.zip",
        "zip",
    )


def _dl_coco(target: Path) -> None:
    """Download COCO 2014 (train + val images and caption / instance annotations)."""
    base = "http://images.cocodataset.org"
    parts: list[tuple[str, str]] = [
        ("zips/train2014.zip",                   "train2014.zip"),
        ("zips/val2014.zip",                     "val2014.zip"),
        ("annotations/annotations_trainval2014.zip", "annotations_trainval2014.zip"),
    ]
    for rel, name in parts:
        _curl_extract(target, f"{base}/{rel}", name, "zip")


def _hf_snapshot(target: Path, repo_id: str) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        sys.exit(f"{repo_id}: pip install huggingface-hub")
    _log("dl", f"{repo_id} -> {target}")
    snapshot_download(repo_id=repo_id, repo_type="dataset", local_dir=str(target))


def _dl_recaptioned_laion(target: Path) -> None:
    """Fetch the Recaptioned LAION webdataset shards from HuggingFace,
    extract them into a flat ``images/`` directory, and rewrite
    ``metadata.csv`` so the text-alignment training loader
    (``LaionFolder``) can consume it directly.

    On disk after this finishes::

        <target>/
        ├── metadata.csv     # columns: filename, image_id, url, caption
        ├── images/<id>.jpg  # 508,039 flat JPGs (per ``image_id``)
        └── README.md

    Each shard is extracted then deleted before the next is opened, so
    peak disk usage stays roughly equal to the 72 GB snapshot rather
    than 2× while extracting. The per-image ``.txt`` files inside the
    shards are skipped (captions live in ``metadata.csv``).
    """
    _hf_snapshot(target, "PuzzleBench/Recaptioned_LAION")

    images_dir = target / "images"
    images_dir.mkdir(exist_ok=True)

    shards = sorted(target.glob("shards-*.tar"))
    for tar_path in shards:
        print(f"[extract] {tar_path.name}")
        with tarfile.open(tar_path) as tf:
            for member in tf.getmembers():
                if member.isfile() and member.name.endswith(".jpg"):
                    tf.extract(member, images_dir)
        tar_path.unlink()

    # Rewrite metadata.csv with a leading ``filename`` column so
    # LaionFolder can look up images by ``row["filename"]``. Existing
    # columns (image_id, url, caption) are preserved for traceability.
    csv_path = target / "metadata.csv"
    if csv_path.exists():
        with open(csv_path, newline="") as fi:
            rows = list(csv.DictReader(fi))
        with open(csv_path, "w", newline="") as fo:
            writer = csv.DictWriter(
                fo, fieldnames=["filename"] + list(rows[0].keys()),
            )
            writer.writeheader()
            for r in rows:
                writer.writerow({"filename": f"{r['image_id']}.jpg", **r})
        print(f"[csv] rewrote {csv_path} ({len(rows)} rows) with leading filename column")


def _dl_puzzle_perception(target: Path) -> None:
    """Fetch the Puzzle-Perception segmentation dataset and extract its splits.

    Pulls ``train.tar``, ``val.tar`` and ``test.tar`` from HuggingFace, then
    extracts them in place to give the on-disk layout that ``dataset.py``
    expects (``train/{images,masks}/``, etc.).
    """
    _hf_snapshot(target, "PuzzleBench/Puzzle_Perception")
    for tar_name in ("train.tar", "val.tar", "test.tar"):
        tar_path = target / tar_name
        if not tar_path.exists():
            print(f"[warn] expected {tar_name} not found in snapshot, skipping")
            continue
        print(f"[extract] {tar_name}")
        with tarfile.open(tar_path) as tf:
            tf.extractall(target)
        tar_path.unlink()


# ---------- registry ----------------------------------------------------------

# Each `target` is repo-root-relative.
DATASETS: dict[str, dict] = {
    "cifar100":          {"target": "Existing_Datasets/Classification/CIFAR-100",            "fn": _dl_cifar100},
    "caltech101":        {"target": "Existing_Datasets/Classification/Caltech-101",          "fn": _dl_caltech101},
    "food101":           {"target": "Existing_Datasets/Classification/Food-101",             "fn": _dl_food101},
    "gtsrb":             {"target": "Existing_Datasets/Classification/GTSRB",                "fn": _dl_gtsrb},
    "imagenet":          {"target": "Existing_Datasets/Classification/ImageNet-1K",          "fn": _dl_imagenet},
    "spair":             {"target": "Existing_Datasets/Keypoint_Matching/SPair-71K",         "fn": _dl_spair},
    "flickr30k":         {"target": "Existing_Datasets/Retrieval/Flickr30K",                 "fn": _dl_flickr30k},
    "ade20k":            {"target": "Existing_Datasets/Segmentation/ADE20K",                 "fn": _dl_ade20k},
    "coco":              {"target": "Existing_Datasets/Vision_Language_Alignment/MS-COCO-2014", "fn": _dl_coco},
    "recaptioned_laion": {"target": "Existing_Datasets/Vision_Language_Alignment/Recaptioned_LAION/data", "fn": _dl_recaptioned_laion},
    "puzzle_perception": {"target": "Puzzle_Perception/Segmentation/data",                   "fn": _dl_puzzle_perception},
}


def _resolve(name: str, args: argparse.Namespace) -> None:
    spec = DATASETS[name]
    target = REPO_ROOT / spec["target"]
    if _is_populated(target) and not args.force:
        _log("skip", f"{name}: already populated at {target.relative_to(REPO_ROOT)}")
        return
    _reset_target(target)
    spec["fn"](target)
    _log("ok", f"{name}: {target.relative_to(REPO_ROOT)}")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument(
        "--dataset",
        choices=sorted(DATASETS),
        help="single dataset to download",
    )
    g.add_argument(
        "--all",
        action="store_true",
        help="download every dataset",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="overwrite an already-populated target",
    )
    args = p.parse_args()

    names = sorted(DATASETS) if args.all else [args.dataset]
    for n in names:
        _resolve(n, args)


if __name__ == "__main__":
    main()
