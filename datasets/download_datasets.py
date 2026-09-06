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
    pascal_voc            -> Existing_Datasets/Segmentation/PASCAL_VOC/
    cityscapes            -> Existing_Datasets/Segmentation/Cityscapes/        [needs Kaggle token]
    coco_stuff            -> Existing_Datasets/Segmentation/COCO_Stuff/
    pascal_context59      -> Existing_Datasets/Segmentation/PASCAL_Context/
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

from PIL import Image

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
    elif kind == "tar":
        with tarfile.open(archive, "r:") as tf:
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
    """Fetch Caltech-101 from the ``HuggingFaceM4/Caltech-101`` mirror
    instead of torchvision's built-in downloader, which hardcodes a
    Google Drive file id that now 404s (link rot, confirmed this
    session). That repo's ``caltech-101.zip`` bundles the exact
    original archive contents (``101_ObjectCategories.tar.gz`` +
    ``Annotations.tar``), so extraction reproduces the same
    ``caltech101/{101_ObjectCategories,Annotations}/`` layout
    ``torchvision.datasets.Caltech101(download=False)`` expects at eval
    time.
    """
    from huggingface_hub import hf_hub_download

    root = target / "caltech101"
    root.mkdir(parents=True, exist_ok=True)

    _log("dl", "HuggingFaceM4/Caltech-101:caltech-101.zip (HF mirror)")
    zip_path = Path(hf_hub_download(
        repo_id="HuggingFaceM4/Caltech-101", filename="caltech-101.zip",
        repo_type="dataset", local_dir=str(target),
    ))
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(target)
    zip_path.unlink()

    # The zip nests its contents one level deeper, under ``caltech-101/``.
    nested = target / "caltech-101"
    for archive_name, kind in (("101_ObjectCategories.tar.gz", "r:gz"), ("Annotations.tar", "r:")):
        archive_path = nested / archive_name
        with tarfile.open(archive_path, kind) as tf:
            tf.extractall(root)

    shutil.rmtree(nested)
    shutil.rmtree(target / "__MACOSX", ignore_errors=True)
    shutil.rmtree(target / ".cache", ignore_errors=True)


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
    """Fetch ADEChallengeData2016.zip from a fast HF mirror instead of the
    official host (``data.csail.mit.edu``), which is throttled to
    ~80 KB/s (hours for the ~967 MB archive) vs ~MB/s over HF's CDN.
    Byte-identical archive/layout, so extraction is unchanged.
    """
    from huggingface_hub import hf_hub_download

    _log("dl", "ranksu/ADE20K:ADEChallengeData2016.zip (HF mirror)")
    archive = Path(hf_hub_download(
        repo_id="ranksu/ADE20K", filename="ADEChallengeData2016.zip",
        repo_type="dataset", local_dir=str(target),
    ))
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(target)
    archive.unlink()


def _dl_pascal_voc(target: Path) -> None:
    """Fetch VOC2012 from a fast mirror instead of torchvision's built-in
    downloader, which hardcodes ``host.robots.ox.ac.uk`` — measured at
    ~150 KB/s-2 MB/s (heavily rate-limited) vs ~30+ MB/s from this mirror.
    Same tar contents/layout (``VOCdevkit/VOC2012/...``), so
    ``torchvision.datasets.VOCSegmentation(download=False)`` reads it
    identically at eval time.
    """
    _curl_extract(
        target,
        "https://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar",
        "VOCtrainval_11-May-2012.tar",
        "tar",
    )


def _dl_cityscapes(target: Path) -> None:
    """Fetch Cityscapes from the ``kavithak1388/cityscapes`` Kaggle mirror
    -- the official host (cityscapes-dataset.com) requires manual account
    registration with no scriptable download. This mirror was verified
    (this session) to ship the genuine ``gtFine/*_labelIds.png`` raw
    annotations at native 2048x1024 resolution (500/500 val images+masks,
    matching the official val split exactly) -- NOT the simplified
    Pix2Pix paired-image reformatting some other Cityscapes Kaggle
    mirrors use, which has no usable per-pixel class labels.

    Requires Kaggle API credentials at ``~/.kaggle/kaggle.json``.
    """
    if not (Path.home() / ".kaggle" / "kaggle.json").exists():
        sys.exit(
            "cityscapes: needs a Kaggle API token at ~/.kaggle/kaggle.json "
            "(https://www.kaggle.com/settings -> Create New Token)"
        )
    _run([
        "kaggle", "datasets", "download", "-d", "kavithak1388/cityscapes",
        "-p", str(target), "--unzip",
    ])
    # This mirror's zip nests everything one level deeper, under
    # ``Cityscape/`` -- flatten it to ``<target>/{gtFine,leftImg8bit}``.
    nested = target / "Cityscape"
    if nested.is_dir():
        for child in nested.iterdir():
            child.rename(target / child.name)
        nested.rmdir()


def _dl_coco_stuff(target: Path) -> None:
    """Fetch COCO-Stuff val2017 images + the raw ``stuffthingmaps``
    annotation release (category ids 0..181, 255=ignore already baked in
    -- ``COCOStuffNativeVal`` remaps these to the standard 171 contiguous
    trainIds at load time via mmsegmentation's own ``clsID_to_trID``
    table). Only val2017 is kept (this repo only ever scores on val); the
    annotation zip also ships train2017, discarded after extraction to
    save ~10x the disk.
    """
    _curl_extract(target, "http://images.cocodataset.org/zips/val2017.zip",
                   "val2017.zip", "zip")
    (target / "images").mkdir(exist_ok=True)
    (target / "val2017").rename(target / "images" / "val2017")

    _curl_extract(
        target,
        "http://calvin.inf.ed.ac.uk/wp-content/uploads/data/cocostuffdataset/stuffthingmaps_trainval2017.zip",
        "stuffthingmaps_trainval2017.zip", "zip",
    )
    (target / "annotations").mkdir(exist_ok=True)
    (target / "val2017").rename(target / "annotations" / "val2017")
    shutil.rmtree(target / "train2017", ignore_errors=True)


def _dl_pascal_context(target: Path) -> None:
    """Fetch PASCAL-Context59: VOC2010 base images (official host) + the
    ``trainval_merged.json`` Context annotations (59-class labels, COCO-
    style RLE segmentation keyed to VOC2010 image ids).

    The canonical ``trainval_merged.json`` host
    (codalabuser.blob.core.windows.net) has gone permanently dead (DNS
    NXDOMAIN, confirmed this session) -- recovered via a Wayback Machine
    snapshot instead. Converts the JSON's per-image RLE annotations into
    dense ``SegmentationClassContext/*.png`` label masks directly (val
    split only), reproducing mmsegmentation's own
    ``tools/dataset_converters/pascal_context.py`` conversion logic
    (verified against that script's actual source, not reimplemented from
    memory) without requiring the (unmaintained, Python-2-era) ``detail``
    package -- COCO-style RLE decode via ``pycocotools`` instead, which
    Detail API's own ``_mask.pyx`` is itself forked from.
    """
    import json
    from collections import defaultdict

    import numpy as np
    try:
        import pycocotools.mask as mask_utils
    except ImportError:
        sys.exit("pascal_context: pip install pycocotools")

    _curl_extract(
        target, "http://host.robots.ox.ac.uk/pascal/VOC/voc2010/VOCtrainval_03-May-2010.tar",
        "VOCtrainval_03-May-2010.tar", "tar",
    )

    json_path = target / "trainval_merged.json"
    _run([
        "curl", "-L", "--fail", "-o", str(json_path),
        "http://web.archive.org/web/20251122105902id_/"
        "https://codalabuser.blob.core.windows.net/public/trainval_merged.json",
    ])

    _log("dl", "decoding trainval_merged.json RLE annotations -> SegmentationClassContext/*.png")
    with open(json_path) as f:
        data = json.load(f)
    json_path.unlink()

    mapping = np.sort(np.array([
        0, 2, 259, 260, 415, 324, 9, 258, 144, 18, 19, 22, 23, 397, 25, 284,
        158, 159, 416, 33, 162, 420, 454, 295, 296, 427, 44, 45, 46, 308, 59,
        440, 445, 31, 232, 65, 354, 424, 68, 326, 72, 458, 34, 207, 80, 355,
        85, 347, 220, 349, 360, 98, 187, 104, 105, 366, 189, 368, 113, 115,
    ]))
    key = np.arange(len(mapping), dtype=np.uint8)

    anns_by_image = defaultdict(list)
    for ann in data["annos_segmentation"]:
        anns_by_image[ann["image_id"]].append(ann)

    devkit = target / "VOCdevkit" / "VOC2010"
    out_dir = devkit / "SegmentationClassContext"
    out_dir.mkdir(parents=True, exist_ok=True)

    val_stems = []
    for img in data["images"]:
        if img["phase"] != "val":
            continue
        h, w = img["height"], img["width"]
        raw_mask = np.zeros((h, w), dtype=np.int32)
        for ann in anns_by_image.get(img["image_id"], []):
            m = mask_utils.decode(ann["segmentation"])
            raw_mask[m.astype(bool)] = ann["category_id"]
        index = np.digitize(raw_mask.ravel(), mapping, right=True)
        idx_mask = key[index].reshape(raw_mask.shape)
        stem = Path(img["file_name"]).stem
        Image.fromarray(idx_mask).save(out_dir / f"{stem}.png")
        val_stems.append(stem)

    imgset_dir = devkit / "ImageSets" / "SegmentationContext"
    imgset_dir.mkdir(parents=True, exist_ok=True)
    (imgset_dir / "val.txt").write_text("".join(s + "\n" for s in sorted(val_stems)))
    _log("dl", f"wrote {len(val_stems)} val masks")


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


def _hf_snapshot(target: Path, repo_id: str, allow_patterns: list[str] | None = None) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        sys.exit(f"{repo_id}: pip install huggingface-hub")
    _log("dl", f"{repo_id} -> {target}" + (f" {allow_patterns}" if allow_patterns else ""))
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(target),
        allow_patterns=allow_patterns,
    )


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
    """Fetch Puzzle-Perception and materialise the layout ``dataset.py`` expects.

    The HuggingFace repo ships a single unified Parquet table covering both the
    segmentation and pVQA halves (built by
    ``Puzzle_Perception/build_hf_release.py``, an internal tool not in this repo),
    with the label maps alongside it as PNG files. It no longer ships
    ``{train,val,test}.tar``.

    ``dataset.py`` and every segmentation probe read
    ``<root>/{train,val,test}/{images,masks}/<task>_<id>.png``, where each mask is
    a single-channel PNG with pixel values ``0..29`` (the unified class ids).
    Reconstruct exactly that: move the shipped RAW masks into place, and write
    the images back out of the Parquet.

    IMPORTANT: the repo ships TWO mask directories, and only one is the raw
    class-id data training needs —

        masks/<split>/*.png       colorized RGB renders, for the HF viewer only
        raw_masks/<split>/*.png   raw single-channel PNGs, values 0..29 <- THIS ONE

    Fetching ``masks/*`` here would silently install 3-channel colorized PNGs as
    training masks — ``dataset.py`` would then read RGB images where it expects
    single-channel label maps, corrupting every downstream segmentation-probe run
    with no obvious error (wrong shape/dtype, not a crash, since PIL opens either
    just fine). Always fetch ``raw_masks/*``, never ``masks/*``.

    Images are copied as **raw bytes** — the ``image`` column is cast with
    ``decode=False`` so PIL never re-encodes them, which keeps the extracted
    PNGs byte-identical to the published ones.

    Only ``type == "segmentation"`` rows are materialised. The pVQA rows are an
    eval-only probe set whose authoritative copy is committed under
    ``Puzzle_Perception/PVQA/``; nothing reading this tree consumes them.
    """
    try:
        from datasets import Image as HFImage
        from datasets import load_dataset
    except ImportError:
        sys.exit("puzzle_perception: pip install datasets")

    repo_id = "PuzzleBench/Puzzle_Perception"

    # raw_masks/ (not masks/, which is the colorized viewer render) and the side
    # files are plain files in the repo. Fetch just those, so this step does not
    # also pull the multi-GB Parquet a second time.
    _hf_snapshot(
        target, repo_id, allow_patterns=["raw_masks/*", "classes.yaml", "manifest.csv"]
    )

    # The shipped directory is named raw_masks/ to distinguish it from the
    # colorized masks/ on the repo; the LOCAL on-disk name dataset.py expects is
    # still masks/ — only the repo-side name changed.
    masks_root = target / "raw_masks"
    for split in ("train", "val", "test"):
        src = masks_root / split
        if not src.is_dir():
            print(f"[warn] no raw_masks/{split}/ in snapshot, skipping")
            continue
        dst = target / split / "masks"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        _log("masks", f"{split}: {len(list(dst.glob('*.png')))} files")
    shutil.rmtree(masks_root, ignore_errors=True)

    for split in ("train", "val", "test"):
        images_dir = target / split / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        ds = load_dataset(repo_id, split=split).cast_column("image", HFImage(decode=False))
        count = 0
        for row in ds:
            if row["type"] != "segmentation":
                continue
            # image_path is "<split>/images/<task>_<id>.png"; keep the stem, as
            # dataset.py derives source_task/source_id by parsing the filename.
            (images_dir / Path(row["image_path"]).name).write_bytes(row["image"]["bytes"])
            count += 1
        _log("extract", f"{split}: {count} images")


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
    "pascal_voc":        {"target": "Existing_Datasets/Segmentation/PASCAL_VOC",              "fn": _dl_pascal_voc},
    "cityscapes":        {"target": "Existing_Datasets/Segmentation/Cityscapes",              "fn": _dl_cityscapes},
    "coco_stuff":        {"target": "Existing_Datasets/Segmentation/COCO_Stuff",              "fn": _dl_coco_stuff},
    "pascal_context59":  {"target": "Existing_Datasets/Segmentation/PASCAL_Context",          "fn": _dl_pascal_context},
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
