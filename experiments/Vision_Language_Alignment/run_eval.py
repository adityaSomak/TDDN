"""Vision_Language_Alignment evaluation entry point.

Dispatches three tasks against any registered model tag:

  - ``classification``  zero-shot / CuPL / TIP-Adapter on CIFAR-100,
                        Caltech-101, Food-101, GTSRB.
  - ``retrieval``       bidirectional Recall@K on Flickr30K (Karpathy test
                        split) and COCO val2014.
  - ``segmentation``    zero-shot open-vocab mIoU on ADE20K, Cityscapes,
                        COCO-Stuff, PASCAL-Context-59 and Puzzle-Perception,
                        scored with a sliding window at each image's own
                        resolution.

Single-combo usage::

    python run_eval.py --task classification --model tdn --dataset cifar100
    python run_eval.py --task classification --model tdn --dataset cifar100 --mode cupl
    python run_eval.py --task classification --model tdn --dataset cifar100 --mode tip --k 16
    python run_eval.py --task retrieval     --model tdn --dataset flickr30k
    python run_eval.py --task segmentation  --model tdn --dataset cityscapes

Sweep usage (any of ``--task`` / ``--model`` / ``--dataset`` / ``--mode``
accepts ``all``; the cross-product is dispatched in one go and a summary table
prints at the end)::

    python run_eval.py --task all --model all --dataset all --limit 200

TIP-Adapter K-sweep (one command per (model, dataset))::

    python run_eval.py --task classification --model tdn --dataset caltech101 \\
        --mode tip --k-sweep 1,2,4,8,16

Results land under ``evaluation/results/<task>/_live/`` unless ``--publish`` is
given, which writes the committed file instead. ``--aggregate`` rebuilds the
headline CSVs from the committed per-model JSONs.

CuPL falls back to template prompts when ``descriptions/<dataset>.json`` is
missing; the fallback is recorded as ``"used_template_fallback": true``.
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Optional

import torch
import yaml
from torch.utils.data import DataLoader

_THIS = Path(__file__).resolve()
_EXPERIMENTS = _THIS.parents[1]
if str(_EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS))

from shared_utils.paths import REPO_ROOT                                # noqa: E402

from evaluation.src.classifier import (                                  # noqa: E402
    build_zero_shot_classifier, build_cupl_classifier, top_k_accuracy,
)
from evaluation.src.tip_adapter import (                                 # noqa: E402
    build_cache, sweep_alpha, DEFAULT_ALPHAS,
)
from evaluation.src.retrieval import bidirectional_recall                # noqa: E402
from evaluation.src.segmentation import (                                # noqa: E402
    accumulate_confusion, miou_from_confusion,
)
from evaluation.src.encoders import (                                    # noqa: E402
    build_alignment_encoder, build_dense_encoder, registry,
)
from evaluation.src.datasets import (                                    # noqa: E402
    build_classification_dataset, build_retrieval_dataset,
    build_segmentation_dataset, retrieval_protocol, seg_target,
)
from evaluation.src.slide_inference import CanvasAccumulator             # noqa: E402

_HERE = _THIS.parent
_CONFIG = _HERE / "configs" / "models.yaml"
_RESULTS = _HERE / "evaluation" / "results"
_PROMPTS = _HERE / "evaluation" / "prompts"

_CLASS_DATASETS = ("cifar100", "caltech101", "food101", "gtsrb")
_RETR_DATASETS = ("flickr30k", "coco")
_SEG_DATASETS = ("ade20k", "cityscapes", "coco_stuff", "context59", "puzzle_perception")
_CLASS_MODES = ("zero_shot", "cupl", "tip")

_TASK_DATASETS = {
    "classification": _CLASS_DATASETS,
    "retrieval": _RETR_DATASETS,
    "segmentation": _SEG_DATASETS,
}

_DATASET_ROOTS = {
    # Torchvision's CIFAR100 / Caltech101 / Food101 / GTSRB descend into their
    # own subdir under the supplied ``root``, so we pass the parent.
    "cifar100":          "datasets/Existing_Datasets/Classification/CIFAR-100",
    "caltech101":        "datasets/Existing_Datasets/Classification/Caltech-101",
    "food101":           "datasets/Existing_Datasets/Classification/Food-101",
    "gtsrb":             "datasets/Existing_Datasets/Classification/GTSRB",
    "flickr30k":         "datasets/Existing_Datasets/Retrieval/Flickr30K",
    "coco":              "datasets/Existing_Datasets/Vision_Language_Alignment/MS-COCO-2014/val2014",
    "ade20k":            "datasets/Existing_Datasets/Segmentation/ADE20K",
    "cityscapes":        "datasets/Existing_Datasets/Segmentation/Cityscapes",
    "coco_stuff":        "datasets/Existing_Datasets/Segmentation/COCO_Stuff",
    "context59":         "datasets/Existing_Datasets/Segmentation/PASCAL_Context",
    "puzzle_perception": "datasets/Puzzle_Perception/Segmentation/data",
}

# Readout per task for the trained tags: classification takes the first half of
# the embedding, retrieval the whole vector.
_CLS_HALF = {"classification": True, "retrieval": False}
# fgclip2 patch budget per task.
_PATCH_MODE = {"classification": "fixed1024", "retrieval": "native"}


def _defaults() -> dict:
    return yaml.safe_load(_CONFIG.read_text()).get("defaults", {}) or {}


def _load_templates() -> list:
    """The 80-element OpenAI prompt template list."""
    if str(_PROMPTS) not in sys.path:
        sys.path.insert(0, str(_PROMPTS))
    return list(importlib.import_module("openai_templates").OPENAI_PROMPT_TEMPLATES)


def _out_path(task: str, tag: str, dataset: str, mode: Optional[str],
              publish: bool) -> Path:
    """``<task>/[_live/]<tag>_<dataset>[_<mode>].json``.

    Zero-shot carries no mode suffix; ``cupl`` / ``tip`` do.
    """
    out_dir = _RESULTS / task if publish else _RESULTS / task / "_live"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{tag}_{dataset}_{mode}" if mode and mode != "zero_shot" else f"{tag}_{dataset}"
    return out_dir / f"{stem}.json"


# ---------------------------------------------------------------------------
# Task dispatchers
# ---------------------------------------------------------------------------

def _extract_classification_features(encoder, dataset, limit, batch_size, device):
    """encoder.encode_image over a classification dataset -> feats + labels."""
    if limit:
        from torch.utils.data import Subset
        dataset = Subset(dataset, range(min(limit, len(dataset))))
    wants_pil = getattr(encoder, "wants_pil", False)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2,
                        pin_memory=not wants_pil,
                        collate_fn=(lambda b: b) if wants_pil else None)
    feats_chunks, label_chunks = [], []
    for batch in loader:
        if wants_pil:
            images = [item[0] for item in batch]
            labels = torch.tensor([int(item[1]) for item in batch])
        else:
            images, labels = batch[0].to(device, non_blocking=True), batch[1]
        feats = encoder.encode_image(images)
        feats_chunks.append(feats.float().cpu())
        label_chunks.append(labels if isinstance(labels, torch.Tensor)
                            else torch.tensor(labels))
    return torch.cat(feats_chunks, dim=0), torch.cat(label_chunks, dim=0)


def _eval_classification(tag: str, dataset: str, mode: str, k: int,
                         limit: Optional[int], encoder, device: str) -> dict:
    root = REPO_ROOT / _DATASET_ROOTS[dataset]
    ds, classes = build_classification_dataset(dataset, root, encoder.image_transform)
    feats, labels = _extract_classification_features(encoder, ds, limit, 64, device)

    def _text_encoder(prompts):
        return encoder.encode_text(prompts).cpu()

    templates = _load_templates()
    if mode == "zero_shot":
        classifier = build_zero_shot_classifier(_text_encoder, classes, templates, device="cpu")
        acc = top_k_accuracy(feats, labels, classifier, ks=(1, 5))
        return {"mode": "zero_shot", "n_samples": int(feats.shape[0]),
                "top1": acc[1], "top5": acc[5]}

    if mode == "cupl":
        desc_path = _PROMPTS / "descriptions" / f"{dataset}.json"
        if not desc_path.is_file():
            print(f"  [warn] {desc_path.name} not found; using template prompts.")
            classifier = build_zero_shot_classifier(_text_encoder, classes,
                                                    templates, device="cpu")
            acc = top_k_accuracy(feats, labels, classifier, ks=(1, 5))
            return {"mode": "cupl", "n_samples": int(feats.shape[0]),
                    "top1": acc[1], "top5": acc[5], "used_template_fallback": True}
        classifier = build_cupl_classifier(_text_encoder, classes, desc_path,
                                           templates=templates, device="cpu")
        acc = top_k_accuracy(feats, labels, classifier, ks=(1, 5))
        return {"mode": "cupl", "n_samples": int(feats.shape[0]),
                "top1": acc[1], "top5": acc[5]}

    if mode == "tip":
        # K-shot cache: the first K examples of each class form the support set,
        # the remainder the query split.
        per_class: dict[int, list[int]] = {}
        for i, lbl in enumerate(labels.tolist()):
            per_class.setdefault(int(lbl), []).append(i)
        cache_idx, query_idx = [], []
        for cls in sorted(per_class):
            ids = per_class[cls]
            cache_idx.extend(ids[:k])
            query_idx.extend(ids[k:])
        cache_idx, query_idx = torch.tensor(cache_idx), torch.tensor(query_idx)
        text_classifier = build_zero_shot_classifier(_text_encoder, classes,
                                                     templates, device="cpu")
        cache_keys, cache_values = build_cache(feats[cache_idx], labels[cache_idx],
                                                n_classes=len(classes))
        sweep = sweep_alpha(feats[query_idx], labels[query_idx], cache_keys,
                            cache_values, text_classifier, alphas=DEFAULT_ALPHAS)
        return {"mode": "tip", "k": k, "n_cache": int(cache_idx.shape[0]),
                "n_query": int(query_idx.shape[0]), **sweep}

    raise ValueError(f"Unknown classification mode {mode!r}")


def _eval_retrieval(tag: str, dataset: str, limit: Optional[int],
                    encoder, device: str, text_chunk: int = 256) -> dict:
    root = REPO_ROOT / _DATASET_ROOTS[dataset]
    ds = build_retrieval_dataset(dataset, root, encoder.image_transform)
    if limit:
        from torch.utils.data import Subset
        ds = Subset(ds, range(min(limit, len(ds))))
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=2,
                        collate_fn=lambda batch: batch)
    wants_pil = getattr(encoder, "wants_pil", False)

    img_feats_chunks: list[torch.Tensor] = []
    all_captions: list[str] = []
    caption_to_image: list[int] = []
    next_image_idx = 0
    for batch in loader:
        imgs = [item[0] for item in batch]
        images = imgs if wants_pil else torch.stack(imgs, dim=0).to(device, non_blocking=True)
        img_feats_chunks.append(encoder.encode_image(images).float().cpu())
        for caps in [item[1] for item in batch]:
            for c in caps:
                all_captions.append(c)
                caption_to_image.append(next_image_idx)
            next_image_idx += 1

    # Captions are encoded in chunks: COCO val2014 is ~200k captions, too many
    # for one forward pass on the heavier text towers.
    txt_chunks = [encoder.encode_text(all_captions[i:i + text_chunk]).float().cpu()
                  for i in range(0, len(all_captions), text_chunk)]

    img_feats = torch.cat(img_feats_chunks, dim=0)
    txt_feats = torch.cat(txt_chunks, dim=0)
    c2i = torch.tensor(caption_to_image, dtype=torch.long)
    k_list = tuple(_defaults().get("retrieval", {}).get("k_list", (1, 5, 10)))
    recall = bidirectional_recall(img_feats, txt_feats, c2i, k_list=k_list)
    return {"protocol": retrieval_protocol(dataset),
            "n_images": int(img_feats.shape[0]),
            "n_captions": len(all_captions), **recall}


def _eval_segmentation(tag: str, dataset: str, limit: Optional[int], encoder,
                       device: str, max_batch: int = 32,
                       images_per_chunk: int = 32) -> dict:
    """Sliding-window open-vocab segmentation.

    Each image is resized so its shorter side is the dataset's target (aspect
    preserved), then covered by ``crop_size`` windows at half-crop stride with
    overlap averaging, and scored at that resolution.
    """
    root = REPO_ROOT / _DATASET_ROOTS[dataset]
    target = seg_target(dataset)
    crop_size = target
    stride = int(crop_size * _defaults().get("segmentation", {}).get("stride_ratio", 0.5))
    ds, classes, n_classes = build_segmentation_dataset(
        dataset, root, encoder.image_transform, target=target)
    indices = list(range(min(limit, len(ds)) if limit else len(ds)))

    classifier = build_zero_shot_classifier(
        lambda p: encoder.encode_text(p).cpu(), classes, templates=_load_templates(),
        device="cpu").to(device)

    def forward_fn(window: torch.Tensor) -> torch.Tensor:
        return encoder.encode_patches_logits(window, classifier)

    conf = None
    for i in range(0, len(indices), images_per_chunk):
        acc = CanvasAccumulator(forward_fn, n_classes, crop_size=crop_size,
                                stride=stride, max_batch=max_batch)
        handles = []
        for idx in indices[i:i + images_per_chunk]:
            image_chw, mask = ds[idx]
            handles.append((acc.add_image(image_chw), mask))
        for canvas_id, mask in handles:
            pred = acc.finalize(canvas_id).argmax(dim=0)
            chunk = accumulate_confusion(pred.reshape(-1), mask.reshape(-1),
                                         n_classes, ignore_index=255)
            conf = chunk if conf is None else conf + chunk
        print(f"  ...{min(i + images_per_chunk, len(indices))}/{len(indices)}", flush=True)

    result = miou_from_confusion(conf)
    return {"protocol": f"slide_{target}_{stride}", "n_classes": n_classes,
            "n_images": len(indices), "miou": 100 * result["miou"],
            "per_class_iou": [round(100 * v, 2) for v in result["per_class_iou"]]}


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def _build_encoder(task: str, tag: str, device: str, crop_size: int | None = None):
    if task == "segmentation":
        return build_dense_encoder(tag, device, crop_size=crop_size or 448)
    return build_alignment_encoder(
        tag, device,
        cls_half=_CLS_HALF[task] and registry()[tag].get("family") == "trained",
        patch_mode=_PATCH_MODE[task])


def _dispatch(task: str, tag: str, dataset: str, mode: Optional[str], k: int,
              limit: Optional[int], encoder, device: str, max_batch: int,
              images_per_chunk: int) -> dict:
    if task == "classification":
        result = _eval_classification(tag, dataset, mode or "zero_shot", k, limit,
                                      encoder, device)
    elif task == "retrieval":
        result = _eval_retrieval(tag, dataset, limit, encoder, device)
    elif task == "segmentation":
        result = _eval_segmentation(tag, dataset, limit, encoder, device,
                                    max_batch=max_batch,
                                    images_per_chunk=images_per_chunk)
    else:
        raise ValueError(f"Unknown task {task!r}")
    return {"model": tag, "dataset": dataset, **result}


def _headline(task: str, result: dict) -> str:
    if task == "classification":
        v = result.get("top1", result.get("best_top1"))
        return f"top1={v:.2f}"
    if task == "retrieval":
        return f"i2t_r1={result['i2t_r1']:.2f}  t2i_r1={result['t2i_r1']:.2f}"
    return f"miou={result['miou']:.2f}"


def _parse_k_sweep(raw: str) -> list[int]:
    return [int(x) for x in raw.split(",") if x.strip()] if raw else []


def _resolve_sweep(args, tags: list[str]) -> list[tuple[str, str, str, Optional[str]]]:
    tasks = list(_TASK_DATASETS) if args.task == "all" else [args.task]
    models = tags if args.model == "all" else [args.model]
    combos = []
    for task in tasks:
        datasets = (_TASK_DATASETS[task] if args.dataset == "all"
                    else [args.dataset])
        for dataset in datasets:
            if dataset not in _TASK_DATASETS[task]:
                if args.dataset != "all":
                    raise SystemExit(
                        f"dataset {dataset!r} is not valid for task {task!r}; "
                        f"choices: {', '.join(_TASK_DATASETS[task])}")
                continue
            modes = ([None] if task != "classification"
                     else (list(_CLASS_MODES) if args.mode == "all" else [args.mode]))
            for m in modes:
                for model in models:
                    combos.append((task, model, dataset, m))
    return combos


def main(argv: list[str] | None = None) -> int:
    reg = registry()
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", choices=(*_TASK_DATASETS, "all"),
                   help="required unless --aggregate")
    p.add_argument("--model", help="a tag from configs/models.yaml, or 'all'")
    p.add_argument("--dataset", help="dataset name, or 'all' to expand within the task")
    p.add_argument("--mode", choices=(*_CLASS_MODES, "all"), default="zero_shot",
                   help="classification only")
    p.add_argument("--k", type=int, default=16, help="TIP shots per class")
    p.add_argument("--k-sweep", dest="k_sweep", default="",
                   help="comma-separated TIP shot counts, e.g. 1,2,4,8,16")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--max-batch", dest="max_batch", type=int, default=32,
                   help="segmentation: max windows per forward")
    p.add_argument("--images-per-chunk", dest="images_per_chunk", type=int, default=32,
                   help="segmentation: images accumulated before scoring")
    p.add_argument("--force", action="store_true",
                   help="recompute even if the output file exists")
    p.add_argument("--publish", action="store_true",
                   help="write the committed result file instead of _live/")
    p.add_argument("--aggregate", action="store_true",
                   help="rebuild the headline CSVs from committed results and exit")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args(argv)

    if args.aggregate:
        from evaluation.src import aggregate
        aggregate.main()
        return 0

    for required in ("task", "model", "dataset"):
        if getattr(args, required) is None:
            p.error(f"--{required} is required unless --aggregate is given")
    if args.model != "all" and args.model not in reg:
        p.error(f"unknown model {args.model!r}; choices: {', '.join(sorted(reg))}")

    tags = sorted(reg)
    combos = _resolve_sweep(args, tags)
    k_sweep = _parse_k_sweep(args.k_sweep)

    encoders: dict[tuple, object] = {}
    summary = []
    for task, model, dataset, mode in combos:
        ks = k_sweep if (task == "classification" and mode == "tip" and k_sweep) else [args.k]
        for k in ks:
            mode_out = mode if mode != "zero_shot" else None
            if task == "classification" and mode == "tip" and k_sweep:
                mode_out = f"tip_k{k}"
            out_path = _out_path(task, model, dataset, mode_out, args.publish)
            if out_path.exists() and not args.force:
                print(f"skip (exists): {out_path.name}  -- use --force to recompute")
                continue

            crop = seg_target(dataset) if task == "segmentation" else None
            key = (task, model, crop)
            if key not in encoders:
                encoders.clear()          # one model resident at a time
                encoders[key] = _build_encoder(task, model, args.device, crop)
            encoder = encoders[key]

            print(f"\n=== {task} | {model} | {dataset}"
                  + (f" | {mode}" + (f" k={k}" if mode == "tip" else "") if mode else "")
                  + " ===", flush=True)
            result = _dispatch(task, model, dataset, mode, k, args.limit, encoder,
                               args.device, args.max_batch, args.images_per_chunk)
            out_path.write_text(json.dumps(result, indent=2) + "\n")
            line = _headline(task, result)
            print(f"{line}\n-> {out_path}", flush=True)
            summary.append((task, model, dataset, mode or "-", line))

    if len(summary) > 1:
        print("\n=== summary ===")
        for task, model, dataset, mode, line in summary:
            print(f"  {task:14s} | {model:14s} | {dataset:18s} | {mode:10s} | {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
