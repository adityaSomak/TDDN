"""Vision_Language_Alignment evaluation entry point.

Dispatches three eval tasks against any of ``clip`` / ``tdn`` / ``tddn``:

  - ``classification``  zero-shot template / CuPL / TIP-Adapter on
                        CIFAR-100, Caltech-101, Food-101, GTSRB, ImageNet-1K.
  - ``retrieval``       bidirectional Recall@K on Flickr30K and COCO val2014.
  - ``segmentation``    zero-shot open-vocab mIoU on ADE20K and Puzzle.

Single-combo usage::

    python run_eval.py --task classification --model tdn --dataset cifar100
    python run_eval.py --task classification --model tdn --dataset cifar100 --mode cupl
    python run_eval.py --task classification --model tdn --dataset cifar100 --mode tip --k 16
    python run_eval.py --task retrieval     --model tdn --dataset flickr30k
    python run_eval.py --task segmentation  --model tdn --dataset ade20k

Sweep usage (any of ``--task`` / ``--model`` / ``--dataset`` / ``--mode``
accepts ``all``; the cross-product is dispatched in one go and a summary
table prints at the end)::

    python run_eval.py --task all --model all --dataset all --limit 200

TIP-Adapter K-sweep (single command per (model, dataset))::

    python run_eval.py --task classification --model tdn --dataset caltech101 \\
        --mode tip --k-sweep 1,2,4,8,16

Live results land under
``evaluation/results/<task>/_live/<model>_<dataset>[_<mode>].json``.
The headline ``*.csv`` files report the published numbers and are NOT
touched by this script.

CuPL falls back to template prompts when ``descriptions/<dataset>.json``
is missing (e.g. ImageNet-1K). The fallback is recorded in the live
JSON as ``"used_template_fallback": true``.
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
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
from evaluation.src.segmentation import zero_shot_predict, compute_miou  # noqa: E402
from evaluation.src.encoders import build_alignment_encoder              # noqa: E402
from evaluation.src.datasets import (                                    # noqa: E402
    build_classification_dataset, build_retrieval_dataset,
    build_segmentation_dataset,
)

_HERE = _THIS.parent
_CONFIG = _HERE / "configs" / "models.yaml"
_RESULTS = _HERE / "evaluation" / "results"
_PROMPTS = _HERE / "evaluation" / "prompts"

_CLASS_DATASETS = ("cifar100", "caltech101", "food101", "gtsrb", "imagenet1k")
_RETR_DATASETS = ("flickr30k", "coco")
_SEG_DATASETS = ("ade20k", "puzzle")
_CLASS_MODES = ("zero_shot", "cupl", "tip")

_DATASET_ROOTS = {
    # Torchvision's CIFAR100 / Caltech101 expect the dataset dir to be
    # one level below the supplied ``root``.
    "cifar100":    "datasets/Existing_Datasets/Classification/CIFAR-100/cifar100",
    "caltech101":  "datasets/Existing_Datasets/Classification/Caltech-101/caltech101",
    # Food101 / GTSRB resolve a nested ``<root>/food-101/`` /
    # ``<root>/gtsrb/GTSRB/`` directory layout, so we point at the
    # symlink's parent.
    "food101":     "datasets/Existing_Datasets/Classification/Food-101",
    "gtsrb":       "datasets/Existing_Datasets/Classification/GTSRB",
    "imagenet1k":  "datasets/Existing_Datasets/Classification/ImageNet-1K/imagenet_hf",
    "flickr30k":   "datasets/Existing_Datasets/Retrieval/Flickr30K/flickr30k",
    "coco":        "datasets/Existing_Datasets/Vision_Language_Alignment/MS-COCO-2014/val2014",
    "ade20k":      "datasets/Existing_Datasets/Segmentation/ADE20K/ade20k",
    "puzzle":      "datasets/Puzzle_Perception/Segmentation/data",
}


def _load_config() -> dict:
    """Parse ``configs/models.yaml`` and flatten the model index."""
    spec = yaml.safe_load(_CONFIG.read_text())
    flat: dict[str, dict] = {}
    for group in ("baselines", "trained"):
        flat.update(spec.get(group, {}) or {})
    flat["_defaults"] = spec.get("defaults", {}) or {}
    return flat


def _load_templates() -> list:
    """Return the 80-element OpenAI prompt template list."""
    sys.path.insert(0, str(_PROMPTS))
    mod = importlib.import_module("openai_templates")
    return list(mod.OPENAI_PROMPT_TEMPLATES)


def _write_live(task: str, payload: dict, tag: str, dataset: str,
                mode: Optional[str] = None) -> Path:
    """Persist a live-run detail JSON under ``_live/``."""
    out_dir = _RESULTS / task / "_live"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{tag}_{dataset}_{mode}" if mode else f"{tag}_{dataset}"
    out_path = out_dir / f"{stem}.json"
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    return out_path


# ---------------------------------------------------------------------------
# Task dispatchers
# ---------------------------------------------------------------------------

def _extract_classification_features(encoder, dataset, limit, batch_size, device):
    """Run encoder.encode_image over a classification dataset → feats + labels."""
    if limit:
        from torch.utils.data import Subset
        dataset = Subset(dataset, range(min(limit, len(dataset))))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=2, pin_memory=True)
    feats_chunks, label_chunks = [], []
    for batch in loader:
        # Some wrappers (e.g. HFImageNet) return (image, label, index).
        images, labels = batch[0], batch[1]
        feats = encoder.encode_image(images.to(device, non_blocking=True))
        feats_chunks.append(feats.float().cpu())
        label_chunks.append(labels if isinstance(labels, torch.Tensor)
                            else torch.tensor(labels))
    return torch.cat(feats_chunks, dim=0), torch.cat(label_chunks, dim=0)


def _eval_classification(
    tag: str, dataset: str, mode: str, k: int, limit: Optional[int],
    encoder, device: str,
) -> dict:
    root = REPO_ROOT / _DATASET_ROOTS[dataset]
    ds, classes = build_classification_dataset(dataset, root, encoder.image_transform)
    feats, labels = _extract_classification_features(encoder, ds, limit, batch_size=64,
                                                     device=device)

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
                    "top1": acc[1], "top5": acc[5],
                    "used_template_fallback": True}
        classifier = build_cupl_classifier(_text_encoder, classes, desc_path,
                                           templates=templates, device="cpu")
        acc = top_k_accuracy(feats, labels, classifier, ks=(1, 5))
        return {"mode": "cupl", "n_samples": int(feats.shape[0]),
                "top1": acc[1], "top5": acc[5]}

    if mode == "tip":
        # K-shot cache: take the first K examples of each class as the
        # support set; the remainder forms the query split. For datasets
        # that ship a separate train split, plug that in instead.
        per_class: dict[int, list[int]] = {}
        for i, lbl in enumerate(labels.tolist()):
            per_class.setdefault(int(lbl), []).append(i)
        cache_idx, query_idx = [], []
        for cls in sorted(per_class):
            ids = per_class[cls]
            cache_idx.extend(ids[:k])
            query_idx.extend(ids[k:])
        cache_idx = torch.tensor(cache_idx)
        query_idx = torch.tensor(query_idx)
        text_classifier = build_zero_shot_classifier(
            _text_encoder, classes, templates, device="cpu",
        )
        cache_keys, cache_values = build_cache(
            feats[cache_idx], labels[cache_idx], n_classes=len(classes),
        )
        sweep = sweep_alpha(feats[query_idx], labels[query_idx],
                            cache_keys, cache_values, text_classifier,
                            alphas=DEFAULT_ALPHAS)
        return {"mode": "tip", "k": k,
                "n_cache": int(cache_idx.shape[0]),
                "n_query": int(query_idx.shape[0]),
                **sweep}

    raise ValueError(f"Unknown classification mode {mode!r}")


def _eval_retrieval(
    tag: str, dataset: str, limit: Optional[int], encoder, device: str,
) -> dict:
    root = REPO_ROOT / _DATASET_ROOTS[dataset]
    ds = build_retrieval_dataset(dataset, root, encoder.image_transform)
    if limit:
        from torch.utils.data import Subset
        ds = Subset(ds, range(min(limit, len(ds))))
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=2,
                        collate_fn=lambda batch: batch)

    img_feats_chunks: list[torch.Tensor] = []
    all_captions: list[str] = []
    caption_to_image: list[int] = []
    next_image_idx = 0
    for batch in loader:
        images = torch.stack([item[0] for item in batch], dim=0).to(device, non_blocking=True)
        feats = encoder.encode_image(images).float().cpu()
        img_feats_chunks.append(feats)
        for caps in [item[1] for item in batch]:
            for c in caps:
                all_captions.append(c)
                caption_to_image.append(next_image_idx)
            next_image_idx += 1

    img_feats = torch.cat(img_feats_chunks, dim=0)
    txt_feats = encoder.encode_text(all_captions).float().cpu()
    c2i = torch.tensor(caption_to_image, dtype=torch.long)
    recall = bidirectional_recall(img_feats, txt_feats, c2i, k_list=(1, 5, 10))
    return {"n_images": int(img_feats.shape[0]), "n_captions": len(all_captions), **recall}


def _eval_segmentation(
    tag: str, dataset: str, limit: Optional[int], encoder, device: str,
) -> dict:
    root = REPO_ROOT / _DATASET_ROOTS[dataset]
    ds, classes, n_classes = build_segmentation_dataset(
        dataset, root, encoder.image_transform, mask_size=512,
    )
    if limit:
        from torch.utils.data import Subset
        ds = Subset(ds, range(min(limit, len(ds))))
    loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=2, pin_memory=True)

    templates = _load_templates()
    classifier = build_zero_shot_classifier(
        lambda p: encoder.encode_text(p).cpu(),
        classes, templates, device="cpu",
    ).to(device)

    # When the text embedding is twice the per-patch channel count, the model
    # uses a [global || patch-mean] split; align per-patch features to the
    # patch half of the text vector. Otherwise text and patches share a space.
    preds, targets = [], []
    text_proj = None
    for batch in loader:
        if len(batch) == 3:
            images, masks, _ = batch
        else:
            images, masks = batch
        images = images.to(device, non_blocking=True)
        patches = encoder.encode_patches(images)
        if text_proj is None:
            patch_dim = patches.shape[1]
            if classifier.shape[-1] == 2 * patch_dim:
                text_proj = F.normalize(classifier[:, patch_dim:], dim=-1)
            else:
                text_proj = classifier
        pred = zero_shot_predict(patches, text_proj, output_size=masks.shape[-1])
        preds.append(pred.cpu())
        targets.append(masks if isinstance(masks, torch.Tensor) else torch.tensor(masks))
    preds_t = torch.cat(preds, dim=0)
    tgt_t = torch.cat(targets, dim=0)
    return {"n_classes": n_classes, "n_images": int(preds_t.shape[0]),
            **compute_miou(preds_t, tgt_t, n_classes)}


# ---------------------------------------------------------------------------
# Sweep helpers
# ---------------------------------------------------------------------------

_ALL_MODELS = ("clip", "tdn", "tddn")
_TASK_DATASETS = {
    "classification": _CLASS_DATASETS,
    "retrieval":      _RETR_DATASETS,
    "segmentation":   _SEG_DATASETS,
}


def _headline(task: str, result: dict) -> str:
    """Format the one-line metric summary for stdout."""
    if task == "classification":
        v = result.get("top1") if result.get("top1") is not None else result.get("best_top1")
        return f"top1={v:.2f}"
    if task == "retrieval":
        return f"i2t_r1={result['i2t_r1']:.2f}  t2i_r1={result['t2i_r1']:.2f}"
    return f"miou={100 * result['miou']:.2f}"


def _dispatch_combo(
    task: str, model: str, dataset: str, mode: Optional[str],
    k: int, limit: Optional[int], encoder, device: str,
) -> tuple[dict, Path]:
    """Run one (task, model, dataset[, mode]) combo and write its live JSON."""
    if task == "classification":
        result = _eval_classification(model, dataset, mode, k, limit, encoder, device)
        out_path = _write_live("classification", result, model, dataset, mode)
    elif task == "retrieval":
        result = _eval_retrieval(model, dataset, limit, encoder, device)
        out_path = _write_live("retrieval", result, model, dataset)
    else:
        result = _eval_segmentation(model, dataset, limit, encoder, device)
        out_path = _write_live("segmentation", result, model, dataset)
    return result, out_path


def _resolve_sweep(args, all_cfg) -> list[tuple[str, str, str, Optional[str]]]:
    """Expand args into the (task, model, dataset, mode) cross-product."""
    tasks = list(_TASK_DATASETS) if args.task == "all" else [args.task]
    models = list(_ALL_MODELS) if args.model == "all" else [args.model]
    for m in models:
        if m not in all_cfg:
            raise SystemExit(f"Unknown model_tag {m!r}.")

    combos: list[tuple[str, str, str, Optional[str]]] = []
    for task in tasks:
        valid_datasets = _TASK_DATASETS[task]
        datasets = list(valid_datasets) if args.dataset == "all" else [args.dataset]
        for d in datasets:
            if d not in valid_datasets:
                if args.task == "all":
                    continue  # silently skip datasets that don't apply to this task
                raise SystemExit(f"dataset {d!r} not valid for {task}")
        for model in models:
            for dataset in datasets:
                if dataset not in valid_datasets:
                    continue
                if task == "classification":
                    modes = (list(_CLASS_MODES) if args.mode == "all"
                             else [args.mode])
                    for mode in modes:
                        combos.append((task, model, dataset, mode))
                else:
                    combos.append((task, model, dataset, None))
    return combos


def _parse_k_sweep(raw: str) -> list[int]:
    """Parse ``--k-sweep "1,2,4,8,16"`` into ``[1, 2, 4, 8, 16]``."""
    out: list[int] = []
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(int(tok))
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry: parse arguments, expand sweep, dispatch each combo."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--task", required=True,
                   choices=("classification", "retrieval", "segmentation", "all"))
    p.add_argument("--model", required=True,
                   help="Model tag (clip / tdn / tddn / all).")
    p.add_argument("--dataset", required=True,
                   help="Dataset name or ``all`` to expand within the task.")
    p.add_argument("--mode", default="zero_shot",
                   choices=(*_CLASS_MODES, "all"),
                   help="Classification only: zero_shot / cupl / tip / all.")
    p.add_argument("--k", type=int, default=16,
                   help="TIP-Adapter shots per class (ignored unless --mode tip).")
    p.add_argument("--k-sweep", default="", dest="k_sweep",
                   help="Comma-separated K list; runs --mode tip once per K "
                        "and writes one JSON per K (e.g. ``1,2,4,8,16``).")
    p.add_argument("--limit", type=int, default=None,
                   help="Cap the number of evaluation samples (for quick runs).")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    all_cfg = _load_config()
    k_sweep = _parse_k_sweep(args.k_sweep)
    combos = _resolve_sweep(args, all_cfg)
    if not combos:
        raise SystemExit("Empty sweep: nothing to run.")

    encoder_cache: dict[str, object] = {}
    summary: list[tuple[str, str, str, str, str]] = []

    for task, model, dataset, mode in combos:
        if model not in encoder_cache:
            encoder_cache[model] = build_alignment_encoder(model, args.device)
        encoder = encoder_cache[model]

        # Choose the K values for this combo. K-sweep applies only to
        # classification × TIP-Adapter; everything else runs once.
        ks = (k_sweep if (task == "classification" and mode == "tip" and k_sweep)
              else [args.k])

        for k in ks:
            print(f"=== task={task} model={model} dataset={dataset} "
                  f"mode={mode or '-'}"
                  f"{f' k={k}' if (task == 'classification' and mode == 'tip') else ''} ===")
            result, out_path = _dispatch_combo(task, model, dataset, mode, k,
                                               args.limit, encoder, args.device)
            if task == "classification" and mode == "tip" and k_sweep:
                # Override the default ``_<mode>`` suffix with ``_tip_k{N}``.
                new_path = out_path.with_name(f"{model}_{dataset}_tip_k{k}.json")
                out_path.rename(new_path)
                out_path = new_path
            line = _headline(task, result)
            print(f"  {line}")
            print(f"  -> {out_path}")
            summary.append((task, model, dataset,
                            f"{mode}" + (f"_k{k}" if (mode == "tip" and k_sweep) else ""),
                            line))

    if len(summary) > 1:
        print()
        print("=== Sweep summary ===")
        for task, model, dataset, mode, line in summary:
            print(f"  {task:14s} | {model:5s} | {dataset:11s} | {mode:14s} | {line}")


if __name__ == "__main__":
    main()
