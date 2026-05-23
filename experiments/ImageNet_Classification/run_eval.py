"""ImageNet_Classification entry point — k-NN classification on ImageNet-1K.

Extracts a single pooled, L2-normalized vector per image with the
``shared_utils.feature_extraction`` backbone registry, then runs cosine
k-NN against a balanced 100/class gallery (100K images) and reports
top-1 / top-5 accuracy on the 50K validation split.

Usage::

    python run_eval.py --model dinov3                # one model_tag
    python run_eval.py --model all                   # sweep, write headline CSV
    python run_eval.py --model dinov3 --val-subset 100 --per-class-train 10
                                                     # smoke test (~30s)

All per-backbone recipes (pooling rule, transform, diffusion timestep)
live in ``configs/models.yaml`` — no per-backbone Python code lives in
this experiment.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

_THIS = Path(__file__).resolve()
_EXPERIMENTS = _THIS.parents[1]
if str(_EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS))

from shared_utils.feature_extraction import (                                  # noqa: E402
    build_extractor, build_transform, fuse_concat_global, pool_to_vector,
)
from shared_utils.paths import REPO_ROOT                                       # noqa: E402

from evaluation.src.knn import knn_classify                                    # noqa: E402


def _load_imagenet_dataset_module():
    """Load ``datasets/Existing_Datasets/Classification/ImageNet-1K/dataset.py``
    by file path — the local ``datasets/`` package shares a top-level name
    with the HuggingFace ``datasets`` library so we can't put REPO_ROOT
    on ``sys.path``."""
    import importlib.util

    module_name = "imagenet_dataset_loader"
    if module_name in sys.modules:
        return sys.modules[module_name]
    path = (REPO_ROOT / "datasets" / "Existing_Datasets" / "Classification"
            / "ImageNet-1K" / "dataset.py")
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


HFImageNet = _load_imagenet_dataset_module().HFImageNet


_HERE = _THIS.parent
_CONFIG = _HERE / "configs" / "models.yaml"
_RESULTS_DIR = _HERE / "evaluation" / "results"
_RESULTS_CSV = _RESULTS_DIR / "imagenet_classification.csv"
_FEATURE_CACHE = _HERE / "features"

_CSV_COLUMNS = ("model", "top1", "top5", "dim", "k", "n_train", "n_val")


def _load_config() -> dict:
    """Return the parsed ``configs/models.yaml`` as a flat dict by model_tag."""
    spec = yaml.safe_load(_CONFIG.read_text())
    flat: dict[str, dict] = {}
    for group in ("baselines", "trained"):
        flat.update(spec.get(group, {}) or {})
    flat.update(spec.get("fusion", {}) or {})
    flat["_defaults"] = spec.get("defaults", {}) or {}
    return flat


def _extract_single(
    tag: str, cfg: dict, args: argparse.Namespace, device: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run a single backbone's extractor over gallery + query splits.

    Returns ``(gallery_feats, gallery_labels, query_feats, query_labels)``
    with L2-normalized features in float16.
    """
    extractor = build_extractor(
        cfg["backbone"], device=device,
        extractor_kwargs=cfg.get("extractor", {}) or {},
    )
    transform = build_transform(
        cfg["backbone"], cfg["transform"]["input_size"], cfg["transform"]["strategy"],
    )

    def _features(split: str, **dataset_kwargs) -> tuple[np.ndarray, np.ndarray]:
        ds = HFImageNet(split, transform=transform, **dataset_kwargs)
        loader = DataLoader(
            ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True,
        )
        feats, labels = [], []
        for imgs, lbls, _ in loader:
            imgs = imgs.to(device, non_blocking=True)
            try:
                out = extractor.extract(imgs, prompt="A photo")
            except TypeError:
                out = extractor.extract(imgs)
            vec = pool_to_vector(out, cfg["pool"])
            feats.append(vec.float().cpu().numpy().astype(np.float16))
            labels.append(lbls.numpy())
        return np.concatenate(feats, axis=0), np.concatenate(labels, axis=0)

    gf, gy = _features("train", per_class=args.per_class_train)
    qf, qy = _features("validation", subset=args.val_subset)
    return gf, gy, qf, qy


def _extract_fused(
    cfg: dict, all_cfg: dict, args: argparse.Namespace, device: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fuse two component model_tags by L2-norm-and-concat at the pooled level."""
    gf_parts, qf_parts = [], []
    gy = qy = None
    for component in cfg["components"]:
        sub_cfg = all_cfg[component["tag"]]
        gf_i, gy_i, qf_i, qy_i = _extract_single(component["tag"], sub_cfg, args, device)
        gf_parts.append(torch.from_numpy(gf_i.astype(np.float32)))
        qf_parts.append(torch.from_numpy(qf_i.astype(np.float32)))
        gy = gy_i if gy is None else gy
        qy = qy_i if qy is None else qy
    weights = [c.get("weight", 1.0) for c in cfg["components"]]
    gf = fuse_concat_global(gf_parts, weights).numpy().astype(np.float16)
    qf = fuse_concat_global(qf_parts, weights).numpy().astype(np.float16)
    return gf, gy, qf, qy


def _features_for(
    tag: str, all_cfg: dict, args: argparse.Namespace, device: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Either load cached features for ``tag`` or extract them now."""
    cache = _FEATURE_CACHE / tag
    if args.use_cached and cache.exists():
        gf = np.load(cache / "gallery_x.npy")
        gy = np.load(cache / "gallery_y.npy")
        qf = np.load(cache / "query_x.npy")
        qy = np.load(cache / "query_y.npy")
        return gf, gy, qf, qy
    cfg = all_cfg[tag]
    if "components" in cfg:
        gf, gy, qf, qy = _extract_fused(cfg, all_cfg, args, device)
    else:
        gf, gy, qf, qy = _extract_single(tag, cfg, args, device)
    if args.cache_features:
        cache.mkdir(parents=True, exist_ok=True)
        np.save(cache / "gallery_x.npy", gf)
        np.save(cache / "gallery_y.npy", gy)
        np.save(cache / "query_x.npy", qf)
        np.save(cache / "query_y.npy", qy)
    return gf, gy, qf, qy


def _update_csv(row: dict) -> None:
    """Upsert one row keyed by ``model`` into the headline CSV."""
    _RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    if _RESULTS_CSV.exists():
        rows = list(csv.DictReader(_RESULTS_CSV.open()))
    rows = [r for r in rows if r["model"] != row["model"]]
    rows.append(row)
    rows.sort(key=lambda r: r["model"])
    with _RESULTS_CSV.open("w") as f:
        w = csv.DictWriter(f, fieldnames=_CSV_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in _CSV_COLUMNS})


def _evaluate(tag: str, all_cfg: dict, args: argparse.Namespace, device: str) -> dict:
    """Extract features (or load cached) → run k-NN → return a CSV row dict."""
    print(f"\n=== {tag} ===")
    t0 = time.time()
    gf, gy, qf, qy = _features_for(tag, all_cfg, args, device)
    print(f"  extract: {time.time() - t0:.1f}s  gallery={gf.shape}  query={qf.shape}")

    defaults = all_cfg["_defaults"]
    k = args.k or defaults.get("k", 20)
    metrics = knn_classify(gf, gy, qf, qy, k=k, device=device)
    print(f"  top1={metrics['top1']:.2f}  top5={metrics['top5']:.2f}  k={metrics['k']}")

    row = {"model": tag, **{k: f"{metrics[k]:.3f}" if isinstance(metrics[k], float) else metrics[k]
                            for k in ("top1", "top5", "dim", "k", "n_train", "n_val")}}
    _update_csv(row)
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (_RESULTS_DIR / f"{tag}.json").write_text(json.dumps(metrics, indent=2))
    return row


def main() -> None:
    """CLI entry: parse arguments and dispatch to ``_evaluate`` per model."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--model", required=True, help="A model_tag or 'all'.")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--per-class-train", type=int, default=None,
                   help="Override gallery size (per-class). Defaults to configs/models.yaml.")
    p.add_argument("--val-subset", type=int, default=None,
                   help="Truncate validation split (smoke testing).")
    p.add_argument("--k", type=int, default=None, help="k-NN neighbours; defaults from config.")
    p.add_argument("--use-cached", action="store_true",
                   help="Skip extraction when features/<tag>/ already exist.")
    p.add_argument("--cache-features", action="store_true",
                   help="Persist extracted features to features/<tag>/ for re-use.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    all_cfg = _load_config()
    defaults = all_cfg["_defaults"]
    if args.per_class_train is None:
        args.per_class_train = defaults.get("per_class_train", 100)
    if args.val_subset is None:
        args.val_subset = defaults.get("val_subset")

    tags = [t for t in all_cfg if not t.startswith("_")] if args.model == "all" else [args.model]
    for tag in tags:
        if tag not in all_cfg:
            raise SystemExit(f"Unknown model_tag {tag!r}. Choices: {sorted(t for t in all_cfg if not t.startswith('_'))}")
        _evaluate(tag, all_cfg, args, args.device)


if __name__ == "__main__":
    main()
