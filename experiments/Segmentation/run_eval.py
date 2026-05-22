"""Segmentation evaluation entry point.

Loads a Lightning checkpoint and reports weighted mIoU and pixel
accuracy on the requested split. Updates the headline CSV at
``evaluation/results/segmentation.csv`` and writes a per-class IoU
breakdown to ``evaluation/results/<model_tag>.json``.

Usage::

    python run_eval.py --model ddn                    # uses training/checkpoints/ddn/best.ckpt
    python run_eval.py --model all                    # sweep all tags whose ckpts exist
    python run_eval.py --model ddn --limit 5          # smoke test
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, Subset

_THIS = Path(__file__).resolve()
_EXPERIMENTS = _THIS.parents[1]
if str(_EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS))

from shared_utils.feature_extraction import build_transform                    # noqa: E402
from shared_utils.paths import DATASETS_ROOT                                    # noqa: E402

from evaluation.src.eval import evaluate                                        # noqa: E402
from training.src.data import PuzzleSegmentationFeatures                        # noqa: E402
from training.src.lightning_module import SegmentationLitModule                 # noqa: E402

_HERE = _THIS.parent
_CONFIG_MODELS = _HERE / "configs" / "models.yaml"
_CONFIG_TRAIN = _HERE / "configs" / "training.yaml"
_CKPT_DIR = _HERE / "training" / "checkpoints"
_RESULTS_DIR = _HERE / "evaluation" / "results"
_RESULTS_CSV = _RESULTS_DIR / "segmentation.csv"
_SEG_ROOT_DEFAULT = DATASETS_ROOT / "Puzzle_Perception" / "Segmentation" / "data"

_CSV_COLUMNS = ("model", "mIoU", "pixel_acc")


def _load_configs() -> tuple[dict, dict]:
    """Parse the model + training YAMLs."""
    models = yaml.safe_load(_CONFIG_MODELS.read_text())
    training = yaml.safe_load(_CONFIG_TRAIN.read_text())
    index: dict[str, dict] = {}
    for group in ("baselines", "trained"):
        index.update(models.get(group, {}) or {})
    index.update(models.get("fusion", {}) or {})
    models["_index"] = index
    models["_default_head"] = models.get("default_head", {"kind": "linear", "hidden_dims": [], "output_size": 512})
    return models, training


def _update_csv(row: dict) -> None:
    """Upsert one row keyed by ``model`` in the headline CSV."""
    _RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(_RESULTS_CSV.open())) if _RESULTS_CSV.exists() else []
    rows = [r for r in rows if r["model"] != row["model"]]
    rows.append(row)
    rows.sort(key=lambda r: r["model"])
    with _RESULTS_CSV.open("w") as f:
        w = csv.DictWriter(f, fieldnames=_CSV_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in _CSV_COLUMNS})


def _evaluate_one(tag: str, checkpoint: Path, all_cfg: dict, training_cfg: dict,
                  args: argparse.Namespace, device: str) -> None:
    """Load one checkpoint, run evaluation, write the CSV row and per-model JSON."""
    model_cfg = all_cfg["_index"][tag]
    head_kwargs = dict(all_cfg["_default_head"])
    transform = build_transform(
        model_cfg.get("backbone", "dinov3-vith16plus"),
        model_cfg["transform"]["input_size"],
        model_cfg["transform"]["strategy"],
    )
    dataset = PuzzleSegmentationFeatures(args.seg_root, args.split, transform,
                                         mask_size=head_kwargs["output_size"])
    if args.limit:
        dataset = Subset(dataset, range(min(args.limit, len(dataset))))
    loader = DataLoader(dataset, batch_size=training_cfg["optim"]["batch_size"],
                        shuffle=False, num_workers=training_cfg["trainer"]["num_workers"])

    classes = dataset.dataset.classes if isinstance(dataset, Subset) else dataset.classes
    pca_path = checkpoint.parent / "pca.pt"
    pca_bases = torch.load(pca_path) if pca_path.exists() else {}
    module = SegmentationLitModule.load_from_checkpoint(
        checkpoint,
        model_tag=tag, model_cfg=model_cfg, model_index=all_cfg["_index"],
        training_cfg=training_cfg,
        n_classes=classes.num_classes, head_kwargs=head_kwargs,
        class_weights=classes.ce_weights(), miou_weights=classes.miou_weights(),
        pca_bases=pca_bases,
    )
    metrics = evaluate(
        module, loader, classes.num_classes,
        miou_weights=classes.miou_weights(),
        ignore_index=training_cfg["loss"]["ignore_index"], device=device,
    )
    row = {"model": tag, "mIoU": f"{100 * metrics['miou']:.2f}",
           "pixel_acc": f"{100 * metrics['pixel_acc']:.2f}"}
    _update_csv(row)
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (_RESULTS_DIR / f"{tag}.json").write_text(json.dumps(metrics, indent=2))
    print(f"  {tag}: mIoU={row['mIoU']}  pixel_acc={row['pixel_acc']}")


def main() -> None:
    """CLI entry: parse arguments and dispatch ``_evaluate_one`` per model_tag."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--model", required=True, help="A model_tag or 'all'.")
    p.add_argument("--checkpoint", type=Path, default=None,
                   help="Lightning .ckpt path. Defaults to training/checkpoints/<tag>/best.ckpt.")
    p.add_argument("--seg-root", type=Path, default=_SEG_ROOT_DEFAULT)
    p.add_argument("--split", default="test", choices=("train", "val", "test"))
    p.add_argument("--limit", type=int, default=None, help="Cap dataset (smoke test).")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    all_cfg, training_cfg = _load_configs()
    tags = list(all_cfg["_index"]) if args.model == "all" else [args.model]
    for tag in tags:
        if tag not in all_cfg["_index"]:
            raise SystemExit(f"Unknown model_tag {tag!r}.")
        ckpt = args.checkpoint if args.checkpoint and len(tags) == 1 else _CKPT_DIR / tag / "best.ckpt"
        if not ckpt.exists():
            print(f"  SKIP {tag}: no checkpoint at {ckpt}")
            continue
        _evaluate_one(tag, ckpt, all_cfg, training_cfg, args, args.device)


if __name__ == "__main__":
    main()
