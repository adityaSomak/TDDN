"""Segmentation training entry point.

Trains a small segmentation head on top of a frozen backbone defined
by ``configs/models.yaml``. Shared training hyperparameters live in
``configs/training.yaml`` and apply uniformly across all model tags.

For diffusion backbones (or fusions containing one), per-layer Global
PCA is fit on a small training subsample before the first training
step. The fitted basis is persisted next to the checkpoint as
``pca.pt`` so the same projection can be applied at eval time.

Usage::

    python run_train.py --model ddn                                # full run
    python run_train.py --model ddn --max-epochs 1                 # smoke
    python run_train.py --model ddn --out-root /tmp/smoke          # sandbox

Logs land in ``<out_root>/logs/<model_tag>/`` and the best checkpoint
is written to ``<out_root>/checkpoints/<model_tag>/best.ckpt``.
``<out_root>`` defaults to ``experiments/Segmentation/training/``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterator

try:
    import lightning.pytorch as pl
except ModuleNotFoundError:
    import pytorch_lightning as pl
import torch
import yaml
from torch.utils.data import DataLoader

_THIS = Path(__file__).resolve()
_EXPERIMENTS = _THIS.parents[1]
if str(_EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS))

from shared_utils.feature_extraction import build_extractor, build_transform  # noqa: E402
from shared_utils.paths import DATASETS_ROOT                                   # noqa: E402

from training.src.data import PuzzleSegmentationFeatures                       # noqa: E402
from training.src.lightning_module import SegmentationLitModule                # noqa: E402
from training.src.pca import fit_layer_pca                                     # noqa: E402

_HERE = _THIS.parent
_CONFIG_MODELS = _HERE / "configs" / "models.yaml"
_CONFIG_TRAIN = _HERE / "configs" / "training.yaml"
_DEFAULT_OUT_ROOT = _HERE / "training"
_SEG_ROOT_DEFAULT = DATASETS_ROOT / "Puzzle_Perception" / "Segmentation" / "data"


def _load_configs() -> tuple[dict, dict]:
    """Parse models + training YAMLs and flatten the model index by tag."""
    models = yaml.safe_load(_CONFIG_MODELS.read_text())
    training = yaml.safe_load(_CONFIG_TRAIN.read_text())
    index: dict[str, dict] = {}
    for group in ("baselines", "trained"):
        index.update(models.get(group, {}) or {})
    index.update(models.get("fusion", {}) or {})
    models["_index"] = index
    models["_default_head"] = models.get(
        "default_head", {"kind": "linear", "hidden_dims": [], "output_size": 512},
    )
    return models, training


def _components_of(model_cfg: dict, tag: str) -> Iterator[dict]:
    """Yield ``{"tag": ..., "weight": ...}`` entries for a single or fused model.

    Single-backbone tags expose themselves as a one-element list so the
    PCA-fit loop can iterate uniformly.
    """
    if "components" in model_cfg:
        yield from model_cfg["components"]
    else:
        yield {"tag": tag, "weight": 1.0}


def _fit_pca_for(
    tag: str, sub_cfg: dict, training_cfg: dict, seg_root: Path,
    head_output_size: int, device: str,
) -> dict[int, dict[str, torch.Tensor]]:
    """Build the component's extractor + dataset and fit per-layer PCA."""
    extractor = build_extractor(
        sub_cfg["backbone"],
        device=device,
        extractor_kwargs=sub_cfg.get("extractor", {}) or {},
        loader_kwargs_override=sub_cfg.get("loader_kwargs", {}) or {},
    )
    transform = build_transform(
        sub_cfg["backbone"],
        sub_cfg["transform"]["input_size"],
        sub_cfg["transform"]["strategy"],
    )
    dataset = PuzzleSegmentationFeatures(
        seg_root, "train", transform, mask_size=head_output_size,
    )
    layers = sub_cfg.get("extractor", {}).get("layers", [2, 5, 8])
    pca_cfg = training_cfg.get("pca", {}) or {}
    print(f"[pca-fit] {tag}: layers={list(layers)} "
          f"n_samples={pca_cfg.get('n_samples', 200)} "
          f"n_components={sub_cfg['pca']['n_components_per_layer']}")
    return fit_layer_pca(
        extractor, dataset, layers,
        n_samples=pca_cfg.get("n_samples", 200),
        n_components=sub_cfg["pca"]["n_components_per_layer"],
        seed=pca_cfg.get("seed", 42),
        device=device,
    )


def main() -> None:
    """CLI entry: parse arguments and launch one Lightning training run."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--model", required=True, help="model_tag from configs/models.yaml.")
    p.add_argument("--config", type=Path, default=_CONFIG_TRAIN,
                   help="Override the training config YAML.")
    p.add_argument("--seg-root", type=Path, default=_SEG_ROOT_DEFAULT,
                   help="Path to the puzzle-perception segmentation data tree.")
    p.add_argument("--out-root", type=Path, default=_DEFAULT_OUT_ROOT,
                   help="Root for checkpoints/<tag>/ and logs/<tag>/ outputs "
                        "(defaults to experiments/Segmentation/training/).")
    p.add_argument("--max-epochs", type=int, default=None,
                   help="Override training.yaml:optim.epochs (smoke testing).")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--devices", default=1,
                   help="Lightning ``Trainer(devices=...)``. Default 1 (single GPU). "
                        "Pass an int (e.g. ``4``) for DDP across N GPUs, or "
                        "``auto`` to use every visible GPU.")
    args = p.parse_args()

    all_cfg, training_cfg = _load_configs()
    if args.config != _CONFIG_TRAIN:
        training_cfg = yaml.safe_load(args.config.read_text())

    if args.model not in all_cfg["_index"]:
        raise SystemExit(f"Unknown model_tag {args.model!r}.")
    model_cfg = all_cfg["_index"][args.model]
    head_kwargs = dict(all_cfg["_default_head"])

    ckpt_dir = args.out_root / "checkpoints" / args.model
    log_dir = args.out_root / "logs" / args.model
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Fit per-layer Global PCA for any component that requests it.
    pca_bases: dict[str, dict] = {}
    for comp in _components_of(model_cfg, args.model):
        sub = all_cfg["_index"].get(comp["tag"])
        if sub is None or "pca" not in sub:
            continue
        pca_bases[comp["tag"]] = _fit_pca_for(
            comp["tag"], sub, training_cfg, args.seg_root,
            head_kwargs["output_size"], args.device,
        )
    if pca_bases:
        torch.save(pca_bases, ckpt_dir / "pca.pt")
        print(f"[pca-fit] saved bases to {ckpt_dir / 'pca.pt'}")

    transform = build_transform(
        model_cfg.get("backbone", "dinov3-vith16plus"),
        model_cfg["transform"]["input_size"],
        model_cfg["transform"]["strategy"],
    )
    train_ds = PuzzleSegmentationFeatures(args.seg_root, "train", transform,
                                          mask_size=head_kwargs["output_size"])
    val_ds = PuzzleSegmentationFeatures(args.seg_root, "val", transform,
                                        mask_size=head_kwargs["output_size"])
    train_loader = DataLoader(train_ds, batch_size=training_cfg["optim"]["batch_size"],
                              shuffle=True, num_workers=training_cfg["trainer"]["num_workers"])
    val_loader = DataLoader(val_ds, batch_size=training_cfg["optim"]["batch_size"],
                            shuffle=False, num_workers=training_cfg["trainer"]["num_workers"])

    classes = train_ds.classes
    module = SegmentationLitModule(
        model_tag=args.model,
        model_cfg=model_cfg,
        model_index=all_cfg["_index"],
        training_cfg=training_cfg,
        n_classes=classes.num_classes,
        head_kwargs=head_kwargs,
        class_weights=classes.ce_weights(),
        miou_weights=classes.miou_weights(),
        pca_bases=pca_bases,
    )

    trainer = pl.Trainer(
        max_epochs=args.max_epochs or training_cfg["optim"]["epochs"],
        precision=training_cfg["trainer"]["precision"],
        val_check_interval=training_cfg["trainer"]["val_check_interval"],
        log_every_n_steps=training_cfg["trainer"]["log_every_n_steps"],
        default_root_dir=str(log_dir),
        callbacks=[pl.callbacks.ModelCheckpoint(
            dirpath=str(ckpt_dir), filename="best",
            monitor="val/miou", mode="max", save_top_k=1, save_last=True,
        )],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=(int(args.devices) if str(args.devices).isdigit() else args.devices),
    )
    trainer.fit(module, train_loader, val_loader)


if __name__ == "__main__":
    main()
