"""Keypoint_Matching entry point — PCK@α on SPair-71K.

For each backbone registered in ``configs/models.yaml``, this script:

1. Loads all (or a subset of) SPair-71K test pairs.
2. Extracts (C, H, W) feature maps for both images via
   ``shared_utils.feature_extraction.build_extractor``.
3. Runs the cosine-NN PCK matcher (``evaluation/src/pck.py``) and
   aggregates per-pair scores into a single per-(model, resolution) row
   in ``evaluation/results/keypoint_matching.csv``.

Two-component fusions (``ddn``, ``sd+dinov2-vitb``, ``sd+dinov2-vitg``)
are composed at runtime via ``fuse_concat`` over the bilinear-aligned
component feature maps.

Usage::

    python run_eval.py --model dinov3                      # full test set
    python run_eval.py --model cd --layer-ablation         # CD per-layer
    python run_eval.py --model all                         # sweep all 11
    python run_eval.py --model dinov3 --categories aeroplane --limit 5
                                                           # smoke test
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

import torch
import torch.nn.functional as F
import yaml
from PIL import Image

_THIS = Path(__file__).resolve()
_EXPERIMENTS = _THIS.parents[1]
if str(_EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS))

from shared_utils.feature_extraction import (                                  # noqa: E402
    build_extractor, build_transform, fuse_concat,
)
from shared_utils.paths import DATASETS_ROOT                                    # noqa: E402

from evaluation.src.layer_ablation import run_layer_ablation, subsample_pairs   # noqa: E402
from evaluation.src.pairs import (                                              # noqa: E402
    PairMeta, load_pairs, transform_keypoints,
)
from evaluation.src.pck import aggregate, match_keypoints, pck_at_alpha         # noqa: E402


_HERE = _THIS.parent
_CONFIG = _HERE / "configs" / "models.yaml"
_RESULTS_CSV = _HERE / "evaluation" / "results" / "keypoint_matching.csv"
_RESULTS_DIR = _HERE / "evaluation" / "results"
_LAYERS_DIR = _HERE / "evaluation" / "results" / "layers"
_SPAIR_ROOT_DEFAULT = (
    DATASETS_ROOT / "Existing_Datasets" / "Keypoint_Matching" / "SPair-71K" / "spair-71k"
)

_CSV_COLUMNS = ("model", "resolution", "pck@0.1", "pck@0.05", "pck@0.01")


def _load_config() -> dict:
    """Parse ``configs/models.yaml`` into a flat dict keyed by model_tag."""
    spec = yaml.safe_load(_CONFIG.read_text())
    flat: dict[str, dict] = {}
    for group in ("baselines", "trained"):
        flat.update(spec.get(group, {}) or {})
    flat.update(spec.get("fusion", {}) or {})
    flat["_defaults"] = spec.get("defaults", {}) or {}
    return flat


def _extract_map(extractor, transform, image_path: Path, device: str) -> torch.Tensor:
    """Extract a single ``(C, H, W)`` spatial feature map for one image."""
    img = transform(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
    try:
        out = extractor.extract(img, prompt="A photo")
    except TypeError:
        out = extractor.extract(img)
    if out.get("patch_tokens") is not None:
        return out["patch_tokens"][0]
    if out.get("per_layer"):
        # Bilinear-align each per-layer map to the largest grid and concat.
        layers = out["per_layer"]
        H = max(f.shape[-2] for f in layers.values())
        W = max(f.shape[-1] for f in layers.values())
        parts = []
        for _, f in sorted(layers.items()):
            if f.shape[-2:] != (H, W):
                f = F.interpolate(f, size=(H, W), mode="bilinear", align_corners=False)
            parts.append(F.normalize(f, dim=1))
        return torch.cat(parts, dim=1)[0]
    raise RuntimeError("Extractor produced no spatial features for keypoint matching.")


def _fused_map(
    component_extractors: list[tuple], image_path: Path, device: str, weights: Sequence[float],
) -> torch.Tensor:
    """Compose a fused ``(C, H, W)`` feature map via ``fuse_concat``."""
    maps = [
        _extract_map(ex, tfm, image_path, device).unsqueeze(0)
        for ex, tfm in component_extractors
    ]
    target_h = max(m.shape[-2] for m in maps)
    target_w = max(m.shape[-1] for m in maps)
    fused = fuse_concat(maps, list(weights), target_grid=(target_h, target_w))
    return fused[0]


def _build_for_tag(tag: str, all_cfg: dict, device: str):
    """Return ``(extract_fn, transform_canvas)`` for the given model_tag.

    ``extract_fn`` accepts an image path and returns a ``(C, H, W)``
    tensor. ``transform_canvas`` is the integer pad-canvas side used
    when remapping keypoints.
    """
    cfg = all_cfg[tag]
    canvas = cfg["transform"]["input_size"]
    strategy = cfg["transform"]["strategy"]

    if "components" in cfg:
        component_extractors = []
        for comp in cfg["components"]:
            sub = all_cfg[comp["tag"]]
            ex = build_extractor(sub["backbone"], device=device,
                                 extractor_kwargs=sub.get("extractor", {}) or {})
            tfm = build_transform(sub["backbone"], canvas, strategy)
            component_extractors.append((ex, tfm))
        weights = [c.get("weight", 1.0) for c in cfg["components"]]

        def extract(path: Path) -> torch.Tensor:
            return _fused_map(component_extractors, path, device, weights)

        return extract, canvas

    extractor = build_extractor(cfg["backbone"], device=device,
                                extractor_kwargs=cfg.get("extractor", {}) or {})
    transform = build_transform(cfg["backbone"], canvas, strategy)

    def extract(path: Path) -> torch.Tensor:
        return _extract_map(extractor, transform, path, device)

    return extract, canvas


def _score_pair(
    pair: PairMeta, extract_fn, canvas: int, alphas: Sequence[float],
) -> dict[float, tuple[int, int]] | None:
    """Per-pair (correct, total) keypoint counts at each α. ``None`` if no visible kps."""
    src_feat = extract_fn(pair.src_path)
    tgt_feat = extract_fn(pair.tgt_path)
    device_kp = src_feat.device
    src_kps = transform_keypoints(pair.src_kps, *pair.src_size, canvas).to(device_kp)
    tgt_kps = transform_keypoints(pair.tgt_kps, *pair.tgt_size, canvas).to(device_kp)
    visible = src_kps[:, 2] * tgt_kps[:, 2] > 0
    if not visible.any():
        return None
    # bbox_max → canvas-pixel space (matches keypoints, mirrors
    # eval_spair.py:111 `bbox * trg_scale`).
    tgt_scale = canvas / max(pair.tgt_size)
    bbox_max = pair.tgt_bbox_max * tgt_scale
    pred = match_keypoints(src_feat, tgt_feat, src_kps[visible], canvas)
    return pck_at_alpha(pred, tgt_kps[visible, :2], bbox_max, alphas)


def _score_by_category(
    pairs: Sequence[PairMeta], extract_fn, canvas: int, alphas: Sequence[float],
) -> dict[str, dict[float, float]]:
    """Run PCK per pair and group the resulting counts by category."""
    by_cat: dict[str, list] = {}
    for pair in pairs:
        row = _score_pair(pair, extract_fn, canvas, alphas)
        if row is not None:
            by_cat.setdefault(pair.category, []).append(row)
    return {cat: aggregate(rows) for cat, rows in by_cat.items()}


def _overall_macro(by_cat: dict[str, dict[float, float]], alphas: Sequence[float]) -> dict[float, float]:
    """Macro-average per-category PCK — the standard SPair-71K protocol."""
    out: dict[float, float] = {}
    for a in alphas:
        vals = [v[a] for v in by_cat.values() if a in v and v[a] == v[a]]
        out[a] = float(sum(vals) / len(vals)) if vals else float("nan")
    return out


def _update_csv(row: dict) -> None:
    """Upsert one row keyed by ``(model, resolution)`` in the headline CSV."""
    _RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    if _RESULTS_CSV.exists():
        rows = list(csv.DictReader(_RESULTS_CSV.open()))
    rows = [r for r in rows if (r["model"], r["resolution"]) != (row["model"], str(row["resolution"]))]
    rows.append(row)
    rows.sort(key=lambda r: (r["model"], int(r["resolution"])))
    with _RESULTS_CSV.open("w") as f:
        w = csv.DictWriter(f, fieldnames=_CSV_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in _CSV_COLUMNS})


def _evaluate(tag: str, all_cfg: dict, args: argparse.Namespace, device: str) -> None:
    """Run full-test (or subset) PCK eval for one model_tag."""
    print(f"\n=== {tag} ===")
    extract_fn, canvas = _build_for_tag(tag, all_cfg, device)
    pairs = load_pairs(args.spair_root, split=args.split, category=None)
    if args.categories:
        pairs = [p for p in pairs if p.category in args.categories]
    if args.n_per_cat and not args.layer_ablation:
        pairs = subsample_pairs(pairs, args.n_per_cat)
    if args.limit:
        pairs = pairs[: args.limit]
    if not pairs:
        raise SystemExit("No pairs to evaluate after filtering.")

    alphas = all_cfg["_defaults"].get("alphas", [0.1, 0.05, 0.01])
    by_cat = _score_by_category(pairs, extract_fn, canvas, alphas)
    pck = _overall_macro(by_cat, alphas)

    row = {
        "model": tag,
        "resolution": canvas,
        "pck@0.1": f"{100 * pck[0.1]:.2f}",
        "pck@0.05": f"{100 * pck[0.05]:.2f}",
        "pck@0.01": f"{100 * pck[0.01]:.2f}",
    }
    _update_csv(row)
    print(f"  PCK@0.1={row['pck@0.1']}  PCK@0.05={row['pck@0.05']}  PCK@0.01={row['pck@0.01']}")

    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _RESULTS_DIR / f"{tag}.json"
    out_path.write_text(json.dumps({
        "model": tag,
        "resolution": canvas,
        "overall": {f"pck@{a}": pck[a] for a in alphas},
        "per_category": {cat: {f"pck@{a}": scores[a] for a in alphas} for cat, scores in by_cat.items()},
    }, indent=2))


def _layer_ablation(tag: str, all_cfg: dict, args: argparse.Namespace, device: str) -> None:
    """Run a per-layer PCK sweep (only meaningful for diffusion backbones)."""
    cfg = all_cfg[tag]
    canvas = cfg["transform"]["input_size"]
    extractor = build_extractor(cfg["backbone"], device=device,
                                extractor_kwargs=cfg.get("extractor", {}) or {})
    transform = build_transform(cfg["backbone"], canvas, cfg["transform"]["strategy"])

    def extract_per_layer(path: Path) -> dict[int, torch.Tensor]:
        img = transform(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
        try:
            out = extractor.extract(img, prompt="A photo")
        except TypeError:
            out = extractor.extract(img)
        if not out.get("per_layer"):
            raise SystemExit(f"--layer-ablation requires per-layer outputs (got {tag!r}).")
        return {idx: f[0] for idx, f in out["per_layer"].items()}

    pairs = load_pairs(args.spair_root, split=args.split, category=None)
    pairs = subsample_pairs(pairs, args.n_per_cat)
    pairs = [PairMeta(
        **{**p.__dict__,
           "src_kps": transform_keypoints(p.src_kps, *p.src_size, canvas),
           "tgt_kps": transform_keypoints(p.tgt_kps, *p.tgt_size, canvas)},
    ) for p in pairs]

    out = run_layer_ablation(extract_per_layer, pairs, canvas)
    _LAYERS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _LAYERS_DIR / f"{tag}_layers_res{canvas}_n{args.n_per_cat}.json"
    out_path.write_text(json.dumps({
        "model": tag, "resolution": canvas, "n_per_cat": args.n_per_cat,
        "per_layer": {str(idx): {f"pck@{a}": v for a, v in scores.items()}
                      for idx, scores in out.items()},
    }, indent=2))
    print(f"  wrote {out_path}")


def main() -> None:
    """CLI entry: parse arguments and dispatch."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--model", required=True, help="A model_tag or 'all'.")
    p.add_argument("--spair-root", type=Path, default=_SPAIR_ROOT_DEFAULT,
                   help="Path to the unpacked SPair-71K dataset.")
    p.add_argument("--split", default="test", choices=("trn", "val", "test"))
    p.add_argument("--categories", nargs="*", default=None,
                   help="Restrict to one or more SPair categories (smoke testing).")
    p.add_argument("--limit", type=int, default=None,
                   help="Cap to N pairs after filtering.")
    p.add_argument("--layer-ablation", action="store_true",
                   help="Diffusion-only: report PCK per U-Net hook layer.")
    p.add_argument("--n-per-cat", type=int, default=None,
                   help="Subsample to N pairs per category (use 20 for fast sweep,"
                        " None for the full split).")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    all_cfg = _load_config()
    tags = [t for t in all_cfg if not t.startswith("_")] if args.model == "all" else [args.model]
    for tag in tags:
        if tag not in all_cfg:
            raise SystemExit(f"Unknown model_tag {tag!r}.")
        if args.layer_ablation:
            _layer_ablation(tag, all_cfg, args, args.device)
        else:
            _evaluate(tag, all_cfg, args, args.device)


if __name__ == "__main__":
    main()
