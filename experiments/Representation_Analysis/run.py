"""Representation_Analysis entry point.

Subcommands:

    activation-maps   Render PCA(3) -> RGB activation maps for one or more
                      (image, model) pairs.
    metrics           Compute CKA and quality metrics; write the result CSVs.
    plots             Render figures from the committed result CSVs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

# Make the experiments tree importable when run as a script.
_THIS = Path(__file__).resolve()
_EXPERIMENTS = _THIS.parents[1]
if str(_EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS))

from shared_utils.paths import REPO_ROOT, DATASETS_ROOT, CHECKPOINTS_ROOT, FEATURES_ROOT

_HERE = _THIS.parent
_CONFIGS = _HERE / "configs"
_QUAL = _HERE / "qualitative"
_QUANT_GLOBAL = _HERE / "quantitative" / "global"
_QUANT_PATCH = _HERE / "quantitative" / "patch"


# ─── plots: render figures from the committed result CSVs. ────────────────


# Plot labels keyed by the canonical CSV `representation` / `pair` strings.
GLOBAL_LABEL = {
    "dino(cls)":   r"DN$_g$",
    "dino(mean)":  r"DN$_{\bar{p}}$",
    "cd(2+5+8)":   r"CD$_{\bar{p}}$",
    "sd(2+5+8)":   r"SD$_{\bar{p}}$",
    "clip(image)": r"CLIP$_g$",
    "ddn_g":       r"DDN",
    "fused":       r"TDDN",
    "vith":        r"TDN",
}

PATCH_LABEL = {
    "dino_p":          r"DN$_p$",
    "cd_p":            r"CD$_p$",
    "sd_p":            r"SD$_p$",
    "clip_p":          r"CLIP$_p$",
    "fused_p":         r"DDN$_p$",
    "fused_trained_p": r"TDDN$_p$",
    "vith_p":          r"TDN$_p$",
}

GLOBAL_PAIR_LABEL = {
    "dino(cls) ↔ dino(mean)":  r"DN$_g$ : DN$_{\bar{p}}$",
    "dino(cls) ↔ cd(2+5+8)":   r"DN$_g$ : CD$_{\bar{p}}$",
    "dino(cls) ↔ DDN_g":       r"DN$_g$ : DDN",
    "dino(mean) ↔ cd(2+5+8)":  r"DN$_{\bar{p}}$ : CD$_{\bar{p}}$",
    "dino(mean) ↔ DDN_g":      r"DN$_{\bar{p}}$ : DDN",
    "cd(2+5+8) ↔ DDN_g":       r"CD$_{\bar{p}}$ : DDN",
    "fused ↔ dino(cls)":       r"TDDN : DN$_g$",
    "fused ↔ dino(mean)":      r"TDDN : DN$_{\bar{p}}$",
    "fused ↔ cd(2+5+8)":       r"TDDN : CD$_{\bar{p}}$",
    "vith ↔ dino(cls)":        r"TDN : DN$_g$",
    "vith ↔ dino(mean)":       r"TDN : DN$_{\bar{p}}$",
}

PATCH_PAIR_LABEL = {
    "dino_p ↔ fused_p":         r"DN$_p$ : DDN$_p$",
    "dino_p ↔ fused_trained_p": r"DN$_p$ : TDDN$_p$",
    "dino_p ↔ vith_p":          r"DN$_p$ : TDN$_p$",
    "cd_p ↔ fused_p":           r"CD$_p$ : DDN$_p$",
    "cd_p ↔ fused_trained_p":   r"CD$_p$ : TDDN$_p$",
    "cd_p ↔ vith_p":            r"CD$_p$ : TDN$_p$",
}

# Bar fill colour by representation family.
FAMILY_COLOR = {
    "DN":   "#1f77b4",
    "CD":   "#ff7f0e",
    "SD":   "#d62728",
    "CLIP": "#bcbd22",
    "TDDN": "#2ca02c",
    "TDN":  "#9467bd",
    "DDN":  "#e377c2",
}

# Row ordering used by both global and patch quality plots.
GLOBAL_ORDER = ["dino(cls)", "dino(mean)", "cd(2+5+8)", "sd(2+5+8)",
                "clip(image)", "ddn_g", "fused", "vith"]
PATCH_ORDER = ["dino_p", "cd_p", "sd_p", "clip_p",
               "fused_p", "fused_trained_p", "vith_p"]


def _family_of(name: str) -> str:
    """Map a CSV `representation` token to a FAMILY_COLOR key."""
    if name in ("fused", "fused_trained_p"):  return "TDDN"
    if name in ("vith",  "vith_p"):           return "TDN"
    if name in ("fused_p", "ddn_g"):          return "DDN"
    if name.startswith("dino"):               return "DN"
    if name.startswith("cd"):                 return "CD"
    if name.startswith("sd"):                 return "SD"
    if name.startswith("clip"):               return "CLIP"
    raise ValueError(f"unknown representation: {name!r}")


def _apply_serif_style(mpl_module) -> None:
    """Set Matplotlib rcParams for paper-style serif figures with dashed gridlines."""
    mpl_module.rcParams.update({
        "font.family":       "serif",
        "font.size":         16,
        "axes.titlesize":    16,
        "axes.labelsize":    18,
        "xtick.labelsize":   18,
        "ytick.labelsize":   18,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.grid":         True,
        "axes.axisbelow":    True,
        "grid.linestyle":    "--",
        "grid.linewidth":    0.8,
        "grid.alpha":        0.55,
        "axes.linewidth":    1.0,
        "pdf.fonttype":      42,
        "ps.fonttype":       42,
    })


def _plot_quality_panel(values, labels, colors, metric, out_stem):
    """One-metric quality bar plot — uniformity or effective_rank."""
    import matplotlib.pyplot as plt
    from matplotlib.transforms import ScaledTranslation

    n = len(labels)
    fig_w = max(7.0, 0.95 * n + 1.5)
    fig, ax = plt.subplots(figsize=(fig_w, 4.7))
    ax.bar(labels, values, color=colors, width=0.7, zorder=2)
    if metric == "uniformity":
        ax.set_ylabel("Uniformity (log)", fontsize=24)
        ax.set_ylim(top=0.0, bottom=-5)
    else:
        ax.set_ylabel("Effective Rank", fontsize=24)
        ax.set_ylim(bottom=0, top=600)
    ax.set_xticks(np.arange(n))
    ax.set_xticklabels(labels, fontsize=24, rotation=45, ha="right",
                       rotation_mode="anchor")
    shift_pt = 8 if metric == "effective_rank" else 20
    offset = ScaledTranslation(shift_pt / 72, 0, fig.dpi_scale_trans)
    for lab in ax.get_xticklabels():
        lab.set_transform(lab.get_transform() + offset)
    ax.tick_params(axis="y", labelsize=24)
    ax.grid(axis="y", linestyle="--", linewidth=0.8, alpha=0.55)
    ax.grid(axis="x", visible=False)
    plt.tight_layout()
    fig.savefig(f"{out_stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{out_stem}.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {out_stem}.pdf / .png")


def _plot_similarity_bars(pairs, linear_cka, pwcca, label_map, out_stem):
    """Grouped CKA + PWCCA bars (oxblood hatch + navy solid)."""
    import matplotlib.pyplot as plt
    from matplotlib.transforms import ScaledTranslation

    labels = [label_map[p] for p in pairs]
    color_cka, color_pwcca = "#8b3a3a", "#1f3a5f"

    n = len(pairs)
    x = np.arange(n)
    width = 0.40
    fig, ax = plt.subplots(figsize=(max(8.0, 0.85 * n + 1.5), 5.0))
    ax.bar(x - width / 2, linear_cka, width, color=color_cka,
           hatch="///", edgecolor="white", linewidth=0.0,
           label="Linear CKA", zorder=2)
    ax.bar(x + width / 2, pwcca, width, color=color_pwcca,
           label="PWCCA", zorder=2)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=20, rotation=45, ha="right",
                       rotation_mode="anchor")
    offset = ScaledTranslation(20 / 72, 0, fig.dpi_scale_trans)
    for lab in ax.get_xticklabels():
        lab.set_transform(lab.get_transform() + offset)
    ax.tick_params(axis="y", labelsize=20)
    ax.set_ylabel("Similarity", fontsize=20)
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", linestyle="--", linewidth=0.8, alpha=0.55)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.22),
              ncol=2, frameon=False, fontsize=20)
    plt.tight_layout()
    fig.savefig(f"{out_stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{out_stem}.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {out_stem}.pdf / .png")


def _order_and_label(df, order, label_map):
    """Reindex `df` by ``order`` and produce (labels, family-colours, df) tuples."""
    df = df.set_index("representation").reindex(order).reset_index().dropna(
        subset=["representation"])
    labels = [label_map[r] for r in df["representation"]]
    colors = [FAMILY_COLOR[_family_of(r)] for r in df["representation"]]
    return df, labels, colors


def cmd_plots(args) -> None:
    """Render the result-CSV-backed plots."""
    import matplotlib
    _apply_serif_style(matplotlib)

    want_global  = args.global_ or not (args.global_ or args.patch)
    want_patch   = args.patch   or not (args.global_ or args.patch)
    want_quality = args.quality or not (args.quality or args.similarity)
    want_sim     = args.similarity or not (args.quality or args.similarity)

    if want_global and want_quality:
        df, labels, colors = _order_and_label(
            pd.read_csv(_QUANT_GLOBAL / "results" / "global_quality.csv"),
            GLOBAL_ORDER, GLOBAL_LABEL)
        _plot_quality_panel(df["uniformity"].values, labels, colors, "uniformity",
                            str(_QUANT_GLOBAL / "plots" / "global_uniformity"))
        _plot_quality_panel(df["effective_rank"].values, labels, colors, "effective_rank",
                            str(_QUANT_GLOBAL / "plots" / "global_effective_rank"))

    if want_global and want_sim:
        df = pd.read_csv(_QUANT_GLOBAL / "results" / "global_similarity.csv")
        _plot_similarity_bars(df["pair"].tolist(),
                              df["linear_cka"].values, df["pwcca"].values,
                              GLOBAL_PAIR_LABEL,
                              str(_QUANT_GLOBAL / "plots" / "global_similarity_bars"))

    if want_patch and want_quality:
        df, labels, colors = _order_and_label(
            pd.read_csv(_QUANT_PATCH / "results" / "patch_quality.csv"),
            PATCH_ORDER, PATCH_LABEL)
        _plot_quality_panel(df["uniformity"].values, labels, colors, "uniformity",
                            str(_QUANT_PATCH / "plots" / "patch_uniformity"))
        _plot_quality_panel(df["effective_rank"].values, labels, colors, "effective_rank",
                            str(_QUANT_PATCH / "plots" / "patch_effective_rank"))

    if want_patch and want_sim:
        df = pd.read_csv(_QUANT_PATCH / "results" / "patch_similarity.csv")
        _plot_similarity_bars(df["pair"].tolist(),
                              df["linear_cka"].values, df["pwcca"].values,
                              PATCH_PAIR_LABEL,
                              str(_QUANT_PATCH / "plots" / "patch_similarity_bars"))


# ─── activation-maps: dispatch render_one per configs/models.yaml ────────────


def _load_models_config() -> dict:
    return yaml.safe_load((_CONFIGS / "models.yaml").read_text())


def _load_activation_maps_config() -> dict:
    return yaml.safe_load((_CONFIGS / "activation_maps.yaml").read_text())


def _output_path_for(model_tag: str, image_stem: str, mode: str, target_size: int,
                     mcfg: dict) -> Path:
    """Build the output path for a given model and image."""
    if model_tag in mcfg["baselines"]:
        group = mcfg["baselines"][model_tag]["group"]
        leaf = model_tag
    elif model_tag in mcfg["trained"]:
        group = mcfg["trained"][model_tag]["group"]
        leaf = {"tdn": "dinov3+roberta",
                "tddn": "dinov3+cd+roberta"}.get(group, group)
    elif model_tag in mcfg["fusion"]:
        entry = mcfg["fusion"][model_tag]
        group = entry["group"]
        leaf = entry["leaf_dir"]
    else:
        raise KeyError(f"unknown model tag {model_tag!r}; check configs/models.yaml")
    return _QUAL / group / "activation-maps" / leaf / f"{image_stem}.png"


def _render_one(model_tag: str, image: Path, mcfg: dict, render_kwargs: dict) -> Path:
    """Resolve per-model kwargs and dispatch to pca_viz.render.render_one."""
    from pca_viz.render import render_one as _do_render

    target_size = render_kwargs["target_size"]
    mode = render_kwargs["mode"]
    out = _output_path_for(model_tag, image.stem, mode, target_size, mcfg)

    if model_tag in mcfg["baselines"]:
        entry = mcfg["baselines"][model_tag]
        models, weights = [entry["backbone"]], None
    elif model_tag in mcfg["trained"]:
        entry = mcfg["trained"][model_tag]
        models, weights = [entry["backbone"]], None
    elif model_tag in mcfg["fusion"]:
        entry = mcfg["fusion"][model_tag]
        models = [c["backbone"] for c in entry["components"]]
        weights = [c["weight"] for c in entry["components"]]
    else:
        raise KeyError(model_tag)

    transform_cfg = entry["transform"]
    call_kwargs = dict(render_kwargs)
    if call_kwargs.get("input_size") is None:
        call_kwargs["input_size"] = transform_cfg["input_size"]

    return _do_render(
        image=image,
        models=models,
        weights=weights,
        output=out,
        strategy=transform_cfg["strategy"],
        **call_kwargs,
    )


def cmd_activation_maps(args) -> None:
    """Render PCA(3) -> RGB activation maps for one or all (image, model) pairs."""
    mcfg = _load_models_config()
    acfg = _load_activation_maps_config()
    # `input_size=None` lets each model's transform.input_size be the default.
    render_kwargs = {
        "input_size": args.input_size,
        "target_size": args.target_size or acfg["render"]["target_size"],
        "mode": args.mode or acfg["render"]["mode"],
        "normalize": acfg["render"]["normalize"],
        "interp": acfg["render"]["interp"],
        "prompt": args.prompt,
    }
    if args.cd_pca_dim is not None:
        render_kwargs["cd_pca_dim"] = args.cd_pca_dim
    if args.normalize_per_layer:
        render_kwargs["normalize_per_layer"] = True

    all_models = list(mcfg["baselines"]) + list(mcfg["trained"]) + list(mcfg["fusion"])
    if args.model == "all":
        models = acfg["models"]
    else:
        models = [args.model]
    for m in models:
        if m not in all_models:
            raise SystemExit(f"unknown model {m!r}; valid: {all_models}")

    if args.image == "all":
        samples_dir = _QUAL / "samples"
        images = [samples_dir / name for name in acfg["images"]]
    else:
        images = [Path(args.image)]
    images = [im for im in images if im.exists()]
    if not images:
        raise SystemExit("no input images found; place samples under qualitative/samples/")

    for image in images:
        for tag in models:
            print(f"\n[activation-maps] {tag} <- {image.name}")
            _render_one(tag, image, mcfg, render_kwargs)


# ─── metrics: extract features + fit PCA + compute CKA/quality + write CSVs ─


def cmd_metrics(args) -> None:
    """Extract features for the configured models, compute CKA + quality
    metrics, and write the result CSVs under
    ``quantitative/{global,patch}/results/``.

    Pipeline:

      1. Load `configs/metrics.yaml` + `configs/models.yaml` +
         `configs/coco_sample_ids.csv`.
      2. For each model in the extraction set, call
         ``metrics.extract.extract_features`` to populate
         ``$EXPERIMENTS_FEATURES_ROOT/<layer>/val/<stem>.npy``.
         (Idempotent: stems already on disk are skipped.)
      3. Call ``metrics.orchestrate.compute_global`` /
         ``compute_patch`` to build the published representations and
         compute uniformity / effective-rank / linear-CKA / PWCCA.
      4. Write the four output CSVs.

    Flags select scope:
      ``--global`` / ``--patch``           which pipelines to run
      ``--similarity`` / ``--quality``     which metric families to emit
                                            (default: both when scope is on)
    """
    import logging

    from metrics.extract import extract_features
    from metrics.orchestrate import compute_global, compute_patch

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(message)s")

    cfg = yaml.safe_load((_CONFIGS / "metrics.yaml").read_text())
    models_cfg = yaml.safe_load((_CONFIGS / "models.yaml").read_text())
    sample_ids = pd.read_csv(_CONFIGS / "coco_sample_ids.csv")["image_stem"].tolist()
    if args.limit is not None:
        sample_ids = sample_ids[: args.limit]

    if not (args.global_ or args.patch):
        raise SystemExit("[metrics] specify at least --global or --patch")
    if not (args.similarity or args.quality):
        # Default: emit both metric families.
        args.similarity = args.quality = True

    coco_dir = DATASETS_ROOT / "Existing_Datasets/Vision_Language_Alignment/MS-COCO-2014/val2014"
    if not coco_dir.is_dir():
        raise SystemExit(f"[metrics] COCO val2014 not found at {coco_dir}")

    print(f"[metrics] n_samples={len(sample_ids)} pca_dim={cfg['pca_dim']} "
          f"spatial_size={cfg['spatial_size']}")
    print(f"[metrics] features cache: {FEATURES_ROOT}")

    # ── Extract features for every model we need on disk ─────────────────
    # ``ddn-cd`` is composed from dinov3 + cd later, so it isn't extracted
    # standalone here.
    extract_tags = ["dinov3", "cd", "sd-2.1", "clip", "tdn", "tddn"]
    for tag in extract_tags:
        for group_key in ("baselines", "trained"):
            group = models_cfg.get(group_key, {}) or {}
            if tag in group:
                extract_features(
                    tag=tag,
                    model_entry=group[tag],
                    image_stems=sample_ids,
                    coco_dir=coco_dir,
                    features_root=FEATURES_ROOT,
                    device="cuda",
                )
                break
        else:
            raise SystemExit(f"[metrics] model tag {tag!r} not found in models.yaml")

    # ── Compose representations + compute metrics ─────────────────────────
    quality_subsample_global = None                # use full N=2000 rows
    quality_subsample_patch = int(cfg.get("uniformity_subsample", 10000))

    def _resolve_out(scope: str, kind: str) -> Path:
        """Choose the destination CSV path; respect --out-root override."""
        if args.out_root is not None:
            return args.out_root / f"{scope}_{kind}.csv"
        parent = _QUANT_GLOBAL if scope == "global" else _QUANT_PATCH
        return parent / "results" / f"{scope}_{kind}.csv"

    if args.global_:
        qg, sg, cd_reducers, sd_reducers = compute_global(
            FEATURES_ROOT, sample_ids,
            pca_dim=int(cfg["pca_dim"]),
            target=int(cfg["spatial_size"]),
            uniformity_subsample=quality_subsample_global,
        )
        if args.quality:
            out = _resolve_out("global", "quality")
            out.parent.mkdir(parents=True, exist_ok=True)
            qg.to_csv(out, index=False)
            print(f"[metrics] wrote {out}")
        if args.similarity:
            out = _resolve_out("global", "similarity")
            out.parent.mkdir(parents=True, exist_ok=True)
            sg.to_csv(out, index=False)
            print(f"[metrics] wrote {out}")
    else:
        cd_reducers = sd_reducers = None

    if args.patch:
        if cd_reducers is None or sd_reducers is None:
            # Patch reps need PCA bases fit on the global features for
            # consistency with the published pipeline. Build them even
            # when --patch is asked for alone.
            _, _, cd_reducers, sd_reducers = compute_global(
                FEATURES_ROOT, sample_ids,
                pca_dim=int(cfg["pca_dim"]),
                target=int(cfg["spatial_size"]),
                uniformity_subsample=quality_subsample_global,
            )
        qp, sp = compute_patch(
            FEATURES_ROOT, sample_ids,
            cd_reducers=cd_reducers, sd_reducers=sd_reducers,
            n_subsample=int(cfg["n_patches_subsample"]),
            pca_dim=int(cfg["pca_dim"]),
            target=int(cfg["spatial_size"]),
            uniformity_subsample=quality_subsample_patch,
            seed=int(cfg.get("seed", 42)),
        )
        if args.quality:
            out = _resolve_out("patch", "quality")
            out.parent.mkdir(parents=True, exist_ok=True)
            qp.to_csv(out, index=False)
            print(f"[metrics] wrote {out}")
        if args.similarity:
            out = _resolve_out("patch", "similarity")
            out.parent.mkdir(parents=True, exist_ok=True)
            sp.to_csv(out, index=False)
            print(f"[metrics] wrote {out}")


# ─── CLI entry point ────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    """Top-level argparse with three subcommands: activation-maps, metrics, plots."""
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    am = sub.add_parser("activation-maps",
                        help="Render PCA(3) -> RGB activation maps for "
                             "(image, model) pairs.")
    am.add_argument("--image", required=True,
                    help="Path to one input image, or 'all' to iterate "
                         "configs/activation_maps.yaml.")
    am.add_argument("--model", required=True,
                    help="Model tag from configs/models.yaml, or 'all'.")
    am.add_argument("--input-size", type=int, default=None,
                    help="Override default input resolution (configs/activation_maps.yaml).")
    am.add_argument("--target-size", type=int, default=None,
                    help="Override default output PNG resolution.")
    am.add_argument("--mode", default=None, choices=("patches", "interpolated"),
                    help="Override default render mode.")
    am.add_argument("--prompt", default=None,
                    help="Text prompt for diffusion extraction (cd, sd-2.1, ddn-cd, "
                         "tddn). If omitted, falls back to filename keyword "
                         "(maze/chess/hanoi) or generic 'A photo'.")
    am.add_argument("--cd-pca-dim", type=int, default=None,
                    help="Opt-in: per-image, per-layer PCA reduction for "
                         "diffusion backbones. Default (omit) is a raw layer "
                         "concat so CD palettes stay consistent with DDN.")
    am.add_argument("--normalize-per-layer", action="store_true",
                    help="L2-normalize each diffusion layer along the channel "
                         "axis before concat (applies only with --cd-pca-dim).")
    am.set_defaults(func=cmd_activation_maps)

    me = sub.add_parser("metrics",
                        help="Compute CKA + quality metrics from features.")
    me.add_argument("--global", dest="global_", action="store_true",
                    help="Run global-level analysis.")
    me.add_argument("--patch", action="store_true",
                    help="Run patch-level analysis.")
    me.add_argument("--similarity", action="store_true",
                    help="Compute CKA / PWCCA pairwise similarities.")
    me.add_argument("--quality", action="store_true",
                    help="Compute uniformity + effective rank.")
    me.add_argument("--cache", action="store_true",
                    help="Persist extracted features to $EXPERIMENTS_FEATURES_ROOT.")
    me.add_argument("--limit", type=int, default=None,
                    help="Cap the number of COCO images analyzed (debug).")
    me.add_argument("--out-root", type=Path, default=None,
                    help="Optional override for the output root. When set, "
                         "writes go to ``<out-root>/{global,patch}_{quality,"
                         "similarity}.csv`` instead of clobbering the "
                         "committed paper-canonical CSVs.")
    me.set_defaults(func=cmd_metrics)

    pl = sub.add_parser("plots",
                        help="Re-render the 5 canonical figures from the committed CSVs.")
    pl.add_argument("--global", dest="global_", action="store_true",
                    help="Render only global plots.")
    pl.add_argument("--patch", action="store_true",
                    help="Render only patch plots.")
    pl.add_argument("--quality", action="store_true",
                    help="Render only uniformity + effective_rank.")
    pl.add_argument("--similarity", action="store_true",
                    help="Render only the similarity-bars plot.")
    pl.set_defaults(func=cmd_plots)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
