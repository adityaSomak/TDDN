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


_QUALITY_LABEL_MAP = {
    "DN_g":  r"DN$_{g}$",
    "DN_p":  r"DN$_{p}$",
    "CD_p":  r"CD$_{p}$",
    "DDN_g": r"DDN$_{g}$",
    "DN":  "DN",
    "CD":  "CD",
    "DDN": "DDN",
}


def _pair_to_label(pair: str) -> str:
    """Format ``A<->B`` as the LaTeX bidirectional arrow used in the figures."""
    a, b = pair.split("<->")
    return rf"{_QUALITY_LABEL_MAP.get(a, a)}$\leftrightarrow${_QUALITY_LABEL_MAP.get(b, b)}"


def _apply_serif_style(mpl_module) -> None:
    mpl_module.rcParams.update({
        "font.family":        "serif",
        "font.serif":         ["Times New Roman", "DejaVu Serif"],
        "font.size":          13,
        "axes.linewidth":     0.8,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "xtick.major.width":  0.8,
        "ytick.major.width":  0.8,
        "xtick.direction":    "out",
        "ytick.direction":    "out",
        "pdf.fonttype":       42,
        "ps.fonttype":        42,
    })


def _plot_quality_bar(values, labels, ylabel, out_stem, value_fmt=".3f"):
    """Single-metric quality bar plot (uniformity or effective rank)."""
    import matplotlib.pyplot as plt

    figsize = (3.8, 2.8)
    bar_w = 0.8
    cmap = plt.get_cmap("Blues")
    tick_fs = 14
    axis_label_fs = 15
    value_fs = 14

    vmin, vmax = min(values), max(values)
    if vmax == vmin:
        colors = [cmap(0.70)] * len(values)
    else:
        colors = [cmap(0.40 + 0.55 * (v - vmin) / (vmax - vmin)) for v in values]

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(values))
    bars = ax.bar(x, values, width=bar_w, color=colors,
                  edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=tick_fs)
    ax.set_ylabel(ylabel, fontsize=axis_label_fs)

    pos_only = all(v >= 0 for v in values)
    span = max(values) - min(values)
    short_bar_thresh = 0.18 * span if span > 0 else 0

    for bar, val in zip(bars, values):
        if val < 0:
            if abs(val) >= short_bar_thresh:
                ax.text(bar.get_x() + bar.get_width() / 2, val / 2,
                        f"{val:{value_fmt}}", ha="center", va="center",
                        fontsize=value_fs, color="black")
            else:
                ax.text(bar.get_x() + bar.get_width() / 2, val - 0.02 * span,
                        f"{val:{value_fmt}}", ha="center", va="top",
                        fontsize=value_fs, color="black")
        else:
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02 * span,
                    f"{val:{value_fmt}}", ha="center", va="bottom",
                    fontsize=value_fs)

    if not pos_only:
        ax.axhline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.4)
    else:
        ax.set_ylim(0, max(values) * 1.18)

    ax.tick_params(axis="y", labelsize=tick_fs)
    ax.yaxis.set_tick_params(width=0.6)
    ax.xaxis.set_tick_params(width=0)

    fig.subplots_adjust(left=0.22, right=0.96, top=0.95, bottom=0.18)
    fig.savefig(f"{out_stem}.pdf", dpi=300)
    fig.savefig(f"{out_stem}.png", dpi=300)
    plt.close(fig)
    print(f"saved -> {out_stem}.pdf / .png")


def _plot_similarity_bars(linear_cka, pwcca, pair_labels, out_stem):
    """Grouped CKA + PWCCA bars per representation pair."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    cmap = plt.get_cmap("Blues")

    def heat(values, vmin=0.0, vmax=1.0, lo=0.20, hi=0.95):
        return [cmap(lo + (hi - lo) * (v - vmin) / (vmax - vmin)) for v in values]

    tick_fs = 11
    axis_label_fs = 12
    value_fs = 12
    legend_fs = 13
    figsize = (6.8, 3.4)
    bar_w = 0.42

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(pair_labels))

    bars_cka = ax.bar(x - bar_w / 2, linear_cka, width=bar_w,
                      color=heat(linear_cka), edgecolor="white", linewidth=0.6,
                      hatch="//", label="Linear CKA")
    bars_pwcca = ax.bar(x + bar_w / 2, pwcca, width=bar_w,
                        color=heat(pwcca), edgecolor="white", linewidth=0.6,
                        label="PWCCA")

    ax.set_xticks(x)
    ax.set_xticklabels([""] * len(pair_labels))
    label_x_shift = 0.35
    label_y = -0.07
    for xi, txt in zip(x, pair_labels):
        ax.text(xi + label_x_shift, label_y, txt,
                fontsize=tick_fs, rotation=20, ha="right", va="top",
                rotation_mode="anchor")
    ax.set_ylabel("Similarity", fontsize=axis_label_fs)
    ax.set_ylim(0, 1.25)
    ax.tick_params(axis="y", labelsize=tick_fs)
    ax.yaxis.set_tick_params(width=0.6)
    ax.xaxis.set_tick_params(width=0)

    for bars, vals in [(bars_cka, linear_cka), (bars_pwcca, pwcca)]:
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02,
                    f"{val:.2f}", ha="center", va="bottom",
                    fontsize=value_fs)

    legend_handles = [
        mpatches.Patch(facecolor=cmap(0.55), edgecolor="white",
                       hatch="//", label="Linear CKA"),
        mpatches.Patch(facecolor=cmap(0.55), edgecolor="white",
                       label="PWCCA"),
    ]
    ax.legend(handles=legend_handles, fontsize=legend_fs, frameon=False,
              loc="upper center", bbox_to_anchor=(0.5, 1.18),
              ncol=2, handlelength=1.4, handleheight=1.0,
              columnspacing=1.5)

    fig.subplots_adjust(left=0.13, right=0.98, top=0.85, bottom=0.30)
    fig.savefig(f"{out_stem}.pdf", dpi=300, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(f"{out_stem}.png", dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"saved -> {out_stem}.pdf / .png")


def cmd_plots(args) -> None:
    """Render the result-CSV-backed plots."""
    import matplotlib
    _apply_serif_style(matplotlib)

    want_global = args.global_ or not (args.global_ or args.patch)
    want_patch = args.patch or not (args.global_ or args.patch)
    want_quality = args.quality or not (args.quality or args.similarity)
    want_sim = args.similarity or not (args.quality or args.similarity)

    if want_global:
        gq = pd.read_csv(_QUANT_GLOBAL / "results" / "global_quality.csv")
        labels = [_QUALITY_LABEL_MAP.get(r, r) for r in gq["representation"]]
        if want_quality:
            _plot_quality_bar(
                gq["uniformity"].tolist(), labels, "Uniformity (log)",
                str(_QUANT_GLOBAL / "plots" / "global_uniformity"),
                value_fmt=".2f",
            )
            _plot_quality_bar(
                gq["effective_rank"].tolist(), labels, "Effective Rank",
                str(_QUANT_GLOBAL / "plots" / "global_effective_rank"),
                value_fmt=".1f",
            )
        if want_sim:
            gs = pd.read_csv(_QUANT_GLOBAL / "results" / "global_similarity.csv")
            _plot_similarity_bars(
                gs["linear_cka"].tolist(),
                gs["pwcca"].tolist(),
                [_pair_to_label(p) for p in gs["pair"]],
                str(_QUANT_GLOBAL / "plots" / "global_similarity_bars"),
            )

    if want_patch:
        pq = pd.read_csv(_QUANT_PATCH / "results" / "patch_quality.csv")
        labels = [_QUALITY_LABEL_MAP.get(r, r) for r in pq["representation"]]
        if want_quality:
            _plot_quality_bar(
                pq["uniformity"].tolist(), labels, "Uniformity (log)",
                str(_QUANT_PATCH / "plots" / "patch_uniformity"),
                value_fmt=".2f",
            )
            _plot_quality_bar(
                pq["effective_rank"].tolist(), labels, "Effective Rank",
                str(_QUANT_PATCH / "plots" / "patch_effective_rank"),
                value_fmt=".1f",
            )


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
    """Compute CKA + quality metrics and write the result CSVs.

    Live feature extraction is not yet wired in this subcommand. The
    committed CSVs under ``quantitative/{global,patch}/results/`` reproduce
    the published numbers; ``python run.py plots`` regenerates the figures
    from them without invoking this path.
    """
    cfg = yaml.safe_load((_CONFIGS / "metrics.yaml").read_text())
    sample_ids = pd.read_csv(_CONFIGS / "coco_sample_ids.csv")["image_stem"].tolist()
    print(f"[metrics] config: n_samples={cfg['n_samples']}, pca_dim={cfg['pca_dim']}, "
          f"spatial_size={cfg['spatial_size']}")
    print(f"[metrics] sample ids: {len(sample_ids)} (first: {sample_ids[0]})")
    print(f"[metrics] features cache: {FEATURES_ROOT}  exists={FEATURES_ROOT.exists()}")

    raise SystemExit(
        "[metrics] live extraction is not yet implemented; edit the CSVs under "
        "quantitative/{global,patch}/results/ and re-run `python run.py plots` "
        "to refresh the figures."
    )


# ─── CLI entry point ────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
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
