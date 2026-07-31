"""Render alignment + uniformity bar charts from ``results/alignment.csv``.

Mirrors the house style of ``run.py``'s ``_plot_quality_panel`` (serif,
per-family bar colours, 45deg rotated labels) and adds mean±std error bars.
Two settings live in the CSV:

    global  — positives = ImageNet same-class (class-alignment / semantic
              clustering); uniformity on the ImageNet global-vector pool.
    patch   — positives = SPair matched keypoints; uniformity on SPair
              keypoint features.

For each present setting we write four figures to ``plots/``::

    <setting>_alignment.{png,pdf}     alignment_mean ± alignment_std   (lower better)
    <setting>_uniformity.{png,pdf}    uniformity_mean ± uniformity_std (lower better)

Usage::

    python plot_alignment.py                 # both settings, both metrics
    python plot_alignment.py --setting global
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_RESULTS = _HERE / "results" / "alignment.csv"
_PLOTS = _HERE / "plots"

# Bar fill colour by representation family (identical palette to run.py).
FAMILY_COLOR = {
    "DN": "#1f77b4", "CD": "#ff7f0e", "SD": "#d62728", "CLIP": "#bcbd22",
    "TDDN": "#2ca02c", "TDN": "#9467bd", "DDN": "#e377c2",
}

# CSV `representation` token -> LaTeX label (matches run.py notation).
LABEL = {
    # global
    "DNg": r"DN$_g$", "CDpbar": r"CD$_{\bar{p}}$", "DDNg": r"DDN",
    "CLIPg": r"CLIP$_g$", "TDNg": r"TDN", "TDDNg": r"TDDN",
    # patch
    "DNp": r"DN$_p$", "CDp": r"CD$_p$", "DDNp": r"DDN$_p$", "SDp": r"SD$_p$",
    "CLIPp": r"CLIP$_p$", "TDNp": r"TDN$_p$", "TDDNp": r"TDDN$_p$",
}

# Bar order per setting (mirrors GLOBAL_ORDER / PATCH_ORDER family grouping).
ORDER = {
    "global": ["DNg", "CDpbar", "CLIPg", "DDNg", "TDDNg", "TDNg"],
    "patch":  ["DNp", "CDp", "SDp", "CLIPp", "DDNp", "TDDNp", "TDNp"],
}


def _family_of(token: str) -> str:
    """Map a CSV representation token to a FAMILY_COLOR key."""
    if token.startswith("DDN"):  return "DDN"
    if token.startswith("TDDN"): return "TDDN"
    if token.startswith("TDN"):  return "TDN"
    if token.startswith("DN"):   return "DN"
    if token.startswith("CD"):   return "CD"
    if token.startswith("SD"):   return "SD"
    if token.startswith("CLIP"): return "CLIP"
    raise ValueError(f"unknown representation token: {token!r}")


def _apply_serif_style(mpl_module) -> None:
    """Paper-style serif rcParams with dashed gridlines (matches run.py)."""
    mpl_module.rcParams.update({
        "font.family": "serif", "font.size": 16, "axes.titlesize": 16,
        "axes.labelsize": 18, "xtick.labelsize": 18, "ytick.labelsize": 18,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.axisbelow": True, "grid.linestyle": "--",
        "grid.linewidth": 0.8, "grid.alpha": 0.55, "axes.linewidth": 1.0,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def _plot_panel(values, errs, labels, colors, metric, out_stem):
    """One-metric bar plot with mean±std error bars (alignment | uniformity)."""
    import matplotlib.pyplot as plt
    from matplotlib.transforms import ScaledTranslation

    n = len(labels)
    fig_w = max(7.0, 0.95 * n + 1.5)
    fig, ax = plt.subplots(figsize=(fig_w, 4.7))
    ax.bar(np.arange(n), values, yerr=errs, color=colors, width=0.7, zorder=2,
           error_kw=dict(ecolor="#333333", elinewidth=1.0, capsize=3, capthick=1.0))

    if metric == "uniformity":
        ax.set_ylabel(r"Uniformity (log, $\downarrow$)", fontsize=24)
        ax.set_ylim(top=0.0, bottom=1.18 * float(np.min(values)))
    else:  # alignment — lower is better
        ax.set_ylabel(r"Alignment ($\downarrow$)", fontsize=24)
        top = 1.15 * float(np.max(np.asarray(values) + np.asarray(errs)))
        ax.set_ylim(bottom=0.0, top=top)

    ax.set_xticks(np.arange(n))
    ax.set_xticklabels(labels, fontsize=24, rotation=45, ha="right",
                       rotation_mode="anchor")
    offset = ScaledTranslation(20 / 72, 0, fig.dpi_scale_trans)
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


def _render_setting(df_all: pd.DataFrame, setting: str) -> None:
    df = df_all[df_all["setting"] == setting]
    if df.empty:
        print(f"[skip] no rows for setting={setting!r}")
        return
    order = [t for t in ORDER[setting] if t in set(df["representation"])]
    df = df.set_index("representation").reindex(order).reset_index()
    labels = [LABEL[t] for t in df["representation"]]
    colors = [FAMILY_COLOR[_family_of(t)] for t in df["representation"]]

    _plot_panel(df["alignment_mean"].values, df["alignment_std"].values,
                labels, colors, "alignment", str(_PLOTS / f"{setting}_alignment"))
    _plot_panel(df["uniformity_mean"].values, df["uniformity_std"].values,
                labels, colors, "uniformity", str(_PLOTS / f"{setting}_uniformity"))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--setting", choices=("global", "patch", "both"), default="both")
    args = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    _apply_serif_style(matplotlib)

    _PLOTS.mkdir(parents=True, exist_ok=True)
    df_all = pd.read_csv(_RESULTS)

    settings = ("global", "patch") if args.setting == "both" else (args.setting,)
    for s in settings:
        _render_setting(df_all, s)


if __name__ == "__main__":
    main()
