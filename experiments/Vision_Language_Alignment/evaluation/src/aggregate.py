"""Rebuild the headline CSVs from the committed per-model result JSONs.

    classification.csv   model,dataset,mode,top1
    tip_k_sweep.csv      model,dataset,k_1,k_2,k_4,k_8,k_16
    retrieval.csv        model,dataset,protocol,n_images,i2t_r1,t2i_r1
    segmentation.csv     model,dataset,miou

Everything here is derived: the CSVs carry nothing the JSONs don't, so they can
be regenerated at any time and are never edited by hand. Files under ``_live/``
are ignored.
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

_RESULTS = Path(__file__).resolve().parents[1] / "results"

# Mode suffix in a result filename -> the value written to the CSV's mode column.
_MODES = {None: "zero_shot", "cupl": "cupl", "tip": "tip"}
_TIP_K_RE = re.compile(r"^(?P<model>.+)_(?P<dataset>[^_]+(?:_[^_]+)*)_tip_k(?P<k>\d+)$")


def _load(task: str) -> list[dict]:
    """Committed result JSONs for ``task``, newest schema only."""
    out = []
    d = _RESULTS / task
    if not d.is_dir():
        return out
    for path in sorted(d.glob("*.json")):
        rec = json.loads(path.read_text())
        if "model" not in rec or "dataset" not in rec:
            continue
        rec["_stem"] = path.stem
        out.append(rec)
    return out


def _sort_key(rec: dict) -> tuple:
    return (rec["dataset"], rec["model"], rec.get("mode", ""))


def write_classification() -> None:
    recs = _load("classification")
    rows: list[dict] = []
    sweep: dict[tuple[str, str], dict[int, float]] = {}
    for rec in recs:
        m = _TIP_K_RE.match(rec["_stem"])
        if m and rec.get("mode") == "tip":
            key = (rec["model"], rec["dataset"])
            sweep.setdefault(key, {})[int(m.group("k"))] = rec.get("best_top1")
            continue
        top1 = rec.get("top1", rec.get("best_top1"))
        if top1 is None:
            continue
        rows.append({"model": rec["model"], "dataset": rec["dataset"],
                     "mode": rec.get("mode", "zero_shot"), "top1": f"{top1:.2f}"})
    rows.sort(key=lambda r: (r["dataset"], r["model"], r["mode"]))
    dst = _RESULTS / "classification" / "classification.csv"
    with open(dst, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "dataset", "mode", "top1"])
        w.writeheader()
        w.writerows(rows)
    print(f"saved -> {dst}  ({len(rows)} rows)")

    if sweep:
        ks = sorted({k for v in sweep.values() for k in v})
        dst = _RESULTS / "classification" / "tip_k_sweep.csv"
        with open(dst, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["model", "dataset"] + [f"k_{k}" for k in ks])
            for (model, dataset) in sorted(sweep):
                vals = sweep[(model, dataset)]
                w.writerow([model, dataset]
                           + ["" if vals.get(k) is None else f"{vals[k]:.2f}" for k in ks])
        print(f"saved -> {dst}  ({len(sweep)} rows)")


def write_retrieval() -> None:
    # ``protocol`` is a column, not a footnote: Recall@K depends on the candidate
    # set size, so the two datasets' numbers are not on a common scale.
    cols = ["model", "dataset", "protocol", "n_images", "i2t_r1", "t2i_r1"]
    rows = [{"model": r["model"], "dataset": r["dataset"],
             "protocol": r.get("protocol", ""), "n_images": r.get("n_images", ""),
             "i2t_r1": f"{r['i2t_r1']:.2f}", "t2i_r1": f"{r['t2i_r1']:.2f}"}
            for r in _load("retrieval") if "i2t_r1" in r]
    rows.sort(key=lambda r: (r["dataset"], r["model"]))
    dst = _RESULTS / "retrieval" / "retrieval.csv"
    with open(dst, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"saved -> {dst}  ({len(rows)} rows)")


def write_segmentation() -> None:
    rows = [{"model": r["model"], "dataset": r["dataset"], "miou": f"{r['miou']:.2f}"}
            for r in _load("segmentation") if "miou" in r]
    rows.sort(key=lambda r: (r["dataset"], r["model"]))
    dst = _RESULTS / "segmentation" / "segmentation.csv"
    with open(dst, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "dataset", "miou"])
        w.writeheader()
        w.writerows(rows)
    print(f"saved -> {dst}  ({len(rows)} rows)")


def coverage() -> dict[str, dict[str, list[str]]]:
    """``{task: {dataset: [models]}}`` over the committed zero-shot/default results."""
    out: dict[str, dict[str, list[str]]] = {}
    for task in ("classification", "retrieval", "segmentation"):
        per_ds: dict[str, list[str]] = {}
        for rec in _load(task):
            if task == "classification" and rec.get("mode") != "zero_shot":
                continue
            per_ds.setdefault(rec["dataset"], []).append(rec["model"])
        out[task] = {k: sorted(v) for k, v in sorted(per_ds.items())}
    return out


def main() -> None:
    write_classification()
    write_retrieval()
    write_segmentation()
    print("\n=== coverage (models per dataset) ===")
    for task, per_ds in coverage().items():
        print(f"\n{task}")
        for dataset, models in per_ds.items():
            print(f"  {dataset:20s} {len(models)}  {', '.join(models)}")
