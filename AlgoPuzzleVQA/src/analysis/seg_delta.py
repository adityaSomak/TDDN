"""Compute the dM / dPM / dPMM table from seg_eval_results JSONs.

Definitions:
    Img    = (raw_M + raw_PM) / 2
    dM     =  seg_M  - Img
    dPM    =  seg_PM - Img
    dPMM   =  dPM - dM   ==  seg_PM - seg_M

M  = <model>_oracle_mask.json   (oracle-mask experiment)
PM = <model>_tddn_mask.json     (TDDN-mask experiment)

Per-task metric:
    maze    -- cell accuracy
    nqueens -- queen F1
    chess   -- mean of black + white piece F1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional


RESULTS_DIR = Path(__file__).resolve().parents[2] / "seg_eval_results"


def _maze(d: dict, variant: str) -> Optional[float]:
    return d.get("maze", {}).get(variant, {}).get("cell")


def _nq(d: dict, variant: str) -> Optional[float]:
    return d.get("nqueens", {}).get(variant, {}).get("f1")


def _chess(d: dict, variant: str) -> Optional[float]:
    vals = [d.get(k, {}).get(variant, {}).get("f1")
            for k in ("chess_black", "chess_white")]
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


METRICS = {"chess": _chess, "maze": _maze, "nqueens": _nq}


def discover_models(model_filter: Optional[list[str]] = None) -> list[str]:
    """Return base model tags with both _oracle_mask.json and _tddn_mask.json."""
    oracle = {p.name[:-len("_oracle_mask.json")] for p in RESULTS_DIR.glob("*_oracle_mask.json")}
    tddn = {p.name[:-len("_tddn_mask.json")] for p in RESULTS_DIR.glob("*_tddn_mask.json")}
    paired = sorted(oracle & tddn)
    if model_filter:
        paired = [m for m in paired if any(f in m for f in model_filter)]
    return paired


def deltas(model: str, task: str) -> Optional[dict]:
    m_path = RESULTS_DIR / f"{model}_oracle_mask.json"
    pm_path = RESULTS_DIR / f"{model}_tddn_mask.json"
    if not (m_path.exists() and pm_path.exists()):
        return None
    M = json.loads(m_path.read_text())
    PM = json.loads(pm_path.read_text())
    fn = METRICS[task]
    raw_M, seg_M = fn(M, "raw"), fn(M, "oracle_mask")
    raw_PM, seg_PM = fn(PM, "raw"), fn(PM, "tddn_mask")
    if None in (raw_M, raw_PM):
        return None
    img = (raw_M + raw_PM) / 2
    return dict(
        raw_M=raw_M, seg_M=seg_M, raw_PM=raw_PM, seg_PM=seg_PM, Img=img,
        dM=(seg_M - img) if seg_M is not None else None,
        dPM=(seg_PM - img) if seg_PM is not None else None,
        dPMM=(seg_PM - seg_M) if (seg_M is not None and seg_PM is not None) else None,
    )


def _f(x: Optional[float], width: int = 5, sign: bool = False) -> str:
    if x is None:
        return f"{'N/A':>{width}}"
    return f"{x:+{width}.1f}" if sign else f"{x:{width}.1f}"


def print_text_table(models: list[str]) -> None:
    print("# Img = (raw_M + raw_PM) / 2;  dM = seg_M - Img;  dPM = seg_PM - Img;  dPMM = dPM - dM")
    print()
    print(f"{'Model':<28} {'TASK':<8} | "
          f"{'raw_M':>6} {'seg_M':>6} | "
          f"{'raw_PM':>6} {'seg_PM':>6} | "
          f"{'Img':>6} {'dM':>6} {'dPM':>6} {'dPMM':>6}")
    print("-" * 100)
    for m in models:
        for t in ("chess", "maze", "nqueens"):
            d = deltas(m, t)
            if d is None:
                continue
            print(f"{m[:28]:<28} {t:<8} | "
                  f"{_f(d['raw_M'])} {_f(d['seg_M'])} | "
                  f"{_f(d['raw_PM'])} {_f(d['seg_PM'])} | "
                  f"{_f(d['Img'])} {_f(d['dM'], sign=True)} "
                  f"{_f(d['dPM'], sign=True)} {_f(d['dPMM'], sign=True)}")
        print()


def print_latex_rows(models: list[str]) -> None:
    """One row per model: Model & Img,dM,dPM,dPMM (chess) & ... (maze) & ... (nq) & meanDPMM."""
    print(r"% Model & chess(Img,dM,dPM,dPMM) & maze(...) & nq(...) & meanDPMM")
    for m in models:
        cells = [m[:28]]
        pmms: list[float] = []
        for t in ("chess", "maze", "nqueens"):
            d = deltas(m, t)
            for k in ("Img", "dM", "dPM", "dPMM"):
                cells.append(_f(d[k] if d else None, sign=(k != "Img")))
            if d and d["dPMM"] is not None:
                pmms.append(d["dPMM"])
        mean = sum(pmms) / len(pmms) if pmms else None
        cells.append(_f(mean, sign=True))
        print(" & ".join(c.strip() for c in cells) + r"  \\")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--models", nargs="*",
                    help="substring filter on model tag")
    ap.add_argument("--latex", action="store_true",
                    help="also print LaTeX rows after the text table")
    args = ap.parse_args()
    models = discover_models(args.models)
    if not models:
        print("No paired (_outline + _tddn_masks) result JSONs found in", RESULTS_DIR)
        return
    print(f"Found {len(models)} model(s) with paired runs:")
    for m in models:
        print(f"  {m}")
    print()
    print_text_table(models)
    if args.latex:
        print()
        print("=" * 100)
        print()
        print_latex_rows(models)


if __name__ == "__main__":
    main()
