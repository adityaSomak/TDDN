"""Aggregate the committed per-model results into crg.csv and crg_table.tex.

Both outputs are *derived*: everything they contain comes from
``results/{chess,nqueens}/<tag>.json``, so the paper table is reproducible from the
repo with no archive and no GPU.

The two tasks reduce differently because their committed granularity differs:

chess    per-board records -> accuracy and bootstrap CIs are recomputed here.
nqueens  per-question metrics with CIs already baked in -> read, not recomputed.
         (The pre-restructure writer never emitted per-board records for N-Queens; a
         fresh --publish run does, and then this path can recompute them too.)
"""
from __future__ import annotations

import csv
import json
import random
import statistics as st

import yaml

from . import config, metrics

# The paper table's row order and display names come from configs/models.yaml
# (paper_row / display), so adding a model there is enough to place it here.


def _registry() -> dict:
    spec = yaml.safe_load(config.MODELS_YAML.read_text())
    out = {}
    for group in ("autoregressive", "diffusion"):
        out.update(spec.get(group) or {})
    return out


def _paper_order() -> list[tuple[str, dict]]:
    reg = _registry()
    rows = [(t, m) for t, m in reg.items() if m.get("paper_row")]
    return sorted(rows, key=lambda kv: kv[1]["paper_row"])


# ---------------------------------------------------------------------------
# chess: recompute from per-board records
# ---------------------------------------------------------------------------
def _chess_correct_by_board(rep: dict) -> dict:
    """qid -> {image_id: {arm: 0/1}}, keeping only boards present in every arm."""
    out = {}
    arms = rep["arms"]
    for qid, blk in rep["per_question"].items():
        per_board: dict[str, dict] = {}
        for arm in arms:
            for bid, pred, lab in blk[arm]:
                per_board.setdefault(bid, {})[arm] = int(pred == lab)
        out[qid] = {b: v for b, v in per_board.items() if set(arms) <= set(v)}
    return out


def chess_summary(rep: dict, n_boot: int = metrics.N_BOOT) -> dict:
    """raw accuracy + macro Δ per CRG arm with board-level bootstrap CIs."""
    crg_arms = [a for a in rep["arms"] if a != "raw"]
    corr = _chess_correct_by_board(rep)
    qids = list(corr)

    def macro(fn):
        return st.mean(st.mean(fn(v) for v in corr[q].values()) for q in qids)

    raw = macro(lambda v: v["raw"])
    deltas = {a: macro(lambda v, a=a: v[a] - v["raw"]) for a in crg_arms}

    rng = random.Random(metrics.BOOT_SEED)
    boot = {a: [] for a in crg_arms}
    pools = {q: list(corr[q]) for q in qids}
    for _ in range(n_boot):
        for a in crg_arms:
            vals = []
            for q in qids:
                pool = pools[q]
                sample = [pool[rng.randrange(len(pool))] for _ in pool]
                vals.append(st.mean(corr[q][p][a] - corr[q][p]["raw"] for p in sample))
            boot[a].append(st.mean(vals))
    return {"n_boards": sum(len(v) for v in corr.values()),
            "n_questions": len(qids),
            "raw_acc": raw,
            "deltas": deltas,
            "delta_ci": {a: metrics.ci(boot[a]) for a in crg_arms}}


# ---------------------------------------------------------------------------
# nqueens: read the precomputed combined block
# ---------------------------------------------------------------------------
def nqueens_summary(rep: dict) -> dict:
    c = rep["combined"]
    crg_arms = [a for a in rep["arms"] if a != "raw"]
    return {"n_boards": rep.get("n_boards"),
            "n_questions": len(rep["per_question"]),
            "raw_acc": c["raw"]["acc"],
            "deltas": {a: c[a]["d_acc"] for a in crg_arms},
            "delta_ci": {a: c[a].get("d_acc_ci") for a in crg_arms}}


SUMMARY = {"chess": chess_summary, "nqueens": nqueens_summary}


def load_summaries() -> dict[str, dict[str, dict]]:
    """{task: {tag: summary}} over every committed per-model JSON."""
    out: dict[str, dict[str, dict]] = {}
    for task in ("chess", "nqueens"):
        out[task] = {}
        d = config.TASK_RESULTS[task]
        for path in sorted(d.glob("*.json")):
            rep = json.loads(path.read_text())
            if "arms" not in rep or "per_question" not in rep:
                continue          # e.g. the LaViDa raw-only reports: different schema
            out[task][path.stem] = SUMMARY[task](rep)
    return out


# ---------------------------------------------------------------------------
# outputs
# ---------------------------------------------------------------------------
_CSV_COLS = ["model", "display", "task", "n_boards", "n_questions", "raw_acc",
             "d_oracle", "d_tddn", "d_diff", "recovery_pct"]


def write_csv(summaries: dict, reg: dict) -> None:
    rows = []
    for task in ("chess", "nqueens"):
        for tag, s in sorted(summaries[task].items()):
            d_o = s["deltas"].get("oracle")
            d_t = s["deltas"].get("tddn")
            # Left blank rather than "nan" when the oracle itself did not help:
            # "TDDN recovered X% of the oracle's gain" is undefined with no gain.
            rp = None if None in (d_o, d_t) else metrics.recovery_pct(d_t, d_o)
            rows.append({
                "model": tag,
                "display": (reg.get(tag) or {}).get("display", tag),
                "task": task,
                "n_boards": s["n_boards"],
                "n_questions": s["n_questions"],
                "raw_acc": round(s["raw_acc"], 4),
                "d_oracle": None if d_o is None else round(d_o, 4),
                "d_tddn": None if d_t is None else round(d_t, 4),
                "d_diff": None if None in (d_o, d_t) else round(d_o - d_t, 4),
                "recovery_pct": None if rp is None or rp != rp else round(rp, 1),
            })
    dst = config.RESULTS_DIR / "crg.csv"
    with open(dst, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_CSV_COLS)
        w.writeheader()
        w.writerows(rows)
    print(f"saved -> {dst}")


_TEX_HEAD = r"""\begin{table}[t]
\centering
\small
\setlength{\tabcolsep}{4pt}
\caption{\textbf{Contrastive Region Guidance (CRG) with TDDN-predicted regions improves VLM perception.} We evaluate CRG on two puzzle datasets---Puzzle Perception (Chess) and AlgoPuzzleVQA* (N-Queens)---reporting VLM question accuracy (\%), macro-averaged over questions. \textbf{Img} is the raw image-only baseline. CRG blacks out the queried region and contrasts it against the full image; the region is taken from ground-truth masks ($\Delta_{\mathrm{o}}$, oracle) or from our TDDN segmenter ($\Delta_{\mathrm{t}}$, predicted, deployable), each reported as the accuracy change relative to Img. $\Delta_{\mathrm{diff}}=\Delta_{\mathrm{o}}-\Delta_{\mathrm{t}}$ is the oracle$\to$TDDN gap: values near $0$ show that cheap TDDN regions recover the ground-truth region's gain (\emph{CRG-TDDN $\approx$ CRG-oracle}). The rightmost column is the mean $\Delta_{\mathrm{diff}}$ across the two puzzles. Rows are ordered by model size; the rule separates models where CRG gives a clear positive gain (above) from the largest models that are already near ceiling with little headroom (below). $\alpha{=}1.0$; deterministic greedy single-forward decode.}
\label{tab:crg}
\begin{tabular}{l cccc cccc c}
\toprule
& \multicolumn{4}{c}{\textbf{Puzzle Perception}} & \multicolumn{4}{c}{\textbf{AlgoPuzzleVQA*}} & \\
\cmidrule(lr){2-5}\cmidrule(lr){6-9}
& \multicolumn{4}{c}{Chess} & \multicolumn{4}{c}{N-Queens} & \textbf{Avg} \\
\cmidrule(lr){2-5}\cmidrule(lr){6-9}
\textbf{Model} & Img & $\Delta_{\mathrm{o}}$ & $\Delta_{\mathrm{t}}$ & $\Delta_{\mathrm{diff}}$ & Img & $\Delta_{\mathrm{o}}$ & $\Delta_{\mathrm{t}}$ & $\Delta_{\mathrm{diff}}$ & $\Delta_{\mathrm{diff}}$ \\
\midrule"""

_TEX_TAIL = r"""\bottomrule
\end{tabular}
\end{table}"""

# Row index after which the near-ceiling rule is drawn (see the caption).
_RULE_AFTER_ROW = 6


def write_tex(summaries: dict, reg: dict) -> None:
    def cells(s):
        d_o, d_t = s["deltas"].get("oracle"), s["deltas"].get("tddn")
        img = f"{100 * s['raw_acc']:.1f}"
        f = lambda v: "---" if v is None else f"${100 * v:+.1f}$".replace("+-", "-")
        diff = None if None in (d_o, d_t) else d_o - d_t
        return [img, f(d_o), f(d_t), f(diff)], diff

    lines = [_TEX_HEAD]
    for i, (tag, meta) in enumerate(_paper_order()):
        ch, nq = summaries["chess"].get(tag), summaries["nqueens"].get(tag)
        if not (ch and nq):
            continue
        c_cells, c_diff = cells(ch)
        n_cells, n_diff = cells(nq)
        avg = ("---" if None in (c_diff, n_diff)
               else f"${100 * (c_diff + n_diff) / 2:+.1f}$".replace("+-", "-"))
        if i == _RULE_AFTER_ROW:
            lines.append(r"\midrule")
        lines.append(f"{meta['display']} & " + " & ".join(c_cells + n_cells + [avg])
                     + r" \\")
    lines.append(_TEX_TAIL)
    dst = config.RESULTS_DIR / "crg_table.tex"
    dst.write_text("\n".join(lines) + "\n")
    print(f"saved -> {dst}")


def print_markdown(summaries: dict, reg: dict) -> None:
    """Human-readable summary to stdout (no file: nothing consumes a markdown table)."""
    for task in ("chess", "nqueens"):
        if not summaries[task]:
            continue
        print(f"\n## {task}\n")
        print("| model | raw acc | Δ oracle [95% CI] | Δ tddn [95% CI] | TDDN/oracle |")
        print("|---|---|---|---|---|")
        for tag, s in sorted(summaries[task].items()):
            d_o, d_t = s["deltas"].get("oracle"), s["deltas"].get("tddn")
            ci_o, ci_t = s["delta_ci"].get("oracle"), s["delta_ci"].get("tddn")
            fmt = lambda d, c: ("—" if d is None else
                                f"{d:+.3f}" + (f" [{c[0]:+.3f}, {c[1]:+.3f}]" if c else ""))
            # recovery_pct is undefined (nan) when the oracle itself did not help:
            # "TDDN recovered X% of the oracle's gain" is meaningless with no gain.
            rp = None if None in (d_o, d_t) else metrics.recovery_pct(d_t, d_o)
            recov = "—" if rp is None or rp != rp else f"{rp:.0f}%"
            print(f"| {tag} | {s['raw_acc']:.3f} | {fmt(d_o, ci_o)} | "
                  f"{fmt(d_t, ci_t)} | {recov} |")


def main() -> None:
    summaries = load_summaries()
    if not any(summaries.values()):
        print(f"no results under {config.RESULTS_DIR}")
        return
    reg = _registry()
    print_markdown(summaries, reg)
    write_csv(summaries, reg)
    write_tex(summaries, reg)
