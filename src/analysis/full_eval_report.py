"""Aggregate full-eval results across the 4 Q1..QN tasks."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


ALGO_ROOT = Path(__file__).resolve().parents[2]
ALL_TASKS = ("checker_move", "maze_solve", "nqueens", "wood_slide")


def _qkey(qid: str) -> int:
    m = re.search(r"\d+", qid)
    return int(m.group()) if m else 0


def collect(tasks: list[str], model_filter: list[str] | None) -> dict:
    """Walk each task's eval_results_low_detail tree.

    Returns ``{(task, model, reasoning): [records, ...]}``.
    """
    out: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for task in tasks:
        root = ALGO_ROOT / task / "eval_results_low_detail"
        if not root.exists():
            continue
        for results_path in root.glob("*/*/results.jsonl"):
            model, reasoning = results_path.parts[-3], results_path.parts[-2]
            if model_filter and not any(f in model for f in model_filter):
                continue
            with open(results_path) as f:
                out[(task, model, reasoning)] = [json.loads(l) for l in f]
    return out


def overall(records: list[dict]) -> tuple[int, int, float]:
    n = len(records)
    c = sum(1 for r in records if r.get("correct"))
    return c, n, (c / n * 100 if n else 0.0)


def per_question(records: list[dict]) -> list[tuple[str, int, int, float]]:
    stats: dict[str, dict] = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in records:
        qpre = r["question_id"].split("_")[0]
        stats[qpre]["total"] += 1
        stats[qpre]["correct"] += int(r.get("correct", False))
    rows = [(q, s["correct"], s["total"], s["correct"] / s["total"] * 100)
            for q, s in stats.items()]
    return sorted(rows, key=lambda x: _qkey(x[0]))


def print_overall(table: dict) -> None:
    print(f"{'Task':<14} {'Model':<32} {'Reasoning':<10} {'Correct':>8} {'Total':>6} {'Acc':>7}")
    print("-" * 80)
    for (task, model, reasoning), recs in sorted(table.items()):
        c, n, acc = overall(recs)
        print(f"{task:<14} {model[:32]:<32} {reasoning:<10} {c:>8} {n:>6} {acc:>6.1f}%")


def print_by_question(table: dict) -> None:
    for (task, model, reasoning), recs in sorted(table.items()):
        rows = per_question(recs)
        if not rows:
            continue
        print(f"\n=== {task} | {model} | {reasoning} ===")
        print(f"{'Q':<8} {'Correct':>8} {'Total':>6} {'Acc':>7}")
        for q, c, n, acc in rows:
            print(f"{q:<8} {c:>8} {n:>6} {acc:>6.1f}%")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tasks", nargs="+", default=list(ALL_TASKS),
                    choices=list(ALL_TASKS))
    ap.add_argument("--models", nargs="*", help="substring filter on model tag")
    ap.add_argument("--by-question", action="store_true")
    args = ap.parse_args()
    table = collect(args.tasks, args.models)
    if not table:
        print("No results.jsonl files found under <task>/eval_results_low_detail/")
        return
    print_overall(table)
    if args.by_question:
        print_by_question(table)


if __name__ == "__main__":
    main()
