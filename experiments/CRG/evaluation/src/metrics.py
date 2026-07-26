"""Scoring + statistics: AUROC, accuracy, board-level bootstrap CIs, recovery%.

The confidence intervals are computed by resampling *boards* (paired raw vs CRG),
not by repeating model runs — each arm is a single deterministic greedy forward, so
the only sampling variation is over the finite board set.
"""
from __future__ import annotations

import random
from typing import Callable, Sequence

N_BOOT = 1000
BOOT_SEED = 12345


def auroc(scores: Sequence[float], labels: Sequence[int]) -> float:
    """Mann-Whitney AUROC with average ranks for ties. NaN if a class is empty."""
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0  # 1-based average rank
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    sum_pos = sum(ranks[i] for i in range(len(labels)) if labels[i] == 1)
    return (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def accuracy(preds: Sequence[int], labels: Sequence[int]) -> float:
    if not preds:
        return float("nan")
    return sum(int(p == l) for p, l in zip(preds, labels)) / len(preds)


def mean_ignore_nan(xs: Sequence[float]) -> float:
    vals = [x for x in xs if x == x]
    return sum(vals) / len(vals) if vals else float("nan")


def percentile(values: Sequence[float], q: float) -> float:
    v = sorted(x for x in values if x == x)
    if not v:
        return float("nan")
    return v[min(len(v) - 1, int(q * len(v)))]


def ci(values: Sequence[float], lo: float = 0.025, hi: float = 0.975) -> list[float]:
    """[lo, hi] percentile interval, rounded to 3dp."""
    return [round(percentile(values, lo), 3), round(percentile(values, hi), 3)]


def bootstrap_resamples(keys: Sequence, n_boot: int = N_BOOT,
                        seed: int = BOOT_SEED) -> list[list]:
    """n_boot resamples (with replacement) of a key list, deterministic by seed."""
    rng = random.Random(seed)
    n = len(keys)
    return [[keys[rng.randrange(n)] for _ in range(n)] for _ in range(n_boot)]


def delta_ci(metric_on: Callable[[str, Sequence], float], arm: str, baseline: str,
             keys: Sequence, resamples: Sequence[Sequence]) -> dict:
    """Point estimate + CI for an arm and for its paired delta vs the baseline.

    metric_on(arm, sample) returns the metric for that arm over a board sample.
    """
    pt = metric_on(arm, keys)
    base_pt = metric_on(baseline, keys)
    return {
        "value": round(pt, 4),
        "value_ci": ci([metric_on(arm, s) for s in resamples]),
        "delta": round(pt - base_pt, 4),
        "delta_ci": ci([metric_on(arm, s) - metric_on(baseline, s) for s in resamples]),
    }


def recovery_pct(d_tddn: float, d_oracle: float) -> float:
    """Fraction of the oracle's gain recovered by TDDN, as a percentage."""
    return 100.0 * d_tddn / d_oracle if d_oracle > 0 else float("nan")
