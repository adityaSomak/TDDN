"""N-Queens CRG evaluation: raw vs CRG-oracle(GT) vs CRG-TDDN.

For each coordinate-free question the negative blacks the queried extreme queen — its
GT cell (oracle) or its TDDN-detected box (tddn). Reports per-question AUROC (binary)
or accuracy (the 3-way q4), plus board-level bootstrap 95% CIs on the paired deltas, a
macro-combined summary, and multi-seed jitter.

Everything is read from the committed dataset: questions, answers, board images and
the cached TDDN detections. Negatives are built in memory per item.

The seed loop measures only VLM forward-pass jitter — the region a negative blacks is
fully determined by the dataset, so it cannot vary between seeds. It is a strict
determinism check that should read ~0.
"""
from __future__ import annotations

import random

import numpy as np
import torch

from . import config, data, decode_engine as de, metrics, negatives

_ARMS = ("raw", "oracle", "tddn")


def _seed_all(s: int) -> None:
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def _chunks(seq, n):
    return [seq[i:i + n] for i in range(0, len(seq), n)]


def _auroc_of(recs: list[dict], binary: bool) -> float:
    """AUROC over records; binary uses P(option 0), 3-way uses macro one-vs-rest."""
    if not recs:
        return float("nan")
    if binary:
        scores = [r["probs"][0] for r in recs]
        labs = [1 if r["label"] == 0 else 0 for r in recs]
        return metrics.auroc(scores, labs)
    nk = len(recs[0]["probs"])
    aucs = []
    for k in range(nk):
        labs = [1 if r["label"] == k else 0 for r in recs]
        if 0 < sum(labs) < len(labs):
            a = metrics.auroc([r["probs"][k] for r in recs], labs)
            if a == a:
                aucs.append(a)
    return metrics.mean_ignore_nan(aucs)


def _acc_of(recs: list[dict]) -> float:
    return metrics.accuracy([r["pred"] for r in recs], [r["label"] for r in recs])


def _run_arm(items, arm, boards, spec, detections, opt_ids, bs, alpha) -> list[dict]:
    recs = []
    for batch in _chunks(items, bs):
        pos = [boards[it["image_id"]] for it in batch]
        prompts = [it["prompt"] for it in batch]
        ni = None if arm == "raw" else [
            negatives.build("nqueens", arm, it, boards[it["image_id"]], spec, detections)
            for it in batch]
        probs, preds = de.decision_batch(pos, prompts, opt_ids, neg_images=ni, alpha=alpha)
        for it, pr, pd in zip(batch, probs, preds):
            recs.append({"image_id": it["image_id"], "probs": pr,
                         "pred": int(pd), "label": it["label"]})
    return recs


def run_eval(model_id: str, arms: list[str], *, family: str = "hf",
             limit: int | None = None, alpha: float = 1.0, batch_size: int = 8,
             no_think: bool = False, seeds: list[int] | None = None,
             load_4bit: bool = False, max_memory: dict | None = None,
             detections: dict | None = None) -> dict:
    """Run the requested arms and return the report (the caller writes it)."""
    arms = ["raw"] + [a for a in arms if a != "raw"]     # raw is always the baseline
    seeds = seeds or [0]
    qspecs, items = data.build_items("nqueens", limit)
    ids = sorted({it["image_id"] for v in items.values() for it in v})
    boards = {i: data.load_board_image("nqueens", i) for i in ids}
    if detections is None:
        detections = data.load_detections("nqueens") if "tddn" in arms else {}

    de.load(model_id=model_id, family=family, no_think=no_think,
            load_4bit=load_4bit, max_memory=max_memory)
    print(f"model={model_id} arms={arms} alpha={alpha} boards={len(ids)} seeds={seeds}",
          flush=True)

    # Seed 0 carries the records used for point estimates + bootstrap; later seeds only
    # re-measure jitter on the macro metrics.
    seed0: dict = {}
    per_seed_macro = []
    for si, seed in enumerate(seeds):
        _seed_all(seed)
        per_q_metric, recs_q = {}, {}
        for qid, spec in qspecs.items():
            its = items.get(qid, [])
            if not its:
                continue
            opt_ids = de.option_token_ids(spec["options"])
            binary = len(spec["options"]) == 2
            recs_q[qid] = {}
            per_q_metric[qid] = {"binary": binary}
            for arm in arms:
                recs = _run_arm(its, arm, boards, spec, detections, opt_ids,
                                batch_size, alpha)
                recs_q[qid][arm] = {r["image_id"]: r for r in recs}
                per_q_metric[qid][arm] = {"auroc": round(_auroc_of(recs, binary), 4),
                                          "acc": round(_acc_of(recs), 4)}
        macro = {arm: {m: round(metrics.mean_ignore_nan(
            [per_q_metric[q][arm][m] for q in per_q_metric]), 4) for m in ("auroc", "acc")}
            for arm in arms}
        per_seed_macro.append(macro)
        if si == 0:
            seed0 = recs_q
        print(f"  [seed {seed}] " + "  ".join(
            f"{a}(au={macro[a]['auroc']:.3f} ac={macro[a]['acc']:.3f})" for a in arms),
            flush=True)

    return _assemble(model_id, arms, qspecs, seed0, per_seed_macro, ids, alpha,
                     no_think, seeds)


def _assemble(model_id, arms, qspecs, seed0, per_seed_macro, ids, alpha,
              no_think, seeds) -> dict:
    crg_arms = [a for a in arms if a != "raw"]
    resamples = metrics.bootstrap_resamples(ids)
    report = {"model": model_id, "alpha": alpha, "no_think": no_think,
              "n_boards": len(ids), "seeds": seeds, "arms": arms,
              "per_question": {}, "combined": {}}

    def metric_on(by_id, binary, want):
        def f(arm, sample):
            recs = [by_id[arm][i] for i in sample if i in by_id[arm]]
            return _auroc_of(recs, binary) if want == "auroc" else _acc_of(recs)
        return f

    for qid in seed0:
        binary = len(qspecs[qid]["options"]) == 2
        by_id = seed0[qid]
        blk = {"binary": binary}
        for want in ("auroc", "acc"):
            mo = metric_on(by_id, binary, want)
            blk.setdefault("raw", {})[want] = round(mo("raw", ids), 4)
            blk["raw"][f"{want}_ci"] = metrics.ci([mo("raw", s) for s in resamples])
            for arm in crg_arms:
                d = metrics.delta_ci(mo, arm, "raw", ids, resamples)
                blk.setdefault(arm, {}).update({
                    want: d["value"], f"{want}_ci": d["value_ci"],
                    f"d_{want}": d["delta"], f"d_{want}_ci": d["delta_ci"]})
        report["per_question"][qid] = blk

    # combined = macro over questions, resampling boards once per draw
    def combined_on(want, arm, sample):
        vals = []
        for qid in seed0:
            binary = len(qspecs[qid]["options"]) == 2
            recs = [seed0[qid][arm][i] for i in sample if i in seed0[qid][arm]]
            vals.append(_auroc_of(recs, binary) if want == "auroc" else _acc_of(recs))
        return metrics.mean_ignore_nan(vals)

    for want in ("auroc", "acc"):
        raw_pt = combined_on(want, "raw", ids)
        report["combined"].setdefault("raw", {}).update({
            want: round(raw_pt, 4),
            f"{want}_ci": metrics.ci([combined_on(want, "raw", s) for s in resamples])})
        for arm in crg_arms:
            pt = combined_on(want, arm, ids)
            report["combined"].setdefault(arm, {}).update({
                want: round(pt, 4),
                f"{want}_ci": metrics.ci([combined_on(want, arm, s) for s in resamples]),
                f"d_{want}": round(pt - raw_pt, 4),
                f"d_{want}_ci": metrics.ci(
                    [combined_on(want, arm, s) - combined_on(want, "raw", s)
                     for s in resamples])})

    report["seed_jitter_combined"] = {
        f"{arm}_{want}": round(max(v) - min(v), 4)
        for arm in arms for want in ("auroc", "acc")
        for v in [[m[arm][want] for m in per_seed_macro]]}
    return report
