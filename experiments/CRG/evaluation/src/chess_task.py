"""Chess CRG evaluation: raw vs CRG-oracle(GT) vs CRG-TDDN.

The negative blacks the queried piece cell(s): the GT cells from answers.csv (oracle)
or the cells implied by the cached TDDN prediction map (tddn). Questions, options and
ablate-pieces come from questions.yaml.

Emits **per-board records** ``[image_id, pred, label]`` per (question, arm) rather than
pre-reduced metrics, so ``aggregate.py`` can re-derive accuracy and bootstrap CIs from
scratch without re-running the model.

Note q8 (`exists_wQ`): on the 50 boards whose answer is "No" there is no white queen,
so ablate_cells is empty and the oracle negative equals the positive — CRG reduces to
the raw logits there by construction.
"""
from __future__ import annotations

import numpy as np

from . import config, data, decode_engine as de, metrics, negatives


def _chunks(seq, n):
    return [seq[i:i + n] for i in range(0, len(seq), n)]


def _run_arm(items, arm, boards, spec, detections, opt_ids, bs, alpha) -> list[list]:
    records = []
    for batch in _chunks(items, bs):
        pos = [boards[it["image_id"]] for it in batch]
        prompts = [it["prompt"] for it in batch]
        ni = None if arm == "raw" else [
            negatives.build("chess", arm, it, boards[it["image_id"]], spec, detections)
            for it in batch]
        probs, _ = de.decision_batch(pos, prompts, opt_ids, neg_images=ni, alpha=alpha)
        for it, pr in zip(batch, probs):
            records.append([it["image_id"], int(np.argmax(pr)), it["label"]])
    return records


def run_eval(model_id: str, arms: list[str], *, family: str = "hf",
             limit: int | None = None, alpha: float = 1.0, batch_size: int = 4,
             no_think: bool = False, load_4bit: bool = False,
             max_memory: dict | None = None, detections: dict | None = None) -> dict:
    """Run the requested arms and return the report (the caller writes it)."""
    arms = ["raw"] + [a for a in arms if a != "raw"]
    qspecs, items = data.build_items("chess", limit)
    ids = sorted({it["image_id"] for v in items.values() for it in v})
    boards = {i: data.load_board_image("chess", i) for i in ids}
    if detections is None:
        detections = data.load_detections("chess") if "tddn" in arms else {}

    de.load(model_id=model_id, family=family, no_think=no_think,
            load_4bit=load_4bit, max_memory=max_memory)
    print(f"model={model_id} arms={arms} alpha={alpha} boards={len(ids)} "
          f"questions={len(items)}", flush=True)

    report = {"model": model_id, "alpha": alpha, "arms": arms, "per_question": {}}
    print(f"\n{'question':10s} {'cat':12s} {'rawAcc':>7} "
          + " ".join(f"{a:>8}" for a in arms if a != "raw"))
    for qid in sorted(items, key=lambda q: int(q[1:])):
        spec = qspecs[qid]
        opt_ids = de.option_token_ids(spec["options"])
        blk = {"cat": spec.get("category", "other"), "options": spec["options"]}
        accs = {}
        for arm in arms:
            recs = _run_arm(items[qid], arm, boards, spec, detections, opt_ids,
                            batch_size, alpha)
            blk[arm] = recs
            accs[arm] = metrics.accuracy([p for _, p, _ in recs], [l for _, _, l in recs])
        report["per_question"][qid] = blk
        deltas = " ".join(f"{accs[a] - accs['raw']:>+8.3f}" for a in arms if a != "raw")
        print(f"{qid:10s} {blk['cat']:12s} {accs['raw']:>7.3f} {deltas}", flush=True)

    return report
