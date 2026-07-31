#!/usr/bin/env python3
"""CRG evaluation entry point: raw vs CRG-oracle vs CRG-TDDN on two puzzle tasks.

    python run_eval.py --task nqueens --model qwen2.5-vl-7b
    python run_eval.py --task chess   --model qwen3.6-27b --arm tddn
    python run_eval.py --task chess   --model all --publish
    python run_eval.py --aggregate

Reads everything from the committed dataset under
``datasets/Puzzle_Perception/PVQA/{chess,nqueens}/``, including the cached TDDN
detections, so no GPU segmenter and no checkpoints are needed. ``--redetect`` is the
only flag that loads the TDDN encoder.

Runs write to ``evaluation/results/<task>/_live/`` unless ``--publish`` is given, so a
smoke test cannot overwrite a paper number.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))              # evaluation.src.*  (CWD-independent)
sys.path.insert(0, str(_HERE.parent))       # shared_utils.*

import yaml  # noqa: E402

from evaluation.src import config, data  # noqa: E402

TASKS = ("nqueens", "chess")


def load_registry() -> tuple[dict, dict]:
    """(models, defaults) from configs/models.yaml, groups flattened."""
    spec = yaml.safe_load(config.MODELS_YAML.read_text())
    return dict(spec.get("autoregressive") or {}), spec["defaults"]


def _parse_max_memory(spec) -> dict | None:
    """'0:42,1:42' or {0: 42} -> {0: '42GiB', 1: '42GiB'}."""
    if not spec:
        return None
    if isinstance(spec, dict):
        return {int(k): f"{v}GiB" for k, v in spec.items()}
    return {int(k): f"{v}GiB" for k, v in (kv.split(":") for kv in spec.split(","))}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", choices=TASKS, help="required unless --aggregate")
    p.add_argument("--model", default=None,
                   help="a tag from configs/models.yaml, or 'all' for the paper sweep")
    p.add_argument("--arm", dest="arms", action="append", choices=["raw", "oracle", "tddn"],
                   help="arm(s) to run; repeatable. raw is always included as the baseline")
    p.add_argument("--alpha", type=float, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--limit", type=int, default=None, help="truncate the row list (smoke test)")
    p.add_argument("--seeds", type=int, nargs="+", default=None, help="N-Queens seeds")
    p.add_argument("--no-think", action="store_true", default=None,
                   help="force enable_thinking=False (overrides the registry)")
    p.add_argument("--load-4bit", action="store_true", default=None)
    p.add_argument("--max-memory", default=None, help="per-GPU caps, e.g. 0:42,1:42")
    p.add_argument("--publish", action="store_true",
                   help="write the committed result file instead of results/<task>/_live/")
    p.add_argument("--aggregate", action="store_true",
                   help="rebuild crg.csv + crg_table.tex from committed results and exit")
    p.add_argument("--validate-dataset", action="store_true",
                   help="check dataset integrity and exit")
    p.add_argument("--redetect", action="store_true",
                   help="re-run TDDN instead of reading the cached detections (needs DINOv3)")
    p.add_argument("--save-detections", action="store_true",
                   help="with --redetect, overwrite the committed tddn_detections.json")
    p.add_argument("--validate-tddn", action="store_true",
                   help="(chess) score TDDN piece detection vs GT and exit")
    return p


def _validate(tasks) -> int:
    bad = 0
    for task in tasks:
        problems = data.validate_dataset(task)
        rows = len(data.load_rows(task))
        if problems:
            bad += 1
            print(f"{task}: {len(problems)} problem(s) over {rows} rows")
            for m in problems:
                print(f"  - {m}")
        else:
            print(f"{task}: OK ({rows} rows)")
    return 1 if bad else 0


def _resolve(args, models, defaults, tag):
    """Merge CLI > per-model > defaults for one model tag."""
    m = models[tag]
    task = args.task
    pick = lambda cli, key, dflt=None: cli if cli is not None else m.get(key, dflt)
    bs = args.batch_size
    if bs is None:
        bs = (m.get("batch_size") or {}).get(task) or defaults["batch_size"][task]
    return {
        "model_id": m["hf_id"],
        "family": m.get("family", "hf"),
        "alpha": args.alpha if args.alpha is not None else defaults["alpha"],
        "batch_size": bs,
        "no_think": bool(pick(args.no_think, "no_think", False)),
        "load_4bit": bool(pick(args.load_4bit, "load_4bit", False)),
        "max_memory": _parse_max_memory(args.max_memory or m.get("max_memory")),
        "limit": args.limit,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    models, defaults = load_registry()

    if args.aggregate:
        from evaluation.src import aggregate
        aggregate.main()
        return 0

    if args.validate_dataset:
        return _validate([args.task] if args.task else TASKS)

    if not args.task:
        build_parser().error("--task is required unless --aggregate is given")

    if args.validate_tddn:
        if args.task != "chess":
            build_parser().error("--validate-tddn is chess-only")
        from evaluation.src import tddn
        out = tddn.validate_chess()
        dst = config.TASK_RESULTS["chess"] / "tddn_validation.json"
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(json.dumps(out, indent=2) + "\n")
        print(f"\npresence {out['presence']} | type={out['type_acc']} "
              f"color={out['color_acc']} exact={out['exact_acc']}")
        print(f"saved -> {dst}")
        return 0

    # A dataset problem invalidates every number below it, so fail before loading a model.
    if problems := data.validate_dataset(args.task):
        print(f"dataset validation failed for {args.task}:")
        for m in problems:
            print(f"  - {m}")
        return 1

    arms = args.arms or list(defaults["arms"])

    detections = None
    if args.redetect:
        if "tddn" not in arms:
            build_parser().error("--redetect only makes sense with the tddn arm")
        from evaluation.src import tddn
        detections = tddn.redetect(args.task)
        if args.save_detections:
            dst = config.detections_path(args.task)
            dst.write_text(json.dumps(detections) + "\n")
            print(f"detections -> {dst}")

    if args.model is None:
        build_parser().error("--model is required (a tag from configs/models.yaml, or 'all')")
    if args.model == "all":
        tags = [t for t, m in sorted(models.items(), key=lambda kv: kv[1].get("paper_row", 99))
                if m.get("engine") == "crg" and m.get("paper_row")]
    else:
        if args.model not in models:
            build_parser().error(
                f"unknown model {args.model!r}. Known tags: {', '.join(sorted(models))}")
        tags = [args.model]

    task_mod = None
    for tag in tags:
        kw = _resolve(args, models, defaults, tag)
        if task_mod is None:
            if args.task == "nqueens":
                from evaluation.src import nqueens_task as task_mod
            else:
                from evaluation.src import chess_task as task_mod
        if args.task == "nqueens":
            kw["seeds"] = args.seeds or list(defaults["seeds"])
        report = task_mod.run_eval(kw.pop("model_id"), arms, detections=detections, **kw)

        out_dir = config.TASK_RESULTS[args.task]
        if not args.publish:
            out_dir = out_dir / "_live"
        out_dir.mkdir(parents=True, exist_ok=True)
        dst = out_dir / f"{tag}.json"
        dst.write_text(json.dumps(report, indent=2) + "\n")
        print(f"saved -> {dst}")

        if len(tags) > 1:
            from evaluation.src import decode_engine as de
            de.free_model()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
