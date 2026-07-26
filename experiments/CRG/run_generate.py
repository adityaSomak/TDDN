#!/usr/bin/env python3
"""Mint a NEW synthetic chess board set. Rarely needed; DESTRUCTIVE.

    python run_generate.py --per 100 --seed 101

The committed 800-board set is the artifact every published chess number was measured
on. This regenerates it from scratch — overwriting
``datasets/Puzzle_Perception/PVQA/chess/{images/,answers.csv}`` — and is only useful
for extending the benchmark or rebuilding it after changing a generator.

Two things it needs that plain evaluation does not:
  * the locally-supplied 269-board segmentation set (piece sprites and board colour
    themes are extracted from those real boards and their GT masks) — see
    ``datasets/_local/README.md``;
  * a follow-up ``run_eval.py --task chess --redetect --save-detections``, because the
    cached TDDN detections are keyed to the old boards.

N-Queens has no generator: its 100 boards come from AlgoPuzzleVQA* and are committed
as-is.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))

from evaluation.src import chess_generate, config  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--per", type=int, default=100,
                   help="boards per question, balanced across answer classes")
    p.add_argument("--seed", type=int, default=101)
    p.add_argument("--force", action="store_true",
                   help="required: confirms overwriting the committed board set")
    args = p.parse_args(argv)

    if not args.force:
        n = len(list((config.task_dir("chess") / "images").glob("*.png")))
        raise SystemExit(
            f"this overwrites the committed chess board set ({n} boards) and desynchronises\n"
            "the cached TDDN detections and every archived per-board result record.\n"
            "Re-run with --force if that is what you want.")

    chess_generate.generate(per=args.per, seed=args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
