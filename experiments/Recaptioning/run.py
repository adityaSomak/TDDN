#!/usr/bin/env python
"""Single entry point for the recaptioning pipeline: download, caption, assemble."""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def _load_config(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def _check_env() -> None:
    try:
        import vllm  # noqa: F401
    except ImportError as e:
        raise SystemExit(
            f"vllm is not importable ({e}). Run: pip install -r requirements.txt"
        ) from e


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["download", "caption", "assemble", "all"])
    p.add_argument("--out", default=os.environ.get("RECAPTION_OUT", "./recaption-data"))
    p.add_argument("--seeds", type=int, default=4)
    p.add_argument("--per-seed", type=int, default=508025)
    p.add_argument("--model", default="google/gemma-3-27b-it")
    p.add_argument("--tensor-parallel", type=int, default=1)
    p.add_argument("--config", default=str(HERE / "config.yaml"))
    args = p.parse_args()

    cfg = _load_config(Path(args.config))
    out_dir = Path(args.out)
    images_dir = out_dir / "images"
    captions_dir = out_dir / "captions"
    target = args.seeds * args.per_seed

    stages = ["download", "caption", "assemble"] if args.stage == "all" else [args.stage]
    for i, stage in enumerate(stages, 1):
        if len(stages) > 1:
            print(f"[{i}/{len(stages)}] {stage}")

        if stage == "download":
            from recaption.download import download_images, load_candidates

            candidates = load_candidates(cfg["source"], out_dir / "source")
            asyncio.run(download_images(
                candidates, images_dir, target,
                cfg["download"]["concurrency"], cfg["download"]["timeout"],
                cfg["download"]["max_side"], cfg["download"]["min_side"],
            ))

        elif stage == "caption":
            _check_env()
            from recaption.caption import caption

            caption(images_dir, captions_dir, args.model, args.tensor_parallel, cfg["caption"])

        elif stage == "assemble":
            from recaption.assemble import assemble

            assemble(images_dir, captions_dir, out_dir, args.seeds, args.per_seed,
                      cfg["assemble"]["master_seed"], cfg["assemble"]["pairs_per_shard"],
                      args.model, cfg["caption"]["prompt"])

    if args.stage == "all":
        print(f"\nDone. {args.seeds} versions in {out_dir}/versions:")
        for s in range(args.seeds):
            print(f"  seed{s}   {args.per_seed:,} images")


if __name__ == "__main__":
    main()
