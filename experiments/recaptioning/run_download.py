#!/usr/bin/env python
"""Download one batch (a pool_rank range) via img2dataset, then apply post-download
gates (sha256 re-verification, phash dedup). Run under the laion_dl venv.

Usage:
    /path/to/laion_dl/bin/python run_download.py --start 0 --end 20000 --batch-name batch_0000
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline.config import load_yaml
from pipeline.download import apply_post_download_gates, download_batch, index_batch
from pipeline.paths import DOWNLOAD_DIR, MANIFEST_DIR, ensure_dirs


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--start", type=int, required=True)
    p.add_argument("--end", type=int, required=True)
    p.add_argument("--batch-name", type=str, required=True)
    p.add_argument("--pool", type=Path, default=None)
    args = p.parse_args()

    ensure_dirs()
    cfg = load_yaml("pipeline.yaml")["download"]
    pool_path = args.pool or (MANIFEST_DIR / "pool.parquet")

    t0 = time.time()
    print(f"[download] batch {args.batch_name}: rows [{args.start}, {args.end}) "
          f"({args.end - args.start:,} rows)")
    batch_dir = download_batch(pool_path, args.start, args.end, args.batch_name, cfg)

    raw = index_batch(batch_dir)
    print(f"[download] {len(raw):,} images downloaded successfully (of {args.end - args.start:,} attempted)")

    clean, stats = apply_post_download_gates(raw)
    print(f"[download] post-download gates: {stats}")

    clean_path = batch_dir / "clean.parquet"
    clean.to_parquet(clean_path)

    progress = {
        "batch_name": args.batch_name, "start": args.start, "end": args.end,
        "downloaded": len(raw), "clean": len(clean), "elapsed_s": round(time.time() - t0, 1),
    }
    (batch_dir / "progress.json").write_text(json.dumps(progress, indent=2))
    print(f"[download] done: {len(clean):,} clean images -> {clean_path}")


if __name__ == "__main__":
    main()
