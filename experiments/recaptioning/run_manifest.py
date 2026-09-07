#!/usr/bin/env python
"""Build the ranked candidate pool: source parquet join valid_ids -> pool.parquet.

Usage:
    python run_manifest.py --valid-ids /path/to/valid_ids.parquet
    python run_manifest.py --valid-ids ... --dry-run
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline.config import load_yaml
from pipeline.manifest import build_pool, download_source_parquet
from pipeline.paths import MANIFEST_DIR, ensure_dirs


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--valid-ids", type=Path, required=True,
                    help="valid_ids.parquet from the URL-liveness pre-check")
    p.add_argument("--dry-run", action="store_true",
                    help="build the pool and print the count, nothing else")
    args = p.parse_args()

    ensure_dirs()
    cfg = load_yaml("pipeline.yaml")["source"]

    print(f"[manifest] fetching source parquet ({cfg['hf_repo_id']}/{cfg['filename']})...")
    source_path = download_source_parquet(cfg["hf_repo_id"], cfg["hf_repo_type"], cfg["filename"])
    print(f"[manifest] source parquet: {source_path}")

    pool_path = MANIFEST_DIR / "pool.parquet"
    funnel_path = MANIFEST_DIR / "funnel.json"
    n = build_pool(source_path, args.valid_ids, pool_path, funnel_path)

    funnel = json.loads(funnel_path.read_text())
    print(f"[manifest] funnel: {funnel}")
    print(f"[manifest] pool size: {n:,} rows -> {pool_path}")


if __name__ == "__main__":
    main()
