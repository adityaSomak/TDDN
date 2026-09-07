#!/usr/bin/env python
"""Assemble the captioned pool into N published versions: metadata.csv +
webdataset shards + provenance.json per version.

Usage:
    python run_assemble.py --clean-parquet .../clean.parquet --n-per-version 70 --num-versions 4
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

from pipeline.config import load_yaml
from pipeline.normalize import VERSION as NORMALIZER_VERSION, normalize_caption
from pipeline.paths import CAPTIONS_DIR, REPO_ROOT, VERSIONS_DIR, ensure_dirs
from pipeline.provenance import git_sha, write_provenance
from pipeline.samplers import partition_disjoint
from pipeline.tars import add_bytes, open_deterministic


def load_captioned_pool(clean_parquets: list[Path]) -> pd.DataFrame:
    frames = [pd.read_parquet(p) for p in clean_parquets]
    downloaded = pd.concat(frames, ignore_index=True)[["key", "jpg_path", "pool_rank"]]

    caption_rows = []
    for chunk_path in sorted(CAPTIONS_DIR.glob("chunk-*.jsonl")):
        with chunk_path.open() as f:
            lines = f.readlines()
        for line in lines[1:]:  # line 0 is the header
            row = json.loads(line)
            caption_rows.append(row)
    captions = pd.DataFrame(caption_rows)

    pool = downloaded.merge(captions, on="key", how="inner")
    return pool.sort_values("pool_rank", kind="stable").reset_index(drop=True)


def emit_version(version_df: pd.DataFrame, name: str, seed: int, cfg: dict) -> Path:
    out_dir = VERSIONS_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)

    n = len(version_df)
    pad = max(6, len(str(n - 1)))
    pairs_per_shard = cfg["shard"]["pairs_per_shard"]

    metadata_path = out_dir / "metadata.csv"
    shard_sha256s = []
    progress_path = out_dir / "progress.json"
    t_start = time.time()
    log_every = 20000

    with metadata_path.open("w", newline="") as meta_f:
        writer = csv.writer(meta_f)
        writer.writerow(["image_id", "url", "caption", "original_caption"])

        shard_idx = 0
        tf = None
        for i, (_, row) in enumerate(version_df.iterrows()):
            if i % pairs_per_shard == 0:
                if tf is not None:
                    tf.close()
                shard_path = out_dir / cfg["shard"]["name_template"].format(idx=shard_idx)
                tf = open_deterministic(shard_path)
                shard_idx += 1

            image_id = str(i).zfill(pad)
            new_caption = normalize_caption(row["new_caption_raw"])

            with open(row["jpg_path"], "rb") as jf:
                jpg_bytes = jf.read()
            add_bytes(tf, f"{image_id}.jpg", jpg_bytes)
            add_bytes(tf, f"{image_id}.txt", new_caption.encode("utf-8"))

            writer.writerow([image_id, row["url"], new_caption, row["original_caption"]])

            if (i + 1) % log_every == 0 or (i + 1) == n:
                elapsed = time.time() - t_start
                rate = (i + 1) / elapsed if elapsed > 0 else 0.0
                eta = (n - i - 1) / rate if rate > 0 else 0.0
                pct = 100.0 * (i + 1) / n
                progress_path.write_text(json.dumps(
                    {"images_written": i + 1, "images_total": n, "current_shard": shard_idx,
                     "rate_img_s": round(rate, 1), "eta_s": round(eta, 1)}, indent=2))
                print(f"[assemble] {name} {i+1}/{n} ({pct:.0f}%) "
                      f"{rate:.1f} img/s  eta {eta/60:.1f}min", flush=True)

        if tf is not None:
            tf.close()

    import hashlib
    for shard_path in sorted(out_dir.glob("shards-*.tar")):
        h = hashlib.sha256(shard_path.read_bytes()).hexdigest()
        shard_sha256s.append({"shard": shard_path.name, "sha256": h})

    write_provenance(
        out_dir / "provenance.json",
        version_name=name, seed=seed, n_images=n, normalizer_version=NORMALIZER_VERSION,
        git_sha=git_sha(REPO_ROOT), shard_sha256s=shard_sha256s,
    )
    print(f"[assemble] DONE {name}: {n} images, {shard_idx} shard(s)")

    return out_dir


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--clean-parquet", type=Path, action="append", required=True)
    p.add_argument("--n-per-version", type=int, default=None)
    p.add_argument("--num-versions", type=int, default=None)
    args = p.parse_args()

    ensure_dirs()
    pipeline_cfg = load_yaml("pipeline.yaml")
    assemble_cfg = load_yaml("assemble.yaml")

    n_per_version = args.n_per_version or pipeline_cfg["n_per_version"]
    num_versions = args.num_versions or pipeline_cfg["num_versions"]
    master_seed = pipeline_cfg["seeds"]["master"]

    pool = load_captioned_pool(args.clean_parquet)
    print(f"[assemble] captioned pool: {len(pool):,} images")

    versions_cfg = assemble_cfg["versions"][:num_versions]
    parts = partition_disjoint(pool, num_versions, n_per_version, seed=master_seed)

    for version_cfg, version_df in zip(versions_cfg, parts):
        t0 = time.time()
        out_dir = emit_version(version_df, version_cfg["name"], version_cfg["seed"], assemble_cfg)
        print(f"[assemble] {version_cfg['name']}: {len(version_df):,} images -> {out_dir} "
              f"({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
