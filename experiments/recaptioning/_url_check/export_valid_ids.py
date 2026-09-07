"""Filter checked*.jsonl shards down to just the ids whose URL was confirmed valid.

check_urls.py records every attempt (valid AND invalid) so the run is resumable;
this script produces the actual deliverable: the set of valid ids/urls, nothing else.
Safe to run while check_urls.py shards are still writing -- a torn last line (from
reading mid-append) is skipped rather than crashing the export.

Usage (crg_env):
    python export_valid_ids.py                                   # globs output/checked*.jsonl
    python export_valid_ids.py --input output/checked_shard0.jsonl --output output/valid_ids.parquet
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=str, default=None,
                    help="glob pattern or single file; default globs output/checked*.jsonl")
    p.add_argument("--output", type=Path, default=HERE / "output" / "valid_ids.parquet")
    args = p.parse_args()

    pattern = args.input or str(HERE / "output" / "checked*.jsonl")
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise SystemExit(f"no files matched: {pattern}")
    print(f"reading {len(paths)} shard file(s): {[Path(p).name for p in paths]}")

    keys, urls, content_lengths, formats = [], [], [], []
    n_total = n_valid = n_torn = 0
    seen_keys: set[str] = set()
    for path in paths:
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    n_torn += 1  # tolerate a torn last line from a live-appending shard
                    continue
                n_total += 1
                if row["valid"] and row["key"] not in seen_keys:
                    seen_keys.add(row["key"])
                    n_valid += 1
                    keys.append(row["key"])
                    urls.append(row["url"])
                    content_lengths.append(row["content_length"])
                    formats.append(row["sniffed_format"])

    table = pa.table({
        "key": keys,
        "url": urls,
        "content_length": content_lengths,
        "sniffed_format": formats,
    })
    pq.write_table(table, args.output)
    print(f"checked={n_total:,}  valid={n_valid:,} ({100*n_valid/max(n_total,1):.1f}%)  torn_lines_skipped={n_torn}")
    print(f"wrote {len(keys):,} valid ids -> {args.output}")


if __name__ == "__main__":
    main()
