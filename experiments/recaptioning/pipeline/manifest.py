"""Build the ranked, deduplicated candidate pool for download+captioning.

Join the source parquet against the URL-liveness pre-check's valid_ids, dedup on
url/sha256 (duplicate images would break the disjoint-partition independence
assumption), and rank deterministically by sha256 so batches are reproducible.
"""
from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

from pipeline.paths import MANIFEST_DIR, SOURCE_DIR


def download_source_parquet(repo_id: str, repo_type: str, filename: str) -> Path:
    path = hf_hub_download(repo_id=repo_id, repo_type=repo_type, filename=filename,
                            local_dir=str(SOURCE_DIR))
    return Path(path)


def build_pool(source_parquet: Path, valid_ids_parquet: Path, out_path: Path,
               funnel_path: Path) -> int:
    """Write the ranked pool parquet; return its row count."""
    funnel: dict[str, int] = {}

    src = pq.read_table(source_parquet, columns=["key", "url", "caption", "sha256"])
    funnel["source_rows"] = src.num_rows

    valid = pq.read_table(valid_ids_parquet, columns=["key"])
    funnel["valid_url_rows"] = valid.num_rows

    joined = src.join(valid, keys="key", join_type="inner")
    funnel["joined_rows"] = joined.num_rows

    # Dedup on url, then on sha256 -- keep the first occurrence of each.
    df = joined.to_pandas()
    before = len(df)
    df = df.drop_duplicates(subset=["url"], keep="first")
    funnel["after_url_dedup"] = len(df)
    df = df.drop_duplicates(subset=["sha256"], keep="first")
    funnel["after_sha256_dedup"] = len(df)

    df = df.sort_values("sha256", kind="stable").reset_index(drop=True)
    df["pool_rank"] = range(len(df))
    df = df.rename(columns={"caption": "original_caption"})

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), out_path)

    funnel_path.parent.mkdir(parents=True, exist_ok=True)
    funnel_path.write_text(json.dumps(funnel, indent=2))

    return len(df)
