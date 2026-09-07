"""img2dataset wrapper: download one batch (a pool_rank row range) into its own
numbered batch directory. Runs under the laion_dl env (img2dataset/pyarrow/pandas,
no torch/GPU needed) -- kept separate from recaption_env so img2dataset's
pandas<3/albumentations<2 pins never touch the vLLM env's dependencies.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from pipeline.paths import DOWNLOAD_DIR


def download_batch(pool_path: Path, start: int, end: int, batch_name: str, cfg: dict) -> Path:
    from img2dataset import download

    pool = pq.read_table(pool_path).to_pandas()
    batch = pool[(pool["pool_rank"] >= start) & (pool["pool_rank"] < end)].copy()
    # img2dataset reserves "key" (its own shard-relative numeric id) and writes the
    # *downloaded* bytes' hash to a "sha256" json field (compute_hash="sha256") --
    # rename our source-parquet columns first so passthrough can't collide with either.
    batch = batch.rename(columns={"key": "source_key", "sha256": "source_sha256"})

    batch_dir = DOWNLOAD_DIR / batch_name
    batch_dir.mkdir(parents=True, exist_ok=True)
    input_parquet = batch_dir / "input.parquet"
    batch.to_parquet(input_parquet)

    download(
        url_list=str(input_parquet),
        output_folder=str(batch_dir / "out"),
        input_format="parquet",
        url_col="url",
        caption_col="original_caption",
        save_additional_columns=["source_key", "source_sha256", "pool_rank"],
        image_size=cfg["image_size"],
        resize_mode=cfg["resize_mode"],
        resize_only_if_bigger=cfg["resize_only_if_bigger"],
        skip_reencode=cfg["skip_reencode"],
        output_format=cfg["output_format"],
        number_sample_per_shard=cfg["number_sample_per_shard"],
        incremental_mode=cfg["incremental_mode"],
        processes_count=cfg["processes_count"],
        thread_count=cfg["thread_count"],
        timeout=cfg["timeout"],
        retries=cfg["retries"],
        min_image_size=cfg.get("min_image_size", 0),
        compute_hash="sha256",
    )
    return batch_dir


def index_batch(batch_dir: Path) -> pd.DataFrame:
    """Scan a completed batch's `files`-format output into a DataFrame of
    successfully downloaded images: key, jpg_path, sha256_downloaded, pool_rank,
    original_caption, url."""
    out_dir = batch_dir / "out"
    rows = []
    for json_path in sorted(out_dir.glob("*/*.json")):
        meta = json.loads(json_path.read_text())
        if meta.get("status") != "success":
            continue
        jpg_path = json_path.with_suffix(".jpg")
        if not jpg_path.exists():
            continue
        rows.append({
            "key": meta.get("source_key"),
            "jpg_path": str(jpg_path),
            "sha256_downloaded": meta.get("sha256"),        # hash of the actual bytes received
            "source_sha256": meta.get("source_sha256"),      # hash claimed by the source parquet
            "pool_rank": meta.get("pool_rank"),
            "original_caption": meta.get("caption"),
            "url": meta.get("url"),
        })
    return pd.DataFrame(rows)


def apply_post_download_gates(df: pd.DataFrame, phash_max_count: int = 10) -> tuple[pd.DataFrame, dict]:
    """Drop repeated-placeholder garbage that downloaded with status=success but
    isn't a real, distinct photo.

    NOT gated on sha256 mismatch vs. the source parquet's 2022-vintage claimed hash:
    tried this first and it dropped ~40% of a smoke batch, nearly all of which turned
    out to be perfectly good photos from real CDNs (Amazon, Shopify, Etsy, ...) that
    had simply been re-compressed/resized server-side sometime in the last 4 years --
    completely normal CDN behavior. Byte-exact hashing can't tell "same photo,
    recompressed" apart from "different photo," so it isn't the right tool here.

    The actual target -- parked-domain placeholders -- IS caught below: the same
    placeholder graphic gets served by many different dead domains, so it shows up
    as one phash value repeating far more than any real, distinct photo would.
    """
    import imagehash
    from PIL import Image

    stats = {"input": len(df)}

    phashes = []
    for path in df["jpg_path"]:
        try:
            phashes.append(str(imagehash.phash(Image.open(path))))
        except Exception:
            phashes.append(None)
    df["phash"] = phashes
    df = df[df["phash"].notna()]

    counts = df["phash"].value_counts()
    repeated = counts[counts > phash_max_count].index
    stats["phash_duplicate_dropped"] = int(df["phash"].isin(repeated).sum())
    df = df[~df["phash"].isin(repeated)].copy()

    stats["output"] = len(df)
    return df.drop(columns=["phash"]), stats
