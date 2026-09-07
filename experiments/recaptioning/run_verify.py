#!/usr/bin/env python
"""Standalone artifact validator for one assembled version. Run before trusting
any output -- the old pipeline's failures (truncated JSON, mixed configs) only
surfaced at consumption time; this catches them immediately after assemble.

Usage:
    python run_verify.py --version-dir .../versions/laion_s0
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from PIL import Image


def fail(msg: str) -> None:
    raise SystemExit(f"[verify] FAIL: {msg}")


def verify(version_dir: Path) -> None:
    metadata_path = version_dir / "metadata.csv"
    provenance_path = version_dir / "provenance.json"
    if not metadata_path.exists():
        fail(f"missing {metadata_path}")
    if not provenance_path.exists():
        fail(f"missing {provenance_path}")

    provenance = json.loads(provenance_path.read_text())
    expected_n = provenance["n_images"]

    with metadata_path.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ["image_id", "url", "caption", "original_caption"]:
            fail(f"metadata.csv header wrong: {reader.fieldnames}")
        rows = list(reader)

    if len(rows) != expected_n:
        fail(f"metadata.csv has {len(rows)} rows, provenance says n_images={expected_n}")

    pad = len(rows[0]["image_id"]) if rows else 6
    if pad < 6:
        fail(f"image_id pad width {pad} < 6")
    expected_ids = [str(i).zfill(pad) for i in range(len(rows))]
    actual_ids = [r["image_id"] for r in rows]
    if actual_ids != expected_ids:
        fail("image_id not contiguous from 0 with consistent pad width")

    shard_paths = sorted(version_dir.glob("shards-*.tar"))
    if not shard_paths:
        fail("no shard tars found")

    shard_sha_by_name = {s["shard"]: s["sha256"] for s in provenance["shard_sha256s"]}
    seen_ids: set[str] = set()
    for shard_path in shard_paths:
        actual_sha = hashlib.sha256(shard_path.read_bytes()).hexdigest()
        expected_sha = shard_sha_by_name.get(shard_path.name)
        if expected_sha != actual_sha:
            fail(f"{shard_path.name} sha256 mismatch: provenance={expected_sha} actual={actual_sha}")

        with tarfile.open(shard_path) as tf:
            names = tf.getnames()
            if any(n.startswith("./") for n in names):
                fail(f"{shard_path.name} has './'-prefixed member names")

            # members must be strictly interleaved <id>.jpg, <id>.txt, ascending
            if len(names) % 2 != 0:
                fail(f"{shard_path.name} has an odd number of members")
            for i in range(0, len(names), 2):
                jpg_name, txt_name = names[i], names[i + 1]
                if not jpg_name.endswith(".jpg") or not txt_name.endswith(".txt"):
                    fail(f"{shard_path.name} member pair {i} not (jpg, txt): {jpg_name}, {txt_name}")
                if jpg_name[:-4] != txt_name[:-4]:
                    fail(f"{shard_path.name} member pair {i} ids don't match: {jpg_name}, {txt_name}")
                img_id = jpg_name[:-4]
                if img_id in seen_ids:
                    fail(f"duplicate image_id {img_id} across shards")
                seen_ids.add(img_id)

            for member in tf.getmembers():
                if member.name.endswith(".txt"):
                    data = tf.extractfile(member).read()
                    if data.endswith(b"\n"):
                        fail(f"{member.name} has a trailing newline")
                    try:
                        data.decode("utf-8")
                    except UnicodeDecodeError:
                        fail(f"{member.name} is not valid UTF-8")
                elif member.name.endswith(".jpg"):
                    data = tf.extractfile(member).read()
                    img = Image.open(io.BytesIO(data))
                    img = img.convert("RGB")
                    if img.mode != "RGB":
                        fail(f"{member.name} did not decode to RGB")

    if seen_ids != set(expected_ids):
        fail(f"tar member ids don't match metadata.csv ids "
             f"(missing={set(expected_ids)-seen_ids}, extra={seen_ids-set(expected_ids)})")

    print(f"[verify] OK: {version_dir.name} -- {len(rows):,} images, "
          f"{len(shard_paths)} shard(s), all checks passed")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--version-dir", type=Path, required=True)
    args = p.parse_args()
    verify(args.version_dir)


if __name__ == "__main__":
    main()
