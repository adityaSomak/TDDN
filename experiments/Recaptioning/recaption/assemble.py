"""Split the captioned pool into disjoint seeded versions and pack each as a
WebDataset: metadata.csv + shards-*.tar + provenance.json, then verify it."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import tarfile
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


_PREAMBLE_RE = re.compile(
    r"^\s*(here'?s?\s+(is\s+)?(a\s+|an\s+|the\s+)?description[:\-]?\s*|sure[,!]?\s*|certainly[,!]?\s*)",
    re.IGNORECASE,
)
_BULLET_RE = re.compile(r"^\s*[-*•]\s+")
_WS_RE = re.compile(r"\s+")


def normalize_caption(raw: str) -> str:
    text = unicodedata.normalize("NFC", raw)
    text = _PREAMBLE_RE.sub("", text)
    text = _BULLET_RE.sub("", text)
    text = text.strip().strip('"').strip("'").strip()
    text = _WS_RE.sub(" ", text)
    return "".join(c for c in text if c == " " or not unicodedata.category(c).startswith("C")).strip()


def _load_pool(images_dir: Path, captions_dir: Path) -> pd.DataFrame:
    with (images_dir / "index.jsonl").open() as f:
        index = pd.DataFrame([json.loads(line) for line in f])
    rows = []
    for path in sorted(captions_dir.glob("chunk-*.jsonl")):
        lines = path.read_text().splitlines()[1:]
        rows.extend(json.loads(line) for line in lines)
    captions = pd.DataFrame(rows)[["key", "caption"]]
    return index.merge(captions, on="key", how="inner")


def _add_bytes(tf: tarfile.TarFile, name: str, data: bytes) -> None:
    ti = tarfile.TarInfo(name=name)
    ti.size = len(data)
    ti.mtime = 0
    ti.uid = ti.gid = 0
    ti.uname = ti.gname = ""
    ti.mode = 0o644
    tf.addfile(ti, io.BytesIO(data))


def _emit_version(df: pd.DataFrame, out_dir: Path, seed: int, pairs_per_shard: int,
                   model: str, prompt: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(df)
    pad = max(6, len(str(n - 1)))
    metadata_path = out_dir / "metadata.csv"
    shard_sha256s = []

    with metadata_path.open("w", newline="") as meta_f:
        writer = csv.writer(meta_f)
        writer.writerow(["image_id", "url", "caption", "original_caption"])
        shard_idx, tf = 0, None
        for i, row in enumerate(df.itertuples(index=False)):
            if i % pairs_per_shard == 0:
                if tf is not None:
                    tf.close()
                shard_path = out_dir / f"shards-{shard_idx:05d}.tar"
                tf = tarfile.open(shard_path, "w", format=tarfile.GNU_FORMAT)
                shard_idx += 1
            image_id = str(i).zfill(pad)
            caption = normalize_caption(row.caption)
            writer.writerow([image_id, row.url, caption, row.original_caption])
            with open(row.image_path, "rb") as jf:
                _add_bytes(tf, f"{image_id}.jpg", jf.read())
            _add_bytes(tf, f"{image_id}.txt", caption.encode("utf-8"))
        if tf is not None:
            tf.close()

    for shard_path in sorted(out_dir.glob("shards-*.tar")):
        shard_sha256s.append({"shard": shard_path.name,
                               "sha256": hashlib.sha256(shard_path.read_bytes()).hexdigest()})

    (out_dir / "provenance.json").write_text(json.dumps({
        "seed": seed, "n_images": n, "model": model, "prompt": prompt,
        "shard_sha256s": shard_sha256s,
    }, indent=2))


def verify(out_dir: Path) -> None:
    def fail(msg: str) -> None:
        raise ValueError(f"verify failed for {out_dir.name}: {msg}")

    provenance = json.loads((out_dir / "provenance.json").read_text())
    with (out_dir / "metadata.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != provenance["n_images"]:
        fail(f"metadata.csv has {len(rows)} rows, provenance says {provenance['n_images']}")

    pad = len(rows[0]["image_id"]) if rows else 6
    expected_ids = [str(i).zfill(pad) for i in range(len(rows))]
    if [r["image_id"] for r in rows] != expected_ids:
        fail("image_id not contiguous from 0")

    sha_by_name = {s["shard"]: s["sha256"] for s in provenance["shard_sha256s"]}
    seen = set()
    for shard_path in sorted(out_dir.glob("shards-*.tar")):
        actual = hashlib.sha256(shard_path.read_bytes()).hexdigest()
        if sha_by_name.get(shard_path.name) != actual:
            fail(f"{shard_path.name} sha256 mismatch")
        with tarfile.open(shard_path) as tf:
            names = tf.getnames()
            if len(names) % 2 != 0:
                fail(f"{shard_path.name} has an odd member count")
            for i in range(0, len(names), 2):
                jpg, txt = names[i], names[i + 1]
                if jpg[:-4] != txt[:-4] or not jpg.endswith(".jpg") or not txt.endswith(".txt"):
                    fail(f"{shard_path.name} member pair {i} malformed: {jpg}, {txt}")
                seen.add(jpg[:-4])
            for member in tf.getmembers():
                if member.name.endswith(".txt"):
                    data = tf.extractfile(member).read()
                    if data.endswith(b"\n"):
                        fail(f"{member.name} has a trailing newline")
                    data.decode("utf-8")
                else:
                    Image.open(io.BytesIO(tf.extractfile(member).read())).convert("RGB")
    if seen != set(expected_ids):
        fail("tar member ids don't match metadata.csv")


def assemble(images_dir: Path, captions_dir: Path, out_dir: Path, seeds: int, per_seed: int,
             master_seed: int, pairs_per_shard: int, model: str, prompt: str) -> None:
    pool = _load_pool(images_dir, captions_dir)
    pool["image_path"] = pool["key"].apply(lambda k: images_dir / f"{k}.jpg")
    needed = seeds * per_seed
    if len(pool) < needed:
        raise ValueError(f"captioned pool has {len(pool)} images, need {needed} "
                          f"for {seeds} seeds of {per_seed}")

    rng = np.random.default_rng(master_seed)
    order = rng.permutation(len(pool))[:needed]
    shuffled = pool.iloc[order].reset_index(drop=True)

    for s in range(seeds):
        version_df = shuffled.iloc[s * per_seed:(s + 1) * per_seed].reset_index(drop=True)
        version_dir = out_dir / "versions" / f"seed{s}"
        _emit_version(version_df, version_dir, s, pairs_per_shard, model, prompt)
        verify(version_dir)
        print(f"  assemble seed{s} {per_seed}/{per_seed} (100%) -> {version_dir}")
