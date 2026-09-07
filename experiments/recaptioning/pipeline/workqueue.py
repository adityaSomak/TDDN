"""Deliberately simple chunked resumability for run_caption.py.

A fixed chunk plan (row ranges into the pool), skip-if-output-exists resume computed
fresh at every startup (so restarting with a different world_size still converges --
no chunk_idx % world_size baked in at plan time), atomic os.rename on completion.

No claims/heartbeat/stale-lock-reaping/backup-tasks: a launcher like DeepSpeed already
kills the whole job the moment any rank dies, so "rerun the same command" is the
recovery path regardless of how fine-grained the workqueue is.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


def plan_chunks(pool_size: int, chunk_size: int) -> list[tuple[int, int, int]]:
    """[(chunk_id, start, end), ...] covering [0, pool_size) in order."""
    chunks = []
    start = 0
    cid = 0
    while start < pool_size:
        end = min(start + chunk_size, pool_size)
        chunks.append((cid, start, end))
        start = end
        cid += 1
    return chunks


def chunk_output_path(captions_dir: Path, chunk_id: int) -> Path:
    return captions_dir / f"chunk-{chunk_id:05d}.jsonl"


def pending_chunks(chunks: list[tuple[int, int, int]], captions_dir: Path) -> list[tuple[int, int, int]]:
    return [c for c in chunks if not chunk_output_path(captions_dir, c[0]).exists()]


def my_chunks(pending: list[tuple[int, int, int]], rank: int, world_size: int) -> list[tuple[int, int, int]]:
    return pending[rank::world_size]


def config_hash(effective_config: dict) -> str:
    blob = json.dumps(effective_config, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def check_config_hash(plan_dir: Path, effective_config: dict, allow_change: bool) -> str:
    h = config_hash(effective_config)
    hash_path = plan_dir / "CONFIG_HASH"
    if hash_path.exists():
        prev = hash_path.read_text().strip()
        if prev != h and not allow_change:
            raise SystemExit(
                f"Caption config changed (hash {prev} -> {h}) since this run started. "
                f"Pass --allow-config-change if this is intentional, otherwise you'd "
                f"produce a mixed-config dataset like the old pipeline did."
            )
    hash_path.parent.mkdir(parents=True, exist_ok=True)
    hash_path.write_text(h)
    return h


def write_chunk_atomic(captions_dir: Path, chunk_id: int, header: dict, rows: list[dict]) -> None:
    final_path = chunk_output_path(captions_dir, chunk_id)
    tmp_path = captions_dir / f"chunk-{chunk_id:05d}.jsonl.tmp.{os.getpid()}"
    with tmp_path.open("w") as f:
        f.write(json.dumps(header) + "\n")
        for row in rows:
            f.write(json.dumps(row) + "\n")
    os.rename(tmp_path, final_path)  # atomic on POSIX and NFS
