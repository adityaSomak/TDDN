"""Fetch candidate images from the source dataset until enough succeed."""
from __future__ import annotations

import asyncio
import io
import json
import time
from pathlib import Path

import aiohttp
import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from PIL import Image


def load_candidates(cfg: dict, cache_dir: Path) -> pd.DataFrame:
    path = hf_hub_download(
        repo_id=cfg["hf_repo_id"], repo_type=cfg["hf_repo_type"], filename=cfg["filename"],
        local_dir=str(cache_dir),
    )
    table = pq.read_table(path, columns=["key", "url", "caption", "sha256"])
    df = table.to_pandas().rename(columns={"caption": "original_caption"})
    df = df.drop_duplicates(subset=["url"]).drop_duplicates(subset=["sha256"])
    return df.sort_values("sha256", kind="stable").drop(columns=["sha256"]).reset_index(drop=True)


def _load_done(index_path: Path) -> tuple[set[str], list[dict]]:
    if not index_path.exists():
        return set(), []
    rows = []
    with index_path.open() as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return {r["key"] for r in rows}, rows


def _save_image(data: bytes, path: Path, max_side: int, min_side: int) -> bool:
    img = Image.open(io.BytesIO(data)).convert("RGB")
    if min(img.size) < min_side:
        return False
    if max(img.size) > max_side:
        img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    img.save(path, format="JPEG", quality=92)
    return True


async def download_images(candidates: pd.DataFrame, images_dir: Path, target: int,
                           concurrency: int, timeout: int, max_side: int, min_side: int) -> int:
    images_dir.mkdir(parents=True, exist_ok=True)
    index_path = images_dir / "index.jsonl"
    done_keys, done_rows = _load_done(index_path)

    remaining = target - len(done_rows)
    if remaining <= 0:
        print(f"  download already has {len(done_rows)}/{target} images")
        return len(done_rows)

    pending = [row for row in candidates.itertuples(index=False)
               if row.key not in done_keys][:remaining * 3]
    queue: asyncio.Queue = asyncio.Queue()
    for row in pending:
        queue.put_nowait(row)

    index_lock = asyncio.Lock()
    index_f = index_path.open("a")
    count = len(done_rows)
    t0 = time.time()

    resolver = aiohttp.resolver.AsyncResolver()
    connector = aiohttp.TCPConnector(limit=0, resolver=resolver)
    headers = {"User-Agent": "Mozilla/5.0"}

    loop = asyncio.get_running_loop()

    async def worker(session: aiohttp.ClientSession) -> None:
        nonlocal count
        while count < target:
            try:
                row = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            try:
                async with session.get(row.url, timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
                    if resp.status != 200:
                        continue
                    data = await resp.read()
                # decode/resize/save is CPU-bound; run off the event loop so a
                # slow-to-decode image (e.g. absurd declared dimensions) can't
                # stall every other worker.
                ok = await asyncio.wait_for(
                    loop.run_in_executor(None, _save_image, data,
                                          images_dir / f"{row.key}.jpg", max_side, min_side),
                    timeout=timeout,
                )
            except Exception:
                ok = False
            if not ok:
                continue
            async with index_lock:
                if count >= target:
                    return
                count += 1
                index_f.write(json.dumps({"key": row.key, "url": row.url,
                                           "original_caption": row.original_caption}) + "\n")
                if count % 50 == 0 or count == target:
                    elapsed = time.time() - t0
                    rate = (count - (target - remaining)) / elapsed if elapsed > 0 else 0.0
                    left = target - count
                    eta = left / rate if rate > 0 else float("inf")
                    pct = 100.0 * count / target
                    eta_str = f"{eta/60:.1f}min" if eta != float("inf") else "?"
                    print(f"  download {count}/{target} ({pct:.0f}%)   {rate:.0f} img/s   eta {eta_str}",
                          flush=True)

    async with aiohttp.ClientSession(connector=connector, headers=headers) as session:
        workers = [asyncio.ensure_future(worker(session)) for _ in range(concurrency)]
        await asyncio.gather(*workers)

    index_f.close()
    print(f"  download done — {count} images")
    return count
