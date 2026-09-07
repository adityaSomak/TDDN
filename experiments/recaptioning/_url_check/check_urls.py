"""Check URL validity across guangyil/laion-coco-aesthetic without keeping any images.

For each row we issue a streaming GET, read a small prefix of the body, sniff it
against known image magic bytes, and immediately close the connection. Nothing is
written to disk except one JSONL result line per row (id, url, validity, and cheap
metadata) -- no image bytes ever touch disk.

Resumable: on startup we read every existing output shard and skip any `key` already
present, so a killed/restarted run just picks up where it left off. Safe to run the
same command again.

Usage (crg_env):
    python check_urls.py --limit 20000                      # pilot
    python check_urls.py --shard-idx 0 --num-shards 1        # full run, foreground
    python check_urls.py --concurrency 800                   # tune concurrency
"""
from __future__ import annotations

import argparse
import asyncio
import json
import signal
import sys
import time
from pathlib import Path

import aiohttp
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
PARQUET_PATH = HERE / "data" / "laion-coco_v3_filter.parquet"
OUTPUT_DIR = HERE / "output"

USER_AGENT = "TDDN-research-dataset-url-check/1.0 (+https://github.com/adityaSomak/TDDN)"

# Magic-byte sniffers for the formats LAION-COCO actually contains (imghdr is gone in 3.13).
_MAGIC = [
    ("jpeg", lambda b: b[:3] == b"\xff\xd8\xff"),
    ("png", lambda b: b[:8] == b"\x89PNG\r\n\x1a\n"),
    ("gif", lambda b: b[:6] in (b"GIF87a", b"GIF89a")),
    ("webp", lambda b: b[:4] == b"RIFF" and b[8:12] == b"WEBP"),
    ("bmp", lambda b: b[:2] == b"BM"),
]


def sniff_format(prefix: bytes) -> str | None:
    for name, test in _MAGIC:
        try:
            if test(prefix):
                return name
        except IndexError:
            continue
    return None


async def check_one(session: aiohttp.ClientSession, key: str, url: str,
                     timeout_s: float, prefix_bytes: int) -> dict:
    t0 = time.monotonic()
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=timeout_s),
                                allow_redirects=True) as resp:
            status = resp.status
            content_length = resp.headers.get("Content-Length")
            content_type = resp.headers.get("Content-Type")
            prefix = b""
            if status == 200:
                async for chunk in resp.content.iter_chunked(prefix_bytes):
                    prefix = chunk
                    break
            fmt = sniff_format(prefix) if prefix else None
            valid = status == 200 and fmt is not None
            return {
                "key": key,
                "url": url,
                "valid": valid,
                "status": status,
                "content_type": content_type,
                "content_length": int(content_length) if content_length and content_length.isdigit() else None,
                "sniffed_format": fmt,
                "elapsed_ms": round((time.monotonic() - t0) * 1000),
                "error": None,
            }
    except asyncio.TimeoutError:
        return _err(key, url, "timeout", t0)
    except aiohttp.ClientError as e:
        return _err(key, url, f"client_error:{type(e).__name__}", t0)
    except Exception as e:  # noqa: BLE001 - record and move on, never crash the run
        return _err(key, url, f"other:{type(e).__name__}", t0)


def _err(key: str, url: str, reason: str, t0: float) -> dict:
    return {
        "key": key, "url": url, "valid": False, "status": None,
        "content_type": None, "content_length": None, "sniffed_format": None,
        "elapsed_ms": round((time.monotonic() - t0) * 1000), "error": reason,
    }


def load_done_keys(out_path: Path) -> set[str]:
    done: set[str] = set()
    if not out_path.exists():
        return done
    with out_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["key"])
            except (json.JSONDecodeError, KeyError):
                continue  # tolerate a torn last line from a killed run
    return done


async def run(args: argparse.Namespace) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / args.output_name
    done_keys = load_done_keys(out_path)
    if done_keys:
        print(f"[resume] {len(done_keys):,} keys already checked in {out_path}, skipping them")

    table = pq.read_table(PARQUET_PATH, columns=["key", "url"])
    keys = table.column("key").to_pylist()
    urls = table.column("url").to_pylist()
    del table

    if args.limit is not None:
        keys, urls = keys[: args.limit], urls[: args.limit]

    if args.num_shards > 1:
        keys = keys[args.shard_idx :: args.num_shards]
        urls = urls[args.shard_idx :: args.num_shards]

    pending = [(k, u) for k, u in zip(keys, urls) if k not in done_keys and u]
    print(f"[plan] {len(pending):,} URLs to check (of {len(keys):,} in this shard)")
    if not pending:
        print("[done] nothing to do")
        return

    # A fixed pool of `concurrency` worker coroutines pulling from a pre-filled queue --
    # NOT one asyncio.Task per URL. With 8.5M+ rows, materializing that many Task/coroutine
    # objects upfront (as `asyncio.ensure_future` per row + semaphore would) wastes several
    # GB of RAM and slows the event loop's own bookkeeping for no benefit; a bounded worker
    # pool keeps memory at O(concurrency) regardless of how many rows are queued.
    queue: asyncio.Queue = asyncio.Queue()
    for item in pending:
        queue.put_nowait(item)

    # AsyncResolver (aiodns/c-ares) does real non-blocking DNS I/O. The default
    # ThreadedResolver instead does blocking getaddrinfo() calls via loop.run_in_executor(),
    # capped at asyncio's default thread-pool size (~32 workers here) regardless of
    # `concurrency` -- with URLs spread across millions of distinct domains, nearly every
    # request needs a fresh lookup, so that 32-thread cap was the true bottleneck, not the
    # network or the target servers.
    resolver = aiohttp.resolver.AsyncResolver()
    connector = aiohttp.TCPConnector(limit=0, ttl_dns_cache=300, use_dns_cache=True, resolver=resolver)
    headers = {"User-Agent": USER_AGENT}

    n_valid = n_invalid = 0
    completed = 0
    t_start = time.monotonic()
    stop = asyncio.Event()

    def _handle_sigterm(*_a):
        print("\n[signal] stopping after in-flight requests drain...", file=sys.stderr)
        stop.set()

    signal.signal(signal.SIGINT, _handle_sigterm)
    signal.signal(signal.SIGTERM, _handle_sigterm)

    out_f = out_path.open("a", buffering=1)
    write_lock = asyncio.Lock()

    async def worker(session: aiohttp.ClientSession) -> None:
        nonlocal n_valid, n_invalid, completed
        while not stop.is_set():
            try:
                key, url = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            result = await check_one(session, key, url, args.timeout, args.prefix_bytes)
            async with write_lock:
                out_f.write(json.dumps(result) + "\n")
                completed += 1
                if result["valid"]:
                    n_valid += 1
                else:
                    n_invalid += 1
                if completed % args.log_every == 0:
                    elapsed = time.monotonic() - t_start
                    rate = completed / elapsed if elapsed > 0 else 0.0
                    pct_valid = 100.0 * n_valid / completed
                    remaining = len(pending) - completed
                    eta_s = remaining / rate if rate > 0 else float("inf")
                    print(
                        f"[{completed:>9,}/{len(pending):,}] "
                        f"valid={pct_valid:5.1f}%  rate={rate:6.1f} req/s  "
                        f"eta={eta_s / 60:6.1f} min",
                        flush=True,
                    )

    async with aiohttp.ClientSession(connector=connector, headers=headers) as session:
        workers = [asyncio.ensure_future(worker(session)) for _ in range(args.concurrency)]
        await asyncio.gather(*workers)

    out_f.close()
    elapsed = time.monotonic() - t_start
    total = n_valid + n_invalid
    print(f"\n[summary] checked={total:,}  valid={n_valid:,} ({100*n_valid/max(total,1):.1f}%)  "
          f"invalid={n_invalid:,}  wall_clock={elapsed/60:.1f} min  "
          f"rate={total/max(elapsed,1e-9):.1f} req/s")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--limit", type=int, default=None, help="only check the first N rows (pilot runs)")
    p.add_argument("--shard-idx", type=int, default=0, help="this process's shard index, for splitting the full run across processes")
    p.add_argument("--num-shards", type=int, default=1, help="total number of shards")
    p.add_argument("--concurrency", type=int, default=500, help="max in-flight requests")
    p.add_argument("--timeout", type=float, default=10.0, help="per-request timeout, seconds")
    p.add_argument("--prefix-bytes", type=int, default=4096, help="bytes read before aborting the connection")
    p.add_argument("--output-name", type=str, default="checked.jsonl", help="output JSONL filename under output/")
    p.add_argument("--log-every", type=int, default=2000, help="print a progress line every N completions")
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
