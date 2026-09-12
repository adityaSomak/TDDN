"""Rank-aware vLLM captioning. Resumable across restarts and across a changed
world_size: at startup every rank recomputes which chunks are still missing from
what's on disk, rather than relying on a fixed assignment made in the past.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import pandas as pd
from PIL import Image


def rank_info() -> tuple[int, int, int, int]:
    def first_int(*names: str, default: int) -> int:
        for name in names:
            if name in os.environ:
                return int(os.environ[name])
        return default

    rank = first_int("RANK", "SLURM_PROCID", default=0)
    world_size = first_int("WORLD_SIZE", "SLURM_NTASKS", default=1)
    local_rank = first_int("LOCAL_RANK", "SLURM_LOCALID", default=0)
    local_size = first_int("LOCAL_SIZE", "LOCAL_WORLD_SIZE", "SLURM_NTASKS_PER_NODE",
                            default=world_size)
    return rank, world_size, local_rank, local_size


def _slice_visible_devices(local_rank: int, tensor_parallel: int) -> None:
    # A launcher may hand every local rank the node's full GPU list, so slice by
    # position rather than range(). Must run before `import vllm`.
    import torch

    inherited = os.environ.get("CUDA_VISIBLE_DEVICES")
    avail = inherited.split(",") if inherited else [str(i) for i in range(torch.cuda.device_count())]
    lo, hi = local_rank * tensor_parallel, (local_rank + 1) * tensor_parallel
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(avail[lo:hi])


def _config_hash(model: str, tensor_parallel: int, cfg: dict) -> str:
    blob = json.dumps({"model": model, "tp": tensor_parallel, **cfg}, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def _plan_chunks(n: int, chunk_size: int) -> list[tuple[int, int, int]]:
    chunks, start, cid = [], 0, 0
    while start < n:
        end = min(start + chunk_size, n)
        chunks.append((cid, start, end))
        start, cid = end, cid + 1
    return chunks


def _chunk_path(captions_dir: Path, chunk_id: int) -> Path:
    return captions_dir / f"chunk-{chunk_id:05d}.jsonl"


def _build_prompt(hf_id: str, prompt_text: str):
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(hf_id)
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt_text}]}]
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def caption(images_dir: Path, captions_dir: Path, model: str, tensor_parallel: int,
            cfg: dict, allow_config_change: bool = False) -> None:
    captions_dir.mkdir(parents=True, exist_ok=True)
    rank, world_size, local_rank, _ = rank_info()
    _slice_visible_devices(local_rank, tensor_parallel)

    config_hash = _config_hash(model, tensor_parallel, cfg)
    hash_path = captions_dir / "CONFIG_HASH"
    if hash_path.exists():
        prev = hash_path.read_text().strip()
        if prev != config_hash and not allow_config_change:
            raise SystemExit(
                f"caption config changed (hash {prev} -> {config_hash}) since this run "
                f"started; pass allow_config_change to mix configs on purpose"
            )
    if rank == 0:
        hash_path.write_text(config_hash)

    with (images_dir / "index.jsonl").open() as f:
        index = pd.DataFrame([json.loads(line) for line in f])
    chunks = _plan_chunks(len(index), cfg["chunk_size"])
    pending = [c for c in chunks if not _chunk_path(captions_dir, c[0]).exists()]
    mine = pending[rank::world_size]
    print(f"  caption   rank{rank}: {len(chunks)} total chunks, {len(mine)} assigned")
    if not mine:
        return

    from vllm import LLM, SamplingParams

    llm = LLM(model=model, tensor_parallel_size=tensor_parallel,
              gpu_memory_utilization=cfg["gpu_memory_utilization"],
              max_model_len=cfg["max_model_len"], max_num_seqs=cfg["max_num_seqs"],
              dtype="bfloat16", limit_mm_per_prompt={"image": 1})
    prompt = _build_prompt(model, cfg["prompt"])
    sampling = SamplingParams(temperature=cfg["temperature"], max_tokens=cfg["max_tokens"],
                               seed=cfg["seed"])

    total = sum(end - start for _, start, end in mine)
    done, t0 = 0, time.time()
    for chunk_id, start, end in mine:
        sub = index.iloc[start:end]
        images, rows = [], []
        for row in sub.itertuples(index=False):
            try:
                img = Image.open(images_dir / f"{row.key}.jpg").convert("RGB")
            except Exception:
                continue
            if min(img.size) < 32:
                continue
            images.append(img)
            rows.append(row)

        outputs = llm.generate(
            [{"prompt": prompt, "multi_modal_data": {"image": img}} for img in images],
            sampling, use_tqdm=False,
        ) if images else []

        out_rows = [{"key": row.key, "url": row.url, "original_caption": row.original_caption,
                     "caption": out.outputs[0].text}
                    for row, out in zip(rows, outputs)]

        tmp = captions_dir / f"chunk-{chunk_id:05d}.jsonl.tmp.{os.getpid()}"
        with tmp.open("w") as f:
            f.write(json.dumps({"chunk_id": chunk_id, "config_hash": config_hash}) + "\n")
            for r in out_rows:
                f.write(json.dumps(r) + "\n")
        os.rename(tmp, _chunk_path(captions_dir, chunk_id))

        done += len(out_rows)
        elapsed = time.time() - t0
        rate = done / elapsed if elapsed > 0 else 0.0
        eta = (total - done) / rate if rate > 0 else 0.0
        pct = 100.0 * done / total if total else 100.0
        print(f"  caption   rank{rank} {done}/{total} ({pct:.0f}%)   {rate:.1f} img/s   "
              f"eta {eta/60:.1f}min", flush=True)

    print(f"  caption done — rank{rank}: {done} captions")
