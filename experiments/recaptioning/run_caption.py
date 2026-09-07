#!/usr/bin/env python
"""Rank-aware vLLM captioning worker.

CUDA_VISIBLE_DEVICES is sliced *before* importing vllm (pipeline.captioner imports
vllm lazily, inside Captioner.__init__, specifically so this ordering can be enforced
here first). Must be launched under a launcher that sets RANK/WORLD_SIZE/LOCAL_RANK
(deepspeed/torchrun/srun), or bare (rank 0 of 1, single GPU).

Usage:
    python run_caption.py --clean-parquet .../clean.parquet --model gemma-3-4b-it
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline.config import load_yaml
from pipeline.distributed import resolve, slice_visible_devices
from pipeline.paths import CAPTIONS_DIR, PLAN_DIR, ensure_dirs
from pipeline.workqueue import (
    check_config_hash, chunk_output_path, my_chunks, pending_chunks, plan_chunks,
    write_chunk_atomic,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--local-rank", type=int, default=None)  # accepted+ignored: some
                                                              # launchers inject this
    p.add_argument("--clean-parquet", type=Path, action="append", required=True,
                    help="one or more download batches' clean.parquet (repeatable)")
    p.add_argument("--model", type=str, default="gemma-3-4b-it")
    p.add_argument("--tensor-parallel-size", type=int, default=None)
    p.add_argument("--allow-config-change", action="store_true")
    args = p.parse_args()

    ensure_dirs()
    dist = resolve()
    caption_cfg = load_yaml("caption.yaml")
    model_cfg = caption_cfg["models"][args.model]
    defaults = caption_cfg["defaults"]
    sampling = caption_cfg["sampling"]
    prompt = caption_cfg["prompt"]

    tp = args.tensor_parallel_size or model_cfg.get("tensor_parallel_size", defaults["tensor_parallel_size"])

    cvd = slice_visible_devices(dist.local_rank, tp)
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = cvd
    print(f"[caption] rank={dist.rank}/{dist.world_size} local_rank={dist.local_rank} "
          f"CUDA_VISIBLE_DEVICES={cvd}", flush=True)

    effective_config = {
        "model": model_cfg["hf_id"], "tp": tp, "prompt_id": prompt["id"],
        "prompt_text": prompt["text"], "temperature": sampling["temperature"],
        "max_tokens": sampling["max_tokens"], "seed": sampling["seed"],
    }
    if dist.rank == 0:
        config_hash = check_config_hash(PLAN_DIR, effective_config, args.allow_config_change)
    else:
        config_hash = None  # each rank checks/writes independently below; fine for a single shared FS

    import pandas as pd
    frames = [pd.read_parquet(p) for p in args.clean_parquet]
    pool = pd.concat(frames, ignore_index=True).reset_index(drop=True)
    pool_size = len(pool)
    print(f"[caption] pool: {pool_size:,} images from {len(args.clean_parquet)} batch(es)")

    chunk_size = defaults["chunk_size"]
    chunks = plan_chunks(pool_size, chunk_size)
    (PLAN_DIR / "chunks.json").write_text(json.dumps(chunks))

    pending = pending_chunks(chunks, CAPTIONS_DIR)
    mine = my_chunks(pending, dist.rank, dist.world_size)
    print(f"[caption] {len(chunks)} total chunks, {len(pending)} pending, "
          f"{len(mine)} assigned to this rank")

    if not mine:
        print("[caption] nothing to do for this rank")
        return

    from pipeline.captioner import Captioner
    from PIL import Image

    hf_id = model_cfg["hf_id"]
    print(f"[caption] loading {hf_id} (tp={tp})...")
    t0 = time.time()
    captioner = Captioner(
        hf_id=hf_id, tensor_parallel_size=tp,
        gpu_memory_utilization=defaults["gpu_memory_utilization"],
        max_model_len=defaults["max_model_len"], max_num_seqs=defaults["max_num_seqs"],
        prompt_text=prompt["text"],
    )
    print(f"[caption] model loaded in {time.time()-t0:.1f}s")

    progress_path = CAPTIONS_DIR / f"progress_rank{dist.rank}.json"
    t_start = time.time()
    images_done = 0
    total_images_mine = sum(end - start for _, start, end in mine)

    for chunk_id, start, end in mine:
        sub = pool.iloc[start:end]
        MIN_SIDE = 32  # degenerate images (e.g. a 1x1 placeholder) crash the VLM
                       # processor and take the whole batch down with them; download
                       # is supposed to filter these via min_image_size, but check
                       # again here as defense in depth against already-downloaded
                       # batches from before that filter existed.
        images, valid_rows = [], []
        for _, row in sub.iterrows():
            try:
                img = Image.open(row["jpg_path"]).convert("RGB")
                if min(img.size) < MIN_SIDE:
                    print(f"[caption] skipping degenerate image {row['jpg_path']} (size={img.size})")
                    continue
                images.append(img)
                valid_rows.append(row)
            except Exception as e:  # noqa: BLE001 -- a corrupt/unreadable file, skip it
                print(f"[caption] skipping unreadable image {row['jpg_path']}: {e}")

        if images:
            results = captioner.caption_batch(
                images, temperature=sampling["temperature"],
                max_tokens=sampling["max_tokens"], seed=sampling["seed"],
            )
        else:
            results = []

        out_rows = [
            {"key": row["key"], "url": row["url"], "original_caption": row["original_caption"],
             "new_caption_raw": res.text}
            for row, res in zip(valid_rows, results)
        ]
        header = {"chunk_id": chunk_id, "n_expected": len(out_rows), "config_hash": config_hash,
                  "rank": dist.rank}
        write_chunk_atomic(CAPTIONS_DIR, chunk_id, header, out_rows)

        images_done += len(out_rows)
        elapsed = time.time() - t_start
        rate = images_done / elapsed if elapsed > 0 else 0.0
        eta = (total_images_mine - images_done) / rate if rate > 0 else float("inf")
        progress_path.write_text(json.dumps({
            "rank": dist.rank, "world_size": dist.world_size,
            "chunks_done": mine.index((chunk_id, start, end)) + 1, "chunks_mine": len(mine),
            "images_done": images_done, "images_mine": total_images_mine,
            "rate_img_s": round(rate, 2), "eta_s": round(eta, 1),
        }, indent=2))
        pct = 100.0 * images_done / total_images_mine if total_images_mine else 100.0
        eta_str = f"{eta/60:.1f}min" if eta != float("inf") else "?"
        print(f"[caption] rank{dist.rank} {images_done}/{total_images_mine} ({pct:.0f}%) "
              f"{rate:.1f} img/s  eta {eta_str}", flush=True)

    print(f"[caption] DONE rank{dist.rank}: {images_done} images in {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
