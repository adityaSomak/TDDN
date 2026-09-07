#!/usr/bin/env python
"""Environment sanity check: load the model and caption one real image. Nothing
else in the pipeline should run until this passes.

Usage: python run_check_env.py --model gemma-3-4b-it
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from PIL import Image

from pipeline.config import load_yaml


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="gemma-3-4b-it")
    args = p.parse_args()

    caption_cfg = load_yaml("caption.yaml")
    model_cfg = caption_cfg["models"][args.model]
    defaults = caption_cfg["defaults"]
    sampling = caption_cfg["sampling"]
    prompt = caption_cfg["prompt"]

    from pipeline.captioner import Captioner

    print(f"loading {model_cfg['hf_id']}...")
    captioner = Captioner(
        hf_id=model_cfg["hf_id"], tensor_parallel_size=model_cfg.get("tensor_parallel_size", 1),
        gpu_memory_utilization=defaults["gpu_memory_utilization"],
        max_model_len=defaults["max_model_len"], max_num_seqs=defaults["max_num_seqs"],
        prompt_text=prompt["text"],
    )
    img = Image.new("RGB", (512, 512), (120, 180, 220))
    result = captioner.caption_batch([img], temperature=sampling["temperature"],
                                      max_tokens=sampling["max_tokens"], seed=sampling["seed"])
    print("caption:", result[0].text)
    print("check-env: OK")


if __name__ == "__main__":
    main()
