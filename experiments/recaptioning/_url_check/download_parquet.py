"""One-off: fetch the guangyil/laion-coco-aesthetic parquet locally.

Run with crg_env:
    /home/shanmukha/miniconda3/envs/crg_env/bin/python download_parquet.py
"""
from pathlib import Path

from huggingface_hub import hf_hub_download

HERE = Path(__file__).resolve().parent
DEST_DIR = HERE / "data"

if __name__ == "__main__":
    path = hf_hub_download(
        repo_id="guangyil/laion-coco-aesthetic",
        repo_type="dataset",
        filename="laion-coco_v3_filter.parquet",
        local_dir=str(DEST_DIR),
    )
    print(f"Downloaded to: {path}")
