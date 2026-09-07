# Recaptioning

Produces N seeded versions of a recaptioned LAION-derived dataset (image + new VLM
caption + original source caption), matching the format of the previously published
`PuzzleBench/Recaptioned_LAION` dataset. See `../../../.claude/plans/...` (or ask
for the design doc) for the full design rationale; this README is just setup + run.

## 1. One-time setup

### 1a. Environments

Two separate Python environments (kept apart deliberately -- img2dataset's pins
would otherwise downgrade the vLLM env's dependencies):

```bash
# vLLM + captioning (torch 2.6.0+cu124, vllm 0.8.5.post1 -- pinned for this driver;
# see env/recaption_env.txt for the exact versions verified to work together)
python3.11 -m venv /path/to/recaption_env
/path/to/recaption_env/bin/pip install -r env/recaption_env.txt

# img2dataset download (no GPU needed)
python3.11 -m venv /path/to/laion_dl
/path/to/laion_dl/bin/pip install -r env/laion_dl.txt
```

Edit `RECAPTION_ENV` / `LAION_DL_ENV` at the top of `run.sh` to point at these paths
(or symlink them to `/home/shanmukha/venvs/recaption_env` and `.../laion_dl`).

**Verify the env works before anything else:**
```bash
/path/to/recaption_env/bin/python run_check_env.py --model gemma-3-4b-it
```
This loads the small model and captions one synthetic image. Must print
`check-env: OK` before proceeding -- if it fails, nothing downstream will work either.

### 1b. Get a HuggingFace token

`google/gemma-3-*` models are gated. Accept the license at
https://huggingface.co/google/gemma-3-27b-it (and the 4b variant, for smoke tests),
then create a token at https://huggingface.co/settings/tokens and export it:
```bash
export HF_TOKEN=hf_...
```

### 1c. URL-liveness pre-check (run once, ~20-30 min at full scale)

The source dataset (`guangyil/laion-coco-aesthetic`, 8.56M rows) has many dead URLs
(2022-vintage crawl). Rather than discover this during the expensive download step,
we check every URL directly first -- streamed GET, magic-byte sniff, connection
closed immediately, no image bytes ever hit disk:

```bash
cd _url_check
/path/to/crg_env/bin/python download_parquet.py          # fetches the source parquet once
/path/to/crg_env/bin/python check_urls.py --limit 20000   # pilot -- confirms it works
# Full run, sharded across N processes for throughput (see check_urls.py --help):
for i in 0 1 2 3; do
  python check_urls.py --shard-idx $i --num-shards 4 --concurrency 800 \
    --output-name checked_shard${i}.jsonl &
done
wait
python export_valid_ids.py   # -> output/valid_ids.parquet
```

Needs `aiodns`/`pycares` installed (`pip install aiodns pycares`) -- without it,
aiohttp's default DNS resolver serializes through a small thread pool and this will
be far slower and less accurate at this URL count (millions of distinct domains
mean almost every request needs a fresh, uncached lookup).

This step is resumable (skips already-checked keys) and safe to kill/restart, and
its output (`valid_ids.parquet`) is a one-time asset -- rerun it only if you want a
fresher liveness snapshot.

## 2. Run it

### Single node (dev box / smoke test)

```bash
export HF_TOKEN=hf_...
export RECAPTION_ROOT=/path/to/scratch   # heavy intermediate data lives here
./run.sh --valid-ids _url_check/output/valid_ids.parquet \
         --pool-size 2000 --n-per-version 300 --num-versions 4 \
         --model gemma-3-4b-it
```

This runs `check-env -> manifest -> download -> caption -> assemble -> verify` in
order. Raw stage output goes to `$RECAPTION_ROOT/logs/<stage>.log`; only a short
summary per stage prints to the terminal. On success:
```
Done. Versions written to $RECAPTION_ROOT/versions:
  laion_s0     300 images
  laion_s1     300 images
  ...
```

For the real run, drop `--pool-size`/`--n-per-version` to the config defaults
(`configs/pipeline.yaml`: `n_per_version: 508025`, `num_versions: 4`) and swap
`--model gemma-3-27b-it`. **Before committing to an exact pool size, check the real
gate-cascade numbers** (`run_manifest.py` prints the post-dedup pool count) against
how many you need (`n_per_version * num_versions`), and consider running a ~50k-row
download batch first to measure the real download-success yield on your network --
see `run_download.py`, which can target any `--start`/`--end` row range.

### Multi-node (production, 2+ nodes)

`run.sh` is single-node only. For multi-node, run `run_caption.py` directly under a
launcher that sets `RANK`/`LOCAL_RANK`/`WORLD_SIZE` (see `pipeline/distributed.py`
for exactly which env vars are read, across DeepSpeed/torchrun/Slurm naming):

```bash
# DeepSpeed launcher (recommended -- see launch/caption_deepspeed.sh)
deepspeed --no_ssh --node_rank 0 --hostfile hostfile.example \
  run_caption.py --clean-parquet /shared/path/clean.parquet --model gemma-3-27b-it
# (same command with --node_rank 1 on the second node)
```

**Requirements for multi-node:**
- `RECAPTION_ROOT` (specifically `plan/` and `captions/`) must be on a filesystem
  shared across all nodes -- resumability and progress tracking both depend on
  every rank being able to see every other rank's completed chunk files. This is
  standard for any multi-node GPU cluster (shared NFS or a parallel FS), but
  confirm it before the real run.
- `HF_TOKEN` and any other env vars must reach every node -- DeepSpeed's default
  pdsh-based launcher only forwards `NCCL*`-prefixed vars; put everything else in a
  `.deepspeed_env` file (see DeepSpeed's docs) or use `--no_ssh` and export them
  yourself on each node before launching.
- Kill-and-restart is safe at any point, including with a *different* number of
  ranks/nodes than the previous attempt -- each rank recomputes which chunks still
  need work at startup rather than relying on a fixed assignment.

### Checking progress mid-run

Every stage writes `progress.json` (or `progress_rank{R}.json` for captioning, one
per rank) under `$RECAPTION_ROOT`. From any shell, including a different node:
```bash
cat $RECAPTION_ROOT/captions/progress_rank*.json
watch -n 5 'cat $RECAPTION_ROOT/captions/progress_rank*.json'
```

## 3. Output format

Each version (`$RECAPTION_ROOT/versions/<name>/`) contains:
- `metadata.csv`: `image_id,url,caption,original_caption` -- `caption` is the new
  VLM recaption, `original_caption` is the source BLIP caption (kept, unlike the
  previously published dataset which dropped it).
- `shards-*.tar`: WebDataset shards, flat, `<image_id>.jpg` + `<image_id>.txt` pairs
  (`.txt` = the new caption only, matching what training actually consumes).
- `provenance.json`: seed, image count, normalizer version, git commit, per-shard
  sha256 (for verifying byte-for-byte reproducibility).

Run `run_verify.py --version-dir <path>` on any version before trusting it --
checks header/pad-width/geometry/UTF-8/shard-sha256 consistency.
