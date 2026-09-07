#!/usr/bin/env bash
# Single entry point for a local (single-node) run of the recaptioning pipeline.
#
# Usage:
#   HF_TOKEN=hf_... RECAPTION_ROOT=/scratch/recap OUTPUT_DIR=/scratch/recap/versions \
#     ./run.sh --valid-ids /path/to/valid_ids.parquet --pool-size 20000 \
#              --n-per-version 3000 --num-versions 4 --model gemma-3-4b-it
#
# Every stage's raw output goes to $RECAPTION_ROOT/logs/<stage>.log. Only a short
# progress line per stage is printed here. Multi-node / multi-rank captioning and
# the top-up-to-exact-count loop are not wired into this script yet -- see the plan
# doc and run_caption.py/run_pipeline.py for that; this is the local smoke-test path.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RECAPTION_ENV=/home/shanmukha/venvs/recaption_env/bin/python
LAION_DL_ENV=/home/shanmukha/venvs/laion_dl/bin/python

VALID_IDS=""
POOL_SIZE=20000
N_PER_VERSION=3000
NUM_VERSIONS=4
MODEL=gemma-3-4b-it

while [[ $# -gt 0 ]]; do
  case "$1" in
    --valid-ids) VALID_IDS="$2"; shift 2 ;;
    --pool-size) POOL_SIZE="$2"; shift 2 ;;
    --n-per-version) N_PER_VERSION="$2"; shift 2 ;;
    --num-versions) NUM_VERSIONS="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    *) echo "unknown flag: $1"; exit 1 ;;
  esac
done

if [[ -z "$VALID_IDS" ]]; then
  echo "error: --valid-ids is required (output of _url_check/export_valid_ids.py)"
  exit 1
fi
if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "error: HF_TOKEN env var is required (gemma models are gated)"
  exit 1
fi

: "${RECAPTION_ROOT:=$HOME/recaption_data}"
: "${OUTPUT_DIR:=$RECAPTION_ROOT/versions}"
export RECAPTION_ROOT

# Go offline ONLY if BOTH the model and the source dataset parquet are already
# cached locally -- a first-time run on a fresh machine (e.g. the prod cluster)
# still needs network to pull them, so this must not be a blanket default. This
# matters in particular because the URL-liveness checker's heavy concurrent DNS
# usage can otherwise starve unrelated lookups (huggingface.co included) on the
# same host. Note the source parquet is cached under $RECAPTION_ROOT/source (a
# project-local dir, via hf_hub_download's local_dir=), NOT the shared
# ~/.cache/huggingface/hub blob store the model uses -- wiping $RECAPTION_ROOT
# clears the dataset cache even if the model stays cached globally.
MODEL_HF_ID="$(grep -A2 "^  $MODEL:" "$HERE/configs/caption.yaml" | grep hf_id | awk '{print $2}')"
MODEL_CACHE_DIR="${HF_HOME:-$HOME/.cache/huggingface}/hub/models--${MODEL_HF_ID//\//--}"
SOURCE_FILENAME="$(grep filename: "$HERE/configs/pipeline.yaml" | awk '{print $2}')"
SOURCE_CACHE_PATH="$RECAPTION_ROOT/source/$SOURCE_FILENAME"
if [[ -d "$MODEL_CACHE_DIR" && -f "$SOURCE_CACHE_PATH" ]]; then
  echo "model + source dataset already cached -- running offline"
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
else
  echo "model and/or source dataset not yet cached -- this run will need network access"
fi
LOG_DIR="$RECAPTION_ROOT/logs"
mkdir -p "$LOG_DIR"

echo "== recaptioning pipeline =="
echo "root: $RECAPTION_ROOT"
echo "output: $OUTPUT_DIR"
echo "pool_size=$POOL_SIZE n_per_version=$N_PER_VERSION num_versions=$NUM_VERSIONS model=$MODEL"
echo

# Every stage's full raw output still lands in $LOG_DIR/<stage>.log for debugging;
# `tee` also lets a filtered live view through to the terminal, so progress (percent
# done, rate, ETA) is visible while a stage runs, not just a summary after it exits.
# The `grep` pattern keeps our own [stage]-prefixed lines and img2dataset's own
# progress lines, and drops vLLM/torch/img2dataset's verbose internal logging.
PROGRESS_PATTERN='^\[(manifest|download|caption|assemble)\]|worker  -|total   -|Sharding file'

run_stage() {
  local label="$1" logfile="$2"; shift 2
  set +e
  "$@" 2>&1 | tee "$logfile" | grep --line-buffered -E "$PROGRESS_PATTERN"
  local status=${PIPESTATUS[0]}
  set -e
  if [[ $status -ne 0 ]]; then
    echo "FAILED: $label -- see $logfile"
    exit 1
  fi
}

echo "[1/5] check-env"
run_stage "check-env" "$LOG_DIR/check_env.log" \
  "$RECAPTION_ENV" "$HERE/run_check_env.py" --model "$MODEL"
echo "  check-env done"

echo "[2/5] manifest"
run_stage "manifest" "$LOG_DIR/manifest.log" \
  "$RECAPTION_ENV" "$HERE/run_manifest.py" --valid-ids "$VALID_IDS"
echo "  manifest done"

echo "[3/5] download"
run_stage "download" "$LOG_DIR/download.log" \
  "$LAION_DL_ENV" "$HERE/run_download.py" --start 0 --end "$POOL_SIZE" --batch-name main
echo "  download done"

CLEAN="$RECAPTION_ROOT/download/main/clean.parquet"

echo "[4/5] caption"
run_stage "caption" "$LOG_DIR/caption.log" \
  env RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 LOCAL_SIZE=1 \
  "$RECAPTION_ENV" "$HERE/run_caption.py" --clean-parquet "$CLEAN" --model "$MODEL"
echo "  captioning done"

echo "[5/5] sampling (assemble + verify)"
run_stage "assemble" "$LOG_DIR/assemble.log" \
  "$RECAPTION_ENV" "$HERE/run_assemble.py" --clean-parquet "$CLEAN" \
  --n-per-version "$N_PER_VERSION" --num-versions "$NUM_VERSIONS"

for d in "$RECAPTION_ROOT"/versions/*/; do
  name="$(basename "$d")"
  "$RECAPTION_ENV" "$HERE/run_verify.py" --version-dir "$d" >> "$LOG_DIR/verify.log" 2>&1 \
    || { echo "FAILED verify $name -- see $LOG_DIR/verify.log"; exit 1; }
done
echo "  sampling done"

echo
echo "Done. Versions written to $RECAPTION_ROOT/versions:"
for d in "$RECAPTION_ROOT"/versions/*/; do
  name="$(basename "$d")"
  n=$(($(wc -l < "$d/metadata.csv") - 1))
  printf "  %-12s %s images\n" "$name" "$n"
done
