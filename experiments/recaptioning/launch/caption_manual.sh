#!/usr/bin/env bash
# Reference implementation of the env contract pipeline/distributed.py expects --
# the fallback when neither DeepSpeed nor torchrun is available/working. Run this
# same script on every node with a different NODE_RANK; it spawns one process per
# local GPU and exports RANK/LOCAL_RANK/WORLD_SIZE/LOCAL_SIZE by hand.
#
# Usage:
#   ./launch/caption_manual.sh <node_rank> <num_nodes> <gpus_per_node> <clean_parquet> [model]
set -euo pipefail

NODE_RANK="${1:?node_rank required}"
NUM_NODES="${2:?num_nodes required}"
GPUS_PER_NODE="${3:?gpus_per_node required}"
CLEAN_PARQUET="${4:?clean parquet path required}"
MODEL="${5:-gemma-3-27b-it}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RECAPTION_ENV="${RECAPTION_ENV:-/home/shanmukha/venvs/recaption_env/bin/python}"

WORLD_SIZE=$((NUM_NODES * GPUS_PER_NODE))
pids=()
for local_rank in $(seq 0 $((GPUS_PER_NODE - 1))); do
  rank=$((NODE_RANK * GPUS_PER_NODE + local_rank))
  RANK=$rank WORLD_SIZE=$WORLD_SIZE LOCAL_RANK=$local_rank LOCAL_SIZE=$GPUS_PER_NODE \
    "$RECAPTION_ENV" "$HERE/run_caption.py" --clean-parquet "$CLEAN_PARQUET" --model "$MODEL" &
  pids+=($!)
done
wait "${pids[@]}"
