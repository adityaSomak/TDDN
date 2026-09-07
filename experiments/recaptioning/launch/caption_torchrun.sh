#!/usr/bin/env bash
# Multi-node caption launcher using torchrun instead of DeepSpeed -- same env
# contract (pipeline/distributed.py reads torchrun's RANK/LOCAL_RANK/WORLD_SIZE
# naming too), useful as a fallback if the DeepSpeed launcher misbehaves.
#
# Usage (run the SAME command on every node, changing only --node_rank):
#   ./launch/caption_torchrun.sh <node_rank> <num_nodes> <nproc_per_node> \
#       <master_addr> <master_port> <clean_parquet> [model]
set -euo pipefail

NODE_RANK="${1:?node_rank required}"
NUM_NODES="${2:?num_nodes required}"
NPROC_PER_NODE="${3:?nproc_per_node required}"
MASTER_ADDR="${4:?master_addr required}"
MASTER_PORT="${5:?master_port required}"
CLEAN_PARQUET="${6:?clean parquet path required}"
MODEL="${7:-gemma-3-27b-it}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RECAPTION_ENV_BIN="${RECAPTION_ENV_BIN:-/home/shanmukha/venvs/recaption_env/bin}"

"$RECAPTION_ENV_BIN/torchrun" \
  --nnodes "$NUM_NODES" --node_rank "$NODE_RANK" --nproc_per_node "$NPROC_PER_NODE" \
  --master_addr "$MASTER_ADDR" --master_port "$MASTER_PORT" \
  "$HERE/run_caption.py" --clean-parquet "$CLEAN_PARQUET" --model "$MODEL"
