#!/usr/bin/env bash
# Multi-node caption launcher using DeepSpeed as a process launcher only (no
# DeepSpeed inference engine involved -- run_caption.py never imports deepspeed,
# it only reads the RANK/LOCAL_RANK/WORLD_SIZE env vars this sets).
#
# Usage (run the SAME command on every node, changing only --node_rank):
#   ./launch/caption_deepspeed.sh <node_rank> <num_nodes> <hostfile> <clean_parquet> [model]
set -euo pipefail

NODE_RANK="${1:?node_rank required}"
NUM_NODES="${2:?num_nodes required}"
HOSTFILE="${3:?hostfile required}"
CLEAN_PARQUET="${4:?clean parquet path required}"
MODEL="${5:-gemma-3-27b-it}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RECAPTION_ENV_BIN="${RECAPTION_ENV_BIN:-/home/shanmukha/venvs/recaption_env/bin}"

# --no_ssh: no pdsh, no passwordless-SSH requirement -- you run this script
# yourself on each node. Even with --no_ssh, the launcher still tries `ssh
# <first_host> hostname -I` to auto-detect a master address unless one is given
# explicitly, so MASTER_ADDR/MASTER_PORT must be passed by hand. --no_local_rank:
# run_caption.py reads LOCAL_RANK from the environment rather than an injected
# --local_rank argv (both are accepted, but this keeps argv clean).
: "${MASTER_ADDR:?export MASTER_ADDR (a reachable IP/hostname for node 0) before running}"
: "${MASTER_PORT:=29500}"

"$RECAPTION_ENV_BIN/deepspeed" \
  --no_ssh --node_rank "$NODE_RANK" --num_nodes "$NUM_NODES" --hostfile "$HOSTFILE" \
  --master_addr "$MASTER_ADDR" --master_port "$MASTER_PORT" \
  --no_local_rank \
  "$HERE/run_caption.py" --clean-parquet "$CLEAN_PARQUET" --model "$MODEL"
