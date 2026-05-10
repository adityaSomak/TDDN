#!/usr/bin/env bash
# Launch one or more vLLM servers for AlgoPuzzleVQA evals.
#
# Usage:
#   scripts/launch_vllm.sh <model_id> [num_servers=4] [tp_per_server=1] [base_port=8001]
#
# Examples:
#   scripts/launch_vllm.sh google/gemma-3-27b-it 4 1
#   scripts/launch_vllm.sh OpenGVLab/InternVL3-78B 1 4
set -euo pipefail

MODEL=${1:?'usage: launch_vllm.sh <model_id> [num_servers=4] [tp_per_server=1] [base_port=8001]'}
N_SERVERS=${2:-4}
TP=${3:-1}
BASE_PORT=${4:-8001}

PY=${VLLM_PY:-/data/shanmukha/vllm_env/bin/python}
HF_HOME=${HF_HOME:-/data/shanmukha/.cache/huggingface}
LOG_DIR=${LOG_DIR:-/tmp/vllm_logs}
mkdir -p "$LOG_DIR"

EXTRA_ARGS="${VLLM_EXTRA_ARGS:---max-num-seqs 64 --max-num-batched-tokens 16384 --trust-remote-code}"

echo "Launching $N_SERVERS server(s) for $MODEL  (tensor-parallel=$TP, ports $BASE_PORT..$((BASE_PORT + N_SERVERS - 1)))"
for i in $(seq 0 $((N_SERVERS - 1))); do
  port=$((BASE_PORT + i))
  gpu_lo=$((i * TP))
  gpu_hi=$((gpu_lo + TP - 1))
  gpus=$(seq -s, "$gpu_lo" "$gpu_hi")
  log="$LOG_DIR/vllm_${port}.log"
  echo "  port=$port  CUDA_VISIBLE_DEVICES=$gpus  log=$log"
  HF_HOME="$HF_HOME" CUDA_VISIBLE_DEVICES="$gpus" \
    nohup "$PY" -m vllm.entrypoints.openai.api_server \
      --model "$MODEL" --port "$port" --tensor-parallel-size "$TP" \
      $EXTRA_ARGS > "$log" 2>&1 &
done

echo
echo "All servers spawned in the background. Tail logs in $LOG_DIR."
echo "Wait for 'Application startup complete' then run e.g.:"
echo "  python -m src.eval.run_seg_eval --backend vllm --model $MODEL --ports $(seq -s' ' $BASE_PORT $((BASE_PORT + N_SERVERS - 1))) --mode oracle_mask --tasks maze nqueens"
