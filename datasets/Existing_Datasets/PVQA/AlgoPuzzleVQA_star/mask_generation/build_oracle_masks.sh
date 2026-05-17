#!/usr/bin/env bash
# Regenerate oracle-mask overlays into ../<puzzle>/seg_data/oracle_mask/.
# Pass-through args go to src.overlays.oracle_mask (e.g. --limit 5 --ids 0029).
#
# Puzzles: maze, nqueens (chess requires CHESS_DATASET env var).
set -euo pipefail
PY=${PY:-python}
cd "$(dirname "$0")"
read -ra PUZZLES <<< "${PUZZLES:-maze nqueens}"
for p in "${PUZZLES[@]}"; do
  echo "=== $p ==="
  "$PY" -m src.overlays.oracle_mask --puzzle "$p" "$@"
done
