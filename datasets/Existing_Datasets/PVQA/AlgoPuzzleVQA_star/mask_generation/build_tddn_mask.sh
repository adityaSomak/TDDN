#!/usr/bin/env bash
# Regenerate TDDN (tip-adapter) mask overlays into ../<puzzle>/seg_data/tddn_mask/.
# Pass-through args go to src.overlays.tddn_mask (e.g. --limit 5 --alpha 2.0).
#
# Puzzles: maze, nqueens (chess requires CHESS_DATASET env var).
# Requires the PuzzleBench text-alignment package on PYTHONPATH and a trained
# alignment checkpoint reachable by src.overlays.tddn_loader.
set -euo pipefail
PY=${PY:-python}
cd "$(dirname "$0")"
read -ra PUZZLES <<< "${PUZZLES:-maze nqueens}"
for p in "${PUZZLES[@]}"; do
  echo "=== $p ==="
  "$PY" -m src.overlays.tddn_mask --puzzle "$p" "$@"
done
