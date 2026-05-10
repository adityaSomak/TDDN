#!/usr/bin/env bash
# Regenerate oracle-mask overlays for all three seg-eligible puzzles.
# Pass-through args go to src.overlays.oracle_mask (e.g. --limit 5 --ids 0029).
set -euo pipefail
PY=${PY:-/data/shanmukha/puzzlebench_venv/.venv/bin/python}
cd "$(dirname "$0")/.."
for p in maze nqueens chess; do
  echo "=== $p ==="
  "$PY" -m src.overlays.oracle_mask --puzzle "$p" "$@"
done
