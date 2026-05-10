#!/usr/bin/env bash
# Regenerate TDDN-mask overlays for all three seg-eligible puzzles.
# Pass-through args go to src.overlays.tddn_mask (e.g. --limit 5 --alpha 2.0).
set -euo pipefail
PY=${PY:-/data/shanmukha/puzzlebench_venv/.venv/bin/python}
cd "$(dirname "$0")/.."
for p in maze nqueens chess; do
  echo "=== $p ==="
  "$PY" -m src.overlays.tddn_mask --puzzle "$p" "$@"
done
