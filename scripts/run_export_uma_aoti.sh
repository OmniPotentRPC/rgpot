#!/usr/bin/env bash
# Run the UMA vesin+AOTI exporter from a pixi metatomicbld tree.
set -euo pipefail
ROOT="${ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
PY="${PY:-$ROOT/.pixi/envs/metatomicbld/bin/python}"
if [[ ! -x "$PY" ]]; then
  PY="${PYTHON:-python3}"
fi
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
mkdir -p "$ROOT/bench_data/uma"
cd "$ROOT"
exec "$PY" scripts/export_uma_aoti.py "$@"
