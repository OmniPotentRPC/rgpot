#!/usr/bin/env bash
# gcov/lcov for the in-tree Fortran 2018 pot kernels (CppCore/rgpot/fortran).
# The kernels build as part of the main meson tree, so coverage is extracted
# from a meson b_coverage build; scripts/run_coverage_cpp.sh produces one.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
OUT="${1:-fortran_lcov.info}"
BUILD="${RGPOT_COV_BUILDDIR:-bbdir-cov}"

if ! command -v lcov >/dev/null 2>&1; then
  echo "ERROR: lcov not on PATH" >&2
  exit 1
fi

if [[ ! -d "$BUILD" ]]; then
  echo "ERROR: no meson coverage tree at $BUILD" >&2
  echo "       run scripts/run_coverage_cpp.sh first" >&2
  exit 1
fi

echo "==> extract Fortran kernels from meson coverage tree $BUILD"
RAW=$(mktemp)
trap 'rm -f "$RAW"' EXIT
lcov --directory "$BUILD" --capture --output-file "$RAW"
lcov --extract "$RAW" '*/CppCore/rgpot/fortran/*' \
  --output-file "$OUT"

test -s "$OUT"
python3 - "$OUT" <<'PY'
import sys
hits = tot = 0
for line in open(sys.argv[1], encoding="utf-8", errors="replace"):
    if line.startswith("DA:"):
        h = int(line.split(":")[1].split(",")[1])
        tot += 1
        hits += h > 0
print(f"fortran lcov {100*hits/tot:.1f}% {hits}/{tot}" if tot else "empty")
if tot == 0:
    raise SystemExit(1)
PY
echo "OK wrote $OUT"
