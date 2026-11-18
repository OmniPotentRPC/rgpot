#!/usr/bin/env bash
# Instrument CppCore (C++ and the Fortran kernels) via Meson b_coverage + lcov.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
OUT="${1:-cpp_lcov.info}"
BUILD="${RGPOT_COV_BUILDDIR:-bbdir-cov}"
COVERAGE_TIMEOUT_MULTIPLIER="${RGPOT_COV_TIMEOUT_MULTIPLIER:-3}"

if ! command -v meson >/dev/null 2>&1; then
  echo "ERROR: meson not on PATH (use pixi -e devbld)" >&2
  exit 1
fi
if ! command -v lcov >/dev/null 2>&1; then
  echo "ERROR: lcov not on PATH" >&2
  exit 1
fi

SETUP_ARGS=(
  -Dwith_tests=true \
  -Dwith_examples=false \
  -Dwith_xtensor=false \
  -Dwith_eigen=false \
  -Dwith_rpc=true \
  -Dwith_cache=false \
  -Dpure_lib=false \
  -Db_coverage=true \
  --buildtype=debug
)
# A path in RGPOT_COV_BUILDDIR is never deleted from here. --wipe is only
# for a directory meson already owns. Anything else is left untouched.
if [[ -e "$BUILD" && ! -d "$BUILD" ]]; then
  echo "ERROR: ${BUILD} exists and is not a directory" >&2
  exit 1
fi
if [[ -d "$BUILD/meson-private" ]]; then
  echo "==> meson setup --wipe ${BUILD} (coverage, tests, rpc)"
  meson setup --wipe "$BUILD" "${SETUP_ARGS[@]}"
elif [[ -d "$BUILD" ]]; then
  echo "ERROR: ${BUILD} exists but is not a meson build (no meson-private)." >&2
  echo "Refusing to remove it. Set RGPOT_COV_BUILDDIR to a new directory or remove it." >&2
  exit 1
else
  echo "==> meson setup ${BUILD} (coverage, tests, rpc)"
  meson setup "$BUILD" "${SETUP_ARGS[@]}"
fi

meson compile -C "$BUILD"
if ! meson test -C "$BUILD" --print-errorlogs \
  --timeout-multiplier "$COVERAGE_TIMEOUT_MULTIPLIER"; then
  python3 - "$BUILD/meson-logs/testlog.json" <<'PY'
import collections
import json
import pathlib
import sys

log = pathlib.Path(sys.argv[1])
unsuccessful = []
if log.is_file():
    for raw in log.read_text(encoding="utf-8", errors="replace").splitlines():
        record = json.loads(raw)
        if record["result"] not in {"OK", "SKIP", "EXPECTEDFAIL"}:
            unsuccessful.append((record["name"], record["result"]))

counts = collections.Counter(result for _, result in unsuccessful)
detail = ", ".join(f"{result}: {count}" for result, count in sorted(counts.items()))
print(
    f"ERROR: {len(unsuccessful)} coverage tests were unsuccessful"
    + (f" ({detail})" if detail else ""),
    file=sys.stderr,
)
for name, result in unsuccessful:
    print(f"  {result}: {name}", file=sys.stderr)
raise SystemExit(1)
PY
  exit 1
fi

# lcov 2 renamed the branch-coverage rc option and rejects the old name. It
# also checks gcov data against source lines and reports template instances in
# toolchain headers as mismatched. lcov 2.0 calls that class "mismatch"; later
# releases also say "inconsistent". That one class is tolerated. Every other
# lcov error still fails the script.
LCOV_MAJOR=$(lcov --version | sed -n 's/.*LCOV version \([0-9][0-9]*\)\..*/\1/p')
LCOV_ARGS=()
if [[ "${LCOV_MAJOR:-0}" -ge 2 ]]; then
  LCOV_ARGS+=(--rc branch_coverage=1 --ignore-errors inconsistent,mismatch)
else
  LCOV_ARGS+=(--rc lcov_branch_coverage=1)
fi

echo "==> lcov capture"
lcov --directory "$BUILD" --capture --output-file "$OUT" "${LCOV_ARGS[@]}"

# Keep library sources: CppCore/rgpot (C++ and Fortran kernels)
TMP_DIR=$(mktemp -d)
TMP="$TMP_DIR/cpp.info"
cleanup_tmp() {
  rm -f -- "$TMP"
  rmdir -- "$TMP_DIR"
}
trap cleanup_tmp EXIT
lcov --extract "$OUT" \
  '*/CppCore/rgpot/*' '*/include/*' \
  --output-file "$TMP" "${LCOV_ARGS[@]}"
mv "$TMP" "$OUT"
cleanup_tmp
trap - EXIT

test -s "$OUT"
python3 - "$OUT" <<'PY'
import sys
hits = tot = 0
sfs = []
for line in open(sys.argv[1], encoding="utf-8", errors="replace"):
    if line.startswith("SF:"):
        sfs.append(line.strip())
    if line.startswith("DA:"):
        h = int(line.split(":")[1].split(",")[1])
        tot += 1
        hits += h > 0
print(f"cpp lcov {100*hits/tot:.1f}% {hits}/{tot}" if tot else "empty lcov")
for s in sfs[:8]:
    print(" ", s)
if tot == 0:
    raise SystemExit("no DA records in cpp LCOV")
PY
echo "OK wrote $OUT"
