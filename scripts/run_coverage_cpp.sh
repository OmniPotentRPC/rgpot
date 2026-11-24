#!/usr/bin/env bash
# Instrument CppCore (C++ and the Fortran kernels) via Meson b_coverage + lcov.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
OUT="${1:-cpp_lcov.info}"
BUILD="${RGPOT_COV_BUILDDIR:-bbdir-cov}"

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

# The workflow sets RGPOT_COV_TIMEOUT_MULTIPLIER once for every job that
# runs this script. NeighbourLayerTest's registry timeout is 120s. Four
# multiples is 480s, and a coverage run used 440s of that. One extra
# multiple is 600s. That stays on this test: raising the shared multiplier
# would stretch every other timeout, including ones already at 1200s.
COV_MULT="${RGPOT_COV_TIMEOUT_MULTIPLIER:-1}"
if [[ ! "$COV_MULT" =~ ^[0-9]+([.][0-9]+)?$ ]] || ! awk -v v="$COV_MULT" 'BEGIN { exit !(v+0 > 0) }'; then
  echo "ERROR: RGPOT_COV_TIMEOUT_MULTIPLIER must be a positive number, got ${COV_MULT}" >&2
  exit 1
fi
NEIGH_MULT="$COV_MULT"
if [[ "$COV_MULT" =~ ^[0-9]+$ ]] && [[ "$COV_MULT" -gt 1 ]]; then
  NEIGH_MULT=$((COV_MULT + 1))
fi
echo "==> meson test timeout multiplier ${COV_MULT} (NeighbourLayerTest ${NEIGH_MULT})"

# meson rewrites testlog.json on every invocation, so report after each one.
report_unsuccessful() {
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
PY
}

status=0
run_meson_test() {
  local rc=0
  meson test -C "$BUILD" --print-errorlogs "$@" || rc=$?
  if [[ "$rc" -ne 0 ]]; then
    report_unsuccessful
    status=$rc
  fi
}

if [[ "$NEIGH_MULT" == "$COV_MULT" ]]; then
  run_meson_test --timeout-multiplier "$COV_MULT"
else
  list_file=$(mktemp)
  meson test -C "$BUILD" --list --no-rebuild > "$list_file"
  rest=()
  have_neigh=0
  while IFS= read -r line; do
    [[ -z "$line" ]] && continue
    name="${line##*:}"
    if [[ "$name" == "NeighbourLayerTest" ]]; then
      have_neigh=1
    else
      rest+=("$name")
    fi
  done < "$list_file"
  rm -f "$list_file"
  if [[ "$have_neigh" -eq 0 || ${#rest[@]} -eq 0 ]]; then
    run_meson_test --timeout-multiplier "$COV_MULT"
  else
    run_meson_test --no-rebuild --timeout-multiplier "$COV_MULT" "${rest[@]}"
    echo "==> NeighbourLayerTest timeout multiplier ${NEIGH_MULT}"
    run_meson_test --no-rebuild --timeout-multiplier "$NEIGH_MULT" NeighbourLayerTest
  fi
fi
if [[ "$status" -ne 0 ]]; then
  exit "$status"
fi

# lcov 2 renamed the branch-coverage rc option and rejects the old name.
# Skipping compiler-generated exception-unwind branches prevents mismatched
# exception-tag errors across inline/template instantiations without needing
# --ignore-errors mismatch; -fprofile-update=atomic prevents negative counts
# under OpenMP, and no --ignore-errors suppressions are needed.
LCOV_MAJOR=$(lcov --version | sed -n 's/.*LCOV version \([0-9][0-9]*\)\..*/\1/p')
LCOV_CAPTURE_ARGS=()
LCOV_FILTER_ARGS=()
if [[ "${LCOV_MAJOR:-0}" -ge 2 ]]; then
  LCOV_CAPTURE_ARGS+=(--rc branch_coverage=1 --rc no_exception_branch=1)
  LCOV_FILTER_ARGS+=(--rc branch_coverage=1)
else
  LCOV_CAPTURE_ARGS+=(--rc lcov_branch_coverage=1)
  LCOV_FILTER_ARGS+=(--rc lcov_branch_coverage=1)
fi

echo "==> lcov capture"
lcov --directory "$BUILD" --capture --output-file "$OUT" "${LCOV_CAPTURE_ARGS[@]}"

# Keep library sources: CppCore/rgpot (C++ and Fortran kernels) and repo headers
# (avoid */include/* which matches toolchain/pixi headers).
EXTRACT_PATTERNS=("${ROOT}/CppCore/rgpot/*")
if grep -q "^SF:${ROOT}/include/" "$OUT"; then
  EXTRACT_PATTERNS+=("${ROOT}/include/*")
fi
# Meson compiles a copy of rgpot_sw.f90 in the build tree, and that copy
# includes sw_centred_body.inc by basename. gcov records the copy under
# CppCore/rgpot/fortran and the include at the build directory root, so
# the source-tree pattern above drops both.
BUILD_ABS="$(cd "$BUILD" && pwd)"
if [[ -d "${BUILD_ABS}/CppCore/rgpot/fortran" ]]; then
  EXTRACT_PATTERNS+=("${BUILD_ABS}/CppCore/rgpot/fortran/*")
fi
if grep -q "^SF:${BUILD_ABS}/sw_centred_body.inc$" "$OUT"; then
  EXTRACT_PATTERNS+=("${BUILD_ABS}/sw_centred_body.inc")
fi
TMP_DIR=$(mktemp -d)
TMP="$TMP_DIR/cpp.info"
cleanup_tmp() {
  rm -f -- "$TMP"
  rmdir -- "$TMP_DIR"
}
trap cleanup_tmp EXIT
lcov --extract "$OUT" \
  "${EXTRACT_PATTERNS[@]}" \
  --output-file "$TMP" "${LCOV_FILTER_ARGS[@]}"
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
