#!/usr/bin/env bash
# Compile the ISA fixtures and check the auditor's pass and fail cases.
# ISA_AUDIT_ARCH=x86_64 runs the x86 fixtures.
# ISA_AUDIT_ARCH=aarch64 runs the SVE fixtures.
# Unset, the host architecture is used, and a cross compiler is used for
# the other architecture when one is on PATH.
# ISA_AUDIT_OUT, when set, receives the object that must pass the audit.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
AUDIT=(python3 "$ROOT/ci/isa-audit/isa_audit.py")
TESTS="$ROOT/ci/isa-audit/tests"
HOST="$(uname -m)"
ARCH="${ISA_AUDIT_ARCH:-$HOST}"

python3 -m pip install -q -r "$ROOT/ci/isa-audit/requirements.txt"

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

fail() {
  echo "FAIL: $*" >&2
  exit 1
}

expect_ok() {
  local out
  if ! out="$("${AUDIT[@]}" "$@" 2>&1)"; then
    printf '%s\n' "$out"
    fail "expected a clean audit: $*"
  fi
  printf '%s\n' "$out"
}

expect_fail() {
  local needle="$1"
  shift
  local out code
  set +e
  out="$("${AUDIT[@]}" "$@" 2>&1)"
  code=$?
  set -e
  printf '%s\n' "$out"
  if [[ "$code" -eq 0 ]]; then
    fail "expected a violation: $*"
  fi
  if ! grep -q -- "$needle" <<<"$out"; then
    fail "violation text missing [$needle]"
  fi
}

run_x86() {
  local cc="${CC_X86:-cc}"
  if ! command -v "$cc" >/dev/null 2>&1; then
    echo "x86_64 fixture not exercised: no C compiler"
    return
  fi
  echo "== x86_64 fixtures ($cc) =="
  "$cc" -c -O2 "$TESTS/ok.c" -o "$work/ok.o"
  "$cc" -c -O3 -mavx2 "$TESTS/bad_global.c" -o "$work/bad_global.o"
  "$cc" -c -O2 "$TESTS/bad_unlisted.c" -o "$work/bad_unlisted.o"
  "$cc" -c -O2 -mpopcnt "$TESTS/v2_popcnt.c" -o "$work/v2.o"
  "$cc" -shared -fPIC -O2 "$TESTS/ok_dyn.c" -o "$work/ok.so"

  expect_ok "$work/ok.o" --allow "$TESTS/allow-ok.toml" --json "$work/ok.json"
  python3 - "$work/ok.json" <<'PY'
import json, sys
report = json.load(open(sys.argv[1]))
counts = report["allowed_feature_counts"]
assert report["violations"] == [], report["violations"]
assert counts.get("avx512f", 0) > 0, counts
assert counts.get("avx", 0) > 0, counts
print("json ok:", counts)
PY

  expect_fail "hot_loop" "$work/bad_global.o" --allow "$TESTS/allow-ok.toml"
  expect_fail "avx2" "$work/bad_global.o" --allow "$TESTS/allow-ok.toml"
  expect_fail "secret_kernel" "$work/bad_unlisted.o" --allow "$TESTS/allow-ok.toml"
  expect_fail "avx2" "$work/bad_unlisted.o" --allow "$TESTS/allow-ok.toml"

  cp "$work/ok.o" "$work/ok.stripped.o"
  strip --strip-all "$work/ok.stripped.o"
  expect_fail "no symbols" "$work/ok.stripped.o" --allow "$TESTS/allow-ok.toml"

  expect_ok "$work/ok.so" --allow "$TESTS/allow-ok.toml"

  cp "$work/ok.o" "$work/libopenblas.so.0"
  expect_ok "$work/libopenblas.so.0" --allow "$TESTS/allow-vendored.toml"
  cp "$work/ok.o" "$work/librgpot.so"
  expect_fail "kern_avx512" "$work/librgpot.so" --allow "$TESTS/allow-vendored.toml"

  expect_fail "popcnt" "$work/v2.o" --allow "$TESTS/allow-ok.toml" --baseline x86-64
  expect_ok "$work/v2.o" --allow "$TESTS/allow-ok.toml" --baseline x86-64-v2

  rm -f "$work/ok.whl"
  python3 - "$work" <<'PY'
import sys, zipfile
from pathlib import Path
root = Path(sys.argv[1])
with zipfile.ZipFile(root / "ok.whl", "w") as zf:
    zf.write(root / "ok.o", "rgpot/ok.o")
PY
  expect_ok "$work/ok.whl" --allow "$TESTS/allow-ok.toml"

  if [[ -n "${ISA_AUDIT_OUT:-}" ]]; then
    mkdir -p "$ISA_AUDIT_OUT"
    cp "$work/ok.o" "$ISA_AUDIT_OUT/ok.o"
  fi
}

run_arm() {
  local cc="${CC_ARM:-}"
  if [[ "$HOST" == "aarch64" || "$HOST" == "arm64" ]]; then
    cc="${cc:-cc}"
  elif command -v aarch64-linux-gnu-gcc >/dev/null 2>&1; then
    cc="${cc:-aarch64-linux-gnu-gcc}"
  fi
  if [[ -z "$cc" ]] || ! command -v "$cc" >/dev/null 2>&1; then
    echo "aarch64 fixture not exercised: no aarch64 C compiler"
    return
  fi
  echo "== aarch64 fixtures ($cc) =="
  "$cc" -c -O2 -march=armv8-a "$TESTS/ok_sve.c" -o "$work/ok_sve.o"
  "$cc" -c -O2 -march=armv8-a "$TESTS/bad_sve.c" -o "$work/bad_sve.o"
  expect_ok "$work/ok_sve.o" --allow "$TESTS/allow-sve.toml" --baseline armv8-a \
    --json "$work/sve.json"
  python3 - "$work/sve.json" <<'PY'
import json, sys
report = json.load(open(sys.argv[1]))
assert report["violations"] == [], report["violations"]
assert report["allowed_feature_counts"].get("sve", 0) > 0, report
print("json ok:", report["allowed_feature_counts"])
PY
  expect_fail "other_sve" "$work/bad_sve.o" --allow "$TESTS/allow-sve.toml" --baseline armv8-a
  expect_fail "sve" "$work/bad_sve.o" --allow "$TESTS/allow-sve.toml" --baseline armv8-a
  if [[ -n "${ISA_AUDIT_OUT:-}" ]]; then
    mkdir -p "$ISA_AUDIT_OUT"
    cp "$work/ok_sve.o" "$ISA_AUDIT_OUT/ok_sve.o"
  fi
}

case "$ARCH" in
  x86_64|amd64)
    run_x86
    if [[ "${ISA_AUDIT_ARCH:-}" == "x86_64" ]]; then
      echo "aarch64 fixture not exercised on this runner"
    elif command -v aarch64-linux-gnu-gcc >/dev/null 2>&1; then
      run_arm
    else
      echo "aarch64 fixture not exercised: no cross compiler"
    fi
    ;;
  aarch64|arm64)
    run_arm
    echo "x86_64 fixture not exercised on this runner"
    ;;
  *)
    fail "unsupported architecture $ARCH"
    ;;
esac

python3 -m unittest discover -s "$TESTS" -p 'test_*.py' -v
echo "self-test passed"
