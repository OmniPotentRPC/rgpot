#!/bin/sh
# Run every Gappa bound, the Sollya exp check, the error-free
# transformation test, and the constant drift guard.
set -eu
ROOT=$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)
cd "$ROOT"
for f in validation/gappa/*.gappa; do
  echo "== $f"
  gappa "$f"
done
echo "== sollya"
out=$(sollya validation/sollya/exp_poly.sollya || true)
printf '%s\n' "$out"
printf '%s\n' "$out" | grep -q '^OK$'
echo "== eft"
cc=${CXX:-g++}
"$cc" -std=c++20 -O2 -ffp-contract=off \
  validation/eft/eft_diff.cc -lquadmath -o "$ROOT/validation/eft/eft_diff"
"$ROOT/validation/eft/eft_diff"
rm -f "$ROOT/validation/eft/eft_diff"
echo "== constants"
python3 scripts/export_constants.py --check
echo "validation ok"
