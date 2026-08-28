#!/usr/bin/env bash
# Generate the first-slice libxckernel C package on rg.terra.
# Families: lda,gga,mgga_tau. max_order=2. Pin: d6a9d57ba3fe0f2763667ce168d0c0ef21cff4a4
set -euo pipefail

PIN=d6a9d57ba3fe0f2763667ce168d0c0ef21cff4a4
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/third_party/libxckernel"
SRC="${LIBXCKERNEL_SRC:-/tmp/libxckernel-src}"
FAMILIES="lda,gga,mgga_tau"
MAX_ORDER=2

host="$(hostname -s 2>/dev/null || hostname)"
if [[ "${RGPOT_XCKERNEL_REGEN_OK:-}" != "1" && "$host" != *terra* ]]; then
  echo "gen_libxckernel.sh runs on rg.terra only" >&2
  exit 2
fi

if [[ ! -d "$SRC/xckernel" ]]; then
  echo "missing generator at $SRC; clone https://github.com/susilehtola/libxckernel @$PIN" >&2
  exit 1
fi

if [[ -d "$SRC/.git" ]]; then
  git -C "$SRC" rev-parse HEAD | grep -q "$PIN" || {
    echo "libxckernel at $SRC is not pin $PIN" >&2
    git -C "$SRC" rev-parse HEAD >&2
    exit 1
  }
else
  echo "note: $SRC has no .git; generating against the rsynced pin $PIN" >&2
fi

if [[ -e "$DEST" ]]; then
  rtrash -rf "$DEST"
fi
mkdir -p "$DEST"
PYTHONPATH="$SRC${PYTHONPATH:+:$PYTHONPATH}" python3 -m xckernel.catalog \
  "$DEST" "$FAMILIES" "$MAX_ORDER" c

# Drop Fortran from the rgpot first slice.
rm -rf "$DEST/fortran"

{
  echo "libxckernel $PIN"
  echo "families $FAMILIES"
  echo "max_order $MAX_ORDER"
  echo "backend c"
} >"$DEST/PIN"

python3 - <<PY
import json
from pathlib import Path
man = json.loads(Path("$DEST/manifest.json").read_text())
c = [k["name"] for k in man["kernels"] if "c" in k.get("backends", ["c"]) or k.get("abi") == "xckernel.h"]
print(f"manifest kernels={len(man['kernels'])} c_abi={len(c)}")
for n in c:
    print(" ", n)
banned = [n for n in c if any(s in n for s in ("_o3", "_o4", "giao", "cmgga", "hmgga"))]
if banned:
    raise SystemExit(f"banned names in first slice: {banned}")
if len(c) != 27:
    raise SystemExit(f"expected 27 C kernels, got {len(c)}")
PY

echo "generated $DEST"
