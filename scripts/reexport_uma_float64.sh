#!/usr/bin/env bash
# Copyright 2023--present rgpot developers
# Re-export refused UMA AOTI packages as float64.
#
# A package is a directory with package.pt2, export.json and the
# weights file that export.json names (weights.pt when it does not).
# package.pt2 is the refused package. It is hashed and not rewritten.
# export.json is the original export config. The weights are the
# checkpoint or state dict the export was traced from.
#
# A directory of *.pt2 files is the same contract laid flat. The
# sidecar is <stem>.export.json.
#
# Usage:
#   scripts/reexport_uma_float64.sh PACKAGE_DIR OUT_DIR RECEIPT
#
# The receipt has one line per package:
#   input_sha256=... output_sha256=... dtype=float64 package=<name>
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 PACKAGE_DIR OUT_DIR RECEIPT" >&2
  exit 2
fi

SRC=$(cd "$1" && pwd)
mkdir -p "$2"
OUT=$(cd "$2" && pwd)
RECEIPT=$3
if [[ "$SRC" == "$OUT" ]]; then
  echo "OUT_DIR must not be PACKAGE_DIR" >&2
  exit 2
fi

ROOT=$(cd "$(dirname "$0")/.." && pwd)
PY=${PYTHON:-python3}

resolve_weights() {
  "$PY" -c '
import json
import sys
from pathlib import Path

config = Path(sys.argv[1])
raw = json.loads(config.read_text())
weights = raw.get("weights") or "weights.pt"
path = Path(weights)
if not path.is_absolute():
    path = config.parent / path
if not path.is_file():
    raise SystemExit(f"weights not found: {path}")
print(path)
' "$1"
}

packages=()
names=()
configs=()
saw_dir=0
while IFS= read -r dir; do
  [[ -z "$dir" ]] && continue
  if [[ "$dir" == "$OUT" ]]; then
    continue
  fi
  saw_dir=1
  if [[ ! -f "$dir/package.pt2" || ! -f "$dir/export.json" ]]; then
    echo "package directory needs package.pt2 and export.json: $dir" >&2
    exit 1
  fi
  packages+=("$dir/package.pt2")
  names+=("$(basename "$dir")")
  configs+=("$dir/export.json")
done < <(find "$SRC" -mindepth 1 -maxdepth 1 -type d -print | sort)

if [[ "$saw_dir" -eq 0 ]]; then
  shopt -s nullglob
  for pt2 in "$SRC"/*.pt2; do
    case "$pt2" in
      "$OUT"/*) continue ;;
    esac
    stem=$(basename "$pt2" .pt2)
    cfg="$SRC/${stem}.export.json"
    if [[ ! -f "$cfg" ]]; then
      echo "missing export config for $pt2" >&2
      exit 1
    fi
    packages+=("$pt2")
    names+=("$stem")
    configs+=("$cfg")
  done
  shopt -u nullglob
fi

if [[ ${#packages[@]} -eq 0 ]]; then
  echo "no packages in $SRC" >&2
  exit 1
fi

receipt_tmp=$(mktemp)
trap 'rm -f "$receipt_tmp"' EXIT

for i in "${!packages[@]}"; do
  src_pt2=${packages[$i]}
  name=${names[$i]}
  cfg=${configs[$i]}
  weights=$(resolve_weights "$cfg")
  dest="$OUT/${name}.pt2"
  if ! log=$("$PY" "$ROOT/scripts/export_uma_aoti.py" --reexport \
      --weights "$weights" --config "$cfg" --out "$dest" 2>&1); then
    printf '%s\n' "$log" >&2
    exit 1
  fi
  pos=$(printf '%s\n' "$log" | sed -n 's/.*pos_dtype=\([^ ]*\).*/\1/p' | tail -1)
  model=$(printf '%s\n' "$log" | sed -n 's/.*model_dtype=\([^ ]*\).*/\1/p' | tail -1)
  if [[ "$pos" != "float64" || "$model" != "float64" ]]; then
    echo "FAIL: $name pos_dtype=$pos model_dtype=$model" >&2
    printf '%s\n' "$log" >&2
    exit 1
  fi
  in_hash=$(sha256sum "$src_pt2" | awk '{print $1}')
  out_hash=$(sha256sum "$dest" | awk '{print $1}')
  printf 'input_sha256=%s output_sha256=%s dtype=%s package=%s\n' \
    "$in_hash" "$out_hash" "$pos" "$name" >> "$receipt_tmp"
done

mv "$receipt_tmp" "$RECEIPT"
trap - EXIT
