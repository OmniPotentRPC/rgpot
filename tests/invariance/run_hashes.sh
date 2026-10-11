#!/bin/bash
# Report-only. A hash that moves is printed, and the script still exits 0
# unless the binary itself fails.
set -euo pipefail
BIN=$1
ROOT=$(mktemp -d)
trap 'rm -rf "$ROOT"' EXIT
HERE=$(cd "$(dirname "$0")" && pwd)

run_one() {
  local n=$1
  local threads=$2
  local dest=$3
  local pots=$4
  mkdir -p "$dest"
  OMP_NUM_THREADS=$threads "$BIN" --n "$n" --pots "$pots" --dump-dir "$dest"
}

# Lennard-Jones and Morse use the neighbour-list pool. The embedded-atom
# cell table stores N*(N+1) integers, so each 32000-atom call is serial
# and they run one at a time.
for n in 4000 32000; do
  for t in 1 2 3 8; do
    run_one "$n" "$t" "$ROOT/n${n}_t${t}" "lj,morse"
  done
done
for n in 4000 32000; do
  for t in 1 2 3 8; do
    run_one "$n" "$t" "$ROOT/n${n}_t${t}" "eam"
  done
done

for t in 1 2 3 8; do
  dest=$ROOT/edge_t${t}
  mkdir -p "$dest"
  OMP_NUM_THREADS=$t "$BIN" --edge --dump-dir "$dest"
done

if command -v mpirun >/dev/null 2>&1; then
  mpi_extra=()
  if [ "$(id -u)" -eq 0 ]; then
    mpi_extra+=(--allow-run-as-root)
  fi
  for n in 4000 32000; do
    for np in 1 2 3; do
      dest=$ROOT/n${n}_np${np}
      mkdir -p "$dest"
      OMP_NUM_THREADS=1 mpirun --oversubscribe "${mpi_extra[@]}" -np "$np" \
        "$BIN" --n "$n" --pots lj,morse --dump-dir "$dest"
    done
  done
  for np in 1 2 3; do
    dest=$ROOT/n4000_np${np}
    OMP_NUM_THREADS=1 mpirun --oversubscribe "${mpi_extra[@]}" -np "$np" \
      "$BIN" --n 4000 --pots eam --dump-dir "$dest"
  done
else
  echo "mpi: not available"
fi

: >"$ROOT/bench.txt"
for rep in 1 2 3 4 5; do
  for t in 1 2 4 8; do
    OMP_NUM_THREADS=$t "$BIN" --n 4000 --pots lj,morse,eam --bench | while read -r kind pot sec; do
      printf '%s %s %s %s\n' "$kind" "$pot" "$t" "$sec"
    done >>"$ROOT/bench.txt"
  done
done

python3 "$HERE/compare_hashes.py" "$ROOT"
