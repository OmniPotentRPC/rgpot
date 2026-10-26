#!/usr/bin/env bash
# MIT License
# Copyright 2023--present rgpot developers
#
# Per-call wall time of MorsePot through the Cap'n Proto RPC path, against
# the same Pt nanoparticles scripts/bench_pair_scaling.sh times in process.
# Usage: scripts/bench_rpc_call.sh BUILDDIR [PORT] [SIZES...]
set -euo pipefail

bdir="${1:?usage: bench_rpc_call.sh BUILDDIR [PORT] [SIZES...]}"
port="${2:-23457}"
shift $(( $# >= 2 ? 2 : 1 ))
sizes=("$@")
if [ "${#sizes[@]}" -eq 0 ]; then
  sizes=(7 100 1000 10000)
fi

"$bdir/CppCore/potserv" "$port" Morse >/dev/null 2>&1 &
server=$!
trap 'kill "$server" 2>/dev/null || true' EXIT
for _ in $(seq 1 50); do
  if "$bdir/CppCore/time_rpc_call" --port "$port" --n 7 --calls 1 --repeats 1 >/dev/null 2>&1; then
    break
  fi
  sleep 0.1
done

header_done=0
for n in "${sizes[@]}"; do
  if [ "$n" -ge 10000 ]; then calls=20
  elif [ "$n" -ge 1000 ]; then calls=200
  else calls=2000; fi
  out="$("$bdir/CppCore/time_rpc_call" --port "$port" --n "$n" --calls "$calls" --repeats 5)"
  if [ "$header_done" -eq 0 ]; then
    printf '%s\n' "$out" | head -n 1
    header_done=1
  fi
  printf '%s\n' "$out" | tail -n 1
done
