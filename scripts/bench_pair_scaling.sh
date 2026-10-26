#!/usr/bin/env bash
# MIT License
# Copyright 2023--present rgpot developers
#
# Per-call wall time of the classical pair kernels across atom counts.
# Usage: scripts/bench_pair_scaling.sh BUILDDIR [SIZES...]
# Prints one CSV table on stdout (header once). Each row is the median of
# five repeats; calls per repeat shrink with size so a row costs about the
# same wall time.
set -euo pipefail

bdir="${1:?usage: bench_pair_scaling.sh BUILDDIR [SIZES...]}"
shift
sizes=("$@")
if [ "${#sizes[@]}" -eq 0 ]; then
  sizes=(7 100 1000 10000)
fi
exe="$bdir/CppCore/time_pair_scaling"

header_done=0
for sys in pt-np ar-np ar-cl pt-bulk al-bulk; do
  for n in "${sizes[@]}"; do
    # Bulk cells need at least two cutoffs across, so tiny bulk sizes are
    # not physical systems; skip them.
    case "$sys" in
      *-bulk) [ "$n" -ge 1000 ] || continue ;;
    esac
    for mode in cold warm walk; do
      if [ "$n" -ge 10000 ]; then calls=20
      elif [ "$n" -ge 1000 ]; then calls=200
      else calls=2000; fi
      out="$("$exe" --system "$sys" --mode "$mode" --n "$n" --calls "$calls" --repeats 5)" || {
        echo "time_pair_scaling failed: $sys $mode $n" >&2
        exit 1
      }
      if [ "$header_done" -eq 0 ]; then
        printf '%s\n' "$out" | head -n 1
        header_done=1
      fi
      printf '%s\n' "$out" | tail -n 1
    done
  done
done
