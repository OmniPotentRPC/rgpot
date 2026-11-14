#!/usr/bin/env bash
# MIT License
# Copyright 2023--present rgpot developers
#
# Neighbour build times, orthorhombic against sheared Cu, from two builds.
# Usage: scripts/bench_nlist_shear.sh BASE_EXE NEW_EXE [ROUNDS] [CALLS]
# Each EXE is a nlist_shear_bench binary; build the same source against
# the baseline tree for BASE_EXE. Every round runs each row once per
# build at one and eight threads, the two builds in alternating order, so
# drift on the machine falls on both. Prints one line per row and build
# with the median of every timed call and the raw per-call seconds.
set -euo pipefail

base="${1:?usage: bench_nlist_shear.sh BASE_EXE NEW_EXE [ROUNDS] [CALLS]}"
new="${2:?usage: bench_nlist_shear.sh BASE_EXE NEW_EXE [ROUNDS] [CALLS]}"
rounds="${3:-5}"
calls="${4:-3}"

raw=$(mktemp)
trap 'rm -f "$raw"' EXIT

for ((r = 0; r < rounds; ++r)); do
  if ((r % 2 == 0)); then order=(base new); else order=(new base); fi
  for threads in 1 8; do
    for n in 4000 11000; do
      for cell in ortho sheared; do
        for kind in stored stream public; do
          for build in "${order[@]}"; do
            if [ "$build" = base ]; then exe="$base"; else exe="$new"; fi
            line=$(OMP_NUM_THREADS="$threads" "$exe" --n "$n" --cell "$cell" --kind "$kind" \
                     --calls "$calls")
            printf 'build=%s %s\n' "$build" "${line#row }" >> "$raw"
          done
        done
      done
    done
  done
done

# Pool the raw calls of every round per row and build.
awk '
  {
    key = ""; times = ""
    for (f = 1; f <= NF; ++f) {
      split($f, kv, "=")
      if (kv[1] == "raw_s") times = kv[2]
      else if (kv[1] != "median_s") key = key (key == "" ? "" : " ") $f
    }
    if (!(key in all)) order[++nrow] = key
    all[key] = all[key] (all[key] == "" ? "" : ",") times
  }
  END {
    for (r = 1; r <= nrow; ++r) {
      key = order[r]
      m = split(all[key], v, ",")
      for (i = 1; i <= m; ++i) s[i] = v[i] + 0
      for (i = 2; i <= m; ++i) {
        x = s[i]
        for (j = i - 1; j >= 1 && s[j] > x; --j) s[j + 1] = s[j]
        s[j + 1] = x
      }
      med = m % 2 ? s[(m + 1) / 2] : 0.5 * (s[m / 2] + s[m / 2 + 1])
      printf "%s median_s=%.6e raw_s=%s\n", key, med, all[key]
    }
  }' "$raw" | sort -t' ' -k5,5V -k2,2V -k3,3 -k4,4 -k1,1
