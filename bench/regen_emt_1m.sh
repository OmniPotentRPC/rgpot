#!/bin/sh
# MIT License
# Copyright 2023--present rgpot developers
#
# EMT at 1,048,576 FCC atoms, develop versus HEAD.
# Median of seven calls after two discarded calls. rss_kb is VmHWM.
# Writes bench/raw/emt_1m.csv. The csv is generated, not committed.

set -e
root=$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)
out="$root/bench/raw/emt_1m.csv"
mkdir -p "$root/bench/raw"

build_one() {
  src=$1
  bin=$2
  extra=$3
  (
    cd "$src/CppCore"
    # shellcheck disable=SC2086
    g++ -O3 -DNDEBUG -std=c++20 -fopenmp -ffp-contract=off -I. -Irgpot/EMT/Asap \
      "$root/CppCore/tests/emt_migrate_bench.cc" \
      rgpot/EMT/EMTPot.cc \
      rgpot/EMT/Asap/NeighborList.cpp \
      rgpot/EMT/Asap/GhostAtoms.cpp \
      rgpot/EMT/Asap/Vec.cpp \
      rgpot/EMT/Asap/Atoms.cpp \
      rgpot/EMT/Asap/Exception.cpp \
      rgpot/EMT/Asap/EMTRasmussenParameterProvider.cpp \
      rgpot/EMT/Asap/EMTDefaultParameterProvider.cpp \
      rgpot/EMT/Asap/EMT.cpp \
      rgpot/EMT/Asap/SuperCell.cpp \
      $extra \
      -o "$bin"
  )
}

dev=$(mktemp -d)
git -C "$root" worktree add --detach "$dev" 127dc2babce6446176225967db5ca10821ee2a13
build_one "$dev" /tmp/emt_bench_develop "rgpot/EMT/Asap/CellList.cpp"
build_one "$root" /tmp/emt_bench_tip \
  "rgpot/nlist/Neighbours.cc rgpot/nlist/NeighbourList.cc rgpot/nlist/MinimumImage.cpp rgpot/simd/exact_fma_hw.cc rgpot/simd/exp.cc"
git -C "$root" worktree remove --force "$dev"

{
  printf '%s\n' 'revision,threads,atoms,median_s,rss_kb,energy'
  for threads in 1 4 8; do
    OMP_NUM_THREADS=$threads /tmp/emt_bench_develop 64 "$threads" | awk -v t="$threads" '
      /median_s/ { printf "develop,%s,%s,%s,%s,%s\n", t, $4, $6, $8, $10 }'
    OMP_NUM_THREADS=$threads /tmp/emt_bench_tip 64 "$threads" | awk -v t="$threads" '
      /median_s/ { printf "tip,%s,%s,%s,%s,%s\n", t, $4, $6, $8, $10 }'
  done
} > "$out"
echo "wrote $out"
