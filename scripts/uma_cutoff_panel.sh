#!/bin/sh
# Copyright 2023--present rgpot developers
# Build the v1.8.3 and v1.8.5 edge timers and interleave them.
# Run from the repository root. Rows are appended to the two paths
# given on the command line: panel file, then builder-check file.
set -e
if [ $# -ne 2 ]; then
  echo "usage: uma_cutoff_panel.sh panel.txt builder.txt" >&2
  exit 2
fi
panel=$1
builder=$2
root=$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)
cd "$root"
v185=$(git rev-parse HEAD)
t185=$(git rev-parse HEAD^{tree})
v183=1202f65625c9adeaf7216510b3e34188886a7cf9
t183=$(git rev-parse "${v183}^{tree}")
work=$(mktemp -d)
mkdir -p "$work/rgpot/UmaPot"
git show "${v183}:CppCore/rgpot/UmaPot/UmaEdges.cc" > "$work/UmaEdges.cc"
git show "${v183}:CppCore/rgpot/UmaPot/UmaEdges.hpp" > "$work/rgpot/UmaPot/UmaEdges.hpp"
git show "${v183}:CppCore/rgpot/UmaPot/UmaLinkedCell.hpp" > "$work/rgpot/UmaPot/UmaLinkedCell.hpp"
git show "${v183}:CppCore/rgpot/UmaPot/UmaContract.hpp" > "$work/rgpot/UmaPot/UmaContract.hpp"
common="-O3 -I CppCore -I third_party/vesin -std=c++20 -pthread"
g++ $common -I "$work" \
  -DRGPOT_ROW_HEAD=\"$v183\" -DRGPOT_ROW_TREE=\"$t183\" \
  CppCore/tests/uma_cutoff_abba.cc "$work/UmaEdges.cc" \
  third_party/vesin/vesin-single-build.cpp -o "$work/abba_v183"
g++ $common \
  -DRGPOT_ROW_HEAD=\"$v185\" -DRGPOT_ROW_TREE=\"$t185\" \
  CppCore/tests/uma_cutoff_abba.cc CppCore/rgpot/UmaPot/UmaEdges.cc \
  third_party/vesin/vesin-single-build.cpp -o "$work/abba_v185"
seed=20261128
{
  echo "# start $(date +%Y-%m-%dT%H:%M:%S%z)"
  echo "# loadavg_before $(cat /proc/loadavg)"
  echo "# nproc $(nproc)"
  echo "# cpu $(awk -F: '/model name/{gsub(/^ /,"",$2); print $2; exit}' /proc/cpuinfo)"
  echo "# v183 $v183 $t183"
  echo "# v185 $v185 $t185"
} > "$panel"
cp "$panel" "$builder"
run_pair() {
  series=$1
  n=$2
  threads=$3
  reuse=$4
  call=$5
  "$work/abba_v183" once "$series"_v183 "$n" "$threads" "$reuse" "$seed" "$call" >> "$panel"
  "$work/abba_v185" once "$series"_v185 "$n" "$threads" "$reuse" "$seed" "$call" >> "$panel"
}
for call in 0 1 2 3 4 5 6 7; do
  run_pair weak 8192 1 0 "$call"
  run_pair weak 16384 2 0 "$call"
  run_pair weak 32768 4 0 "$call"
  run_pair weak 65536 8 0 "$call"
  run_pair strong 65536 1 0 "$call"
  run_pair strong 65536 2 0 "$call"
  run_pair strong 65536 4 0 "$call"
  run_pair strong 65536 8 0 "$call"
done
for call in 0 1 2 3; do
  "$work/abba_v183" once builder_v183 65536 8 0 "$seed" "$call" >> "$builder"
  "$work/abba_v185" once builder_v185 65536 8 0 "$seed" "$call" >> "$builder"
done
{
  echo "# end $(date +%Y-%m-%dT%H:%M:%S%z)"
  echo "# loadavg_after $(cat /proc/loadavg)"
} >> "$panel"
{
  echo "# end $(date +%Y-%m-%dT%H:%M:%S%z)"
  echo "# loadavg_after $(cat /proc/loadavg)"
} >> "$builder"
rm -rf "$work"
