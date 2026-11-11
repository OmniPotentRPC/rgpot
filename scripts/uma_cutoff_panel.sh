#!/bin/sh
# Copyright 2023--present rgpot developers
# One process times v1.8.3, v1.8.4 and v1.8.6 on the same calls.
# Arguments are the output file and the three code commits.
# The script does not read HEAD.
set -e
if [ $# -ne 4 ]; then
  echo "usage: uma_cutoff_panel.sh out.txt v183 v184 v186" >&2
  exit 2
fi
out=$1
v183=$2
v184=$3
v186=$4
root=$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)
cd "$root"
t183=$(git rev-parse "${v183}^{tree}")
t184=$(git rev-parse "${v184}^{tree}")
t186=$(git rev-parse "${v186}^{tree}")
work=build/uma_panel
rm -rf "$work"
mkdir -p "$work"
extract() {
  sha=$1
  dest=$2
  mkdir -p "$dest"
  git archive "$sha" CppCore/rgpot/UmaPot | tar -x -C "$dest"
}
extract "$v183" "$work/v183"
extract "$v184" "$work/v184"
extract "$v186" "$work/v186"
build_one() {
  name=$1
  sha=$2
  flags="-O3 -std=c++20 -pthread -I $work/$name/CppCore -I CppCore -I third_party/vesin"
  g++ $flags -Drgpot="rgpot_${name}" -DRGPOT_ENTRY="rgpot_time_${name}" \
    -c "$work/$name/CppCore/rgpot/UmaPot/UmaEdges.cc" -o "$work/${name}.o"
  g++ $flags -Drgpot="rgpot_${name}" -DRGPOT_ENTRY="rgpot_time_${name}" \
    -c CppCore/tests/uma_cutoff_entry.cc -o "$work/${name}_entry.o"
}
build_one v183 "$v183"
build_one v184 "$v184"
build_one v186 "$v186"
g++ -O3 -std=c++20 -pthread -I CppCore -I third_party/vesin \
  -DRGPOT_V183_HEAD="\"$v183\"" -DRGPOT_V183_TREE="\"$t183\"" \
  -DRGPOT_V184_HEAD="\"$v184\"" -DRGPOT_V184_TREE="\"$t184\"" \
  -DRGPOT_V186_HEAD="\"$v186\"" -DRGPOT_V186_TREE="\"$t186\"" \
  CppCore/tests/uma_cutoff_panel.cc \
  "$work/v183.o" "$work/v183_entry.o" \
  "$work/v184.o" "$work/v184_entry.o" \
  "$work/v186.o" "$work/v186_entry.o" \
  third_party/vesin/vesin-single-build.cpp \
  -o "$work/panel"
"$work/panel" 20261129 > "$out"
rm -rf "$work"
