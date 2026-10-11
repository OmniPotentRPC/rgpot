#!/bin/bash
# Manual Verificarlo study. Thirty MCA samples of one pair term and of the
# fixed 8-wide reduction, at the two sizes used by the hash driver.
# Denis, de Oliveira Castro and Petit, ARITH 2016, doi:10.1109/ARITH.2016.31.
set -euo pipefail
if ! command -v verificarlo-c++ >/dev/null 2>&1; then
  echo "verificarlo-c++ is not on PATH" >&2
  exit 1
fi
ROOT=$(CDPATH= cd -- "$(dirname "$0")/../.." && pwd)
cd "$ROOT"
out=$(mktemp -d)
verificarlo-c++ -std=c++20 -O2 -ffp-contract=off \
  -I CppCore \
  tests/invariance/mca_sites.cc \
  -o "$out/mca"
export VFC_BACKENDS="libinterflop_mca.so --mode=mca --precision-binary64=53"
: > verificarlo-digits.txt
for n in 4000 32000; do
  for sample in $(seq 1 30); do
    "$out/mca" "$n" >> "$out/samples_${n}.txt"
  done
done
python3 - "$out" << 'PY'
import math, pathlib, statistics, sys
root = pathlib.Path(sys.argv[1])
lines = ["pot size site digits"]
for n in (4000, 32000):
    rows = []
    for line in (root / f"samples_{n}.txt").read_text().splitlines():
        parts = line.split()
        rows.append(parts)
    # columns: lj_energy lj_force lj_virial eam_energy eam_force
    cols = list(zip(*rows))
    names = ["lj_energy", "lj_force", "lj_virial", "eam_energy", "eam_force"]
    for name, col in zip(names, cols):
        vals = [float(x) for x in col]
        mean = statistics.fmean(vals)
        std = statistics.pstdev(vals)
        if mean == 0.0 or std == 0.0:
            digits = "exact"
        else:
            digits = f"{-math.log10(std / abs(mean)):.2f}"
        lines.append(f"site {name} n {n} digits {digits}")
print("\n".join(lines))
open("verificarlo-digits.txt", "w").write("\n".join(lines) + "\n")
PY
cat verificarlo-digits.txt
rm -rf "$out"
