#!/bin/sh
# Copyright 2023--present rgpot developers
# Build the linked-cell cover driver with gcov and print line and
# branch coverage. A line counts when any record executed it.
# A branch counts when its taken percentage is above zero.
# Throw edges stay in the denominator.
# The only excluded file is the vendored vesin translation unit.
set -e
root=$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)
cd "$root"
out=${1:-CppCore/tests/uma_cutoff_gcov.txt}
cc=${CXX:-g++}
rm -f uma_cutoff_cover uma_cutoff_cover-*.gcno uma_cutoff_cover-*.gcda *.gcov
"$cc" -I CppCore -I third_party/vesin -std=c++20 -O0 -g --coverage -pthread \
  CppCore/tests/uma_linked_cell_cover.cc \
  CppCore/rgpot/UmaPot/UmaEdges.cc \
  third_party/vesin/vesin-single-build.cpp \
  -o uma_cutoff_cover
./uma_cutoff_cover
gcov -b \
  uma_cutoff_cover-uma_linked_cell_cover.gcno \
  uma_cutoff_cover-UmaEdges.gcno \
  uma_cutoff_cover-vesin-single-build.gcno >/dev/null
python3 - "$out" << 'PY'
import re
import sys
from pathlib import Path

# Vendored generated file. Device backends in it are not buildable here.
EXCLUDED = {
    "vesin-single-build.cpp": (
        "vendored generated translation unit; "
        "non-CPU device paths are not buildable on this host"
    ),
}
SCORED = ("UmaLinkedCell.hpp", "UmaEdges.cc", "UmaGrow.hpp")


def score(path):
    line_hit = {}
    branches = {}
    current = None
    branch_i = 0
    source = {}
    for line in Path(path).read_text(errors="replace").splitlines():
        parts = line.split(":", 2)
        if len(parts) >= 3:
            count = parts[0].strip()
            try:
                number = int(parts[1])
            except ValueError:
                number = None
            if number and number >= 1 and count != "-":
                current = number
                branch_i = 0
                hit = count.rstrip("*").isdigit()
                line_hit[number] = line_hit.get(number, False) or hit
                source[number] = parts[2].strip()
        stripped = line.strip()
        if current and stripped.startswith("branch"):
            match = re.search(r"taken\s+(\d+)%", line)
            if match:
                pct = int(match.group(1))
            elif "never executed" in line:
                pct = 0
            else:
                continue
            key = (current, branch_i)
            if pct > branches.get(key, -1):
                branches[key] = pct
            branch_i += 1
    return line_hit, branches, source


def totals(line_hit, branches):
    lc = sum(line_hit.values())
    lt = len(line_hit)
    bc = sum(1 for pct in branches.values() if pct > 0)
    bt = len(branches)
    return lc, lt, bc, bt


def untaken_lines(branches, source):
    lines = sorted({key[0] for key, pct in branches.items() if pct == 0})
    return " ".join(f"{n}:{source.get(n, '')[:60]}" for n in lines)


text = []
text.append("throw edges are counted. never-executed branches are counted.")
failed = False
for name in SCORED:
    line_hit, branches, source = score(name + ".gcov")
    lc, lt, bc, bt = totals(line_hit, branches)
    lp = 100 * lc / lt
    bp = 100 * bc / bt
    text.append(
        f"{name} lines {lc}/{lt} {lp:.2f}% raw_branches {bc}/{bt} {bp:.2f}%"
    )
    text.append(f"{name} untaken {untaken_lines(branches, source)}")
    if lp < 98.0 or bp < 98.0:
        failed = True
line_hit, branches, source = score("vesin-single-build.cpp.gcov")
lc, lt, bc, bt = totals(line_hit, branches)
text.append(
    f"vesin-single-build.cpp with lines {lc}/{lt} {100 * lc / lt:.2f}% "
    f"raw_branches {bc}/{bt} {100 * bc / bt:.2f}%"
)
text.append(
    "vesin-single-build.cpp without excluded "
    + EXCLUDED["vesin-single-build.cpp"]
)
# The gate fails when any file other than the vendored unit is excluded.
if set(EXCLUDED) != {"vesin-single-build.cpp"}:
    raise SystemExit("exclusion is not the vendored vesin unit")
for name in SCORED:
    if name in EXCLUDED or name + ".gcov" in EXCLUDED:
        raise SystemExit("scored file was excluded")
Path(sys.argv[1]).write_text("\n".join(text) + "\n")
print("\n".join(text))
if failed:
    raise SystemExit("coverage is under 98 percent")
PY
mkdir -p CppCore/tests/uma_gcov_raw
cp -f UmaLinkedCell.hpp.gcov UmaEdges.cc.gcov UmaGrow.hpp.gcov \
  vesin-single-build.cpp.gcov CppCore/tests/uma_gcov_raw/
rm -f uma_cutoff_cover uma_cutoff_cover-*.gcno uma_cutoff_cover-*.gcda *.gcov
