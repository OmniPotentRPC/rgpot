#!/bin/sh
# Copyright 2023--present rgpot developers
# Build the linked-cell cover driver with gcov and print line and
# branch coverage for the two cutoff sources. A line counts as covered
# when any record executed it. A branch counts as covered when its
# taken percentage is above zero.
set -e
root=$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)
cd "$root"
out=${1:-uma_cutoff_gcov.txt}
cc=${CXX:-g++}
"$cc" -I CppCore -I third_party/vesin -std=c++20 -O0 -g --coverage -pthread \
  CppCore/tests/uma_linked_cell_cover.cc \
  CppCore/rgpot/UmaPot/UmaEdges.cc \
  third_party/vesin/vesin-single-build.cpp \
  -o uma_cutoff_cover
./uma_cutoff_cover
gcov -b uma_cutoff_cover-uma_linked_cell_cover.gcno uma_cutoff_cover-UmaEdges.gcno >/dev/null
python3 - "$out" << 'PY'
import re
import sys
from pathlib import Path

def lines_of(path, limit):
    best = {}
    for line in Path(path).read_text(errors="replace").splitlines():
        parts = line.split(":", 2)
        if len(parts) < 3:
            continue
        count = parts[0].strip()
        try:
            number = int(parts[1])
        except ValueError:
            continue
        if number < 1 or number > limit or count == "-":
            continue
        best[number] = best.get(number, False) or count != "#####"
    covered = sum(best.values())
    return covered, len(best)

def branches_of(path, limit):
    current = None
    covered = total = 0
    for line in Path(path).read_text(errors="replace").splitlines():
        parts = line.split(":", 2)
        if len(parts) >= 3:
            try:
                current = int(parts[1])
            except ValueError:
                pass
        match = re.search(r"taken\s+(\d+)%", line)
        if not match or current is None or not 1 <= current <= limit:
            continue
        total += 1
        if int(match.group(1)) > 0:
            covered += 1
    return covered, total

rows = []
for name, limit in (
    ("UmaLinkedCell.hpp.gcov", 320),
    ("UmaEdges.cc.gcov", 530),
):
    lc, lt = lines_of(name, limit)
    bc, bt = branches_of(name, limit)
    rows.append((name, lc, lt, bc, bt))
text = []
line_c = line_t = branch_c = branch_t = 0
for name, lc, lt, bc, bt in rows:
    line_c += lc
    line_t += lt
    branch_c += bc
    branch_t += bt
    text.append(
        f"{name} lines {lc}/{lt} {100 * lc / lt:.2f}% "
        f"branches {bc}/{bt} {100 * bc / bt:.2f}%"
    )
text.append(
    f"combined lines {line_c}/{line_t} {100 * line_c / line_t:.2f}% "
    f"branches {branch_c}/{branch_t} {100 * branch_c / branch_t:.2f}%"
)
Path(sys.argv[1]).write_text("\n".join(text) + "\n")
print("\n".join(text))
PY
rm -f uma_cutoff_cover uma_cutoff_cover-*.gcno uma_cutoff_cover-*.gcda *.gcov
