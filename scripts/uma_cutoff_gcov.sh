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

def score(path):
    line_hit = {}
    branches = {}
    kinds = {}
    current = None
    last = None
    branch_i = 0
    for line in Path(path).read_text(errors="replace").splitlines():
        parts = line.split(":", 2)
        if len(parts) >= 3:
            count = parts[0].strip()
            try:
                number = int(parts[1])
            except ValueError:
                number = None
            if number and number >= 1 and count != "-":
                if last is not None and number < last - 5:
                    pass
                last = number
                current = number
                branch_i = 0
                hit = count != "#####"
                line_hit[number] = line_hit.get(number, False) or hit
        match = re.search(r"taken\s+(\d+)%", line)
        if match and current:
            pct = int(match.group(1))
            kind = "throw" if "throw" in line else "code"
            key = (current, branch_i)
            prev = branches.get(key, -1)
            if pct > prev:
                branches[key] = pct
                kinds[key] = kind
            branch_i += 1
    missed = sorted(n for n, hit in line_hit.items() if not hit)
    untaken = []
    for key, pct in sorted(branches.items()):
        if pct == 0:
            untaken.append((key[0], key[1], kinds[key]))
    return line_hit, branches, missed, untaken

rows = []
notes = []
for name in ("UmaLinkedCell.hpp.gcov", "UmaEdges.cc.gcov"):
    line_hit, branches, missed, untaken = score(name)
    lc = sum(line_hit.values())
    lt = len(line_hit)
    bc = sum(1 for pct in branches.values() if pct > 0)
    bt = len(branches)
    rows.append((name, lc, lt, bc, bt))
    notes.append(f"{name} missed_lines {' '.join(str(n) for n in missed)}")
    throw_n = sum(1 for _, _, kind in untaken if kind == "throw")
    code_n = sum(1 for _, _, kind in untaken if kind != "throw")
    code_lines = sorted({n for n, _, kind in untaken if kind != "throw"})
    notes.append(
        f"{name} untaken_throw {throw_n} untaken_code {code_n} "
        f"code_lines {' '.join(str(n) for n in code_lines)}"
    )
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
text.extend(notes)
Path(sys.argv[1]).write_text("\n".join(text) + "\n")
print("\n".join(text))
PY
rm -f uma_cutoff_cover uma_cutoff_cover-*.gcno uma_cutoff_cover-*.gcda *.gcov
