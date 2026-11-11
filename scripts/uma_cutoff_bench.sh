#!/bin/sh
# Copyright 2023--present rgpot developers
set -e
root=$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)
cd "$root"
python3 scripts/uma_cutoff_report.py \
  CppCore/tests/uma_cutoff_panel_20261129.txt \
  > CppCore/tests/uma_cutoff_bench.txt
