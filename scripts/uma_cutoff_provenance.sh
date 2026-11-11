#!/bin/sh
# Copyright 2023--present rgpot developers
set -e
root=$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)
cd "$root"
python3 scripts/uma_cutoff_sidecar.py \
  CppCore/tests/uma_cutoff_calls.txt \
  CppCore/tests/uma_cutoff_calls_20261121.txt \
  CppCore/tests/uma_cutoff_calls_20261122.txt \
  CppCore/tests/uma_cutoff_calls_20261124.txt \
  CppCore/tests/uma_cutoff_calls_20261125.txt \
  CppCore/tests/uma_cutoff_abba_20261126.txt \
  > CppCore/tests/uma_cutoff_provenance.txt
