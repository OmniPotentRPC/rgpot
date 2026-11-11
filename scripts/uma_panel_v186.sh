#!/bin/sh
# Copyright 2023--present rgpot developers
# Labels are the code commits, not the branch tip.
set -e
root=$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)
cd "$root"
exec scripts/uma_cutoff_panel.sh \
  CppCore/tests/uma_cutoff_panel_20261129.txt \
  5d2849971d372caa18701103710923e92422538f \
  1624b14b158564fe987553cd61ab27a47cbb426d \
  2552b023f64d206d80e1fce0c4a2dcdda4a79a6a
