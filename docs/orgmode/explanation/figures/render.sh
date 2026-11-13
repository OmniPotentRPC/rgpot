#!/usr/bin/env bash
# Render the Graphviz sources in this directory to SVG.
#
#     pixi exec --spec graphviz bash docs/orgmode/explanation/figures/render.sh
set -euo pipefail
cd "$(dirname "$0")"
for src in *.dot; do
    dot -Tsvg "$src" -o "${src%.dot}.svg"
done
