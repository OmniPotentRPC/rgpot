#!/bin/sh
# Print the stable C symbols of librgpot, one per line.
# Neighbour and fast-list names are always exported. Engine and RPC
# names appear only in the builds that include them. A missing baseline
# name fails. An extra name does not: it is an addition.
set -eu
lib=${1:?librgpot.so}
baseline=${2:-}
found=$(mktemp)
trap 'rm -f "$found"' EXIT
nm -D --defined-only "$lib" | awk '$3 ~ /^rgpot_(nlist_|fast_)/ {print $3}' | sort -u > "$found"
if [ -z "$baseline" ]; then
  cat "$found"
  exit 0
fi
missing=$(comm -23 "$baseline" "$found" || true)
if [ -n "$missing" ]; then
  echo "C ABI lost symbols:" >&2
  echo "$missing" >&2
  exit 1
fi
echo "C ABI keeps $(wc -l < "$baseline") baseline symbols"
