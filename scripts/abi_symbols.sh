#!/bin/sh
# Print the stable C symbols of a library, one per line.
# The default names are the neighbour and fast-list exports of librgpot,
# which every build carries. Engine and RPC names appear only in the
# builds that include them. A third argument replaces the name pattern;
# the rgpot-core library is checked with '^rgpot_' against
# abi_core_baseline.txt. A missing baseline name fails. An extra name
# does not: it is an addition.
set -eu
lib=${1:?librgpot.so}
baseline=${2:-}
pattern=${3:-'^rgpot_(nlist_|fast_)'}
found=$(mktemp)
want=$(mktemp)
trap 'rm -f "$found" "$want"' EXIT
nm -D --defined-only "$lib" | awk -v pat="$pattern" '$3 ~ pat {print $3}' | LC_ALL=C sort -u > "$found"
if [ -z "$baseline" ]; then
  cat "$found"
  exit 0
fi
LC_ALL=C sort -u "$baseline" > "$want"
missing=$(LC_ALL=C comm -23 "$want" "$found" || true)
if [ -n "$missing" ]; then
  echo "C ABI lost symbols:" >&2
  echo "$missing" >&2
  exit 1
fi
echo "C ABI keeps $(wc -l < "$want") baseline symbols"
