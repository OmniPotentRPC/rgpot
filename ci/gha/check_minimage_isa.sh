#!/bin/sh
# Compare folded-displacement hashes from binaries built for different
# instruction sets. A binary that prints "skip" cannot run on this CPU.
set -eu
golden=$1
shift
fail=0
for exe in "$@"; do
  out=$("$exe")
  if [ "$out" = skip ]; then
    echo "skip $exe"
    continue
  fi
  if [ "$out" != "$golden" ]; then
    echo "mismatch $exe got $out want $golden" >&2
    fail=1
  else
    echo "match $exe"
  fi
done
exit "$fail"
