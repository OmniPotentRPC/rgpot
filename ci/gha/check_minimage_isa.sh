#!/bin/sh
# Compare folded-displacement hashes from binaries built for different
# instruction sets. A binary that prints "skip" cannot run on this CPU.
# Usage: check_minimage_isa.sh GOLDEN [WRAPPER... --] EXE...
# WRAPPER is a cross build's exe_wrapper, run in front of each EXE.
set -eu
golden=$1
shift
quote() {
  printf "'%s'" "$(printf '%s' "$1" | sed "s/'/'\\\\''/g")"
}
wrapper=
for arg in "$@"; do
  if [ "$arg" = -- ]; then
    while [ "$1" != -- ]; do
      wrapper="$wrapper $(quote "$1")"
      shift
    done
    shift
    break
  fi
done
cr=$(printf '\r')
fail=0
for exe in "$@"; do
  out=$(eval "$wrapper \"\$exe\"")
  # A Windows binary ends its line with CRLF.
  out=${out%"$cr"}
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
