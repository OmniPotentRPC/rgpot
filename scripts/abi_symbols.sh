#!/bin/sh
# Print the stable C symbols of librgpot, one per line.
# With baselines, fail on any baseline name the library does not export.
# Neighbour names are always exported. Fast-list names exist only when
# the fast list is built, so meson passes abi_baseline_fast.txt only then.
# Engine and RPC names appear only in the builds that include them.
# An extra name does not fail: it is an addition.
set -eu
lib=${1:?librgpot.so}
shift
found=$(mktemp)
syms=$(mktemp)
want=$(mktemp)
trap 'rm -f "$found" "$syms" "$want"' EXIT
# Apple nm has no -D and prefixes C names with _. A static librgpot
# (the macOS CI default) has no dynamic table, only member globals.
strip=
if [ "$(uname -s)" = Darwin ]; then
    nm -gU "$lib" > "$syms"
    strip=_
else
    case $lib in
        *.a) nm -g --defined-only "$lib" > "$syms" ;;
        *) nm -D --defined-only "$lib" > "$syms" ;;
    esac
fi
awk -v p="$strip" '{s = $NF; if (p != "" && index(s, p) == 1) s = substr(s, 2)}
    s ~ /^rgpot_(nlist_|fast_)/ {print s}' "$syms" | sort -u > "$found"
if [ $# -eq 0 ]; then cat "$found"; exit 0; fi
sort -u "$@" > "$want"
missing=$(comm -23 "$want" "$found")
if [ -n "$missing" ]; then echo "C ABI lost symbols:" >&2; echo "$missing" >&2; exit 1; fi
echo "C ABI keeps $(wc -l < "$want") baseline symbols"
