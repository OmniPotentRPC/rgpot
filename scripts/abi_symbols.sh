#!/bin/sh
# Print the stable C symbols of librgpot, one per line.
# With baselines, fail on any baseline name the library does not export.
# abi_baseline.txt is every function rgpot/rgpot_api.h marks, and those
# are always exported. Fast-list names exist only when the fast list is
# built, so meson passes abi_baseline_fast.txt only then.
# An extra name does not fail: it is an addition.
set -eu
lib=${1:?librgpot.so}
shift
found=$(mktemp)
syms=$(mktemp)
want=$(mktemp)
trap 'rm -f "$found" "$syms" "$want"' EXIT
# A DLL exports what its export table lists, read with dumpbin (MSVC) or
# objdump (binutils). Apple nm has no -D and prefixes C names with _. A
# static librgpot (the macOS CI default) has no dynamic table, only
# member globals.
strip=
case $lib in
    *.dll | *.DLL)
        if command -v dumpbin > /dev/null 2>&1; then
            dumpbin -nologo -exports "$lib" | tr -d '\r' |
                awk 'NF >= 4 && $1 ~ /^[0-9]+$/ && $3 ~ /^[0-9A-Fa-f]+$/ {print $4}' > "$syms"
        else
            objdump=
            for tool in x86_64-w64-mingw32-objdump objdump; do
                if command -v "$tool" > /dev/null 2>&1; then
                    objdump=$tool
                    break
                fi
            done
            if [ -z "$objdump" ]; then
                echo "no dumpbin or objdump to read $lib" >&2
                exit 2
            fi
            "$objdump" -p "$lib" | tr -d '\r' |
                awk '/^\[Ordinal\/Name Pointer\] Table/ {t = 1; next}
                    t && NF == 0 {t = 0}
                    t {print $NF}' > "$syms"
        fi
        ;;
    *)
        if [ "$(uname -s)" = Darwin ]; then
            nm -gU "$lib" > "$syms"
            strip=_
        else
            case $lib in
                *.a) nm -g --defined-only "$lib" > "$syms" ;;
                *) nm -D --defined-only "$lib" > "$syms" ;;
            esac
        fi
        ;;
esac
awk -v p="$strip" '{s = $NF; if (p != "" && index(s, p) == 1) s = substr(s, 2)}
    s ~ /^rgpot_/ {print s}' "$syms" | sort -u > "$found"
if [ $# -eq 0 ]; then cat "$found"; exit 0; fi
sort -u "$@" > "$want"
missing=$(comm -23 "$want" "$found")
if [ -n "$missing" ]; then echo "C ABI lost symbols:" >&2; echo "$missing" >&2; exit 1; fi
echo "C ABI keeps $(wc -l < "$want") baseline symbols"
