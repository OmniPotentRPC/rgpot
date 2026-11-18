#!/usr/bin/env bash
# Fail if the built librgpot exports any Fortran symbol.
#
# The Fortran kernels are reached only from C++ inside librgpot, so their
# bind(c) entries and every module symbol build with hidden visibility.
# An exported bare Fortran name (edip_, force_, a COMMON block, or a
# module symbol like __rgpot_sw_MOD_...) means a target lost its
# visibility setting, and two potentials exporting the same legacy name
# would then bind to whichever object loaded first.
set -euo pipefail

lib="${1:-}"
if [[ -z "$lib" ]]; then
   echo "usage: $0 <path to librgpot.so>" >&2
   exit 2
fi
if [[ ! -f "$lib" ]]; then
   echo "audit: $lib not found" >&2
   exit 2
fi

if ! command -v nm >/dev/null 2>&1; then
   echo "audit: nm unavailable, skipping" >&2
   exit 0
fi

# Dynamic symbols defined by this object (ignore undefined ones: those
# come from libgfortran and the C library).
defined=$(nm -D --defined-only "$lib" | awk '{print $3}')

# Fortran module symbols carry a compiler infix; bind(c) entries and
# legacy entry points are lowercase C identifiers. C++ symbols (_Z...)
# and the public MD session C API are the library's own interface and are
# expected.
offenders=$(printf '%s\n' "$defined" \
   | grep -v -E '^rgpot_md_session_' \
   | grep -E '__.*_MOD_|^rgpot_[a-z0-9_]*_force$|^[a-z0-9]+_$' || true)

if [[ -n "$offenders" ]]; then
   echo "audit: librgpot exports Fortran symbols:" >&2
   printf '  %s\n' $offenders >&2
   exit 1
fi

echo "audit: no Fortran symbols exported by $(basename "$lib")"
