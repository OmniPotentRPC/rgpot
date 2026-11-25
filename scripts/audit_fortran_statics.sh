#!/usr/bin/env bash
# MIT License
# Copyright 2023--present rgpot developers
#
# Fail if a Fortran kernel archive holds procedure-local static data.
#
# Each C++ potential instance owns its Fortran workspace, and instances
# evaluate on separate threads at once. A local static is one copy for
# every thread. gfortran emits one even under -frecursive for the hidden
# length of a deferred-length function result used inside an expression,
# such as "prefix: "//text_of(ptr). Two threads failing together then
# share that length: one gets a truncated message, the other copies past
# its allocation and corrupts the heap. Module variables are global
# symbols and are not reported here; the kernels write them only under
# std::call_once or from the process-serial entries without a handle.
set -euo pipefail

if [[ $# -lt 1 ]]; then
   echo "usage: $0 <archive or object>..." >&2
   exit 2
fi
for path in "$@"; do
   if [[ ! -f "$path" ]]; then
      echo "audit: $path not found" >&2
      exit 2
   fi
done

if ! command -v nm >/dev/null 2>&1; then
   echo "audit: nm unavailable, skipping" >&2
   exit 0
fi

# Lowercase b, d, g, s and v are local symbols in writable sections.
# Read-only constants (r) are shared safely.
offenders=$(nm -A "$@" | awk 'NF >= 3 && $(NF-1) ~ /^[bdgsv]$/ {print $1, $NF}')

if [[ -n "$offenders" ]]; then
   echo "audit: Fortran kernels keep procedure-local static data:" >&2
   printf '%s\n' "$offenders" | sed 's/^/  /' >&2
   exit 1
fi

echo "audit: no procedure-local static data in the Fortran kernels"
