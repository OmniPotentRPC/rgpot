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
#
# Compiler instrumentation and profiling passes also place per-function
# metadata and counters in writable local symbols:
# - GCC / gfortran (gcc/coverage.cc:build_var): __gcov<N><sep><fn> and
#   __gcov_<sep><fn>, where <sep> is '.', '_' or '$'.
# - LLVM / Clang / Flang / Intel ifx (llvm/ProfileData/InstrProf.h,
#   GCOVProfiling.cpp, SanitizerCoverage.cpp): __profc_, __profd_,
#   __profn_, __profv_, __profvp_, __profbm_, __profvt_, __llvm_gcov_,
#   __llvm_internal_gcov_, __llvm_prf_, __llvm_profile_, __llvm_cov*,
#   __covrec_ and __sancov_.
offenders=$(nm -A "$@" | awk '
   NF >= 3 && $(NF-1) ~ /^[bdgsv]$/ {
      sym = $NF
      if (sym ~ /^__gcov([0-9]+|_)[._$]/) next
      if (sym ~ /^__prof(c|d|n|v|vp|bm|vt)_/) next
      if (sym ~ /^__llvm_(gcov|internal_gcov|prf|profile)_/) next
      if (sym ~ /^__llvm_cov/) next
      if (sym ~ /^__covrec_/) next
      if (sym ~ /^__sancov_/) next
      print $1, sym
   }
')

if [[ -n "$offenders" ]]; then
   echo "audit: Fortran kernels keep procedure-local static data:" >&2
   printf '%s\n' "$offenders" | sed 's/^/  /' >&2
   exit 1
fi

echo "audit: no procedure-local static data in the Fortran kernels"
