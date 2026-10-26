#!/bin/sh
# Fail when the shared library $1 records an MPI or UCX library as NEEDED:
# librgpot loads librgpot_mpi with dlopen, so a host that never asks for
# calculator groups never maps libmpi.
set -eu
lib=$1
needed=$(readelf -d "$lib" | sed -n 's/.*(NEEDED).*\[\(.*\)\]/\1/p')
printf 'NEEDED: %s\n' $needed
if printf '%s\n' "$needed" | grep -E -i '^lib(mpi|mpi_cxx|open-pal|open-rte|pmix|ucp|ucs|uct|ucm|rgpot_mpi)' >/dev/null; then
  echo "$lib links MPI directly" >&2
  exit 1
fi
