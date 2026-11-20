#!/usr/bin/env bash
# MIT License
# Copyright 2023--present rgpot developers

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
staging="${repo_root}/rgpot-core/cppcore-staging"
scratch="$(mktemp -d)"
trap 'rm -rf "${scratch}" "${staging}"' EXIT

pushd "${repo_root}" >/dev/null

scripts/sync_rgpot_core_cppcore.sh --dest "${scratch}"

RGPOT_CPPCORE_ROOT="${scratch}" cargo check -p rgpot-core

scripts/sync_rgpot_core_cppcore.sh

cargo package -p rgpot-core --list > "${scratch}/package-list.txt"
if ! rg '^cppcore-staging/rgpot/EAMTable/EAMTablePot.cc$' "${scratch}/package-list.txt" >/dev/null; then
  echo "packaged crate misses cppcore-staging EAM source files" >&2
  exit 1
fi
if ! rg '^cppcore-staging/rgpot/nlist/NeighbourList.cc$' "${scratch}/package-list.txt" >/dev/null; then
  echo "packaged crate misses cppcore-staging nlist source files" >&2
  exit 1
fi
if ! rg '^cppcore-staging/rgpot/simd/exact_fma_hw.cc$' "${scratch}/package-list.txt" >/dev/null; then
  echo "packaged crate misses cppcore-staging simd source files" >&2
  exit 1
fi

popd >/dev/null

echo "rgpot-core cppcore staging is current and package-visible"
