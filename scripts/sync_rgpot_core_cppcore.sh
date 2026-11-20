#!/usr/bin/env bash
# MIT License
# Copyright 2023--present rgpot developers

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
src_root="${repo_root}/CppCore/rgpot"
dst_root="${repo_root}/rgpot-core/cppcore-staging/rgpot"

if [[ "${1:-}" == "--dest" ]]; then
  if [[ "${2:-}" == "" ]]; then
    echo "--dest needs a path" >&2
    exit 1
  fi
  dst_root="${2}/rgpot"
fi

if [[ ! -d "${src_root}" ]]; then
  echo "missing source tree: ${src_root}" >&2
  exit 1
fi

dirs=(
  "EAMTable"
  "nlist"
  "simd"
  "types"
)
files=(
  "Slices.cc"
  "ForceStructs.hpp"
  "Locality.hpp"
  "ParamHash.hpp"
  "PerAtom.hpp"
  "Potential.hpp"
  "PotHelpers.hpp"
  "Slices.hpp"
  "Staged.hpp"
  "base_types.hpp"
  "pot_caps.hpp"
  "pot_types.hpp"
  "stress.hpp"
)

rm -rf "${dst_root}"
mkdir -p "${dst_root}"

for dir in "${dirs[@]}"; do
  mkdir -p "${dst_root}/${dir}"
  cp -a "${src_root}/${dir}/." "${dst_root}/${dir}/"
done

for file in "${files[@]}"; do
  cp -a "${src_root}/${file}" "${dst_root}/${file}"
done

echo "synced minimal EAM bridge closure ${src_root} -> ${dst_root}"
