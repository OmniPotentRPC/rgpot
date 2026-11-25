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

files=(
  "Slices.cc"
  "Slices.hpp"
  "ForceStructs.hpp"
  "Locality.hpp"
  "PairEval.hpp"
  "ParamHash.hpp"
  "PerAtom.hpp"
  "Potential.hpp"
  "PotHelpers.cc"
  "PotHelpers.hpp"
  "Staged.hpp"
  "base_types.hpp"
  "pot_caps.hpp"
  "pot_types.hpp"
  "rgpot_api.h"
  "stress.hpp"
  "EAMTable/EAMTableImages.cc"
  "EAMTable/EAMTablePot.cc"
  "EAMTable/EAMTablePot.hpp"
  "EAMTable/EAMTableTerms.hpp"
  "nlist/HalfShell.cc"
  "nlist/HalfShell.hpp"
  "nlist/HalfShellDetail.hpp"
  "nlist/HalfShellWalk.cc"
  "nlist/MinimumImage.cpp"
  "nlist/MinimumImage.hpp"
  "nlist/MultiImage.hpp"
  "nlist/NeighbourList.cc"
  "nlist/NeighbourList.hpp"
  "nlist/Neighbours.cc"
  "nlist/Neighbours.hpp"
  "nlist/rgpot_nlist.h"
  "simd/clones.hpp"
  "simd/eam_body.inc"
  "simd/exact_fma.hpp"
  "simd/exact_fma_hw.cc"
  "simd/exp.cc"
  "simd/exp.hpp"
  "simd/pair_loops.hpp"
  "simd/pods.hpp"
  "simd/reduce.hpp"
  "simd/sse_soft.hpp"
  "types/AtomMatrix.hpp"
)

rm -rf "${dst_root}"
mkdir -p "${dst_root}"

for file in "${files[@]}"; do
  mkdir -p "$(dirname "${dst_root}/${file}")"
  cp -a "${src_root}/${file}" "${dst_root}/${file}"
done

mkdir -p "${dst_root}"
cp -a "${repo_root}/include/rgpot/plugin.h" "${dst_root}/plugin.h"

echo "synced minimal EAM bridge closure ${src_root} -> ${dst_root}"
