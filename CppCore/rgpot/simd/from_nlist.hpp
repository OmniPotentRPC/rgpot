#pragma once
// MIT License
// Copyright 2023--present rgpot developers

#include "rgpot/nlist/PairListCache.hpp"
#include "rgpot/simd/pods.hpp"

namespace rgpot::simd {

inline PairView view_from(const nlist::CachedPairList::EvalView &ev,
                          const double *R) {
  PairView v{};
  v.R = R;
  v.rows = ev.rows;
  v.nbr = ev.nbr;
  v.n = ev.n;
  v.cutoff2 = ev.cutoff2;
  v.w[0] = ev.w[0];
  v.w[1] = ev.w[1];
  v.w[2] = ev.w[2];
  v.inv[0] = ev.inv[0];
  v.inv[1] = ev.inv[1];
  v.inv[2] = ev.inv[2];
  v.code = ev.code;
  v.shift = ev.shift;
  v.general = ev.general;
  switch (ev.fold) {
  case nlist::CachedPairList::FoldMode::None:
    v.fold = 0;
    break;
  case nlist::CachedPairList::FoldMode::Round:
    v.fold = 1;
    break;
  case nlist::CachedPairList::FoldMode::Coded:
    v.fold = 2;
    break;
  case nlist::CachedPairList::FoldMode::General:
    v.fold = 3;
    break;
  }
  return v;
}

} // namespace rgpot::simd
