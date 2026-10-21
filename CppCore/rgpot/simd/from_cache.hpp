#pragma once
// MIT License
// Copyright 2023--present rgpot developers

#include "rgpot/nlist/PairListCache.hpp"
#include "rgpot/simd/pods.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace rgpot::simd {

/// Indices from one PairListCache evaluation, in the order the scan
/// reported them. The view points into this object.
struct CachedPairs {
  std::vector<std::int32_t> rows;
  std::vector<std::int32_t> nbr;
  PairView view{};
};

/// Build a half-list for the dispatched pair kernels. The scan stays in
/// PairListCache. fold is the orthorhombic round, with a zero inverse on
/// an axis the options leave unfolded.
inline void fill_view(CachedPairs &out, const double *R, std::size_t n,
                      const double *box,
                      const nlist::CachedPairList::Options &opt) {
  std::vector<std::int32_t> ij;
  nlist::PairListCache::global().evaluate(
      R, n, box, opt,
      [&](std::int32_t i, std::int32_t j, double, double, double, double) {
        ij.push_back(i);
        ij.push_back(j);
      });
  out.rows.assign(n + 1, 0);
  for (std::size_t p = 0; p < ij.size(); p += 2) {
    out.rows[static_cast<std::size_t>(ij[p]) + 1] += 1;
  }
  for (std::size_t i = 0; i < n; ++i) {
    out.rows[i + 1] += out.rows[i];
  }
  out.nbr.assign(ij.size() / 2, 0);
  std::vector<std::int32_t> cursor(out.rows.begin(), out.rows.end() - 1);
  for (std::size_t p = 0; p < ij.size(); p += 2) {
    const auto i = static_cast<std::size_t>(ij[p]);
    out.nbr[static_cast<std::size_t>(cursor[i]++)] = ij[p + 1];
  }
  PairView v{};
  v.R = R;
  v.rows = out.rows.data();
  v.nbr = out.nbr.data();
  v.n = n;
  v.cutoff2 = opt.cutoff * opt.cutoff;
  v.fold = 1;
  v.w[0] = box[0];
  v.w[1] = box[4];
  v.w[2] = box[8];
  for (int k = 0; k < 3; ++k) {
    const double w = v.w[k];
    v.inv[k] = (opt.periodic[static_cast<std::size_t>(k)] && w != 0.0)
                   ? 1.0 / w
                   : 0.0;
  }
  out.view = v;
}

} // namespace rgpot::simd
