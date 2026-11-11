#pragma once
// MIT License
// Copyright 2023--present rgpot developers
//
// Thin copies for callers that already have a neighbour convention.
// Metatomic wants five label columns and the vectors left in the table.
// The labels are (first, second, shift_a, shift_b, shift_c).

#include "rgpot/nlist/NeighbourList.hpp"

namespace rgpot::nlist {

/// One metatomic neighbour-list request. `strict` is distance < cutoff.
/// A loose cut uses distance <= cutoff when the skin is zero.
inline Query metatomicQuery(double cutoff, bool full_list, bool strict,
                            const bool periodic[3]) {
  Query query;
  query.cutoff = cutoff;
  query.shape = Pairs{full_list ? Shell::Full : Shell::Half};
  query.cut = strict ? Cut::Strict : Cut::Loose;
  query.fields.shifts = true;
  query.fields.vectors = true;
  query.fields.distances = false;
  if (periodic != nullptr) {
    query.periodic = {periodic[0], periodic[1], periodic[2]};
  }
  return query;
}

/// One sweep, both shells. `fn` receives the five metatomic label fields
/// and the vector `r_j - r_i + S H`. The distance is not copied.
template <typename Fn>
void metatomicPairs(const Sweep &sweep, bool full_list, Fn &&fn) {
  sweep.each(full_list ? Shell::Full : Shell::Half,
             [&](std::int32_t i, std::int32_t j, int sa, int sb, int sc, double x, double y,
                 double z, double) { fn(i, j, sa, sb, sc, x, y, z); });
}

/// Writes `5 * pairs.first.size()` integers. The vectors stay in the table.
inline void copyMetatomicLabels(const View &pairs, std::int32_t *labels) {
  for (std::size_t k = 0; k < pairs.first.size(); ++k) {
    labels[5 * k] = pairs.first[k];
    labels[5 * k + 1] = pairs.second[k];
    const bool shifts = pairs.shift.size() >= (k + 1) * 3;
    labels[5 * k + 2] = shifts ? pairs.shift[3 * k] : 0;
    labels[5 * k + 3] = shifts ? pairs.shift[3 * k + 1] : 0;
    labels[5 * k + 4] = shifts ? pairs.shift[3 * k + 2] : 0;
  }
}

} // namespace rgpot::nlist
