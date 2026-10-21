// MIT License
// Copyright 2023--present rgpot developers

/**
 * @brief Implementation of the free-boundary Lennard-Jones cluster
 * potential.
 *
 * Ported from eOn (https://github.com/TheochemUI/eOn,
 * client/potentials/LJCluster), BSD-3-Clause licensed, copyright the eOn
 * Development Team.
 */

// clang-format off
#include <cmath>
// clang-format on

#include "rgpot/LennardJones/LJClusterPot.hpp"
#include "rgpot/nlist/PairListCache.hpp"
#include "rgpot/simd/api.hpp"
#include "rgpot/simd/from_cache.hpp"

namespace rgpot {

/**
 * @class LJClusterPot
 * @details
 *
 * Every pair within the cutoff contributes, with vectors taken straight
 * from the coordinates: ``opt.periodic`` is off in all three directions,
 * so the pair scan never folds a separation into the cell. A degenerate
 * input cell (any non-positive diagonal entry) is replaced by a 1e6
 * Angstrom cube, which keeps the pair list cacheable for callers that hand
 * over a cluster without a box.
 */
void LJClusterPot::forceImpl(const ForceInput &in, ForceOut *out) const {
  const auto N = static_cast<long>(in.nAtoms);
  const double *R = in.pos;
  double *F = out->F;
  double *U = &out->energy;
  *U = 0.0;
  for (long i = 0; i < N; i++) {
    F[3 * i] = 0.0;
    F[3 * i + 1] = 0.0;
    F[3 * i + 2] = 0.0;
  }
  if (N < 2) {
    return;
  }

  nlist::CachedPairList::Options opt;
  opt.cutoff = (cuttOffR > 0.0) ? cuttOffR : 1.0e6;
  opt.periodic = {{false, false, false}};

  static constexpr double free_box[9] = {1e6, 0.0, 0.0, 0.0, 1e6,
                                         0.0, 0.0, 0.0, 1e6};
  const bool box_ok = in.box != nullptr && in.box[0] > 0.0 && in.box[4] > 0.0 &&
                      in.box[8] > 0.0;
  const double *box_use = box_ok ? in.box : free_box;

  simd::LjParams params{};
  params.psi2 = psi * psi;
  params.four_u0 = 4.0 * u0;
  params.shift = cuttOffU;
  params.use_switch = 0;
  simd::CachedPairs pairs;
  simd::fill_view(pairs, R, static_cast<std::size_t>(N), box_use, opt);
  double strain[6] = {};
  *U = simd::lj_force(pairs.view, params, F, strain);
}

} // namespace rgpot
