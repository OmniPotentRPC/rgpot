// MIT License
// Copyright 2023--present rgpot developers

/**
 * @brief Implementation of the Lennard-Jones potential methods.
 *
 * This file contains the implementation of the force and energy
 * calculation for the Lennard-Jones potential, including periodic
 * boundary condition handling.
 */

// clang-format off
#include <cmath>
#include <limits>
// clang-format on

#include "rgpot/LennardJones/LJPot.hpp"
#include "rgpot/nlist/PairListCache.hpp"
#include "rgpot/simd/api.hpp"
#include "rgpot/simd/from_cache.hpp"
#include "rgpot/types/AtomMatrix.hpp"
using rgpot::types::AtomMatrix;

namespace rgpot {

/**
 * @class LJPot
 * @details
 *
 * Pairwise interactions within the cutoff radius, minimum image convention
 * on an orthogonal box. Pairs come from the shared
 * ``rgpot::nlist::PairListCache``: an orthorhombic cell takes candidates
 * from the fast list (linkcell when the cell is several cutoffs wide,
 * otherwise every unordered pair) and folds them with minimage. The pair
 * term is the Highway kernel selected for this process.
 *
 * @warning The box is assumed to be orthogonal.
 *
 */
void LJPot::forceImpl(const ForceInput &in, ForceOut *out) const {
  const long N = in.nAtoms;
  const double *R = in.pos;
  const double *box = in.box;
  double *F = out->F;
  double *U = &out->energy;
  *U = 0;
  for (long i = 0; i < N; i++) {
    F[3 * i] = 0;
    F[3 * i + 1] = 0;
    F[3 * i + 2] = 0;
  }
  if (N < 2 || cuttOffR <= 0.0) {
    return;
  }

  nlist::CachedPairList::Options opt;
  opt.cutoff = cuttOffR;

  simd::LjParams params{};
  params.psi2 = psi * psi;
  params.four_u0 = 4.0 * u0;
  params.shift = cuttOffU;
  params.use_switch = 0;
  simd::CachedPairs pairs;
  simd::fill_view(pairs, R, static_cast<std::size_t>(N), box, opt);
  double strain[6] = {};
  *U = simd::lj_force(pairs.view, params, F, strain);
}

} // namespace rgpot
