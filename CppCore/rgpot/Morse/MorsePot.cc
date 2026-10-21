// MIT License
// Copyright 2023--present rgpot developers

/**
 * @brief Implementation of the Morse potential methods.
 *
 * Ported from eOn (https://github.com/TheochemUI/eOn,
 * client/potentials/Morse), BSD-3-Clause licensed, copyright the eOn
 * Development Team.
 */

// clang-format off
#include <cmath>
// clang-format on

#include "rgpot/Morse/MorsePot.hpp"
#include "rgpot/nlist/PairListCache.hpp"
#include "rgpot/simd/api.hpp"
#include "rgpot/simd/from_cache.hpp"

namespace rgpot {

/**
 * @class MorsePot
 * @details
 *
 * Pairwise interactions within the cutoff radius, minimum image convention
 * on an orthogonal box. Pairs come from the shared
 * ``rgpot::nlist::PairListCache``: an orthorhombic cell takes candidates
 * from the fast list (linkcell when the cell is several cutoffs wide,
 * otherwise every unordered pair) and folds them with minimage.
 *
 * @warning The box is assumed to be orthogonal.
 */
void MorsePot::forceImpl(const ForceInput &in, ForceOut *out) const {
  const auto N = static_cast<long>(in.nAtoms);
  const double *R = in.pos;
  const double *box = in.box;
  double *F = out->F;
  double *U = &out->energy;
  *U = 0.0;
  for (long k = 0; k < 3 * N; k++) {
    F[k] = 0.0;
  }
  if (N < 2 || cuttOffR <= 0.0) {
    return;
  }

  nlist::CachedPairList::Options opt;
  opt.cutoff = cuttOffR;

  simd::MorseParams params{};
  params.depth = De;
  params.range = a;
  params.r_eq = re;
  params.two_de_a = 2.0 * De * a;
  params.shift = energyCutoff;
  params.use_switch = 0;
  simd::CachedPairs pairs;
  simd::fill_view(pairs, R, static_cast<std::size_t>(N), box, opt);
  double strain[6] = {};
  *U = simd::morse_force(pairs.view, params, F, strain);
}

} // namespace rgpot
