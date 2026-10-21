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
#include "rgpot/stress.hpp"
#include "rgpot/types/AtomMatrix.hpp"
using rgpot::types::AtomMatrix;

namespace rgpot {

/**
 * @class LJPot
 * @details
 *
 * Pairwise interactions within the cutoff radius, minimum image convention
 * on an orthogonal box. Pairs come from the shared
 * ``rgpot::nlist::PairListCache`` (the eOn PairListCache design): repeated
 * evaluations on nearby geometries reuse a Verlet-skin cached candidate
 * list, and one-shot evaluations run a single fused scan identical in pair
 * content to the historical per-call double loop. An orthorhombic cell
 * takes those candidates from the fast list (linkcell when the cell is
 * several cutoffs wide, otherwise every unordered pair) and folds them
 * with minimage. The pair term is the Highway kernel selected for this
 * process. A nonzero switch width uses that kernel's quintic switch.
 * Cauchy stress is published from the kernel's strain accumulator.
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
  double strain[6] = {};
  const double volume = cellVolume(box);
  *U = 0;
  for (long i = 0; i < N; i++) {
    F[3 * i] = 0;
    F[3 * i + 1] = 0;
    F[3 * i + 2] = 0;
  }
  if (N < 2 || cuttOffR <= 0.0) {
    publishCauchyStress(out, strain, 0.0);
    return;
  }

  nlist::CachedPairList::Options opt;
  opt.cutoff = cuttOffR;

  simd::LjParams params{};
  params.psi2 = psi * psi;
  params.four_u0 = 4.0 * u0;
  params.shift = cuttOffU;
  params.use_switch = 0;
  if (m_config.switch_width != 0.0) {
    params.use_switch = 1;
    params.r_on = m_switch.r_on;
    params.inv_width = m_switch.inv_width;
  }
  simd::CachedPairs pairs;
  simd::fill_view(pairs, R, static_cast<std::size_t>(N), box, opt);
  *U = simd::lj_force(pairs.view, params, F, strain);
  publishCauchyStress(out, strain, volume);
}

} // namespace rgpot
