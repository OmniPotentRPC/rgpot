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
#include "rgpot/simd/from_nlist.hpp"
#include "rgpot/stress.hpp"
#include "rgpot/types/AtomMatrix.hpp"
using rgpot::types::AtomMatrix;

namespace rgpot {

/**
 * @class LJPot
 * @details
 *
 * Pairwise interactions within the cutoff radius, minimum image convention
 * with the full cell. Pairs come from the shared
 * ``rgpot::nlist::PairListCache`` (the eOn PairListCache design): repeated
 * evaluations on nearby geometries reuse a Verlet-skin cached candidate
 * list, and one-shot evaluations run a single fused scan identical in pair
 * content to the historical per-call double loop.
 *
 * With ``LJConfig::switch_width`` set, each pair term is the unshifted
 * 12-6 energy times the C^2 quintic switch of PairSwitch.hpp, so energy,
 * force and curvature reach zero at the cutoff; otherwise the energy is
 * shifted to zero there and the force jumps.
 *
 */
void LJPot::forceImpl(const ForceInput &in, ForceOut *out) const {
  const long N = in.nAtoms;
  const double *R = in.pos;
  const double *box = in.box;
  double *F = out->F;
  double *U = &out->energy;
  double strain[6] = {};
  const double volume = cellVolume(in.box);
  publishCauchyStress(out, strain, volume);
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

  simd::LjParams params;
  params.psi2 = psi * psi;
  params.four_u0 = 4.0 * u0;
  params.shift = cuttOffU;
  params.use_switch = m_config.switch_width != 0.0 ? 1 : 0;
  params.r_on = m_switch.r_on;
  params.inv_width = m_switch.inv_width;
  auto &pool = nlist::PairListCache::global();
  pool.withList(R, static_cast<std::size_t>(N), box, opt,
                [&](const nlist::CachedPairList &list) {
                  const simd::PairView view =
                      simd::view_from(list.evalView(), R);
                  *U = simd::lj_force(view, params, F, strain);
                });
  publishCauchyStress(out, strain, volume);
}

} // namespace rgpot
