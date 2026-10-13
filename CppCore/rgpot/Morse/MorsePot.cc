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
#include "rgpot/simd/from_nlist.hpp"
#include "rgpot/stress.hpp"

namespace rgpot {

/**
 * @class MorsePot
 * @details
 *
 * Pairwise interactions within the cutoff radius, minimum image convention
 * on an orthogonal box. Pairs come from the shared
 * ``rgpot::nlist::PairListCache``, so repeated evaluations on nearby
 * geometries (NEB images, dimer rotations) reuse a Verlet-skin cached
 * candidate list.
 *
 * With ``MorseConfig::switch_width`` set, each pair term is the unshifted
 * Morse energy times the C^2 quintic switch of PairSwitch.hpp; otherwise
 * the energy is shifted to zero at the cutoff and the force jumps there.
 *
 * @warning The box is assumed to be orthogonal.
 */
void MorsePot::forceImpl(const ForceInput &in, ForceOut *out) const {
  const auto N = static_cast<long>(in.nAtoms);
  const double *R = in.pos;
  const double *box = in.box;
  double *F = out->F;
  double *U = &out->energy;
  double strain[6] = {};
  const double volume = cellVolume(in.box);
  publishCauchyStress(out, strain, volume);
  *U = 0.0;
  for (long k = 0; k < 3 * N; k++) {
    F[k] = 0.0;
  }
  if (N < 2 || cuttOffR <= 0.0) {
    return;
  }

  nlist::CachedPairList::Options opt;
  opt.cutoff = cuttOffR;

  simd::MorseParams params;
  params.depth = De;
  params.range = a;
  params.r_eq = re;
  params.two_de_a = 2.0 * De * a;
  params.shift = energyCutoff;
  params.use_switch = m_config.switch_width != 0.0 ? 1 : 0;
  params.r_on = m_switch.r_on;
  params.inv_width = m_switch.inv_width;
  auto &pool = nlist::PairListCache::global();
  pool.withList(R, static_cast<std::size_t>(N), box, opt,
                [&](const nlist::CachedPairList &list) {
                  const simd::PairView view =
                      simd::view_from(list.evalView(), R);
                  *U = simd::morse_force(view, params, F, strain);
                });
  publishCauchyStress(out, strain, volume);
}

} // namespace rgpot
