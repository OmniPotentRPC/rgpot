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
  const auto observe = [&](double fscale, double dx, double dy, double dz) {
    accumulatePairStrain(strain, fscale, dx, dy, dz);
  };
  *U = 0.0;
  for (long k = 0; k < 3 * N; k++) {
    F[k] = 0.0;
  }
  if (N < 2 || cuttOffR <= 0.0) {
    return;
  }

  nlist::CachedPairList::Options opt;
  opt.cutoff = cuttOffR;

  const double twoDeA = 2.0 * De * a;
  const double depth = De;
  const double range = a;
  const double rEq = re;
  const double shiftU = energyCutoff;
  auto &pool = nlist::PairListCache::global();
  if (m_config.switch_width != 0.0) {
    // Unshifted pair term times the C^2 switch S; fscale = -(V S)' / r.
    const QuinticSwitch sw = m_switch;
    *U = pool.accumulate(
        R, static_cast<std::size_t>(N), box, opt, F,
        [=](int32_t, int32_t, double r2) noexcept {
          const double r = std::sqrt(r2);
          const double d = 1.0 - std::exp(-range * (r - rEq));
          const double v = depth * d * d - depth;
          const auto s = sw(r);
          return nlist::PairTerm{
              v * s.s, (twoDeA * d * (d - 1.0) * s.s - v * s.dsdr) / r};
        },
        observe);
    publishCauchyStress(out, strain, volume);
    return;
  }
  *U = pool.accumulate(
      R, static_cast<std::size_t>(N), box, opt, F,
      [=](int32_t, int32_t, double r2) noexcept {
        const double r = std::sqrt(r2);
        const double d = 1.0 - std::exp(-range * (r - rEq));
        // -dU/dr / r: the force on i is fscale * (r_i - r_j).
        return nlist::PairTerm{depth * d * d - depth - shiftU,
                               twoDeA * d * (d - 1.0) / r};
      },
      observe);
  publishCauchyStress(out, strain, volume);
}

} // namespace rgpot
