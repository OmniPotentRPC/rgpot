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
  const auto observe = [&](double fscale, double dx, double dy, double dz) {
    accumulatePairStrain(strain, fscale, dx, dy, dz);
  };
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

  const double psi2 = psi * psi;
  const double fourU0 = 4.0 * u0;
  const double shiftU = cuttOffU;
  auto &pool = nlist::PairListCache::global();
  if (m_config.switch_width != 0.0) {
    // Unshifted pair term times the C^2 switch S; fscale = -(V S)' / r.
    const QuinticSwitch sw = m_switch;
    *U = pool.accumulate(
        R, static_cast<std::size_t>(N), box, opt, F,
        [=](int32_t, int32_t, double r2) noexcept {
          const double invR2 = 1.0 / r2;
          const double sr2 = psi2 * invR2;
          const double a = sr2 * sr2 * sr2;
          const double b = fourU0 * a;
          const double v = b * (a - 1.0);
          const double r = std::sqrt(r2);
          const auto s = sw(r);
          return nlist::PairTerm{v * s.s,
                                 6.0 * b * invR2 * (2.0 * a - 1.0) * s.s -
                                     v * s.dsdr / r};
        },
        observe);
    publishCauchyStress(out, strain, volume);
    return;
  }
  *U = pool.accumulate(
      R, static_cast<std::size_t>(N), box, opt, F,
      [=](int32_t, int32_t, double r2) noexcept {
        const double invR2 = 1.0 / r2;
        const double sr2 = psi2 * invR2;
        const double a = sr2 * sr2 * sr2; // (psi/r)^6 without pow()
        const double b = fourU0 * a;
        // -dU/dr / r: the force on i is fscale * (r_i - r_j).
        return nlist::PairTerm{b * (a - 1.0) - shiftU,
                               6.0 * b * invR2 * (2.0 * a - 1.0)};
      },
      observe);
  publishCauchyStress(out, strain, volume);
  return;
}

} // namespace rgpot
