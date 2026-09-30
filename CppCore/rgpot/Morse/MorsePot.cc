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
  const double volume = cellVolume(box);
  double strainAcc[6] = {};
  if (N < 2 || cuttOffR <= 0.0) {
    publishCauchyStress(out, strainAcc, volume);
    return;
  }

  nlist::CachedPairList::Options opt;
  opt.cutoff = cuttOffR;

  const double twoDeA = 2.0 * De * a;
  double energyAcc = 0.0;
  nlist::PairListCache::global().evaluate(
      R, static_cast<std::size_t>(N), box, opt,
      [&](int32_t i, int32_t j, double dx, double dy, double dz, double r2) {
        const double r = std::sqrt(r2);
        const double d = 1.0 - std::exp(-a * (r - re));
        energyAcc += De * d * d - De - energyCutoff;

        // -dU/dr / r along d = r_i - r_j.
        const double fscale = twoDeA * d * (d - 1.0) / r;
        const double fx = fscale * dx;
        const double fy = fscale * dy;
        const double fz = fscale * dz;

        F[3 * i] += fx;
        F[3 * i + 1] += fy;
        F[3 * i + 2] += fz;
        F[3 * j] -= fx;
        F[3 * j + 1] -= fy;
        F[3 * j + 2] -= fz;
        accumulatePairStrain(strainAcc, fscale, dx, dy, dz);
      });
  *U = energyAcc;
  publishCauchyStress(out, strainAcc, volume);
}

} // namespace rgpot
