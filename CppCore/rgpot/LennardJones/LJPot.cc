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
 * on an orthogonal box. Pairs come from the shared
 * ``rgpot::nlist::PairListCache`` (the eOn PairListCache design): repeated
 * evaluations on nearby geometries reuse a Verlet-skin cached candidate
 * list, and one-shot evaluations run a single fused scan identical in pair
 * content to the historical per-call double loop.
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
  const double volume = cellVolume(box);
  double strainAcc[6] = {};
  if (N < 2 || cuttOffR <= 0.0) {
    publishCauchyStress(out, strainAcc, volume);
    return;
  }

  nlist::CachedPairList::Options opt;
  opt.cutoff = cuttOffR;

  const double psi2 = psi * psi;
  double energyAcc = 0.0;
  nlist::PairListCache::global().evaluate(
      R, static_cast<std::size_t>(N), box, opt,
      [&](int32_t i, int32_t j, double dx, double dy, double dz, double r2) {
        const double invR2 = 1.0 / r2;
        const double sr2 = psi2 * invR2;
        const double a = sr2 * sr2 * sr2; // (psi/r)^6 without pow()
        const double b = 4.0 * u0 * a;
        energyAcc += b * (a - 1.0) - cuttOffU;

        // -dU/dr / r along d = r_i - r_j, matching the historical loop's
        // sign convention.
        const double fscale = 6.0 * b * invR2 * (2.0 * a - 1.0);
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
