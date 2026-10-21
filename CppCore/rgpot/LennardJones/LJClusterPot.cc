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
#include <algorithm>
#include <cmath>
// clang-format on

#include "rgpot/LennardJones/LJClusterKernel.hpp"
#include "rgpot/LennardJones/LJClusterPot.hpp"
#include "rgpot/nlist/PairListCache.hpp"

namespace rgpot {

/**
 * @class LJClusterPot
 * @details
 *
 * Every pair within the cutoff contributes, with vectors taken straight
 * from the coordinates: ``opt.periodic`` is off in all three directions,
 * so the pair list never folds a separation into the cell. A cluster
 * whose bounding box fits inside the cutoff walks ``i < j`` in SoA form,
 * AVX2 or NEON on the inner loop. A larger open system goes through the
 * pair cache. A degenerate input cell (any non-positive diagonal entry)
 * is replaced by a 1e6 Angstrom cube.
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

  double minx = R[0];
  double maxx = minx;
  double miny = R[1];
  double maxy = miny;
  double minz = R[2];
  double maxz = minz;
  for (long i = 1; i < N; ++i) {
    const double x = R[3 * i];
    const double y = R[3 * i + 1];
    const double z = R[3 * i + 2];
    minx = std::min(minx, x);
    maxx = std::max(maxx, x);
    miny = std::min(miny, y);
    maxy = std::max(maxy, y);
    minz = std::min(minz, z);
    maxz = std::max(maxz, z);
  }
  const double ex = maxx - minx;
  const double ey = maxy - miny;
  const double ez = maxz - minz;
  const double cutoff = opt.cutoff;
  // Every pair is at most the box diagonal. When that diagonal sits on
  // the cutoff, the direct walk and the cutoff test agree on the set.
  if ((ex * ex + ey * ey) + ez * ez <= cutoff * cutoff) {
    double energyAcc = 0.0;
    lj_cluster_direct(R, static_cast<std::size_t>(N), cutoff, u0, psi, cuttOffU, F,
                      &energyAcc);
    *U = energyAcc;
    return;
  }

  const double psi2 = psi * psi;
  double energyAcc = 0.0;
  nlist::PairListCache::global().evaluate(
      R, static_cast<std::size_t>(N), box_use, opt,
      [&](int32_t i, int32_t j, double dx, double dy, double dz, double r2) {
        const double invR2 = 1.0 / r2;
        const double sr2 = psi2 * invR2;
        const double a = sr2 * sr2 * sr2; // (psi/r)^6 without pow()
        const double b = 4.0 * u0 * a;
        energyAcc += b * (a - 1.0) - cuttOffU;

        // -dU/dr / r along d = r_i - r_j.
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
      });
  *U = energyAcc;
}

} // namespace rgpot
