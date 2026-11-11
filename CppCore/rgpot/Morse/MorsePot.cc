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
#include <array>
#include <cmath>
#include <vector>
// clang-format on

#include "rgpot/Morse/MorsePot.hpp"
#include "rgpot/nlist/NeighbourList.hpp"
#include "rgpot/simd/exp.hpp"
#include "rgpot/stress.hpp"

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace rgpot {

/**
 * @class MorsePot
 * @details
 *
 * Pairwise interactions within the cutoff radius. The minimum image
 * comes from the shared neighbour layer, on an orthogonal or sheared
 * box. The pair term runs during the cell walk. A stream does not copy
 * a half-list. A stored list keeps candidates out to the cutoff plus a
 * skin of one fifth of the cutoff, and a later call inside half that
 * skin reuses them. The distance filter uses the positions of this call.
 *
 * With ``MorseConfig::switch_width`` set, each pair term is the unshifted
 * Morse energy times the C^2 quintic switch of PairSwitch.hpp; otherwise
 * the energy is shifted to zero at the cutoff and the force jumps there.
 * The pair term is ``simd::exp_fixed``, so a per-pair accumulation and this
 * row return the same bits. Cauchy stress is the derivative of that energy.
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

  const auto n_atoms = static_cast<std::size_t>(N);
  // The cell walk is slower than the stored list from about 20k atoms
  // upward (measured at 108k and 1M). Auto stores there.
  nlist::Mode mode = mode_;
  if (mode == nlist::Mode::Auto && n_atoms >= 20000) {
    mode = nlist::Mode::Stored;
  }
  const double skin = mode == nlist::Mode::Stream ? 0.0 : neighbourSkin(cuttOffR);
  const std::array<bool, 3> periodic{{true, true, true}};
  neighbours_.update(R, n_atoms, box, cuttOffR, skin, nlist::Image::Round,
                     mode, periodic);
  const double cutoff2 = cuttOffR * cuttOffR;
  const double twoDeA = 2.0 * De * a;
  const bool switched = m_config.switch_width != 0.0;
  const int nt = nlist::threadsRequested();
  auto accept = [&](std::int32_t i, std::int32_t j, double dx, double dy, double dz,
                    double r2, double *force, double &energy, double *virial) {
    if (!(i < j) || !(r2 > 0.0) || r2 > cutoff2) {
      return;
    }
    const double r = std::sqrt(r2);
    const double d = 1.0 - simd::exp_fixed(-a * (r - re));
    const double bare = De * d * d - De;
    double pair_e = 0.0;
    double fscale = 0.0;
    if (switched) {
      const auto s = m_switch(r);
      pair_e = bare * s.s;
      fscale = (twoDeA * d * (d - 1.0) * s.s - bare * s.dsdr) / r;
    } else {
      pair_e = bare - energyCutoff;
      fscale = twoDeA * d * (d - 1.0) / r;
    }
    energy += pair_e;
    const double fx = fscale * dx;
    const double fy = fscale * dy;
    const double fz = fscale * dz;
    force[3 * i] += fx;
    force[3 * i + 1] += fy;
    force[3 * i + 2] += fz;
    force[3 * j] -= fx;
    force[3 * j + 1] -= fy;
    force[3 * j + 2] -= fz;
    accumulatePairStrain(virial, fscale, dx, dy, dz);
  };
  double energy = 0.0;
  if (nt <= 1) {
    neighbours_.forHalf(0, N,
                     [&](std::int32_t i, std::int32_t j, double dx, double dy, double dz,
                         double r2) { accept(i, j, dx, dy, dz, r2, F, energy, strain); });
  } else {
    std::vector<double> owned(static_cast<std::size_t>(nt) * n_atoms * 3, 0.0);
    std::vector<double> partial(static_cast<std::size_t>(nt), 0.0);
    std::vector<double> virial(static_cast<std::size_t>(nt) * 6, 0.0);
    neighbours_.forCellTeam(
        true, nt, [&](int t, std::int32_t i, std::int32_t j, double dx, double dy, double dz,
                      double r2) {
          double *local = owned.data() + static_cast<std::size_t>(t) * n_atoms * 3;
          double *st = virial.data() + static_cast<std::size_t>(t) * 6;
          accept(i, j, dx, dy, dz, r2, local, partial[static_cast<std::size_t>(t)], st);
        });
    for (int t = 0; t < nt; ++t) {
      energy += partial[static_cast<std::size_t>(t)];
      const double *local = owned.data() + static_cast<std::size_t>(t) * n_atoms * 3;
      for (std::size_t k = 0; k < n_atoms * 3; ++k) {
        F[k] += local[k];
      }
      for (int c = 0; c < 6; ++c) {
        strain[c] += virial[static_cast<std::size_t>(t) * 6 + static_cast<std::size_t>(c)];
      }
    }
  }
  *U = energy;
  publishCauchyStress(out, strain, volume);
}

} // namespace rgpot
