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
#include "rgpot/simd/exp.hpp"
#include "rgpot/stress.hpp"

#include <cmath>

namespace rgpot {
namespace {

double roundVisit(double t) {
  if (!std::isfinite(t)) {
    return 0.0;
  }
  const double shifted = t + std::copysign(0.5, t);
  if (!std::isfinite(shifted) || shifted >= 9223372036854775808.0 ||
      shifted <= -9223372036854775808.0) {
    return 0.0;
  }
  return static_cast<double>(static_cast<long long>(shifted));
}

void separate(const simd::PairView &v, std::size_t i, std::size_t j, double &dx,
              double &dy, double &dz) {
  dx = v.R[3 * i] - v.R[3 * j];
  dy = v.R[3 * i + 1] - v.R[3 * j + 1];
  dz = v.R[3 * i + 2] - v.R[3 * j + 2];
  if (v.fold == 1) {
    dx -= v.w[0] * roundVisit(dx * v.inv[0]);
    dy -= v.w[1] * roundVisit(dy * v.inv[1]);
    dz -= v.w[2] * roundVisit(dz * v.inv[2]);
  } else if (v.fold == 3 && v.general != nullptr) {
    static_cast<const nlist::MinimumImage *>(v.general)->fold(dx, dy, dz);
  }
}

} // namespace

/**
 * @class MorsePot
 * @details
 *
 * Pairwise interactions within the cutoff radius, minimum image convention
 * on an orthogonal box. Pairs come from the shared neighbour layer, so
 * repeated evaluations on nearby geometries reuse a Verlet list.
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

  simd::PairSpec opt;
  opt.cutoff = cuttOffR;
  const auto n_atoms = static_cast<std::size_t>(N);
  neighbours_.update(R, n_atoms, box,
                     nlist::Spec{opt.cutoff, opt.skin, opt.periodic,
                                 nlist::Image::Round, mode_});
  const bool reuse_pairs = have_cache_ && !neighbours_.streaming() &&
                           neighbours_.builds() == list_stamp_ &&
                           cached_.view.n == n_atoms;
  if (reuse_pairs) {
    cached_.view.R = R;
  } else {
    simd::fill_view(cached_, R, n_atoms, box, opt, mode_, &neighbours_);
    have_cache_ = !neighbours_.streaming();
    list_stamp_ = neighbours_.builds();
  }

  const simd::PairView &view = cached_.view;
  const double cutoff2 = cuttOffR * cuttOffR;
  const double twoDeA = 2.0 * De * a;
  const bool switched = m_config.switch_width != 0.0;
  double energy = 0.0;
  for (std::size_t i = 0; i < view.n; ++i) {
    double fxi = 0.0;
    double fyi = 0.0;
    double fzi = 0.0;
    for (std::int32_t p = view.rows[i]; p < view.rows[i + 1]; ++p) {
      const auto j =
          static_cast<std::size_t>(view.nbr[static_cast<std::size_t>(p)]);
      double dx = 0.0;
      double dy = 0.0;
      double dz = 0.0;
      separate(view, i, j, dx, dy, dz);
      const double r2 = dx * dx + dy * dy + dz * dz;
      if (!(r2 > 0.0) || r2 > cutoff2) {
        continue;
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
      fxi += fscale * dx;
      fyi += fscale * dy;
      fzi += fscale * dz;
      F[3 * j] -= fscale * dx;
      F[3 * j + 1] -= fscale * dy;
      F[3 * j + 2] -= fscale * dz;
      accumulatePairStrain(strain, fscale, dx, dy, dz);
    }
    F[3 * i] += fxi;
    F[3 * i + 1] += fyi;
    F[3 * i + 2] += fzi;
  }
  *U = energy;
  publishCauchyStress(out, strain, volume);
}

} // namespace rgpot
