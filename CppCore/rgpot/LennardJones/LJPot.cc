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
#include <vector>
// clang-format on

#if defined(_OPENMP)
#include <omp.h>
#endif

#include "rgpot/LennardJones/LJPot.hpp"
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
 * on an orthogonal box. Pairs come from the shared neighbour layer: a
 * cell list, a Verlet skin, and either a stored list or a streaming walk.
 * Repeated evaluations on nearby geometries reuse that list. The pair
 * term on one thread is the kernel selected for this process. Several
 * threads split the atoms, keep a private force buffer, and add those
 * buffers in thread order. A nonzero switch width uses the quintic switch.
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
    publishCauchyStress(out, strain, volume);
    return;
  }

  simd::PairSpec opt;
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
  const int nt = nlist::threadsRequested();
  if (nt <= 1) {
    *U = simd::lj_force(view, params, F, strain);
  } else {
    const auto n = static_cast<std::size_t>(N);
    std::vector<double> owned(static_cast<std::size_t>(nt) * n * 3, 0.0);
    std::vector<double> energy(static_cast<std::size_t>(nt), 0.0);
    std::vector<double> virial(static_cast<std::size_t>(nt) * 6, 0.0);
    std::vector<nlist::Range> parts(static_cast<std::size_t>(nt));
    for (int t = 0; t < nt; ++t) {
      parts[static_cast<std::size_t>(t)] = nlist::partition(N, t, nt);
    }
#if defined(_OPENMP)
#pragma omp parallel num_threads(nt)
#endif
    {
#if defined(_OPENMP)
      const int t = omp_get_thread_num();
#else
      const int t = 0;
#endif
      const auto range = parts[static_cast<std::size_t>(t)];
      double *local = owned.data() + static_cast<std::size_t>(t) * n * 3;
      double *st = virial.data() + static_cast<std::size_t>(t) * 6;
      energy[static_cast<std::size_t>(t)] = simd::lj_force_span(
          view, params, local, st, static_cast<std::size_t>(range.begin),
          static_cast<std::size_t>(range.end));
    }
    for (int t = 0; t < nt; ++t) {
      *U += energy[static_cast<std::size_t>(t)];
      const double *local = owned.data() + static_cast<std::size_t>(t) * n * 3;
      for (std::size_t k = 0; k < n * 3; ++k) {
        F[k] += local[k];
      }
      for (int c = 0; c < 6; ++c) {
        strain[c] += virial[static_cast<std::size_t>(t) * 6 + static_cast<std::size_t>(c)];
      }
    }
  }
  publishCauchyStress(out, strain, volume);
}

} // namespace rgpot
