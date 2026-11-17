#pragma once
// MIT License
// Copyright 2023--present rgpot developers

/**
 * @brief Shared cell-team pair walk, free-box fallback, and pair-force scatter
 * helpers for classical pair and EAM potentials.
 */

#include "rgpot/PerAtom.hpp"
#include "rgpot/nlist/NeighbourList.hpp"
#include "rgpot/stress.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace rgpot {

/// Return @p box when it is non-null with positive diagonal entries, else a
/// 1e6 Angstrom cube for free-boundary cell walks.
inline const double *freeBoxOrCell(const double *box) noexcept {
  static constexpr double free_box[9] = {1.0e6, 0.0, 0.0, 0.0, 1.0e6,
                                         0.0,   0.0, 0.0, 1.0e6};
  const bool box_ok =
      box != nullptr && box[0] > 0.0 && box[4] > 0.0 && box[8] > 0.0;
  return box_ok ? box : free_box;
}

/// Add a pair force fscale * d to atoms @p i and @p j, accumulate Cauchy
/// strain into @p virial when non-null, and tally per-atom energy/virial into
/// @p share when non-null.
inline void scatterPairForce(double *force, std::int32_t i, std::int32_t j,
                             double pair_e, double fscale, double dx, double dy,
                             double dz, double *virial,
                             const PerAtomOut *share) {
  const double fx = fscale * dx;
  const double fy = fscale * dy;
  const double fz = fscale * dz;
  force[3 * i] += fx;
  force[3 * i + 1] += fy;
  force[3 * i + 2] += fz;
  force[3 * j] -= fx;
  force[3 * j + 1] -= fy;
  force[3 * j + 2] -= fz;
  if (virial != nullptr) {
    accumulatePairStrain(virial, fscale, dx, dy, dz);
  }
  if (share != nullptr) {
    tallyPair(*share, static_cast<std::size_t>(i),
              static_cast<std::size_t>(j), pair_e, fscale, dx, dy, dz);
  }
}

namespace detail {

template <typename TeamWalkFn>
inline void runTeamAccumulation(int nt, long n_atoms, double *F, double &energy,
                                double *strain, const PerAtomOut *atoms,
                                TeamWalkFn &&walk) {
  const auto n_size = static_cast<std::size_t>(n_atoms);
  std::vector<double> owned(static_cast<std::size_t>(nt) * n_size * 3, 0.0);
  std::vector<double> partial(static_cast<std::size_t>(nt), 0.0);
  std::vector<double> virial(
      strain != nullptr ? static_cast<std::size_t>(nt) * 6 : 0, 0.0);
  PerAtomTeam team(atoms, n_size, nt);
  std::vector<PerAtomOut> slots(static_cast<std::size_t>(nt));
  for (int t = 0; t < nt; ++t) {
    slots[static_cast<std::size_t>(t)] = team.slot(t);
  }
  walk(owned, partial, virial, slots);
  team.reduce();
  for (int t = 0; t < nt; ++t) {
    energy += partial[static_cast<std::size_t>(t)];
    const double *local =
        owned.data() + static_cast<std::size_t>(t) * n_size * 3;
    for (std::size_t k = 0; k < n_size * 3; ++k) {
      F[k] += local[k];
    }
    if (strain != nullptr) {
      for (int c = 0; c < 6; ++c) {
        strain[c] += virial[static_cast<std::size_t>(t) * 6 +
                            static_cast<std::size_t>(c)];
      }
    }
  }
}

} // namespace detail

/// Evaluate half-cell pairs across the cell team (or directly on 1 thread) and
/// reduce thread-private force, energy, strain, and per-atom buffers.
template <typename AcceptFn>
inline void runCellTeamPairs(long n_atoms, const nlist::Neighbours &neighbours,
                             double *F, double &energy, double *strain,
                             const PerAtomOut *atoms, AcceptFn &&accept) {
  const int nt = nlist::threadsRequested();
  if (nt <= 1) {
    nlist::visitHalf(neighbours, 0, n_atoms,
                     [&](std::int32_t i, std::int32_t j, double dx, double dy,
                         double dz, double r2) {
                       accept(i, j, dx, dy, dz, r2, F, energy, strain, atoms);
                     });
    return;
  }
  const auto n_size = static_cast<std::size_t>(n_atoms);
  detail::runTeamAccumulation(
      nt, n_atoms, F, energy, strain, atoms,
      [&](std::vector<double> &owned, std::vector<double> &partial,
          std::vector<double> &virial, std::vector<PerAtomOut> &slots) {
        neighbours.forCellTeam(
            true, nt,
            [&](int t, std::int32_t i, std::int32_t j, double dx, double dy,
                double dz, double r2) {
              double *local =
                  owned.data() + static_cast<std::size_t>(t) * n_size * 3;
              double *st = strain != nullptr
                               ? virial.data() + static_cast<std::size_t>(t) * 6
                               : nullptr;
              accept(i, j, dx, dy, dz, r2, local,
                     partial[static_cast<std::size_t>(t)], st,
                     atoms != nullptr ? &slots[static_cast<std::size_t>(t)]
                                      : nullptr);
            });
      });
}

/// Evaluate directed half-pairs partitioned by atom slice (or directly on 1
/// thread) and reduce thread-private force, energy, and per-atom buffers.
template <typename AcceptFn>
inline void runDirectedTeamPairs(long n_atoms,
                                 const nlist::Neighbours &neighbours, double *F,
                                 double &energy, const PerAtomOut *atoms,
                                 AcceptFn &&accept) {
  const int nt = nlist::threadsRequested();
  if (nt <= 1) {
    nlist::visitHalf(neighbours, 0, n_atoms,
                     [&](std::int32_t i, std::int32_t j, double dx, double dy,
                         double dz, double r2) {
                       accept(i, j, dx, dy, dz, r2, F, energy, atoms);
                     });
    return;
  }
  const auto n_size = static_cast<std::size_t>(n_atoms);
  detail::runTeamAccumulation(
      nt, n_atoms, F, energy, nullptr, atoms,
      [&](std::vector<double> &owned, std::vector<double> &partial,
          std::vector<double> &, std::vector<PerAtomOut> &slots) {
        nlist::forEachSlice(nt, [&](int t) {
          const auto range = nlist::partition(n_atoms, t, nt);
          double *local =
              owned.data() + static_cast<std::size_t>(t) * n_size * 3;
          double local_e = 0.0;
          nlist::visitDirected(
              neighbours, range.begin, range.end,
              [&](std::int32_t i, std::int32_t j, double dx, double dy,
                  double dz, double r2) {
                accept(i, j, dx, dy, dz, r2, local, local_e,
                       atoms != nullptr ? &slots[static_cast<std::size_t>(t)]
                                        : nullptr);
              });
          partial[static_cast<std::size_t>(t)] = local_e;
        });
      });
}

} // namespace rgpot
