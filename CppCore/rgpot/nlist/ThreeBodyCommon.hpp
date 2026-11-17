#pragma once
// MIT License
// Copyright 2023--present rgpot developers

/**
 * @brief Shared neighbour-update, bond-collection, and slice-reduction
 * helpers for three-body potentials (Stillinger-Weber and Tersoff).
 */

#include "rgpot/nlist/HalfShell.hpp"
#include "rgpot/nlist/NeighbourList.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace rgpot::nlist {

/// Update @p cells for a three-body cutoff and synchronize @p cache when it
/// carries a positive skin and the cell list rebuilt.
inline void updateThreeBodyCells(Neighbours &cells, const double *positions,
                                 std::size_t n_atoms, const double *box,
                                 double cutoff, Mode mode,
                                 NeighborList *cache) {
  const double skin = (cache != nullptr && cache->skin > 0.0)
                          ? cache->skin
                          : (mode == Mode::Stream ? 0.0 : cutoff / 5.0);
  const auto before = cells.builds();
  updateCells(cells, positions, n_atoms, box, cutoff, skin, Image::Round, mode,
              {true, true, true});
  if (cache != nullptr && cache->skin > 0.0 &&
      (cells.builds() != before || !cache->built)) {
    cache->ref.assign(positions, positions + 3 * n_atoms);
    std::memcpy(cache->box, box, 9 * sizeof(double));
    cache->n = n_atoms;
    cache->cutoff = cutoff;
    cache->skin_used = cache->skin;
    cache->built = true;
  }
}

/// Collect and sort by partner index all neighbour bonds of atom @p i.
template <typename Bond, typename AcceptFn>
inline void collectAtomBonds(const Neighbours &cells, int i,
                             std::vector<Bond> &nb, AcceptFn &&accept) {
  nb.clear();
  if (cells.streaming()) {
    cells.forAtom(static_cast<std::int32_t>(i), accept);
  } else {
    visitDirected(cells, i, static_cast<std::int64_t>(i) + 1, accept);
  }
  std::sort(nb.begin(), nb.end(),
            [](const Bond &a, const Bond &b) { return a.j < b.j; });
}

/// Clamp cosine of two bond vectors to [-1, 1].
inline double clampCosine(double c) noexcept {
  if (c > 1.0) {
    return 1.0;
  }
  if (c < -1.0) {
    return -1.0;
  }
  return c;
}

/// Run a per-atom slice body either directly on one thread or across an OpenMP
/// slice team with private force, energy, and optional 6-component strain
/// buffers reduced in slice order.
template <typename SliceBody>
inline void runSliceTeam(int n, double *acc, double *energy, double *strain,
                         SliceBody &&body) {
  const int nt = threadsRequested();
  if (nt == 1) {
    body(0, n, acc, *energy, strain);
    return;
  }
  const std::size_t n3 = static_cast<std::size_t>(n) * 3;
  std::vector<double> owned(static_cast<std::size_t>(nt) * n3, 0.0);
  std::vector<double> energies(static_cast<std::size_t>(nt), 0.0);
  std::vector<double> vir(
      strain == nullptr ? 0 : static_cast<std::size_t>(nt) * 6, 0.0);
  forEachSlice(nt, [&](int t) {
    const auto range = partition(n, t, nt);
    double *local = owned.data() + static_cast<std::size_t>(t) * n3;
    double local_e = 0.0;
    double *local_strain =
        strain == nullptr ? nullptr
                          : vir.data() + static_cast<std::size_t>(t) * 6;
    body(static_cast<int>(range.begin), static_cast<int>(range.end), local,
         local_e, local_strain);
    energies[static_cast<std::size_t>(t)] = local_e;
  });
  for (int t = 0; t < nt; ++t) {
    *energy += energies[static_cast<std::size_t>(t)];
    const double *local = owned.data() + static_cast<std::size_t>(t) * n3;
    for (std::size_t k = 0; k < n3; ++k) {
      acc[k] += local[k];
    }
    if (strain != nullptr) {
      for (int k = 0; k < 6; ++k) {
        strain[k] +=
            vir[static_cast<std::size_t>(t) * 6 + static_cast<std::size_t>(k)];
      }
    }
  }
}

} // namespace rgpot::nlist
