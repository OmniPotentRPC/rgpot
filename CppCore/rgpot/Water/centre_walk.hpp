#pragma once
// MIT License
// Copyright 2023--present rgpot developers
//
// Molecule and atom pairs for the water kernels. The search is the shared
// neighbour walk. The pair function still applies its own cutoff, so the
// walk is a hair wider than that cutoff and does not drop a boundary pair.
//
// `fn(i, j, shift)`. In a cell at least twice the cutoff wide each pair
// comes once and `shift` is null: the kernel folds its own minimum image.
// In a thinner cell a pair comes once per image inside the search, a
// molecule may pair with its own images, and `shift` is the Cartesian
// vector added to centre j to reach that image.

#include "rgpot/nlist/NeighbourList.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <vector>

namespace forcefields {

template <typename Fn>
void walkCentres(const double *centres, int n, const double *period, double cutoff, Fn &&fn) {
  if (n < 1 || !(cutoff > 0.0) || centres == nullptr || period == nullptr) {
    return;
  }
  double minc[3] = {centres[0], centres[1], centres[2]};
  double maxc[3] = {minc[0], minc[1], minc[2]};
  for (int i = 1; i < n; ++i) {
    for (int a = 0; a < 3; ++a) {
      const double x = centres[static_cast<std::size_t>(3 * i + a)];
      if (x < minc[a]) {
        minc[a] = x;
      }
      if (x > maxc[a]) {
        maxc[a] = x;
      }
    }
  }
  std::array<bool, 3> periodic{{false, false, false}};
  double box[9] = {};
  std::vector<double> wrapped(static_cast<std::size_t>(n) * 3);
  for (int a = 0; a < 3; ++a) {
    const double length = period[a];
    if (std::isfinite(length) && length > 0.0) {
      periodic[static_cast<std::size_t>(a)] = true;
      box[a * 4] = length;
      for (int i = 0; i < n; ++i) {
        double t = centres[static_cast<std::size_t>(3 * i + a)] / length;
        t -= std::floor(t);
        wrapped[static_cast<std::size_t>(3 * i + a)] = t * length;
      }
    } else {
      const double pad = cutoff + 1.0;
      const double origin = minc[a] - pad;
      const double span = maxc[a] - minc[a];
      box[a * 4] = (span > 0.0 ? span : 0.0) + 2.0 * pad;
      for (int i = 0; i < n; ++i) {
        wrapped[static_cast<std::size_t>(3 * i + a)] =
            centres[static_cast<std::size_t>(3 * i + a)] - origin;
      }
    }
  }
  rgpot::nlist::Neighbours cells;
  const double search = cutoff * (1.0 + 1.0e-8);
  rgpot::nlist::updateCells(cells, wrapped.data(), static_cast<std::size_t>(n), box, search,
                            0.0, rgpot::nlist::Image::Round, rgpot::nlist::Mode::Stream,
                            periodic);
  if (!cells.images()) {
    rgpot::nlist::visitHalf(cells, 0, n,
                            [&](std::int32_t i, std::int32_t j, double, double, double, double) {
                              fn(static_cast<int>(i), static_cast<int>(j),
                                 static_cast<const double *>(nullptr));
                            });
    return;
  }
  rgpot::nlist::visitHalf(
      cells, 0, n, [&](std::int32_t i, std::int32_t j, double dx, double dy, double dz, double) {
        const auto a = static_cast<std::size_t>(3 * i);
        const auto b = static_cast<std::size_t>(3 * j);
        const double shift[3] = {centres[a] - centres[b] - dx,
                                 centres[a + 1] - centres[b + 1] - dy,
                                 centres[a + 2] - centres[b + 2] - dz};
        fn(static_cast<int>(i), static_cast<int>(j), static_cast<const double *>(shift));
      });
}

} // namespace forcefields
