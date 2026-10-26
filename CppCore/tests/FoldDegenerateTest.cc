// MIT License
// Copyright 2023--present rgpot developers
//
// Thin and sheared cells that pass the conditioning check. Their search box
// from the full-cell dual norms used to reach millions of candidates, so a
// fold under some periodic masks never returned. The build registers this
// test with a timeout.

#include <array>
#include <chrono>
#include <cmath>
#include <random>

#include <catch2/catch_all.hpp>

#include "rgpot/nlist/MinimumImage.hpp"

namespace {

double brute_r2(const double cell[9], const std::array<bool, 3> &periodic,
                const double d[3], int span) {
  double best = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
  const int si = periodic[0] ? span : 0;
  const int sj = periodic[1] ? span : 0;
  const int sk = periodic[2] ? span : 0;
  for (int i = -si; i <= si; ++i) {
    for (int j = -sj; j <= sj; ++j) {
      for (int k = -sk; k <= sk; ++k) {
        const double x = d[0] - (i * cell[0] + j * cell[3] + k * cell[6]);
        const double y = d[1] - (i * cell[1] + j * cell[4] + k * cell[7]);
        const double z = d[2] - (i * cell[2] + j * cell[5] + k * cell[8]);
        best = std::min(best, x * x + y * y + z * z);
      }
    }
  }
  return best;
}

} // namespace

TEST_CASE("Fold returns on thin and sheared cells under every periodic mask",
          "[PairList][triclinic][degenerate]") {
  const double cells[2][9] = {
      {1.0, 0.0, 0.0, 0.999, 0.05, 0.0, 0.3, 0.7, 0.01},
      {2.0, 0.0, 0.0, 1.99, 1e-3, 0.0, 0.0, 0.0, 2.0},
  };
  std::mt19937_64 rng(11);
  std::uniform_real_distribution<double> u(-3.0, 3.0);
  const auto t0 = std::chrono::steady_clock::now();
  for (const auto &cell : cells) {
    for (int mask = 1; mask < 8; ++mask) {
      const std::array<bool, 3> periodic{(mask & 1) != 0, (mask & 2) != 0,
                                         (mask & 4) != 0};
      const rgpot::nlist::MinimumImage image(cell, periodic);
      for (int t = 0; t < 24; ++t) {
        double d[3] = {u(rng), u(rng), u(rng)};
        if (t % 6 == 0) {
          // A long displacement along every axis, open ones included.
          for (double &v : d)
            v *= 1e5;
        }
        double g[3] = {d[0], d[1], d[2]};
        image.fold(g[0], g[1], g[2]);
        const double g2 = g[0] * g[0] + g[1] * g[1] + g[2] * g[2];
        INFO("mask=" << mask << " d=" << d[0] << "," << d[1] << "," << d[2]);
        REQUIRE(std::isfinite(g2));
        // The fold subtracts a rounded image and then searches from it, so
        // returning to the input leaves a few ulps of the cell entries.
        const double d2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
        REQUIRE(g2 <= d2 * (1.0 + 1e-9) + 1e-9);
        if (t % 6 != 0) {
          const double b2 = brute_r2(cell, periodic, d, 20);
          REQUIRE(g2 <= b2 * (1.0 + 1e-9) + 1e-9);
        }
      }
    }
  }
  const double wall = std::chrono::duration<double>(
                          std::chrono::steady_clock::now() - t0)
                          .count();
  REQUIRE(wall < 20.0);
}
