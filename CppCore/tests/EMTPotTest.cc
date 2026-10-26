// MIT License
// Copyright 2023--present rgpot developers
//
// Copper FCC cluster pin from eOn EMTCuTest / point_emt_cu.dat.

#include <array>
#include <cmath>
#include <vector>

#include <catch2/catch_all.hpp>

#include "rgpot/EMT/EMTPot.hpp"

using Catch::Matchers::WithinAbs;

TEST_CASE("EMT energy matches the copper cluster pin", "[emt]") {
  const std::vector<std::array<double, 3>> xyz = {{
      {8.1925, 8.1925, 8.1925},
      {10.0, 10.0, 8.1925},
      {10.0, 8.1925, 10.0},
      {8.1925, 10.0, 10.0},
  }};
  rgpot::types::AtomMatrix positions(4, 3);
  std::vector<int> types(4, 29);
  for (int i = 0; i < 4; ++i) {
    for (int k = 0; k < 3; ++k) {
      positions(i, k) = xyz[static_cast<size_t>(i)][static_cast<size_t>(k)];
    }
  }
  const std::array<std::array<double, 3>, 3> box = {{
      {20.0, 0.0, 0.0},
      {0.0, 20.0, 0.0},
      {0.0, 0.0, 20.0},
  }};
  rgpot::EMTPot pot(rgpot::EMTConfig{.rasmussen = false});
  const auto [energy, forces, variance] = pot(positions, types, box);
  REQUIRE_THAT(energy, WithinAbs(5.129167, 5e-4));
  double worst = 0.0;
  for (int i = 0; i < 4; ++i) {
    const double n2 = forces(i, 0) * forces(i, 0) + forces(i, 1) * forces(i, 1) +
                      forces(i, 2) * forces(i, 2);
    worst = std::max(worst, std::sqrt(n2));
  }
  REQUIRE_THAT(worst, WithinAbs(1.914263, 2e-3));
  REQUIRE(pot.config().rasmussen == false);
  REQUIRE(pot.paramsKey() != 0);
}
