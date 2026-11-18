// MIT License
// Copyright 2023--present rgpot developers
//
// Copper FCC cluster pin from eOn EMTCuTest / point_emt_cu.dat.

#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
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

TEST_CASE("EMT reads the species again when only the species change",
          "[emt]") {
  rgpot::types::AtomMatrix positions(4, 3);
  const std::array<std::array<double, 3>, 4> xyz = {{
      {8.1925, 8.1925, 8.1925},
      {10.0, 10.0, 8.1925},
      {10.0, 8.1925, 10.0},
      {8.1925, 10.0, 10.0},
  }};
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
  const std::vector<int> copper(4, 29);
  const std::vector<int> mixed = {29, 79, 29, 79};

  rgpot::EMTPot reused;
  (void)reused(positions, copper, box);
  const auto [e_reused, f_reused, v_reused] = reused(positions, mixed, box);

  rgpot::EMTPot fresh;
  const auto [e_fresh, f_fresh, v_fresh] = fresh(positions, mixed, box);

  REQUIRE_THAT(e_reused, WithinAbs(e_fresh, 1e-12));
  for (int i = 0; i < 4; ++i) {
    for (int k = 0; k < 3; ++k) {
      REQUIRE_THAT(f_reused(i, k), WithinAbs(f_fresh(i, k), 1e-12));
    }
  }
}

namespace {

using Box = std::array<std::array<double, 3>, 3>;

rgpot::types::AtomMatrix copperCluster() {
  const std::array<std::array<double, 3>, 4> xyz = {{
      {8.1925, 8.1925, 8.1925},
      {10.0, 10.0, 8.1925},
      {10.0, 8.1925, 10.0},
      {8.1925, 10.0, 10.0},
  }};
  rgpot::types::AtomMatrix positions(4, 3);
  for (int i = 0; i < 4; ++i) {
    for (int k = 0; k < 3; ++k) {
      positions(i, k) = xyz[static_cast<size_t>(i)][static_cast<size_t>(k)];
    }
  }
  return positions;
}

constexpr Box kCube = {{
    {20.0, 0.0, 0.0},
    {0.0, 20.0, 0.0},
    {0.0, 0.0, 20.0},
}};

} // namespace

TEST_CASE("EMT turns an Asap exception into std::invalid_argument",
          "[emt][errors]") {
  const auto positions = copperCluster();
  const std::vector<int> carbon(4, 6);
  rgpot::EMTPot pot;
  REQUIRE_THROWS_AS(pot(positions, carbon, kCube), std::invalid_argument);
  REQUIRE_THROWS_WITH(pot(positions, carbon, kCube),
                      Catch::Matchers::ContainsSubstring("EMT"));
}

TEST_CASE("EMT keeps no state from a call that threw", "[emt][errors]") {
  const auto positions = copperCluster();
  const std::vector<int> carbon(4, 6);
  const std::vector<int> copper(4, 29);
  rgpot::EMTPot pot;
  for (int attempt = 0; attempt < 3; ++attempt) {
    REQUIRE_THROWS_AS(pot(positions, carbon, kCube), std::invalid_argument);
  }
  const auto [e_after, f_after, v_after] = pot(positions, copper, kCube);
  rgpot::EMTPot fresh;
  const auto [e_fresh, f_fresh, v_fresh] = fresh(positions, copper, kCube);
  REQUIRE(e_after == e_fresh);
  for (int i = 0; i < 4; ++i) {
    for (int k = 0; k < 3; ++k) {
      REQUIRE(f_after(i, k) == f_fresh(i, k));
    }
  }
}

TEST_CASE("EMT recovers after a cell that Asap refuses", "[emt][errors]") {
  const auto positions = copperCluster();
  const std::vector<int> copper(4, 29);
  const Box thin = {{{6.0, 0.0, 0.0}, {0.0, 20.0, 0.0}, {0.0, 0.0, 20.0}}};
  rgpot::EMTPot pot;
  (void)pot(positions, copper, kCube);
  REQUIRE_THROWS_AS(pot(positions, copper, thin), std::invalid_argument);
  const auto [energy, forces, variance] = pot(positions, copper, kCube);
  REQUIRE_THAT(energy, WithinAbs(5.129167, 5e-4));
}

TEST_CASE("EMT of zero atoms is a zero result", "[emt][errors]") {
  rgpot::types::AtomMatrix none(0, 3);
  const std::vector<int> nothing;
  rgpot::EMTPot pot;
  const auto [energy, forces, variance] = pot(none, nothing, kCube);
  REQUIRE(energy == 0.0);
  REQUIRE(forces.rows() == 0);
  REQUIRE(variance == 0.0);
}

TEST_CASE("EMT refuses a cell it cannot invert", "[emt][errors]") {
  const auto positions = copperCluster();
  const std::vector<int> copper(4, 29);
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const std::array<Box, 4> bad = {{
      {{{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}}},
      {{{20.0, 0.0, 0.0}, {40.0, 0.0, 0.0}, {0.0, 0.0, 20.0}}},
      {{{20.0, 0.0, 0.0}, {0.0, 20.0, 0.0}, {20.0, 20.0, 0.0}}},
      {{{20.0, 0.0, 0.0}, {0.0, nan, 0.0}, {0.0, 0.0, 20.0}}},
  }};
  rgpot::EMTPot pot;
  for (const Box &box : bad) {
    REQUIRE_THROWS_AS(pot(positions, copper, box), std::invalid_argument);
  }
  const auto [energy, forces, variance] = pot(positions, copper, kCube);
  REQUIRE_THAT(energy, WithinAbs(5.129167, 5e-4));
}

TEST_CASE("EMT refuses non-finite positions", "[emt][errors]") {
  auto positions = copperCluster();
  positions(2, 1) = std::numeric_limits<double>::infinity();
  const std::vector<int> copper(4, 29);
  rgpot::EMTPot pot;
  REQUIRE_THROWS_AS(pot(positions, copper, kCube), std::invalid_argument);
}
