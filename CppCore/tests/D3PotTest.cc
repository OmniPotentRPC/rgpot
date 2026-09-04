// MIT License
// Copyright 2023--present rgpot developers
//
// s-dftd3 is Fortran with an ISO_C_BINDING C API (dftd3.h). Unit tests
// exercise that C contract:
//   - opaque handle create / destroy
//   - first force: dftd3_new_structure + dftd3_new_d3_model
//   - later: dftd3_update_structure
//   - ATM on vs off is an explicit D3Config.atm flag
// Tags: [dftd3][linked] [dftd3][atm]

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <array>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "rgpot/D3Pot/D3Pot.hpp"
#include "rgpot/types/AtomMatrix.hpp"

using Catch::Matchers::WithinAbs;
using rgpot::types::AtomMatrix;

// Water (Å): O + 2H
static const double kWaterPos[] = {
    0.00000000, 0.00000000,  0.11779000,  // O
    0.00000000, 0.75545000,  -0.47116000, // H
    0.00000000, -0.75545000, -0.47116000  // H
};
static const int kWaterZ[] = {8, 1, 1};

// Water dimer (Å): more ATM triplets than the monomer.
static const double kDimerPos[] = {
    -1.551007, -0.114520, 0.000000,  // O
    -1.934259, 0.762503,  0.000000,  // H
    -0.599677, 0.040712,  0.000000,  // H
    1.350625,  0.111469,  0.000000,  // O
    1.680398,  -0.373741, -0.758561, // H
    1.680398,  -0.373741, 0.758561   // H
};
static const int kDimerZ[] = {8, 1, 1, 8, 1, 1};

static void fill_water(AtomMatrix &positions, std::vector<int> &atmtypes,
                       std::array<std::array<double, 3>, 3> &box) {
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      positions(static_cast<size_t>(i), static_cast<size_t>(j)) =
          kWaterPos[i * 3 + j];
  atmtypes.assign(kWaterZ, kWaterZ + 3);
  box = {{{100.0, 0.0, 0.0}, {0.0, 100.0, 0.0}, {0.0, 0.0, 100.0}}};
}

static void fill_dimer(AtomMatrix &positions, std::vector<int> &atmtypes,
                       std::array<std::array<double, 3>, 3> &box) {
  for (int i = 0; i < 6; ++i)
    for (int j = 0; j < 3; ++j)
      positions(static_cast<size_t>(i), static_cast<size_t>(j)) =
          kDimerPos[i * 3 + j];
  atmtypes.assign(kDimerZ, kDimerZ + 6);
  box = {{{100.0, 0.0, 0.0}, {0.0, 100.0, 0.0}, {0.0, 0.0, 100.0}}};
}

static void require_finite_forces(const AtomMatrix &forces, size_t n) {
  REQUIRE(forces.rows() == n);
  REQUIRE(forces.cols() == 3);
  for (size_t i = 0; i < n; ++i)
    for (size_t j = 0; j < 3; ++j)
      REQUIRE(std::isfinite(forces(i, j)));
}

static void require_zero_net_force(const AtomMatrix &forces, double tol) {
  double fx = 0, fy = 0, fz = 0;
  for (size_t i = 0; i < forces.rows(); ++i) {
    fx += forces(i, 0);
    fy += forces(i, 1);
    fz += forces(i, 2);
  }
  REQUIRE_THAT(fx, WithinAbs(0.0, tol));
  REQUIRE_THAT(fy, WithinAbs(0.0, tol));
  REQUIRE_THAT(fz, WithinAbs(0.0, tol));
}

TEST_CASE("D3Pot default is PBE BJ with ATM on", "[dftd3][linked]") {
  rgpot::D3Pot pot;
  REQUIRE(pot.get_type() == rgpot::PotType::D3);
  REQUIRE(pot.config().damping == rgpot::D3Damping::BJ);
  REQUIRE(pot.config().functional == "pbe");
  REQUIRE(pot.config().atm);
}

TEST_CASE("D3Pot PBE-BJ water energy and force conservation",
          "[dftd3][linked]") {
  rgpot::D3Pot pot;
  AtomMatrix positions(3, 3);
  std::vector<int> atmtypes;
  std::array<std::array<double, 3>, 3> box{};
  fill_water(positions, atmtypes, box);

  auto [energy, forces, variance] = pot(positions, atmtypes, box);
  (void)variance;

  REQUIRE(std::isfinite(energy));
  REQUIRE(energy < 0.0);
  require_finite_forces(forces, 3);
  require_zero_net_force(forces, 1e-8);
}

TEST_CASE("D3Pot ATM on vs off changes water-dimer energy",
          "[dftd3][linked][atm]") {
  rgpot::D3Config on;
  on.damping = rgpot::D3Damping::BJ;
  on.functional = "pbe";
  on.atm = true;
  rgpot::D3Config off = on;
  off.atm = false;

  rgpot::D3Pot pot_on(on);
  rgpot::D3Pot pot_off(off);
  REQUIRE(pot_on.config().atm);
  REQUIRE_FALSE(pot_off.config().atm);

  AtomMatrix positions(6, 3);
  std::vector<int> atmtypes;
  std::array<std::array<double, 3>, 3> box{};
  fill_dimer(positions, atmtypes, box);

  auto [e_on, f_on, v_on] = pot_on(positions, atmtypes, box);
  auto [e_off, f_off, v_off] = pot_off(positions, atmtypes, box);
  (void)v_on;
  (void)v_off;

  REQUIRE(std::isfinite(e_on));
  REQUIRE(std::isfinite(e_off));
  REQUIRE(std::abs(e_on - e_off) > 1e-10);
  require_finite_forces(f_on, 6);
  require_finite_forces(f_off, 6);
}

TEST_CASE("D3Pot zero damping water is finite", "[dftd3][linked][zero]") {
  rgpot::D3Config cfg;
  cfg.damping = rgpot::D3Damping::Zero;
  cfg.functional = "pbe";
  cfg.atm = true;
  rgpot::D3Pot pot(cfg);

  AtomMatrix positions(3, 3);
  std::vector<int> atmtypes;
  std::array<std::array<double, 3>, 3> box{};
  fill_water(positions, atmtypes, box);

  auto [energy, forces, variance] = pot(positions, atmtypes, box);
  (void)variance;
  REQUIRE(std::isfinite(energy));
  REQUIRE(energy < 0.0);
  require_finite_forces(forces, 3);
}

TEST_CASE("D3Pot warm update_structure path is stable",
          "[dftd3][linked][capi]") {
  rgpot::D3Pot pot;
  AtomMatrix positions(3, 3);
  std::vector<int> atmtypes;
  std::array<std::array<double, 3>, 3> box{};
  fill_water(positions, atmtypes, box);

  auto [e0, f0, v0] = pot(positions, atmtypes, box);
  (void)v0;
  REQUIRE(std::isfinite(e0));

  AtomMatrix moved(3, 3);
  for (size_t i = 0; i < 3; ++i)
    for (size_t j = 0; j < 3; ++j)
      moved(i, j) = positions(i, j);
  moved(1, 1) += 0.05;
  auto [e1, f1, v1] = pot(moved, atmtypes, box);
  (void)v1;
  REQUIRE(std::isfinite(e1));
  REQUIRE(std::abs(e1 - e0) > 1e-10);
  require_finite_forces(f1, 3);

  auto [e2, f2, v2] = pot(positions, atmtypes, box);
  (void)f0;
  (void)f2;
  (void)v2;
  REQUIRE_THAT(e2, WithinAbs(e0, 1e-12));
}

TEST_CASE("D3Pot rejects empty functional", "[dftd3][linked]") {
  rgpot::D3Config cfg;
  cfg.functional.clear();
  REQUIRE_THROWS_AS(rgpot::D3Pot(cfg), std::invalid_argument);
}

TEST_CASE("D3Pot rejects unknown functional", "[dftd3][linked]") {
  rgpot::D3Config cfg;
  cfg.functional = "not-a-real-functional";
  REQUIRE_THROWS_AS(rgpot::D3Pot(cfg), std::runtime_error);
}
