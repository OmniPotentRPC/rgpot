// MIT License
// Copyright 2023--present rgpot developers
//
// s-dftd3 ISO C API smoke (dftd3.h). Golden pins live on rgpot-0r9w.
// This suite only checks that D3Pot evaluates water and that the explicit
// ATM flag changes the energy.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <array>
#include <cmath>
#include <vector>

#include "rgpot/D3Pot/D3Pot.hpp"
#include "rgpot/types/AtomMatrix.hpp"

using rgpot::types::AtomMatrix;
using Catch::Matchers::WithinAbs;

// Water (Å): O + 2H
static const double kWaterPos[] = {
    0.00000000, 0.00000000,  0.11779000,  // O
    0.00000000, 0.75545000,  -0.47116000, // H
    0.00000000, -0.75545000, -0.47116000  // H
};
static const int kWaterZ[] = {8, 1, 1};

static void fill_water(AtomMatrix &positions, std::vector<int> &atmtypes,
                       std::array<std::array<double, 3>, 3> &box) {
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      positions(static_cast<size_t>(i), static_cast<size_t>(j)) =
          kWaterPos[i * 3 + j];
  atmtypes.assign(kWaterZ, kWaterZ + 3);
  box = {{{100.0, 0.0, 0.0}, {0.0, 100.0, 0.0}, {0.0, 0.0, 100.0}}};
}

TEST_CASE("D3Pot default is BJ PBE with ATM on", "[d3][config]") {
  rgpot::D3Pot pot;
  REQUIRE(pot.get_type() == rgpot::PotType::D3);
  REQUIRE(pot.config().damping == rgpot::D3Damping::BJ);
  REQUIRE(pot.config().functional == "pbe");
  REQUIRE(pot.config().atm);
}

TEST_CASE("D3Pot water energy is finite and ATM flips it", "[d3][atm]") {
  AtomMatrix positions(3, 3);
  std::vector<int> atmtypes;
  std::array<std::array<double, 3>, 3> box{};
  fill_water(positions, atmtypes, box);

  rgpot::D3Config on;
  on.damping = rgpot::D3Damping::BJ;
  on.functional = "pbe";
  on.atm = true;
  rgpot::D3Config off = on;
  off.atm = false;

  rgpot::D3Pot pot_on(on);
  rgpot::D3Pot pot_off(off);
  auto [e_on, f_on, v_on] = pot_on(positions, atmtypes, box);
  auto [e_off, f_off, v_off] = pot_off(positions, atmtypes, box);
  (void)v_on;
  (void)v_off;

  REQUIRE(std::isfinite(e_on));
  REQUIRE(std::isfinite(e_off));
  REQUIRE(e_on != e_off);
  REQUIRE(std::abs(e_on - e_off) > 1e-12);

  REQUIRE(f_on.rows() == 3);
  REQUIRE(f_off.rows() == 3);
  double f2 = 0.0;
  for (size_t i = 0; i < 3; ++i)
    for (size_t j = 0; j < 3; ++j) {
      REQUIRE(std::isfinite(f_on(i, j)));
      REQUIRE(std::isfinite(f_off(i, j)));
      f2 += f_on(i, j) * f_on(i, j);
    }
  REQUIRE(f2 > 0.0);

  double fx = 0, fy = 0, fz = 0;
  for (size_t i = 0; i < 3; ++i) {
    fx += f_on(i, 0);
    fy += f_on(i, 1);
    fz += f_on(i, 2);
  }
  REQUIRE_THAT(fx, WithinAbs(0.0, 1e-8));
  REQUIRE_THAT(fy, WithinAbs(0.0, 1e-8));
  REQUIRE_THAT(fz, WithinAbs(0.0, 1e-8));
}

TEST_CASE("D3Pot zero damping water is finite", "[d3][zero]") {
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
  REQUIRE(forces.rows() == 3);
}

TEST_CASE("D3Pot warm update_structure path is stable", "[d3][capi]") {
  rgpot::D3Pot pot;
  AtomMatrix positions(3, 3);
  std::vector<int> atmtypes;
  std::array<std::array<double, 3>, 3> box{};
  fill_water(positions, atmtypes, box);

  auto [e1, f1, v1] = pot(positions, atmtypes, box);
  positions(1, 1) += 0.01;
  auto [e2, f2, v2] = pot(positions, atmtypes, box);
  (void)f1;
  (void)f2;
  (void)v1;
  (void)v2;
  REQUIRE(std::isfinite(e1));
  REQUIRE(std::isfinite(e2));
  REQUIRE(e1 != e2);
}
