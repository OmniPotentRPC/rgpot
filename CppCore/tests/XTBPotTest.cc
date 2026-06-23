// MIT License
// Copyright 2023--present rgpot developers

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <array>
#include <cmath>
#include <vector>

#include "rgpot/XTBPot/XTBPot.hpp"
#include "rgpot/types/AtomMatrix.hpp"

using rgpot::types::AtomMatrix;

// Water molecule geometry (Angstrom)
// O at origin, two H atoms
static const double water_pos[] = {
    0.00000000, 0.00000000, 0.11779000, // O
    0.00000000, 0.75545000, -0.47116000, // H
    0.00000000, -0.75545000, -0.47116000  // H
};
static const int water_atmnrs[] = {8, 1, 1};
static const double zero_box[9] = {100.0, 0.0, 0.0, 0.0, 100.0,
                                   0.0,   0.0, 0.0, 100.0};

TEST_CASE("XTBPot GFN2 water energy", "[xtb]") {
  rgpot::XTBPot pot;

  AtomMatrix positions(3, 3);
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      positions(i, j) = water_pos[i * 3 + j];
  std::vector<int> atmtypes(water_atmnrs, water_atmnrs + 3);
  std::array<std::array<double, 3>, 3> box = {
      {{100.0, 0.0, 0.0}, {0.0, 100.0, 0.0}, {0.0, 0.0, 100.0}}};

  auto [energy, forces] = pot(positions, atmtypes, box);

  // GFN2-xTB water energy is approximately -5.07 Hartree = -137.9 eV
  REQUIRE(energy < 0.0);
  REQUIRE_THAT(energy, Catch::Matchers::WithinAbs(-137.9, 1.0));

  // Forces should sum to approximately zero (translational invariance)
  double fx_sum = 0.0, fy_sum = 0.0, fz_sum = 0.0;
  for (size_t i = 0; i < 3; ++i) {
    fx_sum += forces(i, 0);
    fy_sum += forces(i, 1);
    fz_sum += forces(i, 2);
  }
  REQUIRE_THAT(fx_sum, Catch::Matchers::WithinAbs(0.0, 1e-6));
  REQUIRE_THAT(fy_sum, Catch::Matchers::WithinAbs(0.0, 1e-6));
  REQUIRE_THAT(fz_sum, Catch::Matchers::WithinAbs(0.0, 1e-6));
}

TEST_CASE("XTBPot GFNFF water energy", "[xtb]") {
  rgpot::XTBConfig cfg;
  cfg.method = rgpot::GFNMethod::GFNFF;
  rgpot::XTBPot pot(cfg);

  AtomMatrix positions(3, 3);
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      positions(i, j) = water_pos[i * 3 + j];
  std::vector<int> atmtypes(water_atmnrs, water_atmnrs + 3);
  std::array<std::array<double, 3>, 3> box = {
      {{100.0, 0.0, 0.0}, {0.0, 100.0, 0.0}, {0.0, 0.0, 100.0}}};

  auto [energy, forces] = pot(positions, atmtypes, box);

  // GFNFF energy should be negative for a bound molecule
  REQUIRE(energy < 0.0);
}
