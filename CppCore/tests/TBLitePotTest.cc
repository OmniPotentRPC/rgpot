// MIT License
// Copyright 2023--present rgpot developers

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <array>
#include <cmath>
#include <cstring>
#include <vector>

#include "rgpot/TBLitePot/TBLitePot.hpp"
#include "rgpot/types/AtomMatrix.hpp"

using rgpot::types::AtomMatrix;

// Water molecule geometry (Angstrom)
static const double water_pos[] = {
    0.00000000, 0.00000000,  0.11779000,  // O
    0.00000000, 0.75545000,  -0.47116000, // H
    0.00000000, -0.75545000, -0.47116000  // H
};
static const int water_atmnrs[] = {8, 1, 1};

TEST_CASE("TBLitePot GFN2 water energy", "[tblite]") {
  rgpot::TBLitePot pot;

  AtomMatrix positions(3, 3);
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      positions(i, j) = water_pos[i * 3 + j];
  std::vector<int> atmtypes(water_atmnrs, water_atmnrs + 3);
  std::array<std::array<double, 3>, 3> box = {
      {{100.0, 0.0, 0.0}, {0.0, 100.0, 0.0}, {0.0, 0.0, 100.0}}};

  auto [energy, forces, variance] = pot(positions, atmtypes, box);
  (void)variance;

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

TEST_CASE("TBLitePot GFN1 water energy", "[tblite]") {
  rgpot::TBLiteConfig cfg;
  cfg.method = rgpot::TBLiteMethod::GFN1;
  rgpot::TBLitePot pot(cfg);

  AtomMatrix positions(3, 3);
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      positions(i, j) = water_pos[i * 3 + j];
  std::vector<int> atmtypes(water_atmnrs, water_atmnrs + 3);
  std::array<std::array<double, 3>, 3> box = {
      {{100.0, 0.0, 0.0}, {0.0, 100.0, 0.0}, {0.0, 0.0, 100.0}}};

  auto [energy, forces, variance] = pot(positions, atmtypes, box);
  (void)variance;

  // GFN1-xTB energy should be negative for a bound molecule
  REQUIRE(energy < 0.0);
}

TEST_CASE("TBLitePot GFN1 periodic virial matches a strain derivative",
          "[tblite][virial]") {
  // Two Si atoms in a sheared cell. Native libxtb does not return a
  // periodic GFN1 virial (the crystal single point fails, and a molecule
  // comes back with a zero tensor). tblite's dE/dε matches the right
  // strain r -> r(I+ε) on this cell. A filled Si 8-atom crystal does not:
  // that analytic tensor is off by tenths of an eV.
  rgpot::TBLiteConfig cfg;
  cfg.method = rgpot::TBLiteMethod::GFN1;
  cfg.periodic = 1;
  cfg.accuracy = 0.01;
  rgpot::TBLitePot pot(cfg);
  REQUIRE(pot.caps().virial);

  const double pos0[6] = {1.0, 1.2, 0.8, 3.4, 1.5, 1.1};
  const double box0[9] = {12, 0.2, -0.1, 0.15, 12, 0.05, -0.08, 0.12, 12};
  const int types[2] = {14, 14};

  auto energy_of = [&](const double *pos, const double *box) {
    double forces[6];
    rgpot::ForceOut out{forces, 0.0, 0.0, {}, 0};
    pot.forceImpl({2, pos, types, box}, &out);
    return out;
  };

  const rgpot::ForceOut base = energy_of(pos0, box0);
  REQUIRE(base.has_virial == 1);
  REQUIRE(base.has_stress == 1);
  REQUIRE(std::isfinite(base.energy));
  const double volume = std::abs(
      box0[0] * (box0[4] * box0[8] - box0[5] * box0[7]) -
      box0[1] * (box0[3] * box0[8] - box0[5] * box0[6]) +
      box0[2] * (box0[3] * box0[7] - box0[4] * box0[6]));
  for (int i = 0; i < 9; ++i) {
    REQUIRE_THAT(base.stress[i],
                 Catch::Matchers::WithinAbs(-base.virial[i] / volume, 1e-12));
  }

  constexpr double h = 1e-6;
  for (int row = 0; row < 3; ++row) {
    for (int col = 0; col < 3; ++col) {
      auto at = [&](double eps) {
        double pos[6];
        double box[9];
        std::memcpy(pos, pos0, sizeof pos);
        std::memcpy(box, box0, sizeof box);
        for (int atom = 0; atom < 2; ++atom)
          pos[3 * atom + col] += eps * pos0[3 * atom + row];
        for (int k = 0; k < 3; ++k)
          box[3 * k + col] += eps * box0[3 * k + row];
        return energy_of(pos, box).energy;
      };
      const double dE = (at(h) - at(-h)) / (2.0 * h);
      CAPTURE(row, col, dE, base.virial[3 * row + col]);
      REQUIRE_THAT(base.virial[3 * row + col],
                   Catch::Matchers::WithinAbs(-dE, 1e-5));
    }
  }
}
