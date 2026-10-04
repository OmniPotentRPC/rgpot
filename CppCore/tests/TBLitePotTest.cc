// MIT License
// Copyright 2023--present rgpot developers

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

#include "rgpot/TBLitePot/TBLitePot.hpp"
#include "rgpot/types/AtomMatrix.hpp"

using rgpot::types::AtomMatrix;
using Catch::Matchers::WithinAbs;

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

// ---------------------------------------------------------------------------
// Periodic boundary conditions derived from the box
// ---------------------------------------------------------------------------

namespace {

// Water in a 6 A cube with the second hydrogen written as its image one
// lattice vector away; a periodic evaluation cannot tell the two apart.
constexpr double kPeriodicEdge = 6.0;

struct PeriodicResult {
  double energy;
  int has_stress;
  std::array<double, 9> stress;
};

struct PeriodicFixture {
  std::array<double, 9> box{kPeriodicEdge, 0.0, 0.0, 0.0, kPeriodicEdge,
                            0.0,           0.0, 0.0, kPeriodicEdge};
  std::array<double, 9> pos{1.0, 1.0, 1.1177,  1.0, 1.75545, 0.52884,
                            1.0, 0.24455, 0.52884};
  std::array<int, 3> z{8, 1, 1};
};

PeriodicResult eval_tblite(rgpot::TBLitePot &pot, const std::array<double, 9> &pos,
                     const std::array<int, 3> &z,
                     const std::array<double, 9> &box) {
  std::array<double, 9> f{};
  rgpot::ForceOut out{f.data(), 0.0, 0.0, {}, 0};
  pot.forceImpl({3, pos.data(), z.data(), box.data()}, &out);
  return {out.energy, out.has_stress,
          {out.stress[0], out.stress[1], out.stress[2], out.stress[3],
           out.stress[4], out.stress[5], out.stress[6], out.stress[7],
           out.stress[8]}};
}

} // namespace

TEST_CASE("TBLitePot treats a nonzero box as periodic", "[tblite][periodic]") {
  PeriodicFixture fx;
  rgpot::TBLitePot pot;
  const auto base = eval_tblite(pot, fx.pos, fx.z, fx.box);

  auto imaged = fx.pos;
  imaged[6] += kPeriodicEdge; // second H, one lattice vector along x
  imaged[8] -= kPeriodicEdge; // and along -z
  const auto shifted = eval_tblite(pot, imaged, fx.z, fx.box);
  REQUIRE_THAT(shifted.energy, WithinAbs(base.energy, 1e-7));
}

TEST_CASE("TBLitePot periodic stress matches the strain derivative",
          "[tblite][periodic][stress]") {
  PeriodicFixture fx;
  // GFN2 periodic stress departs from the strain derivative of its energy;
  // GFN1 agrees.
  rgpot::TBLiteConfig cfg;
  cfg.method = rgpot::TBLiteMethod::GFN1;
  rgpot::TBLitePot pot(cfg);
  const auto base = eval_tblite(pot, fx.pos, fx.z, fx.box);
  REQUIRE(base.has_stress == 1);

  constexpr double step = 1e-3;
  const double volume = kPeriodicEdge * kPeriodicEdge * kPeriodicEdge;
  double largest = 0.0;
  for (int row = 0; row < 3; ++row) {
    for (int col = 0; col < 3; ++col) {
      const auto energyAt = [&](double strain) {
        auto pos = fx.pos;
        auto box = fx.box;
        for (int atom = 0; atom < 3; ++atom) {
          pos[3 * atom + col] += strain * fx.pos[3 * atom + row];
        }
        for (int axis = 0; axis < 3; ++axis) {
          box[3 * axis + col] += strain * fx.box[3 * axis + row];
        }
        return eval_tblite(pot, pos, fx.z, box).energy;
      };
      const double expected =
          (energyAt(step) - energyAt(-step)) / (2.0 * step * volume);
      CAPTURE(row, col, expected);
      REQUIRE_THAT(base.stress[3 * row + col], WithinAbs(expected, 2e-5));
      largest = std::max(largest, std::abs(expected));
    }
  }
  // A non-periodic evaluation reports no lattice response.
  REQUIRE(largest > 1e-4);
}

TEST_CASE("TBLitePot switches between periodic and isolated boxes",
          "[tblite][periodic]") {
  PeriodicFixture fx;
  rgpot::TBLitePot pot;
  const std::array<double, 9> none{};
  const auto isolated_first = eval_tblite(pot, fx.pos, fx.z, none);
  const auto periodic = eval_tblite(pot, fx.pos, fx.z, fx.box);
  const auto isolated_again = eval_tblite(pot, fx.pos, fx.z, none);
  REQUIRE(isolated_first.has_stress == 0);
  REQUIRE(periodic.has_stress == 1);
  REQUIRE_THAT(isolated_again.energy, WithinAbs(isolated_first.energy, 1e-9));
  REQUIRE(std::abs(periodic.energy - isolated_first.energy) > 1e-6);
}
