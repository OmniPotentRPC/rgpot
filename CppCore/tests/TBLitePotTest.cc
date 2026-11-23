// MIT License
// Copyright 2023--present rgpot developers

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <exception>
#include <stdexcept>
#include <iostream>
#include <vector>

#include "rgpot/TBLitePot/TBLitePot.hpp"
#include "rgpot/stress.hpp"
#include "rgpot/types/AtomMatrix.hpp"

#include "XtbHotClusters.hpp"

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

// A failed SCF leaves its last iterate in the result object. The next call
// has to start from tblite's own guess, as a new instance does. The SCF of
// Si13 frames 2 and 3 does not converge at 300 K. Seeded from frame 2,
// frame 3 used to converge, to -588.78 eV.
TEST_CASE("TBLitePot does not seed an SCF from a failed one", "[tblite][hot]") {
  using namespace rgpot_test_hot;
  auto frame = [](int f) {
    AtomMatrix positions(kSi13Atoms, 3);
    for (int i = 0; i < kSi13Atoms; ++i)
      for (int j = 0; j < 3; ++j)
        positions(i, j) = kSi13Pos[f][i][j];
    return positions;
  };
  const std::vector<int> atmtypes(kSi13Atoms, kSi13Z);
  std::array<std::array<double, 3>, 3> box{};
  box[0][0] = box[1][1] = box[2][2] = 60.0;
  auto evaluate = [&](rgpot::TBLitePot &pot, int f, double *energy) {
    try {
      auto [e, forces, variance] = pot(frame(f), atmtypes, box);
      (void)forces;
      (void)variance;
      *energy = e;
      return true;
    } catch (const std::exception &) {
      return false;
    }
  };

  rgpot::TBLitePot session;
  int checked = 0;
  bool previous_failed = false;
  for (int f = 2; f <= 3; ++f) {
    double e_session = 0.0;
    const bool ok_session = evaluate(session, f, &e_session);
    rgpot::TBLitePot fresh;
    double e_fresh = 0.0;
    const bool ok_fresh = evaluate(fresh, f, &e_fresh);
    UNSCOPED_INFO("frame " << f << " session " << ok_session << " " << e_session
                           << " fresh " << ok_fresh << " " << e_fresh);
    if (previous_failed) {
      ++checked;
      REQUIRE(ok_session == ok_fresh);
      if (ok_session)
        REQUIRE_THAT(e_session, Catch::Matchers::WithinAbs(e_fresh, 1e-6));
    }
    previous_failed = !ok_session;
  }
  REQUIRE(checked > 0);
}

namespace {

// Eight silicon atoms in the cubic diamond cell, rattled so that no
// component of the stress is zero by symmetry.
struct SiliconCell {
  std::array<double, 9> box{5.43, 0, 0, 0, 5.43, 0, 0, 0, 5.43};
  std::vector<double> pos;
  std::vector<int> numbers = std::vector<int>(8, 14);

  SiliconCell() {
    const double a = box[0];
    const double base[24] = {0,   0,   0,   0,   .5,  .5,  .5,  0,
                             .5,  .5,  .5,  0,   .25, .25, .25, .25,
                             .75, .75, .75, .25, .75, .75, .75, .25};
    pos.resize(24);
    for (int i = 0; i < 24; ++i) {
      pos[i] = base[i] * a;
    }
    pos[0] += 0.07;
    pos[4] -= 0.05;
    pos[8] += 0.06;
    pos[13] += 0.04;
    pos[20] -= 0.03;
  }
};

rgpot::ForceOut evaluateCell(const rgpot::TBLitePot &pot,
                             const std::vector<double> &pos,
                             const std::vector<int> &numbers,
                             const std::array<double, 9> &box, bool stress,
                             std::vector<double> &forces) {
  forces.assign(pos.size(), 0.0);
  rgpot::ForceOut out{};
  out.F = forces.data();
  rgpot::ForceInput in{numbers.size(), pos.data(), numbers.data(), box.data()};
  in.compute_stress = stress;
  pot.forceImpl(in, &out);
  return out;
}

double energyDerivative(rgpot::TBLiteMethod method, const SiliconCell &cell,
                        int a, int b, double step) {
  double energies[2]{};
  for (int side = 0; side < 2; ++side) {
    const double delta = side == 0 ? -step : step;
    double t[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    if (a == b) {
      t[3 * a + b] += delta;
    } else {
      t[3 * a + b] += 0.5 * delta;
      t[3 * b + a] += 0.5 * delta;
    }
    std::vector<double> moved(24);
    std::array<double, 9> strained{};
    for (int i = 0; i < 8; ++i) {
      for (int c = 0; c < 3; ++c) {
        moved[3 * i + c] = cell.pos[3 * i] * t[c] +
                           cell.pos[3 * i + 1] * t[3 + c] +
                           cell.pos[3 * i + 2] * t[6 + c];
      }
    }
    for (int r = 0; r < 3; ++r) {
      for (int c = 0; c < 3; ++c) {
        strained[3 * r + c] = cell.box[3 * r] * t[c] +
                              cell.box[3 * r + 1] * t[3 + c] +
                              cell.box[3 * r + 2] * t[6 + c];
      }
    }
    rgpot::TBLiteConfig config;
    config.method = method;
    rgpot::TBLitePot fresh(config);
    std::vector<double> forces;
    energies[side] =
        evaluateCell(fresh, moved, cell.numbers, strained, false, forces)
            .energy;
  }
  return (energies[1] - energies[0]) / (2.0 * step);
}

} // namespace

TEST_CASE("TBLitePot stress is the strain derivative of the energy",
          "[tblite][stress]") {
  rgpot::TBLiteConfig config;
  config.method = rgpot::TBLiteMethod::GFN1;
  const rgpot::TBLitePot pot(config);
  REQUIRE(pot.caps().stress);
  const SiliconCell cell;
  std::vector<double> plain_forces;
  std::vector<double> stress_forces;
  // tblite is not bit-reproducible between runs, so the force-only and the
  // stress call are compared at the level of its own run-to-run scatter.
  const rgpot::TBLitePot plain_pot(config);
  const auto plain = evaluateCell(plain_pot, cell.pos, cell.numbers, cell.box,
                                  false, plain_forces);
  const auto stressed =
      evaluateCell(pot, cell.pos, cell.numbers, cell.box, true, stress_forces);
  REQUIRE(plain.has_stress == 0);
  REQUIRE(stressed.has_stress == 1);
  CHECK_THAT(stressed.energy,
             Catch::Matchers::WithinAbs(plain.energy, 1e-8));
  for (std::size_t i = 0; i < plain_forces.size(); ++i) {
    CHECK_THAT(stress_forces[i],
               Catch::Matchers::WithinAbs(plain_forces[i], 1e-8));
  }

  // The strain steps leave the potential as it was.
  std::vector<double> again;
  const auto after =
      evaluateCell(pot, cell.pos, cell.numbers, cell.box, false, again);
  CHECK_THAT(after.energy, Catch::Matchers::WithinAbs(plain.energy, 1e-6));

  constexpr int rows[6] = {0, 1, 2, 1, 0, 0};
  constexpr int cols[6] = {0, 1, 2, 2, 2, 1};
  constexpr int slots[6] = {0, 4, 8, 5, 2, 1};
  const double volume = rgpot::cellVolume(cell.box.data());
  double worst = 0.0;
  double largest = 0.0;
  for (int k = 0; k < 6; ++k) {
    const double numeric =
        energyDerivative(config.method, cell, rows[k], cols[k], 1e-3) / volume;
    const double analytic = stressed.stress[slots[k]];
    worst = std::max(worst, std::abs(numeric - analytic));
    largest = std::max(largest, std::abs(analytic));
  }
  const double relative = worst / largest;
  std::cout << "STRESS_ERROR TBLite GFN1 " << relative << '\n';
  CHECK(relative < 1e-3);
  CHECK(stressed.stress[1] == stressed.stress[3]);
  CHECK(stressed.stress[2] == stressed.stress[6]);
  CHECK(stressed.stress[5] == stressed.stress[7]);
}

TEST_CASE("TBLitePot refuses stress for the methods whose energy is not "
          "smooth under strain",
          "[tblite][stress]") {
  for (auto method : {rgpot::TBLiteMethod::GFN2, rgpot::TBLiteMethod::IPEA1}) {
    rgpot::TBLiteConfig config;
    config.method = method;
    const rgpot::TBLitePot pot(config);
    CHECK_FALSE(pot.caps().stress);
    const SiliconCell cell;
    std::vector<double> forces;
    rgpot::ForceOut out{};
    forces.assign(cell.pos.size(), 0.0);
    out.F = forces.data();
    rgpot::ForceInput in{cell.numbers.size(), cell.pos.data(),
                         cell.numbers.data(), cell.box.data()};
    in.compute_stress = true;
    CHECK_THROWS_AS(pot.forceImpl(in, &out), std::invalid_argument);
    CHECK(out.has_stress == 0);
  }
}
