// MIT License
// Copyright 2023--present rgpot developers

#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

#include <catch2/catch_all.hpp>

#include "rgpot/EAM/EAMPot.hpp"
#include "rgpot/stress.hpp"

TEST_CASE("Cell-list EAM returns a finite aluminium pair", "[eam]") {
  auto positions = rgpot::types::AtomMatrix::Zero(2, 3);
  positions(1, 0) = 2.5;
  const std::vector<int> types = {13, 13};
  const std::array<std::array<double, 3>, 3> box = {{
      {20.0, 0.0, 0.0},
      {0.0, 20.0, 0.0},
      {0.0, 0.0, 20.0},
  }};
  rgpot::EAMPot pot;
  const auto [energy, forces, variance] = pot(positions, types, box);
  REQUIRE(std::isfinite(energy));
  REQUIRE(std::isfinite(forces(0, 0)));
  REQUIRE(std::abs(forces(0, 0) + forces(1, 0)) < 1e-8);
  REQUIRE(pot.paramsKey() != 0);
  (void)variance;
}

TEST_CASE("Cell-list EAM rejects an element it has no parameters for",
          "[eam]") {
  auto positions = rgpot::types::AtomMatrix::Zero(2, 3);
  positions(1, 0) = 2.5;
  const std::vector<int> types = {13, 1};
  const std::array<std::array<double, 3>, 3> box = {{
      {20.0, 0.0, 0.0},
      {0.0, 20.0, 0.0},
      {0.0, 0.0, 20.0},
  }};
  rgpot::EAMPot pot;
  REQUIRE_THROWS_AS(pot(positions, types, box), std::invalid_argument);
}

TEST_CASE("Cell-list EAM opt-in stress matches triclinic strain",
          "[eam][stress]") {
  std::array<double, 9> box = {16.0, 0.7, 0.2, -0.4, 15.0,
                               0.5,  0.3, -0.6, 17.0};
  std::vector<double> fractional = {0.11, 0.13, 0.17, 0.29, 0.31, 0.23,
                                    0.47, 0.26, 0.38, 0.36, 0.51, 0.44,
                                    0.62, 0.43, 0.29, 0.55, 0.61, 0.57};
  std::vector<double> positions(fractional.size());
  auto carry = [](const double *source, double *target, std::size_t count,
                  const double transform[9]) {
    for (std::size_t i = 0; i < count; ++i) {
      for (int c = 0; c < 3; ++c) {
        target[3 * i + c] =
            source[3 * i] * transform[c] +
            source[3 * i + 1] * transform[3 + c] +
            source[3 * i + 2] * transform[6 + c];
      }
    }
  };
  carry(fractional.data(), positions.data(), positions.size() / 3, box.data());
  const std::vector<int> types(positions.size() / 3, 13);
  rgpot::EAMPot pot;
  auto evaluate = [&](const std::vector<double> &pos,
                      const std::array<double, 9> &cell, bool stress) {
    std::vector<double> forces(pos.size());
    rgpot::ForceOut out{};
    out.F = forces.data();
    rgpot::ForceInput in{types.size(), pos.data(), types.data(), cell.data()};
    in.compute_stress = stress;
    pot.forceImpl(in, &out);
    return std::pair{out, forces};
  };
  const auto [plain, plain_forces] = evaluate(positions, box, false);
  const auto [analytic, analytic_forces] = evaluate(positions, box, true);
  REQUIRE(plain.has_stress == 0);
  REQUIRE(analytic.has_stress == 1);
  REQUIRE(plain.energy == analytic.energy);
  REQUIRE(plain_forces == analytic_forces);
  REQUIRE(analytic.stress[1] == analytic.stress[3]);
  REQUIRE(analytic.stress[2] == analytic.stress[6]);
  REQUIRE(analytic.stress[5] == analytic.stress[7]);

  constexpr int rows[6] = {0, 1, 2, 1, 0, 0};
  constexpr int cols[6] = {0, 1, 2, 2, 2, 1};
  constexpr int slots[6] = {0, 4, 8, 5, 2, 1};
  constexpr double steps[5] = {1e-3, 1e-4, 1e-5, 1e-6, 1e-7};
  const double volume = rgpot::cellVolume(box.data());
  double target_error = 0.0;
  double best_error = std::numeric_limits<double>::infinity();
  for (double step : steps) {
    double max_abs_error = 0.0;
    double max_stress = 0.0;
    for (int component = 0; component < 6; ++component) {
      double energies[2]{};
      for (int side = 0; side < 2; ++side) {
        const double delta = side == 0 ? -step : step;
        double transform[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
        const int a = rows[component];
        const int b = cols[component];
        if (a == b) {
          transform[3 * a + b] += delta;
        } else {
          transform[3 * a + b] += 0.5 * delta;
          transform[3 * b + a] += 0.5 * delta;
        }
        std::vector<double> moved(positions.size());
        std::array<double, 9> strained{};
        carry(positions.data(), moved.data(), positions.size() / 3, transform);
        carry(box.data(), strained.data(), 3, transform);
        energies[side] = evaluate(moved, strained, false).first.energy;
      }
      const double numeric = (energies[1] - energies[0]) / (2 * step * volume);
      max_abs_error =
          std::max(max_abs_error,
                   std::abs(numeric - analytic.stress[slots[component]]));
      max_stress =
          std::max(max_stress, std::abs(analytic.stress[slots[component]]));
    }
    const double error = max_abs_error / std::max(max_stress, 1e-8);
    best_error = std::min(best_error, error);
    if (step == 1e-5) {
      target_error = error;
    }
  }
  std::cout << "STRESS_ERROR EAM " << target_error << '\n';
  REQUIRE(target_error <= 1e-6);
  REQUIRE(best_error <= target_error * 1.25 + 1e-12);
}
