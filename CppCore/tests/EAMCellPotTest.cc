// MIT License
// Copyright 2023--present rgpot developers

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
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

namespace {

struct Evaluation {
  rgpot::ForceOut out;
  std::vector<double> forces;
};

Evaluation evaluate(const rgpot::EAMPot &pot, const std::vector<double> &pos,
                    const std::vector<int> &types,
                    const std::array<double, 9> &cell, bool stress) {
  Evaluation result;
  result.forces.assign(pos.size(), 0.0);
  result.out = {};
  result.out.F = result.forces.data();
  rgpot::ForceInput in{types.size(), pos.data(), types.data(), cell.data()};
  in.compute_stress = stress;
  pot.forceImpl(in, &result.out);
  return result;
}

std::uint64_t bits(double value) {
  std::uint64_t u = 0;
  std::memcpy(&u, &value, sizeof u);
  return u;
}

std::uint64_t forceHash(const std::vector<double> &forces) {
  std::uint64_t hash = 1469598103934665603ULL;
  for (double f : forces) {
    hash ^= bits(f);
    hash *= 1099511628211ULL;
  }
  return hash;
}

std::vector<double> carryRows(const std::vector<double> &fractional,
                              const std::array<double, 9> &cell) {
  std::vector<double> out(fractional.size());
  for (std::size_t i = 0; i < fractional.size() / 3; ++i) {
    for (int c = 0; c < 3; ++c) {
      out[3 * i + c] = fractional[3 * i] * cell[c] +
                       fractional[3 * i + 1] * cell[3 + c] +
                       fractional[3 * i + 2] * cell[6 + c];
    }
  }
  return out;
}

// splitmix64, so the atoms do not depend on a standard library's
// distribution.
std::vector<double> scatter(std::size_t count) {
  std::uint64_t state = 0x9e3779b97f4a7c15ULL;
  std::vector<double> out(3 * count);
  for (double &x : out) {
    state += 0x9e3779b97f4a7c15ULL;
    std::uint64_t z = state;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    z ^= z >> 31;
    x = static_cast<double>(z >> 11) * (1.0 / 9007199254740992.0);
  }
  return out;
}

// Strain the positions and the cell together by 1 + delta in one Voigt
// component and compare dE/d(strain)/V with the published stress.
double strainError(const rgpot::EAMPot &pot, const std::vector<double> &pos,
                   const std::vector<int> &types,
                   const std::array<double, 9> &box, const double *stress,
                   const std::vector<int> &components) {
  constexpr int rows[6] = {0, 1, 2, 1, 0, 0};
  constexpr int cols[6] = {0, 1, 2, 2, 2, 1};
  constexpr int slots[6] = {0, 4, 8, 5, 2, 1};
  const double volume = rgpot::cellVolume(box.data());
  const double step = 1e-5;
  auto carry = [](const double *source, double *target, std::size_t count,
                  const double transform[9]) {
    for (std::size_t i = 0; i < count; ++i) {
      for (int c = 0; c < 3; ++c) {
        target[3 * i + c] = source[3 * i] * transform[c] +
                            source[3 * i + 1] * transform[3 + c] +
                            source[3 * i + 2] * transform[6 + c];
      }
    }
  };
  double worst = 0.0;
  double largest = 0.0;
  for (int component : components) {
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
      std::vector<double> moved(pos.size());
      std::array<double, 9> strained{};
      carry(pos.data(), moved.data(), pos.size() / 3, transform);
      carry(box.data(), strained.data(), 3, transform);
      energies[side] = evaluate(pot, moved, types, strained, false).out.energy;
    }
    const double numeric = (energies[1] - energies[0]) / (2 * step * volume);
    worst = std::max(worst, std::abs(numeric - stress[slots[component]]));
    largest = std::max(largest, std::abs(stress[slots[component]]));
  }
  return worst / std::max(largest, 1e-8);
}

} // namespace

TEST_CASE("Cell-list EAM stress matches the strain derivative on an "
          "orthorhombic cell",
          "[eam][stress]") {
  const rgpot::EAMPot pot;
  const std::vector<int> all_components = {0, 1, 2, 3, 4, 5};
  const std::vector<int> diagonal = {0, 1, 2};

  SECTION("a cluster clear of every image: all six components") {
    const std::array<double, 9> box = {16.0, 0, 0, 0, 15.0, 0, 0, 0, 17.0};
    const std::vector<double> fractional = {
        0.11, 0.13, 0.17, 0.29, 0.31, 0.23, 0.47, 0.26, 0.38,
        0.36, 0.51, 0.44, 0.62, 0.43, 0.29, 0.55, 0.61, 0.57};
    const auto pos = carryRows(fractional, box);
    const std::vector<int> types(pos.size() / 3, 13);
    const auto plain = evaluate(pot, pos, types, box, false);
    const auto stressed = evaluate(pot, pos, types, box, true);
    REQUIRE(plain.out.has_stress == 0);
    REQUIRE(stressed.out.has_stress == 1);
    REQUIRE(bits(plain.out.energy) == bits(stressed.out.energy));
    REQUIRE(forceHash(plain.forces) == forceHash(stressed.forces));
    REQUIRE(stressed.out.stress[1] == stressed.out.stress[3]);
    REQUIRE(stressed.out.stress[2] == stressed.out.stress[6]);
    REQUIRE(stressed.out.stress[5] == stressed.out.stress[7]);
    const double error = strainError(pot, pos, types, box, stressed.out.stress,
                                     all_components);
    std::cout << "STRESS_ERROR EAM " << error << '\n';
    REQUIRE(error <= 1e-6);
  }

  SECTION("a dense periodic cell: the diagonal components") {
    const std::array<double, 9> box = {13.0, 0, 0, 0, 12.5, 0, 0, 0, 14.0};
    const auto pos = carryRows(scatter(40), box);
    const std::vector<int> types(pos.size() / 3, 13);
    const auto plain = evaluate(pot, pos, types, box, false);
    const auto stressed = evaluate(pot, pos, types, box, true);
    REQUIRE(plain.out.has_stress == 0);
    REQUIRE(stressed.out.has_stress == 1);
    REQUIRE(bits(plain.out.energy) == bits(stressed.out.energy));
    REQUIRE(forceHash(plain.forces) == forceHash(stressed.forces));
    const double error =
        strainError(pot, pos, types, box, stressed.out.stress, diagonal);
    std::cout << "STRESS_ERROR EAM periodic " << error << '\n';
    REQUIRE(error <= 1e-6);
  }
}

// The kernel has always read only the diagonal of the cell. A force call on
// any cell must keep producing those numbers, whether or not stress exists.
// The constants come from the library built without stress support.
TEST_CASE("Cell-list EAM force-only result on a triclinic cell is "
          "bit-identical to the one before stress existed",
          "[eam][stress]") {
  const rgpot::EAMPot pot;
  struct Golden {
    std::array<double, 9> box;
    std::uint64_t energy;
    std::uint64_t forces;
    bool orthorhombic;
  };
  const Golden goldens[] = {
      {{16.0, 0.7, 0.2, -0.4, 15.0, 0.5, 0.3, -0.6, 17.0},
       0x4070f8c7baccf85fULL,
       0x937b2ab197f3c62bULL,
       false},
      {{16.0, 0, 0, 2.0, 15.0, 0, 0, 0, 17.0},
       0x4062a751498bfaa9ULL,
       0x9c4b2c4ffa5a3172ULL,
       false},
      {{16.0, 0, 0, 0, 15.0, 0, 0, 0, 17.0},
       0x4075d05b7a65e3d1ULL,
       0x1616e4d9fdb0b4d0ULL,
       true},
  };
  const auto fractional = scatter(60);
  for (const auto &golden : goldens) {
    const auto pos = carryRows(fractional, golden.box);
    const std::vector<int> types(pos.size() / 3, 13);
    const auto plain = evaluate(pot, pos, types, golden.box, false);
    CHECK(bits(plain.out.energy) == golden.energy);
    CHECK(forceHash(plain.forces) == golden.forces);
    CHECK(plain.out.has_stress == 0);

    // A stress request leaves the numbers alone. On a cell the model cannot
    // differentiate it says so, instead of publishing zeros.
    const auto asked = evaluate(pot, pos, types, golden.box, true);
    CHECK(bits(asked.out.energy) == golden.energy);
    CHECK(forceHash(asked.forces) == golden.forces);
    CHECK(asked.out.has_stress == (golden.orthorhombic ? 1 : 0));
    if (!golden.orthorhombic) {
      for (double component : asked.out.stress) {
        CHECK(component == 0.0);
      }
    }
  }
}

TEST_CASE("Cell-list EAM advertises stress", "[eam][stress]") {
  const rgpot::EAMPot pot;
  CHECK(pot.caps().stress);
}
