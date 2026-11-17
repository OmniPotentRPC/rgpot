// MIT License
// Copyright 2023--present rgpot developers

// The analytic forces of the Tersoff, Stillinger-Weber and EAM kernels
// against central differences of their energy, on the displaced 64 and 256
// atom cells.
//
// Step and budget. A central difference of two energies differs from the
// slope by two terms:
//
//   truncation  h^2 |E'''| / 6
//   rounding    c eps |E| sqrt(N) / h
//
// The energy is a sum over N atoms, so each value carries a rounding error
// of about eps |E| sqrt(N), and c = 8 covers both energies and their
// difference. |E'''| along one coordinate is about 4e2 eV/A^3 at most on
// these cells (the Tersoff Si-C bond), found by halving h where truncation
// dominates; the budget uses 6e2. The two terms balance between 2e-5 and
// 6e-5 A. h = 1e-5 A sits on the rounding side, where the budget is 7e-8
// to 3e-6 eV/A and the largest error on these cells is about a tenth of it.
//
// A pair that crosses half a box changes image, and the energy has a kink
// there. The check on the cell requires every such pair to be four steps
// away, so that no stencil spans one.

#include <cmath>
#include <cstddef>
#include <limits>

#include <catch2/catch_all.hpp>

#include "displaced_cell.hpp"

using namespace displaced;

namespace {

constexpr double kStep = 1.0e-5;
constexpr double kThirdDerivative = 6.0e2;
constexpr double kRoundingFactor = 8.0;
constexpr double kClearance = 4.0;
// Wider than every cutoff in the five models.
constexpr double kReach = 6.0;
// One coordinate in seven is differenced on the 256 atom cell, so every
// atom and every axis is sampled. The 64 atom cell is differenced whole.
constexpr std::size_t kStride = 7;

double budget(double energy, std::size_t atoms) {
  const double eps = std::numeric_limits<double>::epsilon();
  const double truncation = kStep * kStep / 6.0 * kThirdDerivative;
  const double rounding = kRoundingFactor * eps * std::abs(energy) *
                          std::sqrt(static_cast<double>(atoms)) / kStep;
  return truncation + rounding;
}

struct Worst {
  double error = 0.0;
  std::size_t coordinate = 0;
  std::size_t sampled = 0;
};

// The largest distance between the force and minus the central difference
// of the energy, over every stride-th coordinate. One object serves all the
// energies, so every call after the first reuses its neighbour list.
Worst worstError(Model model, const Cell &cell, const Result &analytic,
                 std::size_t stride) {
  Engine engine(model);
  AtomMatrix moved = cell.pos;
  Worst worst;
  for (std::size_t k = 0; k < moved.size(); k += stride) {
    double &x = moved.data()[k];
    const double keep = x;
    x = keep + kStep;
    const double up = x - keep;
    const double high = engine(moved, cell.box).energy;
    x = keep - kStep;
    const double down = keep - x;
    const double low = engine(moved, cell.box).energy;
    x = keep;
    const double slope = (high - low) / (up + down);
    const double error = std::abs(-slope - analytic.force[k]);
    ++worst.sampled;
    if (error > worst.error) {
      worst.error = error;
      worst.coordinate = k;
    }
  }
  return worst;
}

} // namespace

TEST_CASE("forces match central differences on displaced cells",
          "[force][fd][displaced]") {
  const Model model = GENERATE(from_range(kModels));
  const int n = GENERATE(64, 256);
  CAPTURE(name(model), n);
  const Threads threads;
  threads.use(1);

  const Cell cell = fcc(n);
  const Tie tie = nearestTie(cell, kReach);
  INFO("the pair " << tie.first << " and " << tie.second << " on axis "
                   << tie.axis << " is " << tie.margin
                   << " A from a half-box tie");
  REQUIRE(tie.margin > kClearance * kStep);

  const Result analytic = Engine(model)(cell.pos, cell.box);
  REQUIRE(analytic.force.size() == cell.pos.size());
  const Worst worst = worstError(model, cell, analytic, n <= 64 ? 1 : kStride);
  const double allowed = budget(analytic.energy, static_cast<std::size_t>(n));
  INFO("coordinate " << worst.coordinate << " of " << worst.sampled
                     << " sampled, force " << analytic.force[worst.coordinate]
                     << ", energy " << analytic.energy);
  REQUIRE(worst.sampled >= 64);
  REQUIRE(worst.error <= allowed);
}

TEST_CASE("the budget rejects a force that is off by 1e-4 eV/A",
          "[force][fd][displaced]") {
  const Model model = GENERATE(from_range(kModels));
  CAPTURE(name(model));
  const Threads threads;
  threads.use(1);

  const Cell cell = fcc(64);
  Result analytic = Engine(model)(cell.pos, cell.box);
  const std::size_t hit = 100;
  analytic.force[hit] += 1.0e-4;
  const Worst worst = worstError(model, cell, analytic, 1);
  REQUIRE(worst.coordinate == hit);
  REQUIRE(worst.error > 100.0 * budget(analytic.energy, 64));
}
