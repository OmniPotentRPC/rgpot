// MIT License
// Copyright 2023--present rgpot developers

// The analytic forces of the Tersoff, Stillinger-Weber and EAM kernels
// against central differences of their energy, on the displaced 64 and 256
// atom cells (plus a displaced 64-atom zincblende Si-C cell so mixed Si-C
// bonds and carbon angular terms are active).
//
// Step and budget. A central difference of two energies differs from the
// slope by two terms:
//
//   truncation  h^2 |E'''| / 6
//   rounding    c eps |E| / h
//
// In a single-coordinate central difference only the local bond terms move
// while the total energy sum over N atoms introduces at most O(eps |E|)
// rounding uncertainty in E(x + h) - E(x - h); scaling by sqrt(N) would
// inflate the budget above 1e-6 eV/A at 256 atoms. |E'''| along one
// coordinate is at most ~4.4e2 eV/A^3 on the FCC cells and ~1.2e3 eV/A^3
// on zincblende Si-C; the budget uses 1.5e3, and c = 24 covers the
// empirical rounding floor (~9 * eps |E| / h worst case on streamed EAM)
// with >2.8x headroom while remaining strictly below 1e-6 eV/A.
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
constexpr double kThirdDerivative = 1.5e3;
constexpr double kRoundingFactor = 24.0;
constexpr double kClearance = 4.0;
// Wider than every cutoff in the five models.
constexpr double kReach = 6.0;

double budget(double energy) {
  const double eps = std::numeric_limits<double>::epsilon();
  const double truncation = kStep * kStep / 6.0 * kThirdDerivative;
  const double rounding = kRoundingFactor * eps * std::abs(energy) / kStep;
  return truncation + rounding;
}

struct Worst {
  double error = 0.0;
  std::size_t coordinate = 0;
  std::size_t sampled = 0;
};

double centralDifference(const Engine &engine, const Cell &cell,
                         const std::vector<int> &z, std::size_t k) {
  AtomMatrix moved = cell.pos;
  double &x = moved.data()[k];
  const double keep = x;
  x = keep + kStep;
  const double up = x - keep;
  const double high = engine(moved, z, cell.box).energy;
  x = keep - kStep;
  const double down = keep - x;
  const double low = engine(moved, z, cell.box).energy;
  return (high - low) / (up + down);
}

// The largest distance between the force and minus the central difference
// of the energy across every coordinate. One object serves all the energies,
// so every call after the first reuses its neighbour list.
Worst worstError(const Engine &engine, const Cell &cell,
                 const std::vector<int> &z, const Result &analytic) {
  AtomMatrix moved = cell.pos;
  Worst worst;
  for (std::size_t k = 0; k < moved.size(); ++k) {
    double &x = moved.data()[k];
    const double keep = x;
    x = keep + kStep;
    const double up = x - keep;
    const double high = engine(moved, z, cell.box).energy;
    x = keep - kStep;
    const double down = keep - x;
    const double low = engine(moved, z, cell.box).energy;
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
  const std::vector<int> z = species(model, static_cast<std::size_t>(n));
  const Tie tie = nearestTie(cell, kReach);
  INFO("the pair " << tie.first << " and " << tie.second << " on axis "
                   << tie.axis << " is " << tie.margin
                   << " A from a half-box tie");
  REQUIRE(tie.margin > kClearance * kStep);

  const Engine engine(model);
  const Result analytic = engine(cell.pos, z, cell.box);
  REQUIRE(analytic.force.size() == cell.pos.size());
  const Worst worst = worstError(engine, cell, z, analytic);
  const double allowed = budget(analytic.energy);
  INFO("coordinate " << worst.coordinate << " of " << worst.sampled
                     << " sampled, force " << analytic.force[worst.coordinate]
                     << ", energy " << analytic.energy << ", budget "
                     << allowed);
  REQUIRE(worst.sampled == cell.pos.size());
  REQUIRE(worst.error <= allowed);
}

TEST_CASE("forces match central differences on zincblende Si-C and streams",
          "[force][fd][displaced]") {
  const Threads threads;
  threads.use(1);

  const Cell zb = zincblende(64);
  const std::vector<int> zb_z = zincblendeSpecies(64);
  const Tie zb_tie = nearestTie(zb, kReach);
  REQUIRE(zb_tie.margin > kClearance * kStep);
  for (bool streamed : {false, true}) {
    CAPTURE(streamed);
    const Engine engine(Model::TersoffSiC, streamed);
    const Result analytic = engine(zb.pos, zb_z, zb.box);
    const Worst worst = worstError(engine, zb, zb_z, analytic);
    const double allowed = budget(analytic.energy);
    INFO("zincblende Si-C error " << worst.error << ", budget " << allowed);
    REQUIRE(worst.sampled == zb.pos.size());
    REQUIRE(worst.error <= allowed);
  }

  const Cell cell64 = fcc(64);
  const Tie tie64 = nearestTie(cell64, kReach);
  REQUIRE(tie64.margin > kClearance * kStep);
  for (Model model : {Model::TersoffSi, Model::StillingerWeber}) {
    CAPTURE(name(model));
    const std::vector<int> z = species(model, 64);
    const Engine engine(model, true);
    const Result analytic = engine(cell64.pos, z, cell64.box);
    const Worst worst = worstError(engine, cell64, z, analytic);
    REQUIRE(worst.error <= budget(analytic.energy));
  }
}

TEST_CASE("the budget rejects a force that is off by 1e-6 eV/A",
          "[force][fd][displaced]") {
  const Model model = GENERATE(from_range(kModels));
  const int n = GENERATE(64, 256);
  CAPTURE(name(model), n);
  const Threads threads;
  threads.use(1);

  const Cell cell = fcc(n);
  const std::vector<int> z = species(model, static_cast<std::size_t>(n));
  const Engine engine(model);
  Result analytic = engine(cell.pos, z, cell.box);
  const double allowed = budget(analytic.energy);
  REQUIRE(allowed < 1.0e-6);

  const std::size_t hit = 100;
  analytic.force[hit] += 1.0e-6;
  const double slope = centralDifference(engine, cell, z, hit);
  const double error = std::abs(-slope - analytic.force[hit]);
  REQUIRE(error > allowed);
}
