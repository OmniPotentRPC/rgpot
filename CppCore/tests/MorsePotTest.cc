// MIT License
// Copyright 2023--present rgpot developers
#include <catch2/catch_all.hpp>
#include <cmath>

#include "rgpot/Morse/MorsePot.hpp"
#include "rgpot/types/AtomMatrix.hpp"

using rgpot::types::AtomMatrix;

namespace {
/// A box large enough that the minimum image convention is inert.
std::array<std::array<double, 3>, 3> free_box() {
  return {{{1000.0, 0, 0}, {0, 1000.0, 0}, {0, 0, 1000.0}}};
}
} // namespace

TEST_CASE("Morse pair at its minimum", "[MorsePot]") {
  // Two points at r0 sit at the bottom of the well: energy is minus the depth
  // and the force vanishes. This fixes depth, position and the sign of the
  // derivative at once.
  auto pot = rgpot::MorsePot();
  AtomMatrix positions{{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}};
  std::vector<int> atmtypes{1, 1};
  auto [energy, forces] = pot(positions, atmtypes, free_box());

  REQUIRE(energy == Catch::Approx(-1.0).margin(1e-9));
  for (int i = 0; i < 2; ++i) {
    for (int k = 0; k < 3; ++k) {
      REQUIRE(forces(i, k) == Catch::Approx(0.0).margin(1e-8));
    }
  }
}

TEST_CASE("Morse forces agree with a finite difference", "[MorsePot]") {
  // The gradient is the part a published energy table cannot check, so it is
  // checked against the energy it is supposed to differentiate.
  auto pot = rgpot::MorsePot();
  AtomMatrix positions{
      {0.0, 0.0, 0.0}, {1.07, 0.11, -0.05}, {0.42, 0.96, 0.13}, {-0.3, 0.5, 0.9}};
  std::vector<int> atmtypes{1, 1, 1, 1};
  auto [energy, forces] = pot(positions, atmtypes, free_box());
  (void)energy;

  const double h = 1e-6;
  for (int i = 0; i < positions.rows(); ++i) {
    for (int k = 0; k < 3; ++k) {
      AtomMatrix plus = positions;
      AtomMatrix minus = positions;
      plus(i, k) += h;
      minus(i, k) -= h;
      auto [ep, fp] = pot(plus, atmtypes, free_box());
      auto [em, fm] = pot(minus, atmtypes, free_box());
      (void)fp;
      (void)fm;
      const double numeric = -(ep - em) / (2 * h);
      REQUIRE(forces(i, k) == Catch::Approx(numeric).margin(1e-5));
    }
  }
}

TEST_CASE("Morse range parameter narrows the well without moving it",
          "[MorsePot]") {
  // rho0 is the whole point of this potential: it changes the range without
  // changing where the minimum sits or how deep it is.
  AtomMatrix at_min{{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}};
  AtomMatrix stretched{{0.0, 0.0, 0.0}, {1.4, 0.0, 0.0}};
  std::vector<int> atmtypes{1, 1};

  auto soft = rgpot::MorsePot(1.0, 3.0, 1.0, 20.0);
  auto hard = rgpot::MorsePot(1.0, 14.0, 1.0, 20.0);

  auto [e_soft_min, f0] = soft(at_min, atmtypes, free_box());
  auto [e_hard_min, f1] = hard(at_min, atmtypes, free_box());
  (void)f0;
  (void)f1;
  REQUIRE(e_soft_min == Catch::Approx(-1.0).margin(1e-9));
  REQUIRE(e_hard_min == Catch::Approx(-1.0).margin(1e-9));

  auto [e_soft_far, f2] = soft(stretched, atmtypes, free_box());
  auto [e_hard_far, f3] = hard(stretched, atmtypes, free_box());
  (void)f2;
  (void)f3;
  // The short-ranged one has all but forgotten the other point by 1.4 r0.
  REQUIRE(e_hard_far > e_soft_far);
  REQUIRE(std::abs(e_hard_far) < 0.02);
}

TEST_CASE("Morse cutoff shift makes the pair term vanish at the cutoff",
          "[MorsePot]") {
  // The same defect that was fixed in the Lennard-Jones potential: an
  // uninitialised or unapplied shift leaves a step at the cutoff.
  const double rc = 1.8;
  auto pot = rgpot::MorsePot(1.0, 6.0, 1.0, rc);
  std::vector<int> atmtypes{1, 1};

  AtomMatrix just_inside{{0.0, 0.0, 0.0}, {rc - 1e-7, 0.0, 0.0}};
  auto [e_in, f] = pot(just_inside, atmtypes, free_box());
  (void)f;
  REQUIRE(e_in == Catch::Approx(0.0).margin(1e-6));
}
