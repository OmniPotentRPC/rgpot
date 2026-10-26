// MIT License
// Copyright 2023--present rgpot developers
//
// Unshifted Lennard-Jones, Morse, binary Lennard-Jones, and TIP4P site
// checks. The unit Lennard-Jones dimer is an integer, so it pins the
// expression tree. Forces are checked by finite differences.

#include <cmath>
#include <vector>

#include <catch2/catch_all.hpp>

#include "rgpot/ReducedPair/ReducedPair.hpp"

using Catch::Matchers::WithinAbs;

namespace {

double lj_numeric(const std::vector<double> &x, int coord) {
  const double h = 1.0e-6;
  std::vector<double> up = x;
  std::vector<double> down = x;
  up[static_cast<std::size_t>(coord)] += h;
  down[static_cast<std::size_t>(coord)] -= h;
  double eu = 0.0;
  double ed = 0.0;
  std::vector<double> g(x.size(), 0.0);
  const auto n = static_cast<int32_t>(x.size() / 3);
  rgpot_lj_gradient(1.0, 1.0, 0.0, n, up.data(), &eu, g.data());
  rgpot_lj_gradient(1.0, 1.0, 0.0, n, down.data(), &ed, g.data());
  return (eu - ed) / (2.0 * h);
}

} // namespace

TEST_CASE("unit Lennard-Jones dimer is exact at r = 1 and at the minimum",
          "[ReducedPair]") {
  const double pos_r1[6] = {0.0, 0.0, 0.0, 1.0, 0.0, 0.0};
  double energy = 0.0;
  double gradient[6] = {};
  REQUIRE(rgpot_lj_gradient(1.0, 1.0, 0.0, 2, pos_r1, &energy, gradient) == 0);
  REQUIRE_THAT(energy, WithinAbs(0.0, 0.0));
  REQUIRE_THAT(gradient[0], WithinAbs(24.0, 0.0));
  REQUIRE_THAT(gradient[3], WithinAbs(-24.0, 0.0));

  const double rmin = std::pow(2.0, 1.0 / 6.0);
  const double pos_min[6] = {0.0, 0.0, 0.0, rmin, 0.0, 0.0};
  REQUIRE(rgpot_lj_gradient(1.0, 1.0, 0.0, 2, pos_min, &energy, gradient) == 0);
  REQUIRE_THAT(energy, WithinAbs(-1.0, 1.0e-15));
  REQUIRE_THAT(gradient[0], WithinAbs(0.0, 1.0e-12));
  REQUIRE_THAT(gradient[3], WithinAbs(0.0, 1.0e-12));
}

TEST_CASE("unit Lennard-Jones gradient matches a central difference",
          "[ReducedPair]") {
  const std::vector<double> x = {0.1,  -0.2, 0.3, 1.2, 0.4,  -0.5,
                                 -0.7, 1.1,  0.2, 0.5, -0.4, 1.3};
  double energy = 0.0;
  std::vector<double> gradient(x.size(), 0.0);
  REQUIRE(rgpot_lj_gradient(1.0, 1.0, 0.0, 4, x.data(), &energy,
                            gradient.data()) == 0);
  for (int coord = 0; coord < 12; ++coord) {
    const double numeric = lj_numeric(x, coord);
    REQUIRE_THAT(gradient[static_cast<std::size_t>(coord)],
                 WithinAbs(numeric, 1.0e-4 * (1.0 + std::abs(numeric))));
  }
}

TEST_CASE("Lennard-Jones pair curvature matches the gradient",
          "[ReducedPair]") {
  const double r2 = 1.7;
  double energy = 0.0;
  double coef = 0.0;
  double curv = 0.0;
  REQUIRE(rgpot_lj_pair(1.0, 1.0, r2, &energy, &coef, &curv) == 0);
  const double h = 1.0e-6;
  const double rp = std::sqrt(r2) + h;
  const double rm = std::sqrt(r2) - h;
  double ep = 0.0;
  double em = 0.0;
  double cp = 0.0;
  double cm = 0.0;
  double unused = 0.0;
  rgpot_lj_pair(1.0, 1.0, rp * rp, &ep, &cp, &unused);
  rgpot_lj_pair(1.0, 1.0, rm * rm, &em, &cm, &unused);
  const double dedr = (ep - em) / (2.0 * h);
  REQUIRE_THAT(coef * std::sqrt(r2), WithinAbs(dedr, 1.0e-6));
  const double d2 = ((cp * rp) - (cm * rm)) / (2.0 * h);
  REQUIRE_THAT(curv, WithinAbs(d2, 1.0e-4));
  const double listed[1] = {r2};
  double batch_energy = 0.0;
  double batch_coef = 0.0;
  double batch_curv = 0.0;
  REQUIRE(rgpot_lj_curvature_pairs(1.0, 1.0, 1, listed, &batch_energy,
                                   &batch_coef, &batch_curv) == 0);
  REQUIRE_THAT(batch_energy, WithinAbs(energy, 0.0));
  REQUIRE_THAT(batch_coef, WithinAbs(coef, 0.0));
  REQUIRE_THAT(batch_curv, WithinAbs(curv, 0.0));
}

TEST_CASE("a cutoff drops the distant pair", "[ReducedPair]") {
  const double pos[6] = {0.0, 0.0, 0.0, 5.0, 0.0, 0.0};
  double energy = 0.0;
  double gradient[6] = {};
  REQUIRE(rgpot_lj_gradient(1.0, 1.0, 3.0, 2, pos, &energy, gradient) == 0);
  REQUIRE_THAT(energy, WithinAbs(0.0, 0.0));
  REQUIRE(rgpot_lj_gradient(1.0, 1.0, 0.0, 2, pos, &energy, gradient) == 0);
  REQUIRE(energy != 0.0);
}

TEST_CASE("Morse dimer sits at depth 1 with no force", "[ReducedPair]") {
  const double pos[6] = {0.0, 0.0, 0.0, 1.0, 0.0, 0.0};
  double energy = 0.0;
  double gradient[6] = {};
  REQUIRE(rgpot_morse_gradient(1.0, 6.0, 1.0, 0.0, 2, pos, &energy, gradient) ==
          0);
  REQUIRE_THAT(energy, WithinAbs(-1.0, 0.0));
  REQUIRE_THAT(gradient[0], WithinAbs(0.0, 0.0));
  double pair_e = 0.0;
  double coef = 0.0;
  double curv = 0.0;
  REQUIRE(rgpot_morse_pair(1.0, 6.0, 1.0, 0.64, &pair_e, &coef, &curv) == 0);
  const double r = 0.8;
  const double a = std::exp(6.0 * (1.0 - r));
  REQUIRE_THAT(pair_e, WithinAbs(a * (a - 2.0), 0.0));
  REQUIRE_THAT(coef, WithinAbs(2.0 * 6.0 * (a - a * a) / r, 0.0));
  REQUIRE_THAT(curv, WithinAbs(2.0 * 36.0 * (2.0 * a * a - a), 0.0));
}

TEST_CASE("equal binary parameters match the unit cluster", "[ReducedPair]") {
  const double pos[12] = {0.2,  0.0, 0.1, 1.1, -0.3, 0.2,
                          -0.4, 0.8, 0.5, 0.6, -0.7, 1.2};
  const double eps[3] = {1.0, 1.0, 1.0};
  const double sig[3] = {1.0, 1.0, 1.0};
  double binary = 0.0;
  double unit = 0.0;
  double gb[12] = {};
  double gu[12] = {};
  REQUIRE(rgpot_binary_lj_gradient(2, eps, sig, 0.0, 4, pos, &binary, gb) == 0);
  REQUIRE(rgpot_lj_gradient(1.0, 1.0, 0.0, 4, pos, &unit, gu) == 0);
  REQUIRE_THAT(binary, WithinAbs(unit, 0.0));
  for (int i = 0; i < 12; ++i) {
    REQUIRE_THAT(gb[i], WithinAbs(gu[i], 0.0));
  }
}

TEST_CASE("atom energies sum to twice the cluster energy", "[ReducedPair]") {
  const double pos[9] = {0.0, 0.0, 0.0, 1.3, 0.1, -0.2, -0.4, 1.2, 0.3};
  double energy = 0.0;
  double gradient[9] = {};
  double atoms[3] = {};
  REQUIRE(rgpot_lj_gradient(1.0, 1.0, 0.0, 3, pos, &energy, gradient) == 0);
  REQUIRE(rgpot_lj_atom_energies(1.0, 1.0, 3, pos, atoms) == 0);
  REQUIRE_THAT(atoms[0] + atoms[1] + atoms[2], WithinAbs(2.0 * energy, 0.0));
}

TEST_CASE("bad arguments are rejected", "[ReducedPair]") {
  double energy = 0.0;
  REQUIRE(rgpot_lj_gradient(1.0, 1.0, 0.0, 2, nullptr, &energy, nullptr) == 1);
  REQUIRE(rgpot_morse_gradient(1.0, 6.0, 0.0, 0.0, 0, nullptr, &energy,
                               nullptr) == 1);
}
