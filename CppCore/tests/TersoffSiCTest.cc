// MIT License
// Copyright 2023--present rgpot developers
//
// Tersoff Si-C, J. Tersoff, Phys. Rev. B 39, 5566 (1989),
// doi:10.1103/PhysRevB.39.5566. The erratum Phys. Rev. B 41, 3248 (1990),
// doi:10.1103/PhysRevB.41.3248.2, sets carbon B to 346.74 eV and d to
// 4.3484. Carbon beta is dimensionless and is not an erratum value.
// The diamond energies are the cohesive energies of that fit: 4.63 eV
// for silicon and 7.37 eV for carbon. The digits below are the closed
// form at a0 = 5.432 A (Si) and a0 = 3.5656 A (C), which are the lattice
// constants tabulated for this parameter set.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <vector>

#include <catch2/catch_all.hpp>

#include "rgpot/Tersoff/TersoffSiC.hpp"
#include "rgpot/nlist/HalfShell.hpp"
#include "rgpot/types/AtomMatrix.hpp"

using Catch::Matchers::WithinAbs;
using rgpot::TersoffSiC;
using rgpot::types::AtomMatrix;

namespace {

struct Homo {
  double big_a, big_b, lambda1, lambda2, beta, n, c, d, h;
};

constexpr Homo kSi = {1830.8,  471.18, 2.4799, 1.7322,  1.1e-6,
                      0.78734, 100390, 16.217, -0.59825};
constexpr Homo kC = {1393.6,  346.74, 3.4879, 2.2119,  1.5724e-7,
                     0.72751, 38049,  4.3484, -0.57058};

double gOf(const Homo &sp, double cth) {
  const double c2 = sp.c * sp.c;
  const double d2 = sp.d * sp.d;
  const double gap = sp.h - cth;
  const double den = d2 + gap * gap;
  return 1.0 + c2 / d2 - c2 / den;
}

/// Per-atom diamond energy. Four neighbours, fc = 1, zeta = 3 g(cos).
double diamondAtom(const Homo &sp, double a0) {
  const double r = a0 * std::sqrt(3.0) / 4.0;
  const double zeta = 3.0 * gOf(sp, -1.0 / 3.0);
  const double b = std::pow(
      1.0 + std::pow(sp.beta, sp.n) * std::pow(zeta, sp.n), -0.5 / sp.n);
  const double f_r = sp.big_a * std::exp(-sp.lambda1 * r);
  const double f_a = -sp.big_b * std::exp(-sp.lambda2 * r);
  return 2.0 * (f_r + b * f_a);
}

std::array<std::array<double, 3>, 3> cubic(double a) {
  return {{{a, 0.0, 0.0}, {0.0, a, 0.0}, {0.0, 0.0, a}}};
}

std::vector<double> diamond(int nrep, double a0) {
  const double basis[8][3] = {{0.0, 0.0, 0.0},    {0.5, 0.5, 0.0},
                              {0.5, 0.0, 0.5},    {0.0, 0.5, 0.5},
                              {0.25, 0.25, 0.25}, {0.75, 0.75, 0.25},
                              {0.75, 0.25, 0.75}, {0.25, 0.75, 0.75}};
  std::vector<double> pos;
  for (int ix = 0; ix < nrep; ++ix) {
    for (int iy = 0; iy < nrep; ++iy) {
      for (int iz = 0; iz < nrep; ++iz) {
        for (const auto &b : basis) {
          pos.push_back((ix + b[0]) * a0);
          pos.push_back((iy + b[1]) * a0);
          pos.push_back((iz + b[2]) * a0);
        }
      }
    }
  }
  return pos;
}

double energyOf(TersoffSiC &pot, const std::vector<double> &pos, int z,
                double box_a) {
  const std::size_t n = pos.size() / 3;
  AtomMatrix positions(n, 3);
  std::copy(pos.begin(), pos.end(), positions.data());
  const std::vector<int> types(n, z);
  auto [energy, forces, variance] = pot(positions, types, cubic(box_a));
  (void)variance;
  // The tabulated lattice constant is not the exact minimum, so individual
  // forces need not vanish. Pair forces still sum to zero.
  for (size_t col = 0; col < 3; ++col) {
    double sum = 0.0;
    for (size_t i = 0; i < forces.rows(); ++i) {
      sum += forces(i, col);
    }
    REQUIRE_THAT(sum, WithinAbs(0.0, 1e-8));
  }
  return energy;
}

} // namespace

TEST_CASE("Tersoff 1989 diamond cohesive energies", "[TersoffSiC]") {
  TersoffSiC pot;
  const double si = diamondAtom(kSi, 5.432);
  const double c = diamondAtom(kC, 3.5656);
  // Closed form at the tabulated lattice constants.
  REQUIRE_THAT(si, WithinAbs(-4.629595012655102, 1e-9));
  REQUIRE_THAT(c, WithinAbs(-7.370514205985597, 1e-9));

  // Eight-atom cells: the second neighbour stays outside the outer cutoff
  // after the minimum-image fold, so the crystal has four neighbours.
  const double e_si = energyOf(pot, diamond(1, 5.432), 14, 5.432);
  const double e_c = energyOf(pot, diamond(1, 3.5656), 6, 3.5656);
  REQUIRE_THAT(e_si / 8.0, WithinAbs(si, 1e-8));
  REQUIRE_THAT(e_c / 8.0, WithinAbs(c, 1e-8));

  const auto big = diamond(3, 5.432);
  double e_big = 0.0;
  std::vector<double> grad(big.size());
  std::vector<int> types(big.size() / 3, 14);
  const double box[9] = {3 * 5.432, 0, 0, 0, 3 * 5.432, 0, 0, 0, 3 * 5.432};
  pot.energyAndGradient(big.data(), types.data(), box, types.size(), &e_big,
                        grad.data());
  REQUIRE_THAT(e_big, WithinAbs(27.0 * e_si, 1e-6));
  REQUIRE(pot.get_type() == rgpot::PotType::TersoffSi);
  REQUIRE(pot.paramsKey() == TersoffSiC{}.paramsKey());
}

TEST_CASE("Tersoff 1989 gradient matches a finite difference", "[TersoffSiC]") {
  const double xyz[4][3] = {{0.0, 0.0, 0.0},
                            {2.30, 0.15, 0.05},
                            {0.40, 2.25, 0.20},
                            {1.15, 0.90, 1.40}};
  const int types[4] = {14, 14, 14, 6};
  std::vector<double> pos(12);
  for (int i = 0; i < 4; ++i) {
    pos[3 * i] = xyz[i][0];
    pos[3 * i + 1] = xyz[i][1];
    pos[3 * i + 2] = xyz[i][2];
  }
  const double box[9] = {20, 0, 0, 0, 20, 0, 0, 0, 20};
  TersoffSiC pot;
  double energy = 0.0;
  std::vector<double> grad(12);
  pot.energyAndGradient(pos.data(), types, box, 4, &energy, grad.data());

  AtomMatrix positions(4, 3);
  std::copy(pos.begin(), pos.end(), positions.data());
  auto [e_call, forces, variance] =
      pot(positions, std::vector<int>{14, 14, 14, 6}, cubic(20.0));
  (void)variance;
  REQUIRE_THAT(e_call, WithinAbs(energy, 1e-12));
  for (int i = 0; i < 12; ++i) {
    REQUIRE_THAT(grad[static_cast<std::size_t>(i)],
                 WithinAbs(-forces.data()[i], 1e-12));
  }

  const double h = 1e-6;
  for (int k = 0; k < 12; ++k) {
    pos[static_cast<std::size_t>(k)] += h;
    double ep = 0.0;
    std::vector<double> scratch(12);
    pot.energyAndGradient(pos.data(), types, box, 4, &ep, scratch.data());
    pos[static_cast<std::size_t>(k)] -= 2.0 * h;
    double em = 0.0;
    pot.energyAndGradient(pos.data(), types, box, 4, &em, scratch.data());
    pos[static_cast<std::size_t>(k)] += h;
    REQUIRE_THAT(grad[static_cast<std::size_t>(k)],
                 WithinAbs((ep - em) / (2.0 * h), 2e-6));
  }
  double net[3] = {};
  for (int i = 0; i < 4; ++i) {
    net[0] += grad[3 * i];
    net[1] += grad[3 * i + 1];
    net[2] += grad[3 * i + 2];
  }
  for (double c : net) {
    REQUIRE_THAT(c, WithinAbs(0.0, 1e-8));
  }
}

TEST_CASE("Tersoff 1989 batch and periodic image", "[TersoffSiC]") {
  TersoffSiC pot;
  const int types[2] = {14, 6};
  const double box[9] = {30, 0, 0, 0, 30, 0, 0, 0, 30};
  const double pos[12] = {0, 0, 0, 1.9, 0, 0, 2, 2, 2, 2, 3.8, 2};
  double energies[2] = {};
  double grads[12];
  pot.energyAndGradientBatch(2, pos, types, box, 2, energies, grads);
  double e0 = 0.0;
  double e1 = 0.0;
  double g0[6];
  double g1[6];
  pot.energyAndGradient(pos, types, box, 2, &e0, g0);
  pot.energyAndGradient(pos + 6, types, box, 2, &e1, g1);
  REQUIRE_THAT(energies[0], WithinAbs(e0, 0.0));
  REQUIRE_THAT(energies[1], WithinAbs(e1, 0.0));

  const double L = 12.0;
  const double cell[9] = {L, 0, 0, 0, L, 0, 0, 0, L};
  const double centered[6] = {0, 0, 0, 2.2, 0, 0};
  const double wrapped[6] = {0.4, 0, 0, 0.4 + 2.2 - L, 0, 0};
  const int si[2] = {14, 14};
  pot.energyAndGradient(centered, si, cell, 2, &e0, g0);
  pot.energyAndGradient(wrapped, si, cell, 2, &e1, g1);
  REQUIRE_THAT(e1, WithinAbs(e0, 1e-10));

  const double skew[9] = {30, 0, 0, 3, 30, 0, 0, 1, 30};
  double e_skew = 0.0;
  pot.energyAndGradient(centered, si, skew, 2, &e_skew, g0);
  REQUIRE_THAT(e_skew, WithinAbs(e0, 1e-10));

  const int bad[2] = {14, 8};
  REQUIRE_THROWS_AS(pot.energyAndGradient(centered, bad, cell, 2, &e0, g0),
                    std::invalid_argument);
}

TEST_CASE("Tersoff 1989 held list matches a rebuild", "[TersoffSiC]") {
  const auto pos0 = diamond(3, 5.432);
  std::vector<double> pos = pos0;
  pos[0] += 0.07;
  pos[1] -= 0.04;
  pos[2] += 0.02;
  const std::size_t n = pos.size() / 3;
  std::vector<int> types(n, 14);
  const double box[9] = {3 * 5.432, 0, 0, 0, 3 * 5.432, 0, 0, 0, 3 * 5.432};
  TersoffSiC pot;
  double e0 = 0.0;
  std::vector<double> g0(pos.size());
  pot.energyAndGradient(pos.data(), types.data(), box, n, &e0, g0.data());
  rgpot::nlist::NeighborList held;
  held.skin = 0.4;
  double e1 = 0.0;
  std::vector<double> g1(pos.size());
  pot.energyAndGradient(pos.data(), types.data(), box, n, &e1, g1.data(),
                        &held);
  REQUIRE(std::memcmp(&e0, &e1, sizeof(double)) == 0);
  REQUIRE(std::memcmp(g0.data(), g1.data(), g0.size() * sizeof(double)) == 0);
  pot.energyAndGradient(pos.data(), types.data(), box, n, &e1, g1.data(),
                        &held);
  REQUIRE(std::memcmp(&e0, &e1, sizeof(double)) == 0);
  REQUIRE(std::memcmp(g0.data(), g1.data(), g0.size() * sizeof(double)) == 0);
}
