// MIT License
// Copyright 2023--present rgpot developers
//
// Central-difference check of the virial for every in-tree potential that
// already publishes a strain derivative. The strain is the ASE right
// strain, r -> r (I+ε). stress = -virial/V.

#include <array>
#include <cmath>
#include <functional>
#include <random>
#include <string>
#include <vector>

#include <catch2/catch_all.hpp>

#include "rgpot/EAM/EAMPot.hpp"
#include "rgpot/EMT/EMTPot.hpp"
#include "rgpot/LennardJones/LJClusterPot.hpp"
#include "rgpot/LennardJones/LJPot.hpp"
#include "rgpot/Morse/MorsePot.hpp"
#include "rgpot/SuttonChen/SuttonChenScales.hpp"
#include "rgpot/Tersoff/TersoffSiC.hpp"
#include "rgpot/ZBL/ZBLPot.hpp"
#include "rgpot/stress.hpp"

#ifdef RGPOT_HAS_FORTRAN_POTS
#include "rgpot/fortran/FortranPots.hpp"
#endif

using Catch::Matchers::WithinAbs;

namespace {

struct Sample {
  std::vector<double> pos;
  std::vector<int> types;
  std::array<double, 9> box{};
};

double volumeOf(const std::array<double, 9> &box) {
  return rgpot::cellVolume(box.data());
}

/// Energy at a right strain ε_row,col = eps. Positions and the cell both
/// transform, so a periodic image moves with the cell.
template <typename EnergyAt>
double centralStress(EnergyAt &&energyAt, const Sample &sample, int row,
                     int col) {
  constexpr double h = 1e-6;
  auto at = [&](double eps) {
    Sample shifted = sample;
    const std::size_t n = sample.pos.size() / 3;
    for (std::size_t a = 0; a < n; ++a) {
      shifted.pos[3 * a + static_cast<std::size_t>(col)] +=
          eps * sample.pos[3 * a + static_cast<std::size_t>(row)];
    }
    for (int k = 0; k < 3; ++k) {
      shifted.box[3 * k + col] += eps * sample.box[3 * k + row];
    }
    return energyAt(shifted);
  };
  return (at(h) - at(-h)) / (2.0 * h * volumeOf(sample.box));
}

template <typename Force>
void checkSample(const char *name, Force &&force, const Sample &sample,
                 double tol) {
  std::vector<double> forces(sample.pos.size());
  rgpot::ForceOut out{forces.data(), 0.0, 0.0, {}, 0};
  force(sample, &out);
  REQUIRE(out.has_virial == 1);
  REQUIRE(out.has_stress == 1);
  const double volume = volumeOf(sample.box);
  const double scale = -1.0 / volume;
  for (int i = 0; i < 9; ++i) {
    REQUIRE(out.stress[i] == out.virial[i] * scale);
  }
  auto energyAt = [&](const Sample &s) {
    std::vector<double> f(s.pos.size());
    rgpot::ForceOut e{f.data(), 0.0, 0.0, {}, 0};
    force(s, &e);
    return e.energy;
  };
  for (int row = 0; row < 3; ++row) {
    for (int col = 0; col < 3; ++col) {
      const double fd = centralStress(energyAt, sample, row, col);
      CAPTURE(name, row, col, fd, out.stress[3 * row + col]);
      REQUIRE_THAT(out.stress[3 * row + col], WithinAbs(fd, tol));
    }
  }
}

Sample pairInBox(double x, double y, double z) {
  Sample s;
  s.pos = {1.0, 1.2, 0.8, 1.0 + x, 1.2 + y, 0.8 + z};
  s.types = {14, 14};
  s.box = {12, 0, 0, 0.4, 11, 0, -0.2, 0.3, 10};
  return s;
}

Sample rattledDiamond() {
  Sample s;
  const double a = 5.43;
  const double basis[8][3] = {{0, 0, 0},         {0.5, 0.5, 0},
                              {0.5, 0, 0.5},     {0, 0.5, 0.5},
                              {0.25, 0.25, 0.25}, {0.75, 0.75, 0.25},
                              {0.75, 0.25, 0.75}, {0.25, 0.75, 0.75}};
  for (const auto &b : basis) {
    s.pos.push_back(b[0] * a);
    s.pos.push_back(b[1] * a);
    s.pos.push_back(b[2] * a);
    s.types.push_back(14);
  }
  std::mt19937_64 gen(11);
  for (double &v : s.pos) {
    v += 0.08 * (static_cast<double>(gen() >> 11) * 0x1.0p-53 - 0.5);
  }
  s.box = {a, 0, 0, 0.15, a, 0, 0.05, -0.1, a};
  return s;
}

} // namespace

TEST_CASE("LJ virial matches a central strain difference", "[virial][LJ]") {
  rgpot::LJPot pot;
  REQUIRE(pot.caps().virial);
  checkSample(
      "lj",
      [&](const Sample &s, rgpot::ForceOut *out) {
        pot.forceImpl({s.pos.size() / 3, s.pos.data(), s.types.data(),
                       s.box.data()},
                      out);
      },
      pairInBox(1.3, 0.4, -0.25), 1e-6);
}

TEST_CASE("Morse virial matches a central strain difference",
          "[virial][Morse]") {
  rgpot::MorsePot pot;
  REQUIRE(pot.caps().virial);
  checkSample(
      "morse",
      [&](const Sample &s, rgpot::ForceOut *out) {
        pot.forceImpl({s.pos.size() / 3, s.pos.data(), s.types.data(),
                       s.box.data()},
                      out);
      },
      pairInBox(0.4, 2.6, 0.2), 1e-6);
}

TEST_CASE("ZBL virial matches a central strain difference", "[virial][ZBL]") {
  rgpot::ZBLPot pot;
  REQUIRE(pot.caps().virial);
  checkSample(
      "zbl",
      [&](const Sample &s, rgpot::ForceOut *out) {
        pot.forceImpl({s.pos.size() / 3, s.pos.data(), s.types.data(),
                       s.box.data()},
                      out);
      },
      pairInBox(0.9, 0.4, 0.3), 1e-6);
}

TEST_CASE("Sutton-Chen virial matches a central strain difference",
          "[virial][SuttonChen]") {
  rgpot::SuttonChenConfig cfg = rgpot::sutton_chen_platinum();
  cfg.cutoff = 6.0;
  cfg.periodic = true;
  cfg.stress = true;
  rgpot::SuttonChenPot pot(cfg);
  REQUIRE(pot.caps().virial);
  Sample s;
  const double a = 3.92;
  s.pos = {0, 0, 0, 0, a / 2, a / 2, a / 2, 0, a / 2, a / 2, a / 2, 0};
  s.types.assign(4, 78);
  s.pos[3] += 0.04;
  s.pos[7] -= 0.03;
  s.box = {a, 0, 0, 0.2, a, 0, 0.1, -0.15, a};
  checkSample(
      "sutton-chen",
      [&](const Sample &sample, rgpot::ForceOut *out) {
        pot.forceImpl({sample.pos.size() / 3, sample.pos.data(),
                       sample.types.data(), sample.box.data()},
                      out);
      },
      s, 1e-5);
}

TEST_CASE("Tersoff Si virial matches a central strain difference",
          "[virial][Tersoff]") {
  rgpot::TersoffSiC pot;
  REQUIRE(pot.caps().virial);
  checkSample(
      "tersoff",
      [&](const Sample &s, rgpot::ForceOut *out) {
        pot.forceImpl({s.pos.size() / 3, s.pos.data(), s.types.data(),
                       s.box.data()},
                      out);
      },
      rattledDiamond(), 1e-5);
}

#ifdef RGPOT_HAS_FORTRAN_POTS
TEST_CASE("Fortran SW virial matches a central strain difference",
          "[virial][SW]") {
  rgpot::fortranpots::SWPot pot;
  REQUIRE(pot.caps().virial);
  checkSample(
      "sw-fortran",
      [&](const Sample &s, rgpot::ForceOut *out) {
        pot.forceImpl({s.pos.size() / 3, s.pos.data(), s.types.data(),
                       s.box.data()},
                      out);
      },
      rattledDiamond(), 1e-5);
}
#endif

TEST_CASE("EMT virial matches a central strain difference", "[virial][EMT]") {
  rgpot::EMTPot pot;
  REQUIRE(pot.caps().virial);
  Sample s;
  s.pos = {8.1925, 8.1925, 8.1925, 10.0, 10.0, 8.1925,
           10.0,   8.1925, 10.0,   8.1925, 10.0, 10.0};
  s.types.assign(4, 29);
  s.box = {20, 0, 0, 0.4, 20, 0, 0.2, -0.3, 20};
  checkSample(
      "emt",
      [&](const Sample &sample, rgpot::ForceOut *out) {
        pot.forceImpl({sample.pos.size() / 3, sample.pos.data(),
                       sample.types.data(), sample.box.data()},
                      out);
      },
      s, 1e-5);
}

TEST_CASE("kernels without a strain derivative leave the virial unavailable",
          "[virial]") {
  double pos[6] = {0, 0, 0, 1.2, 0.1, 0};
  int types[2] = {13, 13};
  double box[9] = {20, 0, 0, 0, 20, 0, 0, 0, 20};
  double forces[6] = {};
  rgpot::LJClusterPot cluster;
  REQUIRE_FALSE(cluster.caps().virial);
  rgpot::ForceOut out{forces, 0.0, 0.0, {}, 0};
  cluster.forceImpl({2, pos, types, box}, &out);
  REQUIRE(out.has_virial == 0);

  rgpot::EAMPot eam;
  REQUIRE_FALSE(eam.caps().virial);
  rgpot::ForceOut eamOut{forces, 0.0, 0.0, {}, 0};
  eam.forceImpl({2, pos, types, box}, &eamOut);
  REQUIRE(eamOut.has_virial == 0);
}
