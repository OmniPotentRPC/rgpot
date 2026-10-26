// MIT License
// Copyright 2023--present rgpot developers
//
// Sutton-Chen checks: an analytic dimer, a finite-difference force, and
// the Doye and Wales 10-8 cluster energies.

#include <array>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <catch2/catch_all.hpp>

#include "rgpot/SuttonChen/SuttonChenPot.hpp"
#include "rgpot/SuttonChen/SuttonChenScales.hpp"
#include "rgpot/types/AtomMatrix.hpp"

using Catch::Matchers::WithinAbs;
using rgpot::types::AtomMatrix;

namespace {

AtomMatrix load_points(const std::string &path, double *epsilon, double *c,
                       double *a, int *nexp, int *mexp) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("cannot open " + path);
  }
  double nraw = 0.0;
  double mraw = 0.0;
  in >> *epsilon >> *c >> *a >> nraw >> mraw;
  *nexp = static_cast<int>(nraw);
  *mexp = static_cast<int>(mraw);
  std::vector<double> xyz;
  double x = 0.0;
  double y = 0.0;
  double z = 0.0;
  while (in >> x >> y >> z) {
    xyz.push_back(x);
    xyz.push_back(y);
    xyz.push_back(z);
  }
  AtomMatrix pos(xyz.size() / 3, 3);
  for (size_t i = 0; i < xyz.size(); ++i) {
    pos(i / 3, i % 3) = xyz[i];
  }
  return pos;
}

std::array<std::array<double, 3>, 3> cube(double length) {
  return {{{length, 0.0, 0.0}, {0.0, length, 0.0}, {0.0, 0.0, length}}};
}

/// Independent dimer energy and dE/dr. Not the library kernel.
void dimer_oracle(double r, double epsilon, double c, double a, int nexp,
                  int mexp, double *energy, double *dEdr) {
  const double s = a / r;
  const double rep = std::pow(s, nexp);
  const double phi = std::pow(s, mexp);
  *energy = epsilon * rep - 2.0 * epsilon * c * std::sqrt(phi);
  *dEdr = -epsilon * nexp * rep / r +
          epsilon * c * mexp * std::sqrt(phi) / r;
}

} // namespace

TEST_CASE("Sutton-Chen dimer matches an independent derivative",
          "[SuttonChen]") {
  const rgpot::SuttonChenConfig cfg{.cutoff = 0.0};
  rgpot::SuttonChenPot pot(cfg);
  const double r = 0.8;
  const AtomMatrix pos{{0.0, 0.0, 0.0}, {r, 0.0, 0.0}};
  const std::vector<int> types{78, 78};
  auto [energy, forces, variance] = pot(pos, types, cube(40.0));
  (void)variance;

  double expect_e = 0.0;
  double dEdr = 0.0;
  dimer_oracle(r, cfg.epsilon, cfg.c, cfg.a, cfg.n, cfg.m, &expect_e, &dEdr);
  REQUIRE_THAT(energy, WithinAbs(expect_e, 1e-9));
  // Atom 0 sits at the origin and atom 1 at +r. Attraction pushes atom 0
  // toward +x, so F_0x = dE/dr.
  REQUIRE_THAT(forces(0, 0), WithinAbs(dEdr, 1e-8));
  REQUIRE_THAT(forces(1, 0), WithinAbs(-dEdr, 1e-8));
  REQUIRE_THAT(forces(0, 1), WithinAbs(0.0, 1e-12));
  REQUIRE_THAT(forces(0, 2), WithinAbs(0.0, 1e-12));
}

TEST_CASE("Sutton-Chen forces match a central difference", "[SuttonChen]") {
  rgpot::SuttonChenPot pot(rgpot::SuttonChenConfig{.cutoff = 0.0});
  const int n = 5;
  AtomMatrix pos(static_cast<size_t>(n), 3);
  for (int i = 0; i < n; ++i) {
    const double t = 0.37 * i;
    pos(static_cast<size_t>(i), 0) = 0.55 * std::cos(t);
    pos(static_cast<size_t>(i), 1) = 0.55 * std::sin(t);
    pos(static_cast<size_t>(i), 2) = 0.21 * (i % 3);
  }
  const std::vector<int> types(static_cast<size_t>(n), 78);
  const auto box = cube(30.0);
  auto [energy, forces, variance] = pot(pos, types, box);
  (void)energy;
  (void)variance;

  const double h = 1e-6;
  for (int i = 0; i < n; ++i) {
    for (int k = 0; k < 3; ++k) {
      AtomMatrix plus = pos;
      AtomMatrix minus = pos;
      plus(static_cast<size_t>(i), static_cast<size_t>(k)) += h;
      minus(static_cast<size_t>(i), static_cast<size_t>(k)) -= h;
      const double e_plus = std::get<0>(pot(plus, types, box));
      const double e_minus = std::get<0>(pot(minus, types, box));
      const double numeric = (e_plus - e_minus) / (2.0 * h);
      const double analytic = -forces(static_cast<size_t>(i),
                                      static_cast<size_t>(k));
      REQUIRE_THAT(analytic, WithinAbs(numeric, 1e-4 * (1.0 + std::abs(numeric))));
    }
  }
}

TEST_CASE("Sutton-Chen reproduces the Doye 10-8 cluster energies",
          "[SuttonChen]") {
  const char *files[] = {"CppCore/tests/data/sc108/pt38",
                         "CppCore/tests/data/sc108/pt75"};
  const double published[] = {-10117.2454, -20431.3452};
  for (int which = 0; which < 2; ++which) {
    double epsilon = 0.0;
    double c = 0.0;
    double a = 0.0;
    int nexp = 0;
    int mexp = 0;
    const AtomMatrix pos =
        load_points(files[which], &epsilon, &c, &a, &nexp, &mexp);
    REQUIRE(epsilon == 1.0);
    REQUIRE(c == 34.408);
    REQUIRE(a == 1.0);
    REQUIRE(nexp == 10);
    REQUIRE(mexp == 8);

    const rgpot::SuttonChenConfig cfg{
        .epsilon = epsilon, .c = c, .a = a, .n = nexp, .m = mexp, .cutoff = 40.0};
    rgpot::SuttonChenPot pot(cfg);
    const std::vector<int> types(pos.rows(), 78);
    auto [energy, forces, variance] = pot(pos, types, cube(80.0));
    (void)variance;
    // The published tables print four digits. The geometry is the
    // deposited point file for that row.
    REQUIRE_THAT(energy, WithinAbs(published[which], 5e-5));

    rgpot::SuttonChenPot full(rgpot::SuttonChenConfig{
        .epsilon = epsilon, .c = c, .a = a, .n = nexp, .m = mexp, .cutoff = 0.0});
    const double energy_full = std::get<0>(full(pos, types, cube(80.0)));
    REQUIRE_THAT(energy_full, WithinAbs(energy, 1e-9));

    double net[3] = {0.0, 0.0, 0.0};
    for (size_t i = 0; i < forces.rows(); ++i) {
      net[0] += forces(i, 0);
      net[1] += forces(i, 1);
      net[2] += forces(i, 2);
    }
    for (double component : net) {
      REQUIRE_THAT(component, WithinAbs(0.0, 1e-6));
    }
  }
}

TEST_CASE("periodic Sutton-Chen folds a pair and agrees in a wide cell",
          "[SuttonChen]") {
  const AtomMatrix pos{{0.1, 0.0, 0.0}, {4.9, 0.0, 0.0}};
  const std::vector<int> types{78, 78};
  rgpot::SuttonChenPot cluster(rgpot::SuttonChenConfig{.cutoff = 3.0});
  rgpot::SuttonChenPot periodic(
      rgpot::SuttonChenConfig{.cutoff = 3.0, .periodic = true});
  const double e_cluster = std::get<0>(cluster(pos, types, cube(5.0)));
  const double e_periodic = std::get<0>(periodic(pos, types, cube(5.0)));
  // Direct separation is 4.8, outside the cutoff. The periodic image is
  // 0.2. That image is on the repulsive wall, so the energy is large
  // and positive, and it matches an unwrapped pair at the same distance.
  REQUIRE_THAT(e_cluster, WithinAbs(0.0, 1e-15));
  const AtomMatrix close{{0.0, 0.0, 0.0}, {0.2, 0.0, 0.0}};
  const double e_close = std::get<0>(cluster(close, types, cube(40.0)));
  REQUIRE_THAT(e_periodic, WithinAbs(e_close, 1e-6));

  const AtomMatrix wide{{0.0, 0.0, 0.0}, {0.9, 0.2, -0.1}};
  const auto box = cube(40.0);
  const double e_free = std::get<0>(cluster(wide, types, box));
  const double e_boxed = std::get<0>(periodic(wide, types, box));
  REQUIRE_THAT(e_boxed, WithinAbs(e_free, 1e-9));
}

TEST_CASE("the C entry point matches the C++ potential", "[SuttonChen]") {
  const rgpot_sc_config cfg{1.0, 34.408, 1.0, 10, 8, 40.0, 0};
  const double pos[] = {0.0, 0.0, 0.0, 0.7, 0.1, -0.2, -0.4, 0.5, 0.3};
  const double box[] = {20, 0, 0, 0, 20, 0, 0, 0, 20};
  double energy = 0.0;
  double forces[9] = {};
  REQUIRE(rgpot_sutton_chen_energy_forces(&cfg, 3, pos, box, &energy, forces) == 0);

  AtomMatrix m(3, 3);
  for (int i = 0; i < 9; ++i) {
    m(static_cast<size_t>(i / 3), static_cast<size_t>(i % 3)) = pos[i];
  }
  rgpot::SuttonChenPot pot;
  auto [e2, f2, v2] = pot(m, std::vector<int>{78, 78, 78}, cube(20.0));
  (void)v2;
  REQUIRE_THAT(energy, WithinAbs(e2, 1e-12));
  for (int i = 0; i < 3; ++i) {
    for (int k = 0; k < 3; ++k) {
      REQUIRE_THAT(forces[3 * i + k], WithinAbs(f2(static_cast<size_t>(i),
                                                   static_cast<size_t>(k)),
                                                1e-12));
    }
  }
}

TEST_CASE("Sutton-Chen parameter fingerprint tracks the config",
          "[SuttonChen]") {
  rgpot::SuttonChenPot defaults;
  rgpot::SuttonChenPot same{rgpot::SuttonChenConfig{}};
  rgpot::SuttonChenPot other{rgpot::SuttonChenConfig{.c = 39.432, .n = 9, .m = 6}};
  REQUIRE(defaults.paramsKey() == same.paramsKey());
  REQUIRE(defaults.paramsKey() != other.paramsKey());
  REQUIRE(defaults.get_type() == rgpot::PotType::SuttonChen);
  REQUIRE_THROWS(rgpot::SuttonChenPot(
      rgpot::SuttonChenConfig{.cutoff = 0.0, .periodic = true}));
}

TEST_CASE("Sutton-Chen bulk scales and the gold and platinum rows",
          "[SuttonChen]") {
  const rgpot::SuttonChenConfig reduced;
  const auto scales = rgpot::sutton_chen_scales(reduced);
  REQUIRE(scales.nearest_neighbor > 0.5);
  REQUIRE(scales.nearest_neighbor < 0.6);
  REQUIRE(scales.well_depth > 200.0);
  const double here =
      rgpot::sutton_chen_fcc_energy(reduced, scales.nearest_neighbor);
  const double left = rgpot::sutton_chen_fcc_energy(
      reduced, scales.nearest_neighbor * (1.0 - 1.0e-4));
  const double right = rgpot::sutton_chen_fcc_energy(
      reduced, scales.nearest_neighbor * (1.0 + 1.0e-4));
  REQUIRE(here < left);
  REQUIRE(here < right);
  REQUIRE_THAT(here, WithinAbs(-scales.well_depth, 1e-9));

  const auto pt = rgpot::sutton_chen_platinum();
  const auto au = rgpot::sutton_chen_gold();
  REQUIRE(pt.epsilon == 1.98330e-2);
  REQUIRE(pt.a == 3.92);
  REQUIRE(pt.c == 34.408);
  REQUIRE(pt.n == 10);
  REQUIRE(pt.m == 8);
  REQUIRE(au.epsilon == 1.27930e-2);
  REQUIRE(au.a == 4.08);
  REQUIRE(au.c == 34.408);
  REQUIRE(au.n == 10);
  REQUIRE(au.m == 8);
  const auto pt_scales = rgpot::sutton_chen_scales(pt);
  const auto au_scales = rgpot::sutton_chen_scales(au);
  REQUIRE_THAT(au_scales.nearest_neighbor / pt_scales.nearest_neighbor,
               WithinAbs(au.a / pt.a, 1e-12));

  rgpot_sc_config raw{1.0, 34.408, 1.0, 10, 8, 40.0, 0};
  rgpot_sc_scales exported{};
  REQUIRE(rgpot_sutton_chen_scales(&raw, &exported) == 0);
  REQUIRE_THAT(exported.nearest_neighbor,
               WithinAbs(scales.nearest_neighbor, 0.0));
  REQUIRE_THAT(exported.well_depth, WithinAbs(scales.well_depth, 0.0));
  REQUIRE(rgpot_sutton_chen_scales(nullptr, &exported) == 1);
}

TEST_CASE("gold clusters relax from the rescaled platinum geometries",
          "[SuttonChen]") {
  const auto pt = rgpot::sutton_chen_platinum();
  const auto au = rgpot::sutton_chen_gold();
  const char *files[] = {"CppCore/tests/data/sc108/pt38",
                         "CppCore/tests/data/sc108/pt75"};
  for (const char *file : files) {
    double epsilon = 0.0;
    double c = 0.0;
    double a = 0.0;
    int nexp = 0;
    int mexp = 0;
    const AtomMatrix deposited =
        load_points(file, &epsilon, &c, &a, &nexp, &mexp);
    const std::vector<int> types(deposited.rows(), 79);
    rgpot::SuttonChenPot reduced(rgpot::SuttonChenConfig{
        .epsilon = epsilon, .c = c, .a = a, .n = nexp, .m = mexp, .cutoff = 0.0});
    const double e_reduced = std::get<0>(reduced(deposited, types, cube(200.0)));

    // The deposited file is in units of a = 1. Platinum in Angstrom is
    // that file times a_Pt, and the gold copy is that geometry times
    // a_Au / a_Pt.
    const double factor = au.a / pt.a;
    AtomMatrix pos = deposited;
    for (size_t i = 0; i < pos.rows(); ++i) {
      for (size_t k = 0; k < 3; ++k) {
        pos(i, k) *= factor;
      }
    }
    rgpot::SuttonChenConfig cfg = au;
    cfg.cutoff = 0.0;
    rgpot::SuttonChenPot pot(cfg);
    auto [energy, forces, variance] = pot(pos, types, cube(200.0));
    (void)variance;
    const double move_cap = 0.05 * cfg.a;
    double step = 1.0e-3 * cfg.a;
    for (int iter = 0; iter < 2000; ++iter) {
      double mf = 0.0;
      for (size_t i = 0; i < forces.rows(); ++i) {
        for (size_t k = 0; k < 3; ++k) {
          mf = std::max(mf, std::abs(forces(i, k)));
        }
      }
      if (mf < 1.0e-8) {
        break;
      }
      double disp = step;
      if (disp * mf > move_cap) {
        disp = move_cap / mf;
      }
      AtomMatrix trial = pos;
      for (size_t i = 0; i < trial.rows(); ++i) {
        for (size_t k = 0; k < 3; ++k) {
          trial(i, k) += disp * forces(i, k);
        }
      }
      auto [next, next_forces, next_variance] = pot(trial, types, cube(200.0));
      (void)next_variance;
      if (next < energy) {
        pos = trial;
        forces = next_forces;
        energy = next;
        step = disp * 1.2;
      } else {
        step = disp * 0.5;
      }
    }
    REQUIRE_THAT(energy, WithinAbs(au.epsilon * e_reduced, 1e-6));
  }
}
