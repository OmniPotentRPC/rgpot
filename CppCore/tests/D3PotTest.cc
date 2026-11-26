// MIT License
// Copyright 2023--present rgpot developers
//
// s-dftd3 ISO C API smoke (dftd3.h). Golden pins live on rgpot-0r9w.
// Water-octamer forces are compared to a live dftd3_get_dispersion
// gradient (sigma non-NULL) at the library sqrt(eps) Hartree/Bohr bar.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <regex>
#include <string>
#include <type_traits>
#include <vector>

#include "rgpot/D3Pot/D3Pot.hpp"
#include "rgpot/stress.hpp"
#include "rgpot/types/AtomMatrix.hpp"
#include "rgpot/units.hpp"

using rgpot::types::AtomMatrix;
using Catch::Matchers::WithinAbs;

// Water (Å): O + 2H
static const double kWaterPos[] = {
    0.00000000, 0.00000000,  0.11779000,  // O
    0.00000000, 0.75545000,  -0.47116000, // H
    0.00000000, -0.75545000, -0.47116000  // H
};
static const int kWaterZ[] = {8, 1, 1};

static void fill_water(AtomMatrix &positions, std::vector<int> &atmtypes,
                       std::array<std::array<double, 3>, 3> &box) {
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      positions(static_cast<size_t>(i), static_cast<size_t>(j)) =
          kWaterPos[i * 3 + j];
  atmtypes.assign(kWaterZ, kWaterZ + 3);
  box = {{{100.0, 0.0, 0.0}, {0.0, 100.0, 0.0}, {0.0, 0.0, 100.0}}};
}

// s-dftd3 test/unit/test_dftd3.f90 gradient bar (not looser).
constexpr double kGradThrHaBohr = 1.4901161193847656e-08; // sqrt(eps)
constexpr const char *kOctamerXyz =
    "CppCore/tests/data/dftd/water_octamer.xyz";

struct Octamer {
  AtomMatrix positions;
  std::vector<int> numbers;
  std::array<std::array<double, 3>, 3> box{};
};

Octamer load_octamer() {
  REQUIRE(std::filesystem::exists(kOctamerXyz));
  REQUIRE(std::filesystem::file_size(kOctamerXyz) > 0);
  std::ifstream in(kOctamerXyz);
  REQUIRE(in);
  int nat = 0;
  in >> nat;
  REQUIRE(nat == 24);
  std::string line;
  std::getline(in, line);
  std::getline(in, line);
  Octamer g;
  g.positions = AtomMatrix(static_cast<size_t>(nat), 3);
  g.numbers.resize(static_cast<size_t>(nat));
  for (int i = 0; i < nat; ++i) {
    std::string sym;
    double x = 0.0, y = 0.0, z = 0.0;
    in >> sym >> x >> y >> z;
    REQUIRE(in);
    REQUIRE((sym == "O" || sym == "H"));
    g.numbers[static_cast<size_t>(i)] = (sym == "O") ? 8 : 1;
    g.positions(static_cast<size_t>(i), 0) = x;
    g.positions(static_cast<size_t>(i), 1) = y;
    g.positions(static_cast<size_t>(i), 2) = z;
  }
  g.box = {{{100.0, 0.0, 0.0}, {0.0, 100.0, 0.0}, {0.0, 0.0, 100.0}}};
  return g;
}

TEST_CASE("D3 stress is opt-in and matches triclinic strain", "[d3][stress]") {
  const Octamer geometry = load_octamer();
  std::vector<double> positions(
      geometry.positions.data(),
      geometry.positions.data() + 3 * geometry.numbers.size());
  std::array<double, 9> box = {
      18.0, 0.7, -0.2, -0.4, 17.0, 0.5, 0.3, -0.6, 19.0};
  rgpot::D3Pot pot;
  auto evaluate = [&](const std::vector<double> &pos,
                      const std::array<double, 9> &cell, bool stress) {
    std::vector<double> forces(pos.size());
    rgpot::ForceOut out{};
    out.F = forces.data();
    rgpot::ForceInput in{
        geometry.numbers.size(), pos.data(), geometry.numbers.data(), cell.data()};
    in.compute_stress = stress;
    pot.forceImpl(in, &out);
    return std::pair{out, forces};
  };
  const auto [plain, plain_forces] = evaluate(positions, box, false);
  const auto [analytic, analytic_forces] = evaluate(positions, box, true);
  REQUIRE(plain.has_stress == 0);
  REQUIRE(analytic.has_stress == 1);
  REQUIRE_THAT(plain.energy, WithinAbs(analytic.energy, 1e-15));
  for (std::size_t i = 0; i < plain_forces.size(); ++i) {
    REQUIRE_THAT(plain_forces[i], WithinAbs(analytic_forces[i], 1e-15));
  }

  constexpr int rows[6] = {0, 1, 2, 1, 0, 0};
  constexpr int cols[6] = {0, 1, 2, 2, 2, 1};
  constexpr int slots[6] = {0, 4, 8, 5, 2, 1};
  constexpr double step = 1e-5;
  const double volume = rgpot::cellVolume(box.data());
  double largest = 0.0;
  double error = 0.0;
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
      auto apply = [&](const double *source, double *target,
                       std::size_t count) {
        for (std::size_t i = 0; i < count; ++i) {
          for (int c = 0; c < 3; ++c) {
            target[3 * i + c] =
                source[3 * i] * transform[c] +
                source[3 * i + 1] * transform[3 + c] +
                source[3 * i + 2] * transform[6 + c];
          }
        }
      };
      std::vector<double> moved(positions.size());
      std::array<double, 9> strained{};
      apply(positions.data(), moved.data(), geometry.numbers.size());
      apply(box.data(), strained.data(), 3);
      energies[side] = evaluate(moved, strained, false).first.energy;
    }
    const double numerical =
        (energies[1] - energies[0]) / (2.0 * step * volume);
    const double value = analytic.stress[slots[component]];
    largest = std::max(largest, std::abs(value));
    error = std::max(error, std::abs(value - numerical));
  }
  const double relative = error / std::max(largest, 1e-20);
  std::cout << "STRESS_ERROR D3 " << relative << '\n';
  REQUIRE(relative <= 1e-5);
}

struct CapiDisp {
  double energy_ha = 0.0;
  std::vector<double> grad;
};

CapiDisp capi_d3_bj_pbe(const Octamer &g, bool atm, bool pass_sigma) {
  dftd3_error err = dftd3_new_error();
  REQUIRE(err);
  std::vector<char> method{'p', 'b', 'e', '\0'};
  dftd3_param param = dftd3_load_rational_damping(err, method.data(), atm);
  REQUIRE(dftd3_check_error(err) == 0);
  REQUIRE(param);

  const size_t nat = g.numbers.size();
  const size_t n3 = 3 * nat;
  std::vector<double> pos_bohr(n3);
  for (size_t i = 0; i < nat; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      pos_bohr[3 * i + j] =
          g.positions(i, j) * rgpot::units::ANGSTROM_TO_BOHR;
    }
  }
  double box_bohr[9];
  const double box_a[9] = {100.0, 0.0, 0.0, 0.0, 100.0, 0.0, 0.0, 0.0, 100.0};
  for (int i = 0; i < 9; ++i) {
    box_bohr[i] = box_a[i] * rgpot::units::ANGSTROM_TO_BOHR;
  }
  const bool periodic[3] = {false, false, false};

  dftd3_structure mol = dftd3_new_structure(
      err, static_cast<int>(nat), g.numbers.data(), pos_bohr.data(), box_bohr,
      periodic);
  REQUIRE(dftd3_check_error(err) == 0);
  REQUIRE(mol);
  dftd3_model model = dftd3_new_d3_model(err, mol);
  REQUIRE(dftd3_check_error(err) == 0);
  REQUIRE(model);

  CapiDisp out;
  out.grad.assign(n3, 123.0);
  double sigma[9] = {};
  dftd3_get_dispersion(err, mol, model, param, &out.energy_ha, out.grad.data(),
                       pass_sigma ? sigma : nullptr);
  REQUIRE(dftd3_check_error(err) == 0);

  dftd3_delete_param(&param);
  dftd3_delete_model(&model);
  dftd3_delete_structure(&mol);
  dftd3_delete_error(&err);
  return out;
}

TEST_CASE("D3Pot is a Potential and PotentialConfig.d3 keeps its defaults",
          "[d3][schema]") {
  STATIC_REQUIRE(
      std::is_base_of_v<rgpot::Potential<rgpot::D3Pot>, rgpot::D3Pot>);
  std::ifstream in("CppCore/rgpot/rpc/Potentials.capnp");
  REQUIRE(in);
  const std::string schema((std::istreambuf_iterator<char>(in)),
                           std::istreambuf_iterator<char>());
  // Upstream aligns the arm columns, so match any run of blanks.
  REQUIRE(std::regex_search(schema, std::regex(R"(\bd3\s+@12\s*:D3Params;)")));
  const auto begin = schema.find("struct D3Params {");
  REQUIRE(begin != std::string::npos);
  const std::string params =
      schema.substr(begin, schema.find('}', begin) - begin);
  const rgpot::D3Config defaults;
  REQUIRE(defaults.functional == "pbe");
  REQUIRE(std::regex_search(
      params, std::regex(R"(functional\s+@0\s*:Text\s*=\s*"pbe";)")));
  REQUIRE(defaults.damping == rgpot::D3Damping::BJ);
  REQUIRE(std::regex_search(params,
                            std::regex(R"(damping\s+@1\s*:Text\s*=\s*"bj";)")));
  REQUIRE(defaults.atm);
  REQUIRE(
      std::regex_search(params, std::regex(R"(atm\s+@2\s*:Bool\s*=\s*true;)")));
}

TEST_CASE("D3Pot default is BJ PBE with ATM on", "[d3][config]") {
  rgpot::D3Pot pot;
  REQUIRE(pot.getType() == rgpot::PotType::D3);
  REQUIRE(pot.config().damping == rgpot::D3Damping::BJ);
  REQUIRE(pot.config().functional == "pbe");
  REQUIRE(pot.config().atm);
}

TEST_CASE("D3Pot water energy is finite and ATM flips it", "[d3][atm]") {
  AtomMatrix positions(3, 3);
  std::vector<int> atmtypes;
  std::array<std::array<double, 3>, 3> box{};
  fill_water(positions, atmtypes, box);

  rgpot::D3Config on;
  on.damping = rgpot::D3Damping::BJ;
  on.functional = "pbe";
  on.atm = true;
  rgpot::D3Config off = on;
  off.atm = false;

  rgpot::D3Pot pot_on(on);
  rgpot::D3Pot pot_off(off);
  auto [e_on, f_on, v_on] = pot_on(positions, atmtypes, box);
  auto [e_off, f_off, v_off] = pot_off(positions, atmtypes, box);
  (void)v_on;
  (void)v_off;

  REQUIRE(std::isfinite(e_on));
  REQUIRE(std::isfinite(e_off));
  REQUIRE(e_on != e_off);
  REQUIRE(std::abs(e_on - e_off) > 1e-12);

  REQUIRE(f_on.rows() == 3);
  REQUIRE(f_off.rows() == 3);
  double f2 = 0.0;
  double df2 = 0.0;
  for (size_t i = 0; i < 3; ++i)
    for (size_t j = 0; j < 3; ++j) {
      REQUIRE(std::isfinite(f_on(i, j)));
      REQUIRE(std::isfinite(f_off(i, j)));
      f2 += f_on(i, j) * f_on(i, j);
      const double d = f_on(i, j) - f_off(i, j);
      df2 += d * d;
    }
  REQUIRE(f2 > 0.0);
  REQUIRE(df2 > 0.0);

  double fx = 0, fy = 0, fz = 0;
  for (size_t i = 0; i < 3; ++i) {
    fx += f_on(i, 0);
    fy += f_on(i, 1);
    fz += f_on(i, 2);
  }
  REQUIRE_THAT(fx, WithinAbs(0.0, 1e-8));
  REQUIRE_THAT(fy, WithinAbs(0.0, 1e-8));
  REQUIRE_THAT(fz, WithinAbs(0.0, 1e-8));
}

TEST_CASE("D3Pot zero damping water is finite", "[d3][zero]") {
  rgpot::D3Config cfg;
  cfg.damping = rgpot::D3Damping::Zero;
  cfg.functional = "pbe";
  cfg.atm = true;
  rgpot::D3Pot pot(cfg);

  AtomMatrix positions(3, 3);
  std::vector<int> atmtypes;
  std::array<std::array<double, 3>, 3> box{};
  fill_water(positions, atmtypes, box);

  auto [energy, forces, variance] = pot(positions, atmtypes, box);
  (void)variance;
  REQUIRE(std::isfinite(energy));
  REQUIRE(forces.rows() == 3);
}

TEST_CASE("D3Pot warm update_structure path is stable", "[d3][capi]") {
  rgpot::D3Pot pot;
  AtomMatrix positions(3, 3);
  std::vector<int> atmtypes;
  std::array<std::array<double, 3>, 3> box{};
  fill_water(positions, atmtypes, box);

  auto [e1, f1, v1] = pot(positions, atmtypes, box);
  positions(1, 1) += 0.01;
  auto [e2, f2, v2] = pot(positions, atmtypes, box);
  (void)f1;
  (void)f2;
  (void)v1;
  (void)v2;
  REQUIRE(std::isfinite(e1));
  REQUIRE(std::isfinite(e2));
  REQUIRE(e1 != e2);
}

TEST_CASE("D3Pot water-octamer forces match s-dftd3 C-API at sqrt(eps)",
          "[d3][capi][octamer]") {
  auto geom = load_octamer();
  const double gtol =
      kGradThrHaBohr * std::abs(rgpot::units::NEG_GRAD_TO_FORCE);
  const double etol = 100.0 * std::numeric_limits<double>::epsilon() *
                      rgpot::units::HARTREE_TO_EV;

  // s-dftd3 leaves the gradient sentinel untouched when sigma is NULL.
  auto skipped = capi_d3_bj_pbe(geom, true, false);
  REQUIRE(skipped.grad[0] == 123.0);

  rgpot::D3Config on;
  on.damping = rgpot::D3Damping::BJ;
  on.functional = "pbe";
  on.atm = true;
  rgpot::D3Config off = on;
  off.atm = false;

  rgpot::D3Pot pot_on(on);
  rgpot::D3Pot pot_off(off);
  auto [e_on, f_on, v_on] = pot_on(geom.positions, geom.numbers, geom.box);
  auto [e_off, f_off, v_off] = pot_off(geom.positions, geom.numbers, geom.box);
  (void)v_on;
  (void)v_off;

  auto capi_on = capi_d3_bj_pbe(geom, true, true);
  auto capi_off = capi_d3_bj_pbe(geom, false, true);

  REQUIRE_THAT(e_on, WithinAbs(capi_on.energy_ha * rgpot::units::HARTREE_TO_EV,
                               etol));
  REQUIRE_THAT(e_off, WithinAbs(capi_off.energy_ha * rgpot::units::HARTREE_TO_EV,
                                etol));

  REQUIRE(f_on.rows() == 24);
  REQUIRE(f_off.rows() == 24);
  REQUIRE(capi_on.grad.size() == 72);
  REQUIRE(capi_off.grad.size() == 72);
  double df2 = 0.0;
  for (size_t i = 0; i < 24; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      const double ref_on =
          capi_on.grad[3 * i + j] * rgpot::units::NEG_GRAD_TO_FORCE;
      const double ref_off =
          capi_off.grad[3 * i + j] * rgpot::units::NEG_GRAD_TO_FORCE;
      REQUIRE(std::isfinite(f_on(i, j)));
      REQUIRE(std::isfinite(f_off(i, j)));
      REQUIRE_THAT(f_on(i, j), WithinAbs(ref_on, gtol));
      REQUIRE_THAT(f_off(i, j), WithinAbs(ref_off, gtol));
      const double d = f_on(i, j) - f_off(i, j);
      df2 += d * d;
    }
  }
  REQUIRE(df2 > 0.0);
}

TEST_CASE("D3Pot reads the species again when only the species change",
          "[d3]") {
  AtomMatrix positions(3, 3);
  std::vector<int> water;
  std::array<std::array<double, 3>, 3> box{};
  fill_water(positions, water, box);
  const std::vector<int> swapped = {16, 1, 1};

  rgpot::D3Pot reused;
  (void)reused(positions, water, box);
  const auto [e_reused, f_reused, v_reused] = reused(positions, swapped, box);

  rgpot::D3Pot fresh;
  const auto [e_fresh, f_fresh, v_fresh] = fresh(positions, swapped, box);

  REQUIRE_THAT(e_reused, WithinAbs(e_fresh, 1e-12));
  for (size_t i = 0; i < 3; ++i) {
    for (size_t k = 0; k < 3; ++k) {
      REQUIRE_THAT(f_reused(i, k), WithinAbs(f_fresh(i, k), 1e-12));
    }
  }
}
