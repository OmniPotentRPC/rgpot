// MIT License
// Copyright 2023--present rgpot developers
//
// Golden masters for D3 BJ ATM on/off and D4 vs s-dftd3 / dftd4 C-API
// refs. Pins are Hartree / Hartree/Bohr. Fail closed if a named file is
// missing. Tolerances are the library energy bar (100*eps), not looser.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <array>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#ifdef RGPOT_HAS_DFTD3
#include "rgpot/D3Pot/D3Pot.hpp"
#endif
#ifdef RGPOT_HAS_DFTD4
#include "rgpot/D4Pot/D4Pot.hpp"
#endif

#include "npy_io.hpp"
#include "rgpot/types/AtomMatrix.hpp"
#include "rgpot/units.hpp"

using Catch::Matchers::WithinAbs;
using rgpot::testio::NpyArray;
using rgpot::testio::load_npy;
using rgpot::types::AtomMatrix;
using rgpot::units::EV_TO_HARTREE;
using rgpot::units::HARTREE_BOHR_TO_EV_ANGSTROM;

namespace {

constexpr const char *kData = "CppCore/tests/data/dftd";
// s-dftd3 test/unit/test_dftd3.f90 and dftd4 unit tests: thr = 100*epsilon
// against library reference energies. Pins are the analytic C-API gradient,
// so the same bar applies (the looser sqrt(eps) bar is vs finite difference).
constexpr double kThrHartree = 100.0 * std::numeric_limits<double>::epsilon();

const char *kRequired[] = {
    "CppCore/tests/data/dftd/MANIFEST.json",
    "CppCore/tests/data/dftd/water.xyz",
    "CppCore/tests/data/dftd/water_pos.npy",
    "CppCore/tests/data/dftd/water_z.npy",
    "CppCore/tests/data/dftd/d3_bj_pbe_atm_off_energy.npy",
    "CppCore/tests/data/dftd/d3_bj_pbe_atm_off_grad.npy",
    "CppCore/tests/data/dftd/d3_bj_pbe_atm_on_energy.npy",
    "CppCore/tests/data/dftd/d3_bj_pbe_atm_on_grad.npy",
    "CppCore/tests/data/dftd/d4_pbe_energy.npy",
    "CppCore/tests/data/dftd/d4_pbe_grad.npy",
};

void require_file(const std::string &path) {
  REQUIRE(std::filesystem::exists(path));
  REQUIRE(std::filesystem::file_size(path) > 0);
}

std::string slurp(const std::string &path) {
  require_file(path);
  std::ifstream in(path);
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

void load_water(AtomMatrix &positions, std::vector<int> &atmtypes,
                std::array<std::array<double, 3>, 3> &box) {
  require_file(std::string(kData) + "/water_pos.npy");
  require_file(std::string(kData) + "/water_z.npy");
  const NpyArray pos = load_npy(std::string(kData) + "/water_pos.npy");
  const NpyArray z = load_npy(std::string(kData) + "/water_z.npy");
  REQUIRE(pos.shape.size() == 2);
  REQUIRE(pos.shape[0] == 3);
  REQUIRE(pos.shape[1] == 3);
  REQUIRE(z.shape.size() == 1);
  REQUIRE(z.shape[0] == 3);
  positions = AtomMatrix(3, 3);
  atmtypes.resize(3);
  for (std::size_t i = 0; i < 3; ++i) {
    atmtypes[i] = static_cast<int>(z.data[i]);
    for (std::size_t j = 0; j < 3; ++j) {
      positions(i, j) = pos.data[i * 3 + j];
    }
  }
  box = {{{100.0, 0.0, 0.0}, {0.0, 100.0, 0.0}, {0.0, 0.0, 100.0}}};
}

void check_energy_grad(double energy_ev, const AtomMatrix &forces,
                       const std::string &etag, const std::string &gtag) {
  require_file(etag);
  require_file(gtag);
  const NpyArray eref = load_npy(etag);
  const NpyArray gref = load_npy(gtag);
  REQUIRE(eref.data.size() == 1);
  REQUIRE(gref.shape.size() == 2);
  REQUIRE(gref.shape[0] == 3);
  REQUIRE(gref.shape[1] == 3);
  REQUIRE(forces.rows() == 3);

  const double energy_h = energy_ev * EV_TO_HARTREE;
  REQUIRE_THAT(energy_h, WithinAbs(eref.data[0], kThrHartree));

  for (std::size_t i = 0; i < 3; ++i) {
    for (std::size_t j = 0; j < 3; ++j) {
      const double grad_h = -forces(i, j) / HARTREE_BOHR_TO_EV_ANGSTROM;
      REQUIRE_THAT(grad_h, WithinAbs(gref.data[i * 3 + j], kThrHartree));
    }
  }
}

} // namespace

TEST_CASE("dftd golden fixtures exist (fail closed)", "[dftd][golden]") {
  for (const char *p : kRequired) {
    require_file(p);
  }
  const std::string man = slurp(std::string(kData) + "/MANIFEST.json");
  REQUIRE(man.find("sha256") != std::string::npos);
  REQUIRE(man.find("100*epsilon") != std::string::npos);
  REQUIRE(man.find("pbe") != std::string::npos);
  for (const char *p : kRequired) {
    const char *base = std::strrchr(p, '/');
    REQUIRE(base != nullptr);
    const char *name = base + 1;
    if (std::strcmp(name, "MANIFEST.json") == 0) {
      continue;
    }
    REQUIRE(man.find(name) != std::string::npos);
  }
}

#ifdef RGPOT_HAS_DFTD3
TEST_CASE("D3 BJ PBE ATM off matches s-dftd3 pin", "[dftd][d3][golden]") {
  AtomMatrix positions(3, 3);
  std::vector<int> atmtypes;
  std::array<std::array<double, 3>, 3> box{};
  load_water(positions, atmtypes, box);

  rgpot::D3Config cfg;
  cfg.damping = rgpot::D3Damping::BJ;
  cfg.functional = "pbe";
  cfg.atm = false;
  rgpot::D3Pot pot(cfg);
  auto [energy, forces, variance] = pot(positions, atmtypes, box);
  (void)variance;
  check_energy_grad(energy, forces,
                    std::string(kData) + "/d3_bj_pbe_atm_off_energy.npy",
                    std::string(kData) + "/d3_bj_pbe_atm_off_grad.npy");
}

TEST_CASE("D3 BJ PBE ATM on matches s-dftd3 pin", "[dftd][d3][golden]") {
  AtomMatrix positions(3, 3);
  std::vector<int> atmtypes;
  std::array<std::array<double, 3>, 3> box{};
  load_water(positions, atmtypes, box);

  rgpot::D3Config cfg;
  cfg.damping = rgpot::D3Damping::BJ;
  cfg.functional = "pbe";
  cfg.atm = true;
  rgpot::D3Pot pot(cfg);
  auto [energy, forces, variance] = pot(positions, atmtypes, box);
  (void)variance;
  check_energy_grad(energy, forces,
                    std::string(kData) + "/d3_bj_pbe_atm_on_energy.npy",
                    std::string(kData) + "/d3_bj_pbe_atm_on_grad.npy");
}

TEST_CASE("D3 ATM on pin differs from ATM off pin", "[dftd][d3][atm]") {
  require_file(std::string(kData) + "/d3_bj_pbe_atm_on_energy.npy");
  require_file(std::string(kData) + "/d3_bj_pbe_atm_off_energy.npy");
  const NpyArray on = load_npy(std::string(kData) + "/d3_bj_pbe_atm_on_energy.npy");
  const NpyArray off =
      load_npy(std::string(kData) + "/d3_bj_pbe_atm_off_energy.npy");
  REQUIRE(on.data.size() == 1);
  REQUIRE(off.data.size() == 1);
  REQUIRE(std::abs(on.data[0] - off.data[0]) > kThrHartree);

  AtomMatrix positions(3, 3);
  std::vector<int> atmtypes;
  std::array<std::array<double, 3>, 3> box{};
  load_water(positions, atmtypes, box);
  rgpot::D3Config cfg_on;
  cfg_on.damping = rgpot::D3Damping::BJ;
  cfg_on.functional = "pbe";
  cfg_on.atm = true;
  rgpot::D3Config cfg_off = cfg_on;
  cfg_off.atm = false;
  rgpot::D3Pot pot_on(cfg_on);
  rgpot::D3Pot pot_off(cfg_off);
  auto [e_on, f_on, v_on] = pot_on(positions, atmtypes, box);
  auto [e_off, f_off, v_off] = pot_off(positions, atmtypes, box);
  (void)f_on;
  (void)f_off;
  (void)v_on;
  (void)v_off;
  REQUIRE(std::abs(e_on - e_off) * EV_TO_HARTREE > kThrHartree);
}
#endif

#ifdef RGPOT_HAS_DFTD4
TEST_CASE("D4 PBE default matches dftd4 pin", "[dftd][d4][golden]") {
  AtomMatrix positions(3, 3);
  std::vector<int> atmtypes;
  std::array<std::array<double, 3>, 3> box{};
  load_water(positions, atmtypes, box);

  rgpot::D4Config cfg;
  cfg.functional = "pbe";
  cfg.charge = 0.0;
  cfg.atm = true;
  rgpot::D4Pot pot(cfg);
  auto [energy, forces, variance] = pot(positions, atmtypes, box);
  (void)variance;
  check_energy_grad(energy, forces, std::string(kData) + "/d4_pbe_energy.npy",
                    std::string(kData) + "/d4_pbe_grad.npy");
}
#endif
