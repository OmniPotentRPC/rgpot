// MIT License
// Copyright 2023--present rgpot developers
//
// Tabulated EAM against LAMMPS pair_style eam/alloy and eam/fs on the
// cells in data/eam_table (MANIFEST.json names the LAMMPS build and the
// input deck), and against central differences of the energy.

#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <catch2/catch_all.hpp>

#include "rgpot/EAM/EAMTable.hpp"
#include "rgpot/types/AtomMatrix.hpp"

using rgpot::types::AtomMatrix;

namespace {

const std::string kDir = "CppCore/tests/data/eam_table/";

struct Ref {
  rgpot::EAMTableConfig cfg;
  std::array<std::array<double, 3>, 3> box{};
  AtomMatrix pos, force;
  std::vector<int> z;
  double energy = 0.0;
};

Ref load(const std::string &name) {
  std::ifstream in(kDir + name + ".ref");
  std::string tag, fmt;
  while (in.peek() == '#') {
    std::getline(in, tag);
  }
  Ref r;
  std::size_t n = 0;
  in >> tag >> r.cfg.path >> fmt >> r.cfg.periodic >> tag;
  for (auto &row : r.box) {
    in >> row[0] >> row[1] >> row[2];
  }
  in >> tag >> r.energy >> tag >> n;
  r.cfg.path = kDir + r.cfg.path;
  r.cfg.format = fmt == "fs" ? rgpot::EAMTableFormat::FinnisSinclair
                             : rgpot::EAMTableFormat::Setfl;
  r.z.resize(n);
  r.pos = AtomMatrix(n, 3);
  r.force = AtomMatrix(n, 3);
  for (std::size_t i = 0; i < n; ++i) {
    in >> r.z[i] >> r.pos(i, 0) >> r.pos(i, 1) >> r.pos(i, 2) >>
        r.force(i, 0) >> r.force(i, 1) >> r.force(i, 2);
  }
  REQUIRE(in);
  return r;
}

} // namespace

TEST_CASE("EAMTable reads setfl and eam/fs tables", "[eam_table]") {
  const auto al = rgpot::read_eam_table(kDir + "Al_EA.eam.alloy",
                                        rgpot::EAMTableFormat::Setfl);
  REQUIRE((al.nrho == 500 && al.nr == 500 && al.atomic_numbers[0] == 13));
  REQUIRE(al.cut == 0.5558054418218096e+01);
  REQUIRE(al.frho[0][2] == -0.4371148081215313e-01);
  const auto fs = rgpot::read_eam_table(kDir + "synth.eam.fs",
                                        rgpot::EAMTableFormat::FinnisSinclair);
  REQUIRE((fs.nrho == 8 && fs.nr == 8 && fs.cut > (fs.nr - 1) * fs.dr));
  REQUIRE(fs.atomic_numbers == std::vector<int>{29, 47});
  REQUIRE(fs.rhor[1] != fs.rhor[2]);
}

TEST_CASE("EAMTable needs one name per element", "[eam_table]") {
  const std::string names = GENERATE("2 Cu", "2 Cu Ag Au", "2");
  CAPTURE(names);
  std::ifstream in(kDir + "synth.eam.fs");
  std::ostringstream text;
  int k = 0;
  for (std::string s; std::getline(in, s); ++k) {
    text << (k == 3 ? names : s) << '\n';
  }
  const std::string path =
      (std::filesystem::temp_directory_path() / "rgpot_eam_names.eam.fs")
          .string();
  std::ofstream(path) << text.str();
  REQUIRE_THROWS_AS(
      rgpot::read_eam_table(path, rgpot::EAMTableFormat::FinnisSinclair),
      std::runtime_error);
  std::filesystem::remove(path);
}

TEST_CASE("EAMTable matches LAMMPS", "[eam_table]") {
  const std::string name = GENERATE("al_fcc", "al_cluster", "alpb_fcc",
                                    "alpb_tri", "al_dense", "fs_clamp");
  CAPTURE(name);
  const Ref r = load(name);
  rgpot::EAMTablePot pot(r.cfg);
  const auto [energy, forces, variance] = pot(r.pos, r.z, r.box);
  REQUIRE_THAT(energy, Catch::Matchers::WithinRel(r.energy, 1e-12));
  for (std::size_t c = 0; c < 3 * r.z.size(); ++c) {
    REQUIRE(std::abs(forces.data()[c] - r.force.data()[c]) < 1e-10);
  }
}

// Not fs_clamp: past the last r knot LAMMPS keeps rho and r phi at their
// last knot values but uses a nonzero slope, so forces are not a gradient.
TEST_CASE("EAMTable forces are the energy gradient", "[eam_table]") {
  const std::string name = GENERATE("al_fcc", "alpb_tri", "al_dense");
  CAPTURE(name);
  const Ref r = load(name);
  rgpot::EAMTablePot pot(r.cfg);
  const auto [e0, forces, v0] = pot(r.pos, r.z, r.box);
  const double h = 1e-5;
  for (std::size_t c = 0; c < 18; ++c) {
    AtomMatrix p = r.pos;
    p(c / 3, c % 3) += h;
    const double up = std::get<0>(pot(p, r.z, r.box));
    p(c / 3, c % 3) -= 2.0 * h;
    const double down = std::get<0>(pot(p, r.z, r.box));
    REQUIRE_THAT(forces(c / 3, c % 3),
                 Catch::Matchers::WithinAbs((down - up) / (2.0 * h), 1e-6));
  }
}

TEST_CASE("EAMTable rejects what it cannot evaluate", "[eam_table]") {
  const Ref r = load("alpb_fcc");
  rgpot::EAMTablePot pot(r.cfg);
  REQUIRE(pot.get_type() == rgpot::PotType::EAMTable);
  std::vector<int> unknown = r.z;
  unknown[0] = 29;
  REQUIRE_THROWS_AS(pot(r.pos, unknown, r.box), std::invalid_argument);
  auto narrow = r.box;
  narrow[2][2] = 10.0;
  REQUIRE_THROWS_AS(pot(r.pos, r.z, narrow), std::invalid_argument);
  rgpot::EAMTableConfig cfg = r.cfg;
  cfg.atomic_numbers = {13};
  REQUIRE_THROWS_AS(rgpot::EAMTablePot(cfg), std::invalid_argument);
}
