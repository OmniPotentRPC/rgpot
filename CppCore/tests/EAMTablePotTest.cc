// MIT License
// Copyright 2023--present rgpot developers
//
// Tabulated EAM checks against the Large-scale Atomic/Molecular Massively
// Parallel Simulator (LAMMPS): Cu_u3.eam through pair_style eam,
// CuNi.eam.alloy through eam/alloy (data/eam_table/make_lammps_ref.py),
// and the Sutton-Chen eam/fs run of data/sutton_chen/make_lammps_ref.py.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <catch2/catch_all.hpp>

#include "rgpot/EAMTable/EAMTablePot.hpp"

using Catch::Matchers::WithinAbs;
using rgpot::EAMTableFormat;

namespace {

const std::string kData = "CppCore/tests/data/";

/// One LAMMPS run. Stress is xx yy zz yz xz xy in eV/Angstrom^3. Rows
/// are x y z fx fy fz, led by the atomic number when `with_z`.
struct LammpsRef {
  double energy = 0.0, box[9] = {}, stress[6] = {};
  std::vector<double> pos, forces;
  std::vector<int> z;
};

LammpsRef load_ref(const std::string &path, bool with_z) {
  std::ifstream in(kData + path);
  REQUIRE(in.good());
  LammpsRef ref;
  std::string key;
  std::size_t n = 0;
  while (in >> key) {
    if (key == "energy") {
      in >> ref.energy;
    } else if (key == "box") {
      for (double &v : ref.box)
        in >> v;
    } else if (key == "stress") {
      for (double &v : ref.stress)
        in >> v;
    } else if (key == "atoms" && in >> n) {
      for (std::size_t i = 0; i < n; ++i) {
        int z = 78;
        if (with_z)
          in >> z;
        ref.z.push_back(z);
        for (int k = 0; k < 6; ++k) {
          double v = 0.0;
          in >> v;
          (k < 3 ? ref.pos : ref.forces).push_back(v);
        }
      }
    } else {
      std::getline(in, key);
    }
  }
  return ref;
}

/// Row-major indices of the Voigt order xx yy zz yz xz xy.
constexpr int kVoigt[6][2] = {{0, 0}, {1, 1}, {2, 2}, {1, 2}, {0, 2}, {0, 1}};

struct Eval {
  double energy = 0.0, stress[9] = {};
  int has_stress = 0;
  std::vector<double> forces;
};

Eval eval(const rgpot::EAMTablePot &pot, const std::vector<double> &pos,
          const std::vector<int> &z, const double *box) {
  Eval e;
  e.forces.assign(pos.size(), 0.0);
  rgpot::ForceOut out{};
  out.F = e.forces.data();
  std::fill(out.stress, out.stress + 9, 7.0);
  pot.forceImpl(rgpot::ForceInput{z.size(), pos.data(), z.data(), box}, &out);
  e.energy = out.energy;
  std::copy(out.stress, out.stress + 9, e.stress);
  e.has_stress = out.has_stress;
  return e;
}

void check_against(const rgpot::EAMTablePot &pot, const LammpsRef &ref,
                   double etol, double ftol, double stol) {
  const Eval e = eval(pot, ref.pos, ref.z, ref.box);
  REQUIRE_THAT(e.energy, WithinAbs(ref.energy, etol));
  for (std::size_t k = 0; k < ref.forces.size(); ++k) {
    REQUIRE_THAT(e.forces[k], WithinAbs(ref.forces[k], ftol));
  }
  REQUIRE(e.has_stress == 1);
  for (int v = 0; v < 6; ++v) {
    REQUIRE_THAT(e.stress[3 * kVoigt[v][0] + kVoigt[v][1]],
                 WithinAbs(ref.stress[v], stol));
  }
}

/// The eam/fs table write_table in data/sutton_chen/make_lammps_ref.py
/// gives LAMMPS: platinum, n = 10, m = 8, 50001 points.
std::string sutton_chen_table() {
  const double eps = 1.98330e-2, c = 34.408, a = 3.92, cut = 5.5;
  const int nr = 50001;
  const double dr = cut / (nr - 1), drho = 1000.0 / (nr - 1);
  std::string s = "Sutton-Chen platinum\nn 10 m 8\nmake_lammps_ref.py\n1 Pt\n";
  char buf[64];
  std::snprintf(buf, sizeof buf, "%d %.17g %d %.17g %.17g\n", nr, drho, nr, dr,
                cut);
  s += buf;
  s += "78 195.084 3.92 fcc\n";
  for (int col = 0; col < 3; ++col) {
    for (int m = 0; m < nr; ++m) {
      const double r = std::max(m * dr, 1.0);
      const double v = col == 0   ? -eps * c * std::sqrt(m * drho)
                       : col == 1 ? std::pow(a / r, 8)
                                  : r * eps * std::pow(a / r, 10);
      std::snprintf(buf, sizeof buf, "%.17g\n", v);
      s += buf;
    }
  }
  return s;
}

/// Two-element eam/fs table of constants: F_A = -rho, F_B = -2 rho, no
/// pair term, rho_AA = 1, rho_BA = 3, rho_AB = 5, rho_BB = 7, with
/// rho_ab the density a b neighbor puts at an a site.
std::string two_element_fs() {
  std::ostringstream s;
  auto row = [&](auto f) {
    for (int m = 0; m < 11; ++m)
      s << f(m) << ' ';
    s << '\n';
  };
  s << "c\nc\nc\n2 A B\n11 1.0 11 0.5 4.0\n";
  s << "13 1.0 1.0 fcc\n";
  row([](int m) { return -1.0 * m; });
  row([](int) { return 1.0; });
  row([](int) { return 3.0; });
  s << "28 1.0 1.0 fcc\n";
  row([](int m) { return -2.0 * m; });
  row([](int) { return 5.0; });
  row([](int) { return 7.0; });
  for (int k = 0; k < 3; ++k)
    row([](int) { return 0.0; });
  return s.str();
}

rgpot::EAMTablePot cu_u3(bool periodic, bool stress) {
  return rgpot::EAMTablePot({.path = kData + "eam_table/Cu_u3.eam",
                             .format = EAMTableFormat::Funcfl,
                             .periodic = periodic,
                             .stress = stress});
}

} // namespace

TEST_CASE("funcfl Cu_u3 matches LAMMPS pair_style eam", "[EAMTable]") {
  check_against(cu_u3(true, true), load_ref("eam_table/cu_u3_cubic.ref", true),
                1e-10, 1e-10, 1e-13);
}

TEST_CASE("setfl CuNi matches LAMMPS pair_style eam/alloy", "[EAMTable]") {
  const rgpot::EAMTablePot pot(
      {.path = kData + "eam_table/CuNi.eam.alloy", .stress = true});
  check_against(pot, load_ref("eam_table/cuni_sheared.ref", true), 1e-10, 1e-10,
                1e-13);
}

TEST_CASE("eam/fs reader matches the LAMMPS Sutton-Chen run", "[EAMTable]") {
  std::istringstream in(sutton_chen_table());
  const rgpot::EAMTablePot pot(
      rgpot::read_eam_table(in, EAMTableFormat::FinnisSinclair), true, true);
  check_against(pot, load_ref("sutton_chen/pt_cubic.ref", false), 1e-10, 1e-10,
                1e-13);
}

TEST_CASE("eam/fs density blocks follow the LAMMPS order", "[EAMTable]") {
  std::istringstream in(two_element_fs());
  const rgpot::EAMTablePot pot(
      rgpot::read_eam_table(in, EAMTableFormat::FinnisSinclair), false, false);
  const std::vector<double> pos{0.0, 0.0, 0.0, 2.0, 0.0, 0.0};
  // A at an A site 1, B at an A site 5, A at a B site 3, B at a B site 7.
  REQUIRE(eval(pot, pos, {13, 13}, nullptr).energy == -2.0);
  REQUIRE(eval(pot, pos, {13, 28}, nullptr).energy == -5.0 - 2.0 * 3.0);
  REQUIRE(eval(pot, pos, {28, 28}, nullptr).energy == -28.0);
}

TEST_CASE("EAM table stress flag leaves energy and forces alone",
          "[EAMTable]") {
  const LammpsRef ref = load_ref("eam_table/cu_u3_cubic.ref", true);
  const Eval off = eval(cu_u3(true, false), ref.pos, ref.z, ref.box);
  const Eval on = eval(cu_u3(true, true), ref.pos, ref.z, ref.box);
  REQUIRE(off.energy == on.energy);
  REQUIRE(off.forces == on.forces);
  REQUIRE(off.has_stress == 0);
  REQUIRE(std::all_of(off.stress, off.stress + 9,
                      [](double v) { return v == 7.0; }));
  REQUIRE(cu_u3(true, false).paramsKey() != cu_u3(true, true).paramsKey());
}

TEST_CASE("EAM table rejects bad input", "[EAMTable]") {
  const LammpsRef ref = load_ref("eam_table/cu_u3_cubic.ref", true);
  std::vector<int> nickel(ref.z.size(), 28);
  REQUIRE_THROWS_AS(eval(cu_u3(true, false), ref.pos, nickel, ref.box),
                    std::invalid_argument);
  const double small[9] = {9.0, 0.0, 0.0, 0.0, 9.0, 0.0, 0.0, 0.0, 9.0};
  REQUIRE_THROWS_AS(eval(cu_u3(true, false), ref.pos, ref.z, small),
                    std::invalid_argument);
  std::ifstream file(kData + "eam_table/Cu_u3.eam");
  std::string head, line;
  for (int k = 0; k < 40 && std::getline(file, line); ++k)
    head += line + "\n";
  std::istringstream cut(head);
  REQUIRE_THROWS_AS(rgpot::read_eam_table(cut, EAMTableFormat::Funcfl),
                    std::invalid_argument);
}
