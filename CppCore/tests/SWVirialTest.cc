// MIT License
// Copyright 2023--present rgpot developers
//
// Analytic Stillinger-Weber virial. Stress is -virial/V, the ASE sign
// (sigma = (1/V) dE/dε). Central differences use the same right strain
// as ase.calculators.fd.calculate_numerical_stress.

#include <array>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include <catch2/catch_all.hpp>

#include "rgpot/LennardJones/LJPot.hpp"
#include "rgpot/StillingerWeber/SWSi.hpp"
#include "rgpot/stress.hpp"

using Catch::Matchers::WithinAbs;
using rgpot::StillingerWeberSi;

namespace {

double pairMinimumU() {
  auto df = [](double u) {
    const double s = 1.0 / (u - StillingerWeberSi::kCut);
    const double inv = 1.0 / u;
    const double um4 = inv * inv * inv * inv;
    const double g = StillingerWeberSi::kB * um4 - 1.0;
    const double es = std::exp(s);
    const double dgdu =
        -StillingerWeberSi::kP * StillingerWeberSi::kB * um4 * inv;
    return StillingerWeberSi::kA * es * (dgdu - g * s * s);
  };
  double u = 1.12;
  for (int it = 0; it < 20; ++it) {
    const double h = 1e-8;
    u -= df(u) / ((df(u + h) - df(u - h)) / (2.0 * h));
  }
  return u;
}

double diamondA0() {
  return pairMinimumU() * StillingerWeberSi::kSigma * 4.0 / std::sqrt(3.0);
}

void appendDiamond(std::vector<double> &pos, int nrep, double a0) {
  const double basis[8][3] = {{0, 0, 0},         {0.5, 0.5, 0},
                              {0.5, 0, 0.5},     {0, 0.5, 0.5},
                              {0.25, 0.25, 0.25}, {0.75, 0.75, 0.25},
                              {0.75, 0.25, 0.75}, {0.25, 0.75, 0.75}};
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
}

struct Eval {
  double energy = 0;
  std::array<double, 9> stress{};
  std::array<double, 9> virial{};
  int has_stress = 0;
  int has_virial = 0;
};

Eval evaluate(const std::vector<double> &pos, const std::array<double, 9> &box) {
  const std::size_t n = pos.size() / 3;
  std::vector<double> forces(pos.size());
  std::vector<int> types(n, 14);
  rgpot::ForceOut out{forces.data(), 0.0, 0.0, {}, 0};
  StillingerWeberSi{}.forceImpl({n, pos.data(), types.data(), box.data()}, &out);
  Eval e;
  e.energy = out.energy;
  e.has_stress = out.has_stress;
  e.has_virial = out.has_virial;
  for (int i = 0; i < 9; ++i) {
    e.stress[static_cast<std::size_t>(i)] = out.stress[i];
    e.virial[static_cast<std::size_t>(i)] = out.virial[i];
  }
  return e;
}

/// (1/V) dE/dε_row,col by a two-point central difference. The strain sends
/// every row vector r to r (I+ε), the deformation in stress.hpp and in ASE.
double centralStress(const std::vector<double> &pos,
                     const std::array<double, 9> &box, int row, int col) {
  constexpr double h = 1e-6;
  const std::size_t n = pos.size() / 3;
  auto energyAt = [&](double eps) {
    std::vector<double> p = pos;
    std::array<double, 9> cell = box;
    for (std::size_t a = 0; a < n; ++a) {
      p[3 * a + static_cast<std::size_t>(col)] +=
          eps * pos[3 * a + static_cast<std::size_t>(row)];
    }
    for (int k = 0; k < 3; ++k) {
      cell[3 * k + col] += eps * box[3 * k + row];
    }
    return evaluate(p, cell).energy;
  };
  return (energyAt(h) - energyAt(-h)) /
         (2.0 * h * rgpot::cellVolume(box.data()));
}

void checkAgainstCentral(const std::vector<double> &pos,
                         const std::array<double, 9> &box, double tol) {
  const Eval got = evaluate(pos, box);
  REQUIRE(got.has_virial == 1);
  REQUIRE(got.has_stress == 1);
  const double volume = rgpot::cellVolume(box.data());
  REQUIRE(volume > 0.0);
  const double scale = -1.0 / volume;
  for (int i = 0; i < 9; ++i) {
    REQUIRE(got.stress[static_cast<std::size_t>(i)] ==
            got.virial[static_cast<std::size_t>(i)] * scale);
  }
  for (int row = 0; row < 3; ++row) {
    for (int col = 0; col < 3; ++col) {
      const double fd = centralStress(pos, box, row, col);
      CAPTURE(row, col, fd, got.stress[3 * row + col]);
      REQUIRE_THAT(got.stress[3 * row + col], WithinAbs(fd, tol));
    }
  }
}

std::string lammpsBinary() {
  const char *names[] = {"lmp", "lmp_serial", "lmp_mpi", "lammps"};
  for (const char *name : names) {
    const std::string probe =
        std::string("command -v ") + name + " >/dev/null 2>&1";
    if (std::system(probe.c_str()) == 0) {
      return name;
    }
  }
  return {};
}

} // namespace

TEST_CASE("publishVirialStress uses the ASE sign and an explicit absence",
          "[SWSi][virial]") {
  rgpot::ForceOut out{};
  const double w[9] = {2, 0.5, 0, 0.5, 2, 0, 0, 0, 2};
  rgpot::publishVirialStress(&out, w, 0.0);
  REQUIRE(out.has_virial == 1);
  REQUIRE(out.has_stress == 0);
  REQUIRE(out.virial[0] == 2.0);
  rgpot::publishVirialStress(&out, w, 4.0);
  REQUIRE(out.has_stress == 1);
  REQUIRE_THAT(out.stress[0], WithinAbs(-0.5, 0.0));
  REQUIRE_THAT(out.stress[1], WithinAbs(-0.125, 0.0));
  REQUIRE_THAT(out.stress[4], WithinAbs(-0.5, 0.0));
}

TEST_CASE("virial is unavailable unless the potential implements it",
          "[SWSi][virial]") {
  struct Bare : rgpot::Potential<Bare> {
    Bare() : Potential(rgpot::PotType::UNKNOWN) {}
    void forceImpl(const rgpot::ForceInput &,
                   rgpot::ForceOut *out) const override {
      out->energy = 1.0;
    }
  };
  Bare bare;
  REQUIRE_FALSE(bare.caps().stress);
  REQUIRE_FALSE(bare.caps().virial);
  double pos[3] = {0, 0, 0};
  int type = 1;
  double box[9] = {10, 0, 0, 0, 10, 0, 0, 0, 10};
  double forces[3] = {};
  rgpot::ForceInput in{1, pos, &type, box};
  rgpot::ForceOut slot{forces, 0.0, 0.0, {}, 0};
  slot.has_virial = 1;
  slot.virial[0] = 9.0;
  slot.has_stress = 1;
  const rgpot::ForceInput inputs[] = {in};
  rgpot::ForceOut outputs[] = {slot};
  const rgpot::ForceBatch cleared{1, inputs, outputs};
  bare.forceBatch(cleared);
  REQUIRE(outputs[0].has_virial == 0);
  REQUIRE(outputs[0].has_stress == 0);
  REQUIRE(outputs[0].virial[0] == 0.0);
  REQUIRE(outputs[0].energy == 1.0);

  rgpot::LJPot lj;
  REQUIRE(lj.caps().stress);
  REQUIRE_FALSE(lj.caps().virial);
  double pair[6] = {0, 0, 0, 1.2, 0.1, 0};
  int types[2] = {1, 1};
  double ljBox[9] = {20, 0, 0, 0, 20, 0, 0, 0, 20};
  double ljF[6] = {};
  rgpot::ForceOut ljOut{ljF, 0.0, 0.0, {}, 0};
  lj.forceImpl({2, pair, types, ljBox}, &ljOut);
  REQUIRE(ljOut.has_stress == 1);
  REQUIRE(ljOut.has_virial == 0);
  for (double component : ljOut.virial) {
    REQUIRE(component == 0.0);
  }
}

TEST_CASE("SW 1985 virial matches central strain differences",
          "[SWSi][virial]") {
  constexpr double kTol = 1e-6; // eV/Angstrom^3
  const double a0 = diamondA0();

  SECTION("perfect diamond, compressed") {
    const double a = 0.98 * a0;
    std::vector<double> pos;
    appendDiamond(pos, 2, a);
    const std::array<double, 9> box{2 * a, 0, 0, 0, 2 * a, 0, 0, 0, 2 * a};
    checkAgainstCentral(pos, box, kTol);
    const Eval got = evaluate(pos, box);
    for (int k : {0, 4, 8}) {
      REQUIRE(got.stress[k] < -1e-2);
    }
  }

  SECTION("sheared diamond") {
    const double l = 2 * a0;
    std::vector<double> cubic;
    appendDiamond(cubic, 2, a0);
    const std::array<double, 9> box{l, 0,        0,         0.12 * l, l,
                                    0, 0.04 * l, -0.08 * l, l};
    std::vector<double> pos(cubic.size());
    for (std::size_t a = 0; a < pos.size() / 3; ++a) {
      for (int c = 0; c < 3; ++c) {
        pos[3 * a + c] =
            (cubic[3 * a] * box[static_cast<std::size_t>(c)] +
             cubic[3 * a + 1] * box[3 + c] +
             cubic[3 * a + 2] * box[6 + c]) /
            l;
      }
    }
    checkAgainstCentral(pos, box, kTol);
    const Eval got = evaluate(pos, box);
    REQUIRE(std::abs(got.stress[1]) > 1e-3);
    REQUIRE(std::abs(got.stress[2]) > 1e-3);
    REQUIRE(std::abs(got.stress[5]) > 1e-3);
  }

  SECTION("rattled diamond") {
    std::vector<double> pos;
    appendDiamond(pos, 2, a0);
    std::mt19937_64 gen(19850415);
    for (double &x : pos) {
      x += 0.15 * (static_cast<double>(gen() >> 11) * 0x1.0p-53 - 0.5);
    }
    const std::array<double, 9> box{2 * a0, 0, 0, 0, 2 * a0, 0, 0, 0, 2 * a0};
    checkAgainstCentral(pos, box, kTol);
  }

  SECTION("vacancy") {
    std::vector<double> pos;
    appendDiamond(pos, 2, a0);
    pos.erase(pos.begin(), pos.begin() + 3);
    std::mt19937_64 gen(7);
    for (double &x : pos) {
      x += 0.12 * (static_cast<double>(gen() >> 11) * 0x1.0p-53 - 0.5);
    }
    const std::array<double, 9> box{2 * a0, 0, 0, 0, 2 * a0, 0, 0, 0, 2 * a0};
    checkAgainstCentral(pos, box, kTol);
    const Eval got = evaluate(pos, box);
    double peak = 0.0;
    for (double component : got.stress) {
      peak = std::max(peak, std::abs(component));
    }
    REQUIRE(peak > 1e-3);
  }
}

TEST_CASE("SW 1985 stress is zero at the equilibrium lattice constant",
          "[SWSi][virial]") {
  const double a = diamondA0();
  std::vector<double> pos;
  appendDiamond(pos, 2, a);
  const std::array<double, 9> box{2 * a, 0, 0, 0, 2 * a, 0, 0, 0, 2 * a};
  const Eval got = evaluate(pos, box);
  REQUIRE(got.has_stress == 1);
  REQUIRE(got.has_virial == 1);
  for (int i = 0; i < 9; ++i) {
    CAPTURE(i, got.stress[static_cast<std::size_t>(i)]);
    REQUIRE_THAT(got.stress[static_cast<std::size_t>(i)], WithinAbs(0.0, 1e-8));
    REQUIRE_THAT(got.virial[static_cast<std::size_t>(i)], WithinAbs(0.0, 1e-4));
  }
}

TEST_CASE("SW 1985 virial uses the folded image vector", "[SWSi][virial]") {
  const double r = 2.5;
  const double box[9] = {30, 0, 0, 0, 30, 0, 0, 0, 30};
  const double centered[6] = {10, 10, 10, 10 + r, 10, 10};
  const double wrapped[6] = {0.2, 10, 10, 0.2 + r - 30, 10, 10};
  std::vector<double> a(centered, centered + 6);
  std::vector<double> b(wrapped, wrapped + 6);
  const std::array<double, 9> cell{30, 0, 0, 0, 30, 0, 0, 0, 30};
  (void)box;
  const Eval ea = evaluate(a, cell);
  const Eval eb = evaluate(b, cell);
  REQUIRE(ea.has_virial == 1);
  REQUIRE(eb.has_virial == 1);
  for (int i = 0; i < 9; ++i) {
    REQUIRE_THAT(eb.virial[static_cast<std::size_t>(i)],
                 WithinAbs(ea.virial[static_cast<std::size_t>(i)], 1e-10));
  }
  REQUIRE(std::abs(ea.virial[0]) > 1e-3);
}

TEST_CASE("SW 1985 virial matches LAMMPS when lmp is installed",
          "[SWSi][virial][lammps]") {
  const std::string lmp = lammpsBinary();
  if (lmp.empty()) {
    SKIP("neither LAMMPS (lmp) nor a LAMMPS binary is installed");
  }
  const double a = 0.98 * diamondA0();
  std::vector<double> pos;
  appendDiamond(pos, 1, a);
  const Eval got = evaluate(pos, {a, 0, 0, 0, a, 0, 0, 0, a});
  const std::string dir = "sw-virial-lammps";
  std::system(("rm -rf " + dir + " && mkdir -p " + dir).c_str());
  {
    std::ofstream sw(dir + "/Si.sw");
    sw.setf(std::ios::scientific);
    sw.precision(16);
    sw << "# Stillinger-Weber 1985, epsilon in eV, sigma in Angstrom\n";
    sw << "Si Si Si " << StillingerWeberSi::kEpsilon << " "
       << StillingerWeberSi::kSigma << " " << StillingerWeberSi::kCut << " "
       << StillingerWeberSi::kLambda << " " << StillingerWeberSi::kGamma
       << " " << (-1.0 / 3.0) << " " << StillingerWeberSi::kA << " "
       << StillingerWeberSi::kB << " " << StillingerWeberSi::kP
       << " 0 0\n";
  }
  {
    std::ofstream data(dir + "/data.si");
    data.setf(std::ios::scientific);
    data.precision(16);
    data << pos.size() / 3 << " atoms\n1 atom types\n\n";
    data << "0.0 " << a << " xlo xhi\n";
    data << "0.0 " << a << " ylo yhi\n";
    data << "0.0 " << a << " zlo zhi\n\n";
    data << "Masses\n\n1 28.0855\n\nAtoms # atomic\n\n";
    for (std::size_t i = 0; i < pos.size() / 3; ++i) {
      data << (i + 1) << " 1 " << pos[3 * i] << " " << pos[3 * i + 1] << " "
           << pos[3 * i + 2] << "\n";
    }
  }
  {
    std::ofstream in(dir + "/in.sw");
    in << "units metal\natom_style atomic\nboundary p p p\n"
          "atom_modify map yes\nread_data data.si\n"
          "pair_style sw\npair_coeff * * Si.sw Si\nrun 0\n"
          "variable pxx equal pxx\nvariable pyy equal pyy\n"
          "variable pzz equal pzz\nvariable pxy equal pxy\n"
          "variable pxz equal pxz\nvariable pyz equal pyz\n"
          "print \"RGPOT_LAMMPS ${pxx} ${pyy} ${pzz} ${pxy} ${pxz} ${pyz}\"\n";
  }
  const std::string cmd = "cd " + dir + " && " + lmp +
                          " -in in.sw -log none -screen none > lmp.out 2> lmp.err";
  const int status = std::system(cmd.c_str());
  if (status != 0) {
    std::ifstream err(dir + "/lmp.err");
    std::stringstream buffer;
    buffer << err.rdbuf();
    SKIP("LAMMPS did not accept the SW script: " << buffer.str());
  }
  std::ifstream out(dir + "/lmp.out");
  std::string line;
  std::array<double, 6> pressure{};
  bool found = false;
  while (std::getline(out, line)) {
    if (line.rfind("RGPOT_LAMMPS", 0) == 0) {
      std::istringstream row(line.substr(std::string("RGPOT_LAMMPS").size()));
      for (double &p : pressure) {
        row >> p;
      }
      found = static_cast<bool>(row);
    }
  }
  REQUIRE(found);
  // LAMMPS metal pressure is bars, positive in compression.
  // 1 eV/Angstrom^3 = 1.602176634e6 bar. ASE stress = -P.
  constexpr double kBarPerEV = 1.602176634e6;
  const int index[6] = {0, 4, 8, 1, 2, 5};
  for (int k = 0; k < 6; ++k) {
    const double ase = -pressure[static_cast<std::size_t>(k)] / kBarPerEV;
    CAPTURE(k, ase, got.stress[index[k]]);
    REQUIRE_THAT(got.stress[index[k]], WithinAbs(ase, 1e-5));
  }
  std::system(("rm -rf " + dir).c_str());
}
