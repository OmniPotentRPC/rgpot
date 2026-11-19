// MIT License
// Copyright 2023--present rgpot developers
//
// rgpot::D2CpmdPot against OpenCPMD's own empirical D2 (062582b, ALL DFT-D2
// with BLYP) on water and methane in cubic, orthorhombic and triclinic
// cells, including a pair on either side of half the cell and pairs on
// either side of CPMD's 6 R0 truncation; finite differences for its forces
// and stress; the cusp at half the cell that rgpot::D2Pot does not have;
// and the composite that swaps CPMD's term for the lattice sum.
// scripts/regen_d2_cpmd_references.py writes CppCore/tests/data/d2cpmd/.

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <catch2/catch_all.hpp>

#include "dftd_tolerances.hpp"
#include "rgpot/D2Pot/D2CpmdPot.hpp"
#include "rgpot/D2Pot/D2Pot.hpp"
#include "rgpot/SumPot/SumPot.hpp"
#include "rgpot/units.hpp"

using rgpot::testtol::kEnergyThr;
using rgpot::testtol::kGradThr;
using rgpot::units::BOHR_TO_ANGSTROM;
using rgpot::units::HARTREE_TO_EV;

namespace {

// OpenCPMD's bohr (1 / fbohr), which the reference files and D2CpmdPot's
// input conversion use; the outputs use rgpot's, as cpmdc does.
constexpr double kCpmdBohr = 0.529177210859;
constexpr double kCpmdC6O = 0.70;
constexpr double kCpmdR0O = 1.342;
constexpr double kCpmdC6Conv = 1.733979e1;
constexpr double kCpmdAlpha = 20.0;
constexpr double kCpmdS6 = 1.2;

struct Reference {
  std::vector<int> z;
  std::vector<double> pos; // Angstrom
  std::array<double, 9> box{};
  double energy = 0.0;  // Ha
  double printed = 0.0; // Ha, CPMD's VDW ENERGY line
  std::vector<double> force; // Ha/bohr
};

Reference load(const std::string &name) {
  const std::string path = "CppCore/tests/data/d2cpmd/" + name + ".txt";
  std::ifstream in(path);
  INFO(path);
  REQUIRE(in.good());
  Reference r;
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') {
      continue;
    }
    std::istringstream ss(line);
    std::string key;
    ss >> key;
    if (key == "box") {
      for (double &v : r.box) {
        ss >> v;
      }
    } else if (key == "atom") {
      int z = 0;
      double x = 0, y = 0, w = 0;
      ss >> z >> x >> y >> w;
      r.z.push_back(z);
      r.pos.insert(r.pos.end(), {x, y, w});
    } else if (key == "energy") {
      ss >> r.energy;
    } else if (key == "printed") {
      ss >> r.printed;
    } else if (key == "force") {
      double a = 0, b = 0, c = 0;
      ss >> a >> b >> c;
      r.force.insert(r.force.end(), {a, b, c});
    }
  }
  REQUIRE(r.force.size() == r.pos.size());
  return r;
}

struct Result {
  double energy = 0.0; // eV
  std::vector<double> forces;
  std::array<double, 9> stress{};
  bool has_stress = false;
};

template <typename Pot>
Result run(Pot &&pot, const std::vector<int> &z, const std::vector<double> &pos,
           const std::array<double, 9> &box) {
  Result r;
  r.forces.assign(pos.size(), 0.0);
  const rgpot::ForceInput in{z.size(), pos.data(), z.data(), box.data()};
  rgpot::ForceOut out{};
  out.F = r.forces.data();
  pot.forceBatch(rgpot::ForceBatch{.nSystems = 1, .in = &in, .out = &out});
  r.energy = out.energy;
  r.has_stress = out.has_stress != 0;
  std::copy(out.stress, out.stress + 9, r.stress.begin());
  return r;
}

double volume(const std::array<double, 9> &h) {
  return std::abs(h[0] * (h[4] * h[8] - h[5] * h[7]) - h[1] * (h[3] * h[8] - h[5] * h[6]) +
                  h[2] * (h[3] * h[7] - h[4] * h[6]));
}

std::array<double, 3> cartesian_from_fractional(const std::array<double, 9> &h,
                                                const std::array<double, 3> &s) {
  return {s[0] * h[0] + s[1] * h[3] + s[2] * h[6],
          s[0] * h[1] + s[1] * h[4] + s[2] * h[7],
          s[0] * h[2] + s[1] * h[5] + s[2] * h[8]};
}

double norm(const std::array<double, 3> &v) {
  return std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

double brute_minimum_image_distance(const std::array<double, 9> &h,
                                    const std::array<double, 3> &fractional_delta) {
  double best = std::numeric_limits<double>::infinity();
  for (int nx = -1; nx <= 1; ++nx) {
    for (int ny = -1; ny <= 1; ++ny) {
      for (int nz = -1; nz <= 1; ++nz) {
        const std::array<double, 3> shifted{fractional_delta[0] - static_cast<double>(nx),
                                            fractional_delta[1] - static_cast<double>(ny),
                                            fractional_delta[2] - static_cast<double>(nz)};
        best = std::min(best, norm(cartesian_from_fractional(h, shifted)));
      }
    }
  }
  return best;
}

double cpmd_two_oxygen_energy_ev(double distance_angstrom) {
  const double r_bohr = distance_angstrom / kCpmdBohr;
  const double c6 = std::sqrt(kCpmdC6O * kCpmdC6O) * kCpmdC6Conv;
  const double rr = (kCpmdR0O + kCpmdR0O) / kCpmdBohr;
  const double damp = 1.0 + std::exp(-kCpmdAlpha * (r_bohr / rr - 1.0));
  const double hartree = -kCpmdS6 * c6 / (std::pow(r_bohr, 6) * damp);
  return hartree * HARTREE_TO_EV;
}

struct CellCase {
  const char *name;
  std::array<double, 9> box;
  std::array<double, 3> fractional_delta;
};

const std::array<CellCase, 4> &pbc3_cell_cases() {
  static const std::array<CellCase, 4> kCases{{
      {"fcc", {0.0, 2.0, 2.0, 2.0, 0.0, 2.0, 2.0, 2.0, 0.0},
       {-0.13670875044881015, 0.26099462104158744, 0.45981805354846594}},
      {"trigonal",
       {3.0, 0.0, 0.0, 1.2, 2.749545416973504, 0.0, 1.2, 0.7855844048495726,
        2.6349301969610393},
       {-0.13895857209664464, 0.42371828161708736, 0.4258527661071376}},
      {"hexagonal",
       {3.0, 0.0, 0.0, -1.5, 2.598076211, 0.0, 0.0, 0.0, 5.0},
       {-0.4908678763235309, 0.18057788068661407, 0.38419771091555066}},
      {"monoclinic", {4.0, 0.0, 0.0, 0.0, 5.0, 0.0, 1.8, 0.0, 6.0},
       {-0.39289603625100367, -0.2542605353421451, -0.24891177114900753}},
  }};
  return kCases;
}

const char *kSystems[] = {"cubic_water",         "cubic_below_half",
                          "cubic_above_half",    "ortho_water_methane",
                          "triclinic_water_methane", "cubic_truncation"};

} // namespace

TEST_CASE("CPMD's D2 matches OpenCPMD's empirical correction", "[d2cpmd][reference]") {
  rgpot::D2CpmdPot pot;
  for (const char *name : kSystems) {
    INFO(name);
    const Reference ref = load(name);
    const Result r = run(pot, ref.z, ref.pos, ref.box);
    // CPMD prints eight decimals, rounded.
    REQUIRE(std::abs(ref.printed - ref.energy) <= 0.5e-8);
    REQUIRE(std::abs(r.energy / HARTREE_TO_EV - ref.energy) <= kEnergyThr);
    const double to_au = BOHR_TO_ANGSTROM / HARTREE_TO_EV;
    for (std::size_t k = 0; k < ref.pos.size(); ++k) {
      INFO("component " << k);
      REQUIRE(std::abs(r.forces[k] * to_au - ref.force[k]) <= kGradThr);
    }
    REQUIRE(r.has_stress);
  }
}

TEST_CASE("CPMD's D2 forces and stress are derivatives of its energy", "[d2cpmd][fd]") {
  rgpot::D2CpmdPot pot;
  // Positions go in with CPMD's bohr and forces come out with rgpot's, so
  // the exact derivative of the energy is the force times this ratio.
  const double ratio = BOHR_TO_ANGSTROM / kCpmdBohr;
  const double step = std::cbrt(std::numeric_limits<double>::epsilon());
  const double bar = kGradThr * HARTREE_TO_EV / BOHR_TO_ANGSTROM;
  for (const char *name : {"cubic_water", "ortho_water_methane", "triclinic_water_methane"}) {
    INFO(name);
    const Reference ref = load(name);
    const Result r = run(pot, ref.z, ref.pos, ref.box);
    for (std::size_t k = 0; k < ref.pos.size(); ++k) {
      std::vector<double> p = ref.pos;
      p[k] += step;
      const double ep = run(pot, ref.z, p, ref.box).energy;
      p[k] -= 2.0 * step;
      const double em = run(pot, ref.z, p, ref.box).energy;
      INFO("component " << k);
      REQUIRE(std::abs(-(ep - em) / (2.0 * step) - r.forces[k] * ratio) <= bar);
    }
    const double v = volume(ref.box);
    for (int a = 0; a < 3; ++a) {
      for (int b = 0; b < 3; ++b) {
        auto strained = [&](double e) {
          std::array<double, 9> box = ref.box;
          std::vector<double> p = ref.pos;
          for (int i = 0; i < 3; ++i) {
            box[3 * i + b] += e * ref.box[3 * i + a];
          }
          for (std::size_t n = 0; n < ref.z.size(); ++n) {
            p[3 * n + b] += e * ref.pos[3 * n + a];
          }
          return run(pot, ref.z, p, box).energy;
        };
        const double dE = (strained(step) - strained(-step)) / (2.0 * step);
        INFO("stress " << a << b);
        REQUIRE(std::abs(dE - v * r.stress[3 * a + b] * ratio * ratio * ratio) <=
                bar * std::cbrt(v));
      }
    }
  }
}

TEST_CASE("CPMD's minimum image has a cusp at half the cell", "[d2cpmd][cusp]") {
  // Two O atoms on the x axis of a 7 Angstrom cube, a hair either side of
  // 3.5 Angstrom apart. The minimum image switches sides between the two, so
  // CPMD's force on the pair flips sign while its energy does not move; the
  // lattice sum has both images at all times and stays smooth.
  const double length = 7.0;
  const std::array<double, 9> box{length, 0, 0, 0, length, 0, 0, 0, length};
  const std::vector<int> z{8, 8};
  const double delta = std::cbrt(std::numeric_limits<double>::epsilon());
  auto at = [&](double x) { return std::vector<double>{0.0, 0.0, 0.0, x, 0.0, 0.0}; };
  const std::vector<double> below = at(0.5 * length - delta);
  const std::vector<double> above = at(0.5 * length + delta);

  rgpot::D2CpmdPot cpmd;
  const Result cb = run(cpmd, z, below, box);
  const Result ca = run(cpmd, z, above, box);
  INFO("CPMD forces " << cb.forces[3] << " and " << ca.forces[3] << " eV/A");
  REQUIRE(std::abs(cb.energy - ca.energy) <= kGradThr * HARTREE_TO_EV);
  REQUIRE(cb.forces[3] < 0.0);
  REQUIRE(ca.forces[3] > 0.0);
  REQUIRE(std::abs(cb.forces[3] + ca.forces[3]) <= 4.0 * delta * std::abs(cb.forces[3]));

  rgpot::D2Pot sum;
  const Result sb = run(sum, z, below, box);
  const Result sa = run(sum, z, above, box);
  INFO("lattice sum forces " << sb.forces[3] << " and " << sa.forces[3] << " eV/A");
  REQUIRE(std::abs(sb.forces[3] - sa.forces[3]) <= kGradThr * HARTREE_TO_EV / BOHR_TO_ANGSTROM);
  REQUIRE(std::abs(sb.forces[3]) <= kGradThr * HARTREE_TO_EV / BOHR_TO_ANGSTROM);
}

TEST_CASE("CPMD's D2 tables, functionals and limits", "[d2cpmd]") {
  REQUIRE(rgpot::d2_cpmd_config_for("blyp").s6 == 1.2);
  REQUIRE(rgpot::d2_cpmd_config_for("PBE").s6 == 0.75);
  REQUIRE(rgpot::d2_cpmd_config_for("revpbe").s6 == 1.25);
  REQUIRE_THROWS_AS(rgpot::d2_cpmd_config_for("b97-d"), std::invalid_argument);
  REQUIRE_THROWS_AS(rgpot::D2CpmdPot(rgpot::D2CpmdConfig{.vdweps = 0.0}),
                    std::invalid_argument);

  rgpot::D2CpmdPot pot;
  const std::array<double, 9> none{};
  const std::array<double, 9> box{30.0, 0, 0, 0, 30.0, 0, 0, 0, 30.0};
  const std::vector<double> pos{0.0, 0.0, 0.0, 3.0, 0.0, 0.0};
  // Francium to einsteinium have no parameters and give nothing.
  const Result fr = run(pot, std::vector<int>{87, 8}, pos, box);
  REQUIRE(fr.energy == 0.0);
  REQUIRE_THROWS_AS(run(pot, std::vector<int>{100, 8}, pos, box), std::invalid_argument);
  // An all-zero box is a molecule: no image at all.
  const Result mol = run(pot, std::vector<int>{8, 8}, pos, none);
  REQUIRE(mol.energy < 0.0);
  REQUIRE_FALSE(mol.has_stress);
  REQUIRE(mol.energy == run(pot, std::vector<int>{8, 8}, pos, box).energy);

  // The 6 R0 truncation: the O-O sum radius is 2 x 1.342 Angstrom.
  const double cut = 6.0 * 2.684;
  const double rel = std::cbrt(std::numeric_limits<double>::epsilon());
  const Result in = run(pot, std::vector<int>{8, 8},
                        std::vector<double>{0.0, 0.0, 0.0, cut * (1.0 - rel), 0.0, 0.0}, none);
  const Result out = run(pot, std::vector<int>{8, 8},
                         std::vector<double>{0.0, 0.0, 0.0, cut * (1.0 + rel), 0.0, 0.0}, none);
  REQUIRE(in.energy < 0.0);
  REQUIRE(out.energy == 0.0);
}

TEST_CASE("a composite swaps CPMD's D2 for the lattice sum", "[d2cpmd][composite]") {
  const Reference ref = load("cubic_water");
  rgpot::D2Pot lattice;
  rgpot::D2CpmdPot cpmd;
  const Result l = run(lattice, ref.z, ref.pos, ref.box);
  const Result c = run(cpmd, ref.z, ref.pos, ref.box);

  rgpot::SumPot sum;
  sum.add({.name = "undo cpmd", .weight = -1.0}, std::make_unique<rgpot::D2CpmdPot>());
  sum.add({.name = "d2"}, std::make_unique<rgpot::D2Pot>());
  REQUIRE(sum.caps().stress);
  const Result s = run(sum, ref.z, ref.pos, ref.box);
  REQUIRE(s.energy == Catch::Approx(l.energy - c.energy).margin(kEnergyThr * HARTREE_TO_EV));
  for (std::size_t k = 0; k < ref.pos.size(); ++k) {
    REQUIRE(s.forces[k] ==
            Catch::Approx(l.forces[k] - c.forces[k]).margin(kGradThr * HARTREE_TO_EV));
  }
  REQUIRE(sum.lastComponents()[0].energy == c.energy);
}

TEST_CASE("CPMD pbc3 image selection matches brute-force on key cell symmetries",
          "[d2cpmd][pbc3]") {
  rgpot::D2CpmdPot pot;
  for (const auto &sample : pbc3_cell_cases()) {
    INFO(sample.name);
    const auto delta = cartesian_from_fractional(sample.box, sample.fractional_delta);
    const std::vector<double> pos{0.0, 0.0, 0.0, delta[0], delta[1], delta[2]};
    const std::vector<int> z{8, 8};
    const Result got = run(pot, z, pos, sample.box);
    const double r = brute_minimum_image_distance(sample.box, sample.fractional_delta);
    const double want = cpmd_two_oxygen_energy_ev(r);
    INFO("distance " << std::setprecision(16) << r << " A");
    REQUIRE(got.energy == Catch::Approx(want).margin(kEnergyThr * HARTREE_TO_EV));
  }
}

TEST_CASE("CPMD pbc3 folds a displacement of several cells like one inside the cell",
          "[d2cpmd][pbc3]") {
  const std::array<std::array<double, 3>, 3> kShifts{{
      {2.0, -2.0, 2.0},
      {3.0, -4.0, 2.0},
      {-7.0, 5.0, -6.0},
  }};
  rgpot::D2CpmdPot pot;
  const std::vector<int> z{8, 8};
  for (const auto &sample : pbc3_cell_cases()) {
    const auto inside = cartesian_from_fractional(sample.box, sample.fractional_delta);
    const std::vector<double> pos_inside{0.0, 0.0, 0.0, inside[0], inside[1], inside[2]};
    const Result want = run(pot, z, pos_inside, sample.box);
    for (const auto &shift : kShifts) {
      INFO(sample.name << " shift " << shift[0] << " " << shift[1] << " " << shift[2]);
      const std::array<double, 3> far_fractional{sample.fractional_delta[0] + shift[0],
                                                 sample.fractional_delta[1] + shift[1],
                                                 sample.fractional_delta[2] + shift[2]};
      const auto far = cartesian_from_fractional(sample.box, far_fractional);
      const std::vector<double> pos_far{0.0, 0.0, 0.0, far[0], far[1], far[2]};
      const Result got = run(pot, z, pos_far, sample.box);
      REQUIRE(got.energy == Catch::Approx(want.energy).margin(kEnergyThr * HARTREE_TO_EV));
      REQUIRE(got.forces.size() == want.forces.size());
      for (std::size_t k = 0; k < want.forces.size(); ++k) {
        REQUIRE(got.forces[k] ==
                Catch::Approx(want.forces[k]).margin(kGradThr * HARTREE_TO_EV / BOHR_TO_ANGSTROM));
      }
    }
  }
}
