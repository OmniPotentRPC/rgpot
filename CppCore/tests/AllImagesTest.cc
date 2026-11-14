// MIT License
// Copyright 2023--present rgpot developers
//
// All-images pair sums on cells narrower than twice the cutoff. Each
// kernel is checked against the minimum image on a supercell wide enough
// for one image per pair, against central differences of its energy, and
// against ASE (data/all_images/make_ase_ref.py) for Lennard-Jones and a
// tabulated EAM. The default minimum image must refuse those cells.

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <catch2/catch_all.hpp>

#include "rgpot/EAM/EAMPot.hpp"
#ifdef RGPOT_HAS_EAM_TABLE
#include "rgpot/EAMTable/EAMTablePot.hpp"
#endif
#include "rgpot/LennardJones/LJPot.hpp"
#include "rgpot/Morse/MorsePot.hpp"
#include "rgpot/SuttonChen/SuttonChenPot.hpp"
#include "rgpot/nlist/NeighbourList.hpp"

using rgpot::nlist::Images;

namespace {

const std::string kData = "CppCore/tests/data/";

struct Cell {
  std::vector<double> pos;
  std::vector<int> z;
  std::array<double, 9> box{};
  [[nodiscard]] std::size_t size() const { return z.size(); }
};

struct Result {
  double energy = 0.0;
  std::vector<double> forces;
  double stress[9] = {};
  int has_stress = 0;
};

template <class Pot> Result run(const Pot &pot, const Cell &c) {
  Result r;
  r.forces.assign(3 * c.size(), 0.0);
  rgpot::ForceOut out{};
  out.F = r.forces.data();
  pot.forceImpl(rgpot::ForceInput{c.size(), c.pos.data(), c.z.data(), c.box.data()},
                &out);
  r.energy = out.energy;
  r.has_stress = out.has_stress;
  std::copy(out.stress, out.stress + 9, r.stress);
  return r;
}

double volume(const std::array<double, 9> &h) {
  return std::abs(h[0] * (h[4] * h[8] - h[5] * h[7]) - h[1] * (h[3] * h[8] - h[5] * h[6]) +
                  h[2] * (h[3] * h[7] - h[4] * h[6]));
}

/// Small deterministic displacements, so the forces are not zero by symmetry.
void jitter(Cell &c, double size) {
  for (std::size_t k = 0; k < c.pos.size(); ++k) {
    c.pos[k] += size * std::sin(1.7 * static_cast<double>(k) + 0.3);
  }
}

/// One atom of an fcc primitive cell under a small shear. The atom sits
/// outside the cell: the kernels must not assume wrapped positions.
Cell fccPrimitive(double a, int z) {
  Cell c;
  const double h[9] = {0.5 * a, 0.5 * a, 0, 0, 0.5 * a, 0.5 * a, 0.5 * a, 0, 0.5 * a};
  const double strain[9] = {1.01, 0.02, -0.01, 0.0, 0.985, 0.03, 0.0, 0.0, 1.02};
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      double s = 0.0;
      for (int k = 0; k < 3; ++k) {
        s += h[3 * i + k] * strain[3 * k + j];
      }
      c.box[static_cast<std::size_t>(3 * i + j)] = s;
    }
  }
  c.pos = {0.3 * a, -0.2 * a, 1.1 * a};
  c.z = {z};
  return c;
}

/// Three atoms in a sheared cell; the third is one cell vector out.
Cell triclinic(double a, std::vector<int> z) {
  Cell c;
  c.box = {a, 0, 0, 0.35 * a, 0.9 * a, 0, -0.2 * a, 0.3 * a, 1.1 * a};
  const double frac[3][3] = {{0.1, 0.2, 0.3}, {0.6, 0.55, 0.4}, {1.35, 0.8, 0.85}};
  for (const auto &f : frac) {
    for (int k = 0; k < 3; ++k) {
      c.pos.push_back(f[0] * c.box[static_cast<std::size_t>(k)] +
                      f[1] * c.box[static_cast<std::size_t>(3 + k)] +
                      f[2] * c.box[static_cast<std::size_t>(6 + k)]);
    }
  }
  c.z = std::move(z);
  jitter(c, 0.03 * a);
  return c;
}

/// fcc(100) layers with a one-atom lateral cell and vacuum wider than
/// twice the cutoff.
Cell slab(double a, std::vector<int> z, double cutoff) {
  Cell c;
  const double side = a / std::sqrt(2.0);
  const auto layers = z.size();
  for (std::size_t k = 0; k < layers; ++k) {
    const double off = (k % 2 == 1) ? 0.5 * side : 0.0;
    c.pos.insert(c.pos.end(), {off, off, 0.5 * a * static_cast<double>(k)});
  }
  const double height = 0.5 * a * static_cast<double>(layers - 1) + 2.0 * cutoff + 1.0;
  c.box = {side, 0, 0, 0, side, 0, 0, 0, height};
  c.z = std::move(z);
  jitter(c, 0.03 * a);
  return c;
}

/// Simple cubic, one atom: the EAM cell kernel takes the box diagonal only.
Cell simpleCubic(double a, int z) {
  Cell c;
  c.box = {a, 0, 0, 0, a, 0, 0, 0, a};
  c.pos = {0.2, -0.1, 0.4};
  c.z = {z};
  return c;
}

/// Conventional fcc cube, four atoms.
Cell fccCube(double a, std::vector<int> z) {
  Cell c;
  c.box = {a, 0, 0, 0, a, 0, 0, 0, a};
  const double basis[4][3] = {{0, 0, 0}, {0, 0.5, 0.5}, {0.5, 0, 0.5}, {0.5, 0.5, 0}};
  for (const auto &b : basis) {
    c.pos.insert(c.pos.end(), {b[0] * a, b[1] * a, b[2] * a});
  }
  c.z = std::move(z);
  jitter(c, 0.03 * a);
  return c;
}

/// Copies per cell vector so that every perpendicular width of the
/// supercell clears twice the cutoff.
std::array<int, 3> copiesFor(const Cell &c, double cutoff) {
  double width[3];
  rgpot::nlist::perpendicularWidths(c.box.data(), width);
  std::array<int, 3> n{};
  for (int k = 0; k < 3; ++k) {
    n[static_cast<std::size_t>(k)] =
        std::max(1, static_cast<int>(std::ceil((2.0 * cutoff + 0.5) / width[k])));
  }
  return n;
}

/// Supercell with the original atoms first.
Cell replicate(const Cell &c, const std::array<int, 3> &n) {
  Cell out;
  for (int a = 0; a < n[0]; ++a) {
    for (int b = 0; b < n[1]; ++b) {
      for (int d = 0; d < n[2]; ++d) {
        for (std::size_t i = 0; i < c.size(); ++i) {
          for (int k = 0; k < 3; ++k) {
            const auto uk = static_cast<std::size_t>(k);
            out.pos.push_back(c.pos[3 * i + uk] + a * c.box[uk] + b * c.box[3 + uk] +
                              d * c.box[6 + uk]);
          }
          out.z.push_back(c.z[i]);
        }
      }
    }
  }
  for (int k = 0; k < 3; ++k) {
    for (int j = 0; j < 3; ++j) {
      out.box[static_cast<std::size_t>(3 * k + j)] =
          n[static_cast<std::size_t>(k)] * c.box[static_cast<std::size_t>(3 * k + j)];
    }
  }
  return out;
}

void sameAsSupercell(const Result &small, const Result &big, std::size_t n, int copies,
                     double etol, double ftol, double stol) {
  CAPTURE(small.energy, big.energy, copies);
  REQUIRE_THAT(small.energy, Catch::Matchers::WithinAbs(big.energy / copies, etol));
  for (std::size_t k = 0; k < 3 * n; ++k) {
    CAPTURE(k);
    REQUIRE_THAT(small.forces[k], Catch::Matchers::WithinAbs(big.forces[k], ftol));
  }
  REQUIRE(small.has_stress == big.has_stress);
  if (small.has_stress != 0) {
    for (int k = 0; k < 9; ++k) {
      CAPTURE(k);
      REQUIRE_THAT(small.stress[k], Catch::Matchers::WithinAbs(big.stress[k], stol));
    }
  }
}

template <class Pot>
void forcesByDifference(const Pot &pot, const Cell &c, const Result &got, double tol) {
  constexpr double h = 1e-5;
  for (std::size_t k = 0; k < c.pos.size(); ++k) {
    Cell plus = c;
    Cell minus = c;
    plus.pos[k] += h;
    minus.pos[k] -= h;
    const double expected = -(run(pot, plus).energy - run(pot, minus).energy) / (2.0 * h);
    CAPTURE(k, expected);
    REQUIRE_THAT(got.forces[k], Catch::Matchers::WithinAbs(expected, tol));
  }
}

/// sigma_rc = (1/V) dE/d eps for x_c += eps x_r on atoms and cell rows.
template <class Pot>
void stressByDifference(const Pot &pot, const Cell &c, const Result &got, double tol) {
  REQUIRE(got.has_stress == 1);
  constexpr double h = 1e-6;
  const double v = volume(c.box);
  for (int row = 0; row < 3; ++row) {
    for (int col = 0; col < 3; ++col) {
      auto energyAt = [&](double eps) {
        Cell s = c;
        for (std::size_t i = 0; i < c.size(); ++i) {
          s.pos[3 * i + static_cast<std::size_t>(col)] +=
              eps * c.pos[3 * i + static_cast<std::size_t>(row)];
        }
        for (int k = 0; k < 3; ++k) {
          s.box[static_cast<std::size_t>(3 * k + col)] +=
              eps * c.box[static_cast<std::size_t>(3 * k + row)];
        }
        return run(pot, s).energy;
      };
      const double expected = (energyAt(h) - energyAt(-h)) / (2.0 * h * v);
      CAPTURE(row, col, expected);
      REQUIRE_THAT(got.stress[3 * row + col], Catch::Matchers::WithinAbs(expected, tol));
    }
  }
}

struct Tolerances {
  double energy, force, stress, fd_force, fd_stress;
};

/// The checks every kernel runs on a narrow cell.
template <class Pot>
void checkCell(const Pot &all, const Pot &minimum, const Cell &c, double cutoff,
               const Tolerances &tol) {
  REQUIRE_THROWS_AS(run(minimum, c), std::invalid_argument);
  const auto n = copiesFor(c, cutoff);
  const int copies = n[0] * n[1] * n[2];
  REQUIRE(copies > 1);
  const Cell big = replicate(c, n);
  const Result small = run(all, c);
  const Result folded = run(minimum, big);
  sameAsSupercell(small, folded, c.size(), copies, tol.energy, tol.force, tol.stress);
  // On the supercell one image per pair is the whole sum, so both modes agree.
  const Result listed = run(all, big);
  sameAsSupercell(listed, folded, big.size(), 1, tol.energy * copies, tol.force,
                  tol.stress);
  forcesByDifference(all, c, small, tol.fd_force);
  if (small.has_stress != 0) {
    stressByDifference(all, c, small, tol.fd_stress);
  }
}

struct AseRef {
  Cell cell;
  double energy = 0.0;
  double stress[6] = {};
  std::vector<double> forces;
};

AseRef loadAse(const std::string &name) {
  std::ifstream in(kData + "all_images/" + name + ".ref");
  REQUIRE(in.good());
  AseRef ref;
  std::string line, key;
  std::size_t n = 0;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') {
      continue;
    }
    std::istringstream row(line);
    row >> key;
    if (key == "box") {
      for (double &v : ref.cell.box) {
        row >> v;
      }
    } else if (key == "energy") {
      row >> ref.energy;
    } else if (key == "stress") {
      for (double &v : ref.stress) {
        row >> v;
      }
    } else if (key == "atoms") {
      row >> n;
      break;
    }
  }
  for (std::size_t i = 0; i < n; ++i) {
    int z = 0;
    double x[3], f[3];
    in >> z >> x[0] >> x[1] >> x[2] >> f[0] >> f[1] >> f[2];
    ref.cell.z.push_back(z);
    ref.cell.pos.insert(ref.cell.pos.end(), {x[0], x[1], x[2]});
    ref.forces.insert(ref.forces.end(), {f[0], f[1], f[2]});
  }
  REQUIRE(ref.cell.size() == n);
  return ref;
}

template <class Pot>
void sameAsAse(const Pot &pot, const std::string &name, double etol, double ftol,
               double stol) {
  CAPTURE(name);
  const AseRef ref = loadAse(name);
  const Result got = run(pot, ref.cell);
  REQUIRE_THAT(got.energy, Catch::Matchers::WithinAbs(ref.energy, etol));
  for (std::size_t k = 0; k < ref.forces.size(); ++k) {
    CAPTURE(k);
    REQUIRE_THAT(got.forces[k], Catch::Matchers::WithinAbs(ref.forces[k], ftol));
  }
  REQUIRE(got.has_stress == 1);
  // ASE Voigt order: xx yy zz yz xz xy.
  const int voigt[6] = {0, 4, 8, 5, 2, 1};
  for (int k = 0; k < 6; ++k) {
    CAPTURE(k);
    REQUIRE_THAT(got.stress[voigt[k]], Catch::Matchers::WithinAbs(ref.stress[k], stol));
  }
}

constexpr rgpot::LJConfig kLJ{.u0 = 1.0, .cutoff = 2.5, .psi = 1.0};
constexpr Tolerances kPairTol{1e-11, 1e-10, 1e-11, 1e-6, 1e-7};

#ifdef RGPOT_HAS_EAM_TABLE
rgpot::EAMTablePot table(const std::string &file, rgpot::EAMTableFormat format,
                         Images images) {
  return rgpot::EAMTablePot({.path = kData + file,
                             .format = format,
                             .periodic = true,
                             .stress = true,
                             .images = images});
}
#endif

} // namespace

TEST_CASE("image pairs match an explicit lattice sum", "[nlist][images]") {
  // Lennard-Jones energy as a direct sum over shifts far past the cutoff.
  const Cell c = triclinic(1.6, {18, 18, 18});
  const double rc = kLJ.cutoff;
  const double shift = 4.0 * std::pow(1.0 / rc, 12) - 4.0 * std::pow(1.0 / rc, 6);
  double direct = 0.0;
  constexpr int m = 6;
  for (std::size_t i = 0; i < c.size(); ++i) {
    for (std::size_t j = 0; j < c.size(); ++j) {
      for (int a = -m; a <= m; ++a) {
        for (int b = -m; b <= m; ++b) {
          for (int d = -m; d <= m; ++d) {
            if (i == j && a == 0 && b == 0 && d == 0) {
              continue;
            }
            double r2 = 0.0;
            for (int k = 0; k < 3; ++k) {
              const auto uk = static_cast<std::size_t>(k);
              const double x = c.pos[3 * j + uk] - c.pos[3 * i + uk] + a * c.box[uk] +
                               b * c.box[3 + uk] + d * c.box[6 + uk];
              r2 += x * x;
            }
            if (r2 <= rc * rc) {
              const double s6 = 1.0 / (r2 * r2 * r2);
              direct += 0.5 * (4.0 * (s6 * s6 - s6) - shift);
            }
          }
        }
      }
    }
  }
  const rgpot::LJPot pot(kLJ, Images::All);
  REQUIRE_THAT(run(pot, c).energy, Catch::Matchers::WithinRel(direct, 1e-12));

  // Moving an atom by whole cell vectors changes nothing.
  Cell moved = c;
  for (int k = 0; k < 3; ++k) {
    const auto uk = static_cast<std::size_t>(k);
    moved.pos[3 + uk] += 3.0 * c.box[uk] - 2.0 * c.box[6 + uk];
  }
  const Result a = run(pot, c);
  const Result b = run(pot, moved);
  REQUIRE_THAT(b.energy, Catch::Matchers::WithinRel(a.energy, 1e-12));
  for (std::size_t k = 0; k < a.forces.size(); ++k) {
    REQUIRE_THAT(b.forces[k], Catch::Matchers::WithinAbs(a.forces[k], 1e-10));
  }
}

TEST_CASE("the minimum image refuses only cells where it drops pairs",
          "[nlist][images]") {
  // A cutoff of 15 in a 20 A cell is beyond half the width, but this pair
  // has no second image inside it: the minimum image is exact and runs.
  Cell pair;
  pair.box = {20, 0, 0, 0, 21, 0, 0, 0, 22};
  pair.pos = {0.0, 0.0, 0.0, 1.3, 0.4, -0.2};
  pair.z = {18, 18};
  const rgpot::LJPot minimum{rgpot::LJConfig{.cutoff = 15.0}};
  const rgpot::LJPot all(rgpot::LJConfig{.cutoff = 15.0}, Images::All);
  const Result a = run(minimum, pair);
  const Result b = run(all, pair);
  REQUIRE_THAT(a.energy, Catch::Matchers::WithinRel(b.energy, 1e-13));
  // Moved to 5.5 A apart, the pair has a second image at 14.5 A.
  pair.pos[3] = 5.5;
  REQUIRE_THROWS_WITH(run(minimum, pair),
                      Catch::Matchers::ContainsSubstring("two periodic images"));
  REQUIRE_NOTHROW(run(all, pair));
  // One atom in an 8 A cell meets its own image.
  Cell one;
  one.box = {8, 0, 0, 0, 30, 0, 0, 0, 30};
  one.pos = {0.0, 0.0, 0.0};
  one.z = {18};
  const rgpot::LJPot nine{rgpot::LJConfig{.cutoff = 9.0}};
  REQUIRE_THROWS_WITH(run(nine, one), Catch::Matchers::ContainsSubstring("own periodic image"));
  REQUIRE(rgpot::LJPot(kLJ, Images::All).paramsKey() != rgpot::LJPot(kLJ).paramsKey());
  // Two atoms on one point give a NaN energy in both modes.
  Cell same;
  same.box = {3, 0, 0, 0, 3, 0, 0, 0, 3};
  same.pos = {0.5, 0.5, 0.5, 0.5, 0.5, 0.5};
  same.z = {18, 18};
  REQUIRE(std::isnan(run(rgpot::LJPot(kLJ, Images::All), same).energy));
  same.box = {6, 0, 0, 0, 6, 0, 0, 0, 6};
  REQUIRE(std::isnan(run(rgpot::LJPot(kLJ), same).energy));
}

TEST_CASE("all-images Lennard-Jones", "[LJPot][images]") {
  const rgpot::LJPot all(kLJ, Images::All);
  const rgpot::LJPot minimum(kLJ);
  SECTION("fcc primitive cell") { checkCell(all, minimum, fccPrimitive(1.56, 18), 2.5, kPairTol); }
  SECTION("triclinic cell") {
    checkCell(all, minimum, triclinic(1.6, {18, 18, 18}), 2.5, kPairTol);
  }
  SECTION("thin slab") { checkCell(all, minimum, slab(1.56, {18, 18, 18}, 2.5), 2.5, kPairTol); }
  SECTION("ASE LennardJones") {
    for (const char *name : {"lj_primitive", "lj_triclinic", "lj_slab"}) {
      sameAsAse(all, name, 1e-11, 1e-11, 1e-11);
    }
  }
}

TEST_CASE("all-images Morse", "[MorsePot][images]") {
  const rgpot::MorseConfig cfg{.cutoff = 6.0};
  const rgpot::MorsePot all(cfg, Images::All);
  const rgpot::MorsePot minimum(cfg);
  SECTION("fcc primitive cell") { checkCell(all, minimum, fccPrimitive(3.92, 78), 6.0, kPairTol); }
  SECTION("triclinic cell") {
    checkCell(all, minimum, triclinic(3.6, {78, 78, 78}), 6.0, kPairTol);
  }
  SECTION("thin slab") { checkCell(all, minimum, slab(3.92, {78, 78, 78}, 6.0), 6.0, kPairTol); }
}

TEST_CASE("all-images Sutton-Chen", "[SuttonChen][images]") {
  rgpot::SuttonChenConfig cfg{.cutoff = 3.0, .periodic = true, .stress = true};
  const rgpot::SuttonChenPot minimum(cfg);
  cfg.images = Images::All;
  const rgpot::SuttonChenPot all(cfg);
  const Tolerances tol{1e-10, 1e-9, 1e-10, 1e-5, 1e-6};
  // Off sqrt(2), so no lattice vector lands on the unshifted cutoff of 3.
  const double a = 1.43;
  SECTION("fcc primitive cell") { checkCell(all, minimum, fccPrimitive(a, 78), 3.0, tol); }
  SECTION("triclinic cell") { checkCell(all, minimum, triclinic(1.05, {78, 78, 78}), 3.0, tol); }
  SECTION("thin slab") { checkCell(all, minimum, slab(a, {78, 78, 78}, 3.0), 3.0, tol); }
  SECTION("C entry point") {
    const Cell c = fccPrimitive(a, 78);
    rgpot_sc_config raw{1.0, 34.408, 1.0, 10, 8, 3.0, RGPOT_SC_ALL_IMAGES};
    double energy = 0.0;
    double forces[3] = {};
    REQUIRE(rgpot_sutton_chen_energy_forces(&raw, 1, c.pos.data(), c.box.data(), &energy,
                                            forces) == 0);
    REQUIRE_THAT(energy, Catch::Matchers::WithinRel(run(all, c).energy, 1e-13));
    raw.periodic = 1;
    REQUIRE(rgpot_sutton_chen_energy_forces(&raw, 1, c.pos.data(), c.box.data(), &energy,
                                            forces) != 0);
  }
  SECTION("all images need a cutoff") {
    cfg.cutoff = 0.0;
    REQUIRE_THROWS_AS(run(rgpot::SuttonChenPot(cfg), fccPrimitive(a, 78)),
                      std::invalid_argument);
  }
}

#ifdef RGPOT_HAS_EAM_TABLE
TEST_CASE("all-images tabulated EAM", "[EAMTable][images]") {
  const Tolerances tol{1e-10, 1e-9, 1e-11, 1e-5, 1e-6};
  SECTION("funcfl Cu_u3") {
    const auto all = table("eam_table/Cu_u3.eam", rgpot::EAMTableFormat::Funcfl, Images::All);
    const auto minimum =
        table("eam_table/Cu_u3.eam", rgpot::EAMTableFormat::Funcfl, Images::Minimum);
    checkCell(all, minimum, fccPrimitive(3.615, 29), 4.95, tol);
    checkCell(all, minimum, triclinic(3.5, {29, 29, 29}), 4.95, tol);
    checkCell(all, minimum, slab(3.615, {29, 29, 29}, 4.95), 4.95, tol);
  }
  SECTION("setfl CuNi") {
    const auto all =
        table("eam_table/CuNi.eam.alloy", rgpot::EAMTableFormat::Setfl, Images::All);
    const auto minimum =
        table("eam_table/CuNi.eam.alloy", rgpot::EAMTableFormat::Setfl, Images::Minimum);
    const double rc = 6.394332378;
    checkCell(all, minimum, fccPrimitive(3.56, 28), rc, tol);
    checkCell(all, minimum, triclinic(3.5, {28, 29, 28}), rc, tol);
    checkCell(all, minimum, slab(3.56, {29, 28, 29}, rc), rc, tol);
  }
  SECTION("ASE EAM") {
    const auto all = table("all_images/synthetic.eam.alloy", rgpot::EAMTableFormat::Setfl,
                           Images::All);
    for (const char *name : {"eam_primitive", "eam_triclinic", "eam_slab"}) {
      sameAsAse(all, name, 1e-8, 1e-7, 5e-9);
    }
  }
}

#endif

TEST_CASE("all-images cell-list EAM", "[EAMPot][images]") {
  const rgpot::EAMPot all(rgpot::EAMCellConfig{.images = Images::All});
  const rgpot::EAMPot minimum;
  const Tolerances tol{1e-10, 1e-9, 0.0, 1e-5, 0.0};
  const double rc = 5.555;
  SECTION("one atom") { checkCell(all, minimum, simpleCubic(2.7, 13), rc, tol); }
  SECTION("fcc cube") { checkCell(all, minimum, fccCube(4.05, {13, 13, 13, 13}), rc, tol); }
  SECTION("mixed fcc cube") {
    // A mixed pair takes its pair term from the lower-indexed atom, so a
    // supercell, which renumbers the images, is not the same sum. The
    // two modes still agree on a wide cell.
    const Cell c = fccCube(4.0, {13, 46, 79, 13});
    REQUIRE_THROWS_AS(run(minimum, c), std::invalid_argument);
    const Cell big = replicate(c, copiesFor(c, rc));
    sameAsSupercell(run(all, big), run(minimum, big), big.size(), 1, 1e-9, tol.force, 0.0);
    forcesByDifference(all, c, run(all, c), tol.fd_force);
  }
  SECTION("thin slab") { checkCell(all, minimum, slab(4.05, {13, 13, 13}, rc), rc, tol); }
}
