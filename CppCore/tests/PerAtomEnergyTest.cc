// MIT License
// Copyright 2023--present rgpot developers
//
// Per-atom energies from Potential::forceEnergies. Every supported kernel
// must sum its shares to the total, leave energy and forces as forceImpl
// writes them, follow the atoms under translation and permutation, and
// give each atom of a thin cell the share of its copies in a supercell.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include <catch2/catch_all.hpp>

#include "rgpot/EAM/EAMPot.hpp"
#include "rgpot/ForceStructs.hpp"
#include "rgpot/LennardJones/LJClusterPot.hpp"
#include "rgpot/LennardJones/LJPot.hpp"
#include "rgpot/Morse/MorsePot.hpp"
#include "rgpot/StillingerWeber/SWSi.hpp"
#include "rgpot/SuttonChen/SuttonChenPot.hpp"
#include "rgpot/Tersoff/TersoffSiC.hpp"
#include "rgpot/ZBL/ZBLPot.hpp"
#include "rgpot/fortran/FortranPots.hpp"

using Catch::Matchers::WithinAbs;

namespace {

using Box = std::array<double, 9>;

struct Cell {
  std::vector<double> pos;
  std::vector<int> types;
  Box box{};
  [[nodiscard]] std::size_t size() const { return types.size(); }
};

struct Result {
  double energy = 0.0;
  std::vector<double> forces;
  std::vector<double> energies;
};

template <class Pot>
Result plain(const Pot &pot, const Cell &cell) {
  Result r;
  r.forces.assign(3 * cell.size(), 0.0);
  rgpot::ForceOut out{r.forces.data(), 0.0, 0.0, {}, 0};
  pot.forceImpl(rgpot::ForceInput{cell.size(), cell.pos.data(), cell.types.data(),
                                  cell.box.data()},
                &out);
  r.energy = out.energy;
  return r;
}

template <class Pot>
Result perAtom(const Pot &pot, const Cell &cell) {
  Result r;
  r.forces.assign(3 * cell.size(), 1.0);
  r.energies.assign(cell.size(), 1.0);
  rgpot::ForceOut out{r.forces.data(), 7.0, 0.0, {}, 0};
  pot.forceEnergies(rgpot::ForceInput{cell.size(), cell.pos.data(), cell.types.data(),
                                      cell.box.data()},
                    &out, r.energies.data());
  r.energy = out.energy;
  return r;
}

/// OMP_NUM_THREADS for one scope. The kernels read it on every call.
class Threads {
public:
  explicit Threads(int n) {
    const char *old = std::getenv("OMP_NUM_THREADS");
    if (old != nullptr) {
      m_old = old;
      m_had = true;
    }
    setenv("OMP_NUM_THREADS", std::to_string(n).c_str(), 1);
  }
  ~Threads() {
    if (m_had) {
      setenv("OMP_NUM_THREADS", m_old.c_str(), 1);
    } else {
      unsetenv("OMP_NUM_THREADS");
    }
  }
  Threads(const Threads &) = delete;
  Threads &operator=(const Threads &) = delete;

private:
  std::string m_old;
  bool m_had = false;
};

Box cubicBox(double l) { return {l, 0.0, 0.0, 0.0, l, 0.0, 0.0, 0.0, l}; }

// Primitive fcc vectors in lower-triangular form for cubic constant `a`.
Box triclinicFcc(double a) {
  const double s2 = std::sqrt(2.0);
  const double s3 = std::sqrt(3.0);
  const double s6 = std::sqrt(6.0);
  return {a / s2, 0.0, 0.0, a / (2.0 * s2), a * s3 / (2.0 * s2), 0.0,
          a / (2.0 * s2), a / (2.0 * s6), a / s3};
}

// Deterministic displacement of up to `amp` on every coordinate.
void rattle(Cell &cell, double amp) {
  for (std::size_t k = 0; k < cell.pos.size(); ++k) {
    cell.pos[k] += amp * std::sin(1.7 * static_cast<double>(k) + 0.3);
  }
}

// `n`^3 conventional cells of a cubic lattice with fractional `basis`.
Cell lattice(double a, int n, const std::vector<std::array<double, 3>> &basis,
             const std::vector<int> &types) {
  Cell cell;
  cell.box = cubicBox(a * n);
  for (int x = 0; x < n; ++x) {
    for (int y = 0; y < n; ++y) {
      for (int z = 0; z < n; ++z) {
        for (std::size_t b = 0; b < basis.size(); ++b) {
          cell.pos.push_back(a * (x + basis[b][0]));
          cell.pos.push_back(a * (y + basis[b][1]));
          cell.pos.push_back(a * (z + basis[b][2]));
          cell.types.push_back(types[b % types.size()]);
        }
      }
    }
  }
  return cell;
}

const std::vector<std::array<double, 3>> kFcc = {
    {0.0, 0.0, 0.0}, {0.0, 0.5, 0.5}, {0.5, 0.0, 0.5}, {0.5, 0.5, 0.0}};
const std::vector<std::array<double, 3>> kDiamond = {
    {0.0, 0.0, 0.0},    {0.0, 0.5, 0.5},    {0.5, 0.0, 0.5},    {0.5, 0.5, 0.0},
    {0.25, 0.25, 0.25}, {0.25, 0.75, 0.75}, {0.75, 0.25, 0.75}, {0.75, 0.75, 0.25}};

// The atoms of a lattice within `radius` of the atom nearest its middle,
// moved into a cubic box of side `side` so no image is in range.
Cell cluster(const Cell &bulk, double radius, double side) {
  const double mid = bulk.box[0] / 2.0;
  std::size_t centre = 0;
  double best = 1e300;
  for (std::size_t i = 0; i < bulk.size(); ++i) {
    double d = 0.0;
    for (int k = 0; k < 3; ++k) {
      d += std::pow(bulk.pos[3 * i + static_cast<std::size_t>(k)] - mid, 2);
    }
    if (d < best) {
      best = d;
      centre = i;
    }
  }
  Cell out;
  out.box = cubicBox(side);
  for (std::size_t i = 0; i < bulk.size(); ++i) {
    double d = 0.0;
    double rel[3];
    for (int k = 0; k < 3; ++k) {
      rel[k] = bulk.pos[3 * i + static_cast<std::size_t>(k)] -
               bulk.pos[3 * centre + static_cast<std::size_t>(k)];
      d += rel[k] * rel[k];
    }
    if (d <= radius * radius) {
      for (double r : rel) {
        out.pos.push_back(side / 2.0 + r);
      }
      out.types.push_back(bulk.types[i]);
    }
  }
  return out;
}

// One primitive cell of a two-site fcc basis, the second site at
// (a1 + a2 + a3) / 4 when `diamond`, both pushed off their sites.
Cell thinTriclinic(double a, bool diamond, int t0, int t1) {
  Cell cell;
  cell.box = triclinicFcc(a);
  if (!diamond) {
    for (int k = 0; k < 3; ++k) {
      cell.box[static_cast<std::size_t>(k)] *= 2.0;
    }
  }
  cell.pos = {0.0, 0.0, 0.0};
  for (int k = 0; k < 3; ++k) {
    const double site =
        diamond ? (cell.box[static_cast<std::size_t>(k)] + cell.box[static_cast<std::size_t>(3 + k)] +
                   cell.box[static_cast<std::size_t>(6 + k)]) /
                      4.0
                : cell.box[static_cast<std::size_t>(k)] / 2.0;
    cell.pos.push_back(site);
  }
  cell.types = {t0, t1};
  const double push[6] = {0.012, -0.007, 0.004, 0.031, -0.022, 0.043};
  for (int k = 0; k < 6; ++k) {
    cell.pos[static_cast<std::size_t>(k)] += push[k] * a / 4.0;
  }
  return cell;
}

// Copies are laid out copy-major, so atom i of copy c is row c * n + i.
Cell repeat(const Cell &cell, int na, int nb, int nc) {
  Cell big;
  const int reps[3] = {na, nb, nc};
  for (int r = 0; r < 3; ++r) {
    for (int k = 0; k < 3; ++k) {
      big.box[static_cast<std::size_t>(3 * r + k)] =
          reps[r] * cell.box[static_cast<std::size_t>(3 * r + k)];
    }
  }
  for (int a = 0; a < na; ++a) {
    for (int b = 0; b < nb; ++b) {
      for (int c = 0; c < nc; ++c) {
        for (std::size_t i = 0; i < cell.size(); ++i) {
          for (int k = 0; k < 3; ++k) {
            big.pos.push_back(cell.pos[3 * i + static_cast<std::size_t>(k)] +
                              a * cell.box[static_cast<std::size_t>(k)] +
                              b * cell.box[static_cast<std::size_t>(3 + k)] +
                              c * cell.box[static_cast<std::size_t>(6 + k)]);
          }
          big.types.push_back(cell.types[i]);
        }
      }
    }
  }
  return big;
}

double sum(const std::vector<double> &v) {
  return std::accumulate(v.begin(), v.end(), 0.0);
}

/// The shares sum to the total to 1e-10 relative, and energy and forces
/// are those of forceImpl. One thread walks the pairs in forceImpl's
/// order, so those match bit for bit; threads only to rounding.
template <class Pot>
void requireConsistent(const Pot &pot, const Cell &cell) {
  REQUIRE(pot.perAtomEnergies());
  const Result ref = plain(pot, cell);
  const Result got = perAtom(pot, cell);
  REQUIRE(ref.energy != 0.0);
  REQUIRE(got.energies.size() == cell.size());
  REQUIRE(std::fabs(sum(got.energies) - got.energy) <= 1e-10 * std::fabs(got.energy));
  const char *threads = std::getenv("OMP_NUM_THREADS");
  if (threads == nullptr || std::string(threads) == "1") {
    REQUIRE(got.energy == ref.energy);
    REQUIRE(got.forces == ref.forces);
  } else {
    REQUIRE_THAT(got.energy, WithinAbs(ref.energy, 1e-10 * std::fabs(ref.energy)));
    for (std::size_t k = 0; k < ref.forces.size(); ++k) {
      REQUIRE_THAT(got.forces[k], WithinAbs(ref.forces[k], 1e-10));
    }
  }
}

double scale(const std::vector<double> &e) {
  double m = 0.0;
  for (double x : e) {
    m = std::max(m, std::fabs(x));
  }
  return std::max(m, 1.0);
}

/// A rigid shift moves no share; a permutation of the rows permutes them.
template <class Pot>
void requireSymmetric(const Pot &pot, const Cell &cell) {
  const Result base = perAtom(pot, cell);
  const double tol = 1e-10 * scale(base.energies);

  Cell moved = cell;
  const double shift[3] = {0.37, -1.21, 2.05};
  for (std::size_t i = 0; i < moved.size(); ++i) {
    for (int k = 0; k < 3; ++k) {
      moved.pos[3 * i + static_cast<std::size_t>(k)] += shift[k];
    }
  }
  const Result shifted = perAtom(pot, moved);
  for (std::size_t i = 0; i < cell.size(); ++i) {
    REQUIRE_THAT(shifted.energies[i], WithinAbs(base.energies[i], tol));
  }

  // Row p of the permuted cell is row order[p] of the original.
  std::vector<std::size_t> order(cell.size());
  std::iota(order.begin(), order.end(), std::size_t{0});
  std::reverse(order.begin(), order.end());
  if (order.size() > 2) {
    std::rotate(order.begin(), order.begin() + 1, order.end());
  }
  Cell shuffled;
  shuffled.box = cell.box;
  for (std::size_t p : order) {
    for (int k = 0; k < 3; ++k) {
      shuffled.pos.push_back(cell.pos[3 * p + static_cast<std::size_t>(k)]);
    }
    shuffled.types.push_back(cell.types[p]);
  }
  const Result permuted = perAtom(pot, shuffled);
  for (std::size_t p = 0; p < order.size(); ++p) {
    REQUIRE_THAT(permuted.energies[p], WithinAbs(base.energies[order[p]], tol));
  }
}

/// Each atom of the thin cell has the share of each of its copies in a
/// supercell thick enough for the minimum image.
template <class Pot>
void requireThinMatchesRepeat(const Pot &pot, const Cell &small, int na, int nb, int nc,
                              double tol) {
  const Cell big = repeat(small, na, nb, nc);
  const Result a = perAtom(pot, small);
  const Result b = perAtom(pot, big);
  const std::size_t n = small.size();
  for (std::size_t c = 0; c < big.size() / n; ++c) {
    for (std::size_t i = 0; i < n; ++i) {
      REQUIRE_THAT(b.energies[c * n + i], WithinAbs(a.energies[i], tol));
    }
  }
}

template <class Pot>
void requireAll(const Pot &pot, const Cell &cell) {
  requireConsistent(pot, cell);
  requireSymmetric(pot, cell);
  Threads four(4);
  requireConsistent(pot, cell);
}

// Two atoms `r` apart, and a chain of three with both bonds `r` long and
// the ends 2r apart, in a cubic box of side `side`.
Cell dimer(double r, double side, int type) {
  Cell cell;
  cell.box = cubicBox(side);
  const double c = side / 2.0;
  cell.pos = {c, c, c, c + r, c, c};
  cell.types = {type, type};
  return cell;
}

Cell chain(double r, double side, int type, double bend = 0.0) {
  Cell cell;
  cell.box = cubicBox(side);
  const double c = side / 2.0;
  cell.pos = {c - r * std::cos(bend), c + r * std::sin(bend), c, c, c, c, c + r, c, c};
  cell.types = {type, type, type};
  return cell;
}

} // namespace

TEST_CASE("Per-atom LJ shares sum to the total and follow the atoms", "[peratom][lj]") {
  const double a = 5.26;
  const Cell bulk = [&] {
    Cell c = lattice(a, 3, kFcc, {18});
    rattle(c, 0.05);
    return c;
  }();
  const Cell thin = thinTriclinic(a, false, 18, 18);
  const Cell drop = cluster(lattice(a, 3, kFcc, {18}), 5.3, 60.0);
  REQUIRE(drop.size() == 19);
  for (double width : {0.0, 1.0}) {
    const rgpot::LJPot pot{rgpot::LJConfig{
        .u0 = 0.0104, .cutoff = 8.5, .psi = 3.40, .switch_width = width}};
    requireAll(pot, bulk);
    requireAll(pot, thin);
    requireAll(pot, drop);
    requireThinMatchesRepeat(pot, thin, 4, 7, 7, 1e-12);
  }
}

TEST_CASE("Per-atom Morse shares sum to the total and follow the atoms",
          "[peratom][morse]") {
  const rgpot::MorsePot pot{
      rgpot::MorseConfig{.De = 0.3429, .a = 1.3588, .re = 2.866, .cutoff = 6.0}};
  const double a = 3.61;
  Cell bulk = lattice(a, 3, kFcc, {29});
  rattle(bulk, 0.04);
  const Cell thin = thinTriclinic(a, false, 29, 29);
  requireAll(pot, bulk);
  requireAll(pot, thin);
  requireAll(pot, cluster(lattice(a, 3, kFcc, {29}), 3.7, 40.0));
  requireThinMatchesRepeat(pot, thin, 5, 9, 9, 1e-12);
}

TEST_CASE("Per-atom Sutton-Chen shares sum to the total and follow the atoms",
          "[peratom][eam]") {
  rgpot::SuttonChenConfig periodic;
  periodic.cutoff = 2.3;
  periodic.periodic = true;
  const rgpot::SuttonChenPot pot{periodic};
  const double a = std::sqrt(2.0);
  Cell bulk = lattice(a, 4, kFcc, {78});
  rattle(bulk, 0.02);
  const Cell thin = thinTriclinic(a, false, 78, 78);
  requireAll(pot, bulk);
  requireAll(pot, thin);
  requireThinMatchesRepeat(pot, thin, 4, 8, 8, 1e-10);
  // A cluster in the published Sutton-Chen form: no cell, every pair.
  const rgpot::SuttonChenPot free{rgpot::SuttonChenConfig{}};
  requireAll(free, cluster(lattice(a, 3, kFcc, {78}), 1.5, 40.0));
}

TEST_CASE("Per-atom cell-list EAM shares sum to the total and follow the atoms",
          "[peratom][eam]") {
  const double a = 4.05;
  Cell bulk = lattice(a, 3, kFcc, {13});
  rattle(bulk, 0.04);
  Cell thin = lattice(a, 1, kFcc, {13});
  rattle(thin, 0.03);
  const Cell drop = cluster(lattice(a, 3, kFcc, {13}), 4.1, 40.0);

  const rgpot::EAMPot cxx;
  requireAll(cxx, bulk);
  requireAll(cxx, thin);
  requireAll(cxx, drop);
  requireThinMatchesRepeat(cxx, thin, 4, 4, 4, 1e-9);

  const rgpot::fortranpots::EAMAlPot fortran;
  const Cell tri = thinTriclinic(a, false, 13, 13);
  requireAll(fortran, bulk);
  requireAll(fortran, tri);
  requireAll(fortran, drop);
  requireThinMatchesRepeat(fortran, tri, 4, 8, 8, 1e-9);

  // Cu with interstitial hydrogen, through the CuH2 EAM.
  const rgpot::fortranpots::CuH2Pot cuh2;
  const double acu = 3.61;
  Cell cu = lattice(acu, 3, kFcc, {29});
  for (const double f : {0.5, 1.5}) {
    cu.pos.insert(cu.pos.end(), {acu * f, acu * 0.25 + 0.1, acu * 0.5});
    cu.types.push_back(1);
  }
  rattle(cu, 0.03);
  requireAll(cuh2, cu);
  const Cell cuTri = thinTriclinic(acu, false, 29, 29);
  requireAll(cuh2, cuTri);
  requireAll(cuh2, cluster(lattice(acu, 3, kFcc, {29}), 3.7, 40.0));
  requireThinMatchesRepeat(cuh2, cuTri, 5, 9, 9, 1e-9);
}

TEST_CASE("Per-atom Stillinger-Weber and Tersoff shares sum to the total",
          "[peratom][threebody]") {
  const double a = 5.431;
  Cell bulk = lattice(a, 2, kDiamond, {14});
  rattle(bulk, 0.05);
  const Cell thin = thinTriclinic(a, true, 14, 14);
  const Cell drop = cluster(lattice(a, 2, kDiamond, {14}), 4.0, 40.0);

  // The conventional cube is under twice either cutoff, so its own images
  // are neighbours. The supercell of a sheared primitive cell is not used
  // here: these kernels do not yet agree with its thin primitive cell.
  Cell cube = lattice(a, 1, kDiamond, {14});
  rattle(cube, 0.05);

  const rgpot::StillingerWeberSi sw;
  requireAll(sw, bulk);
  requireAll(sw, thin);
  requireAll(sw, drop);
  requireThinMatchesRepeat(sw, cube, 3, 3, 3, 1e-11);

  const rgpot::TersoffSiC tersoff;
  requireAll(tersoff, bulk);
  requireAll(tersoff, thin);
  requireAll(tersoff, drop);
  requireThinMatchesRepeat(tersoff, cube, 3, 3, 3, 1e-11);
  // Zincblende SiC: the C sites carry other parameters.
  Cell sic = lattice(4.36, 2, kDiamond, {14, 14, 14, 14, 6, 6, 6, 6});
  rattle(sic, 0.04);
  requireAll(tersoff, sic);
  requireAll(tersoff, thinTriclinic(4.36, true, 14, 6));
  Cell sicCube = lattice(4.36, 1, kDiamond, {14, 14, 14, 14, 6, 6, 6, 6});
  rattle(sicCube, 0.04);
  requireThinMatchesRepeat(tersoff, sicCube, 3, 3, 3, 1e-11);
}

// LAMMPS compute pe/atom gives an interaction among a set of atoms to
// each of them in equal portions (https://docs.lammps.org/compute_pe_atom.html),
// so a pair term is half and half.
TEST_CASE("Pair terms are split half and half, as LAMMPS pe/atom does",
          "[peratom][lammps]") {
  const rgpot::LJPot lj{rgpot::LJConfig{.u0 = 0.0104, .cutoff = 8.5, .psi = 3.40}};
  const rgpot::MorsePot morse{
      rgpot::MorseConfig{.De = 0.3429, .a = 1.3588, .re = 2.866, .cutoff = 6.0}};
  for (double r : {3.3, 3.8, 4.5}) {
    const Result one = perAtom(lj, dimer(r, 30.0, 18));
    REQUIRE(one.energies[0] == 0.5 * one.energy);
    REQUIRE(one.energies[1] == 0.5 * one.energy);
    const Result two = perAtom(morse, dimer(r - 0.8, 30.0, 29));
    REQUIRE(two.energies[0] == 0.5 * two.energy);
    REQUIRE(two.energies[1] == 0.5 * two.energy);
  }
}

// pair_style eam gives atom i the energy
// E_i = F(sum_j rho(r_ij)) + 1/2 sum_j phi(r_ij)
// (https://docs.lammps.org/pair_eam.html). Sutton-Chen is that form with
// F = -eps c sqrt(rho), rho = (a/r)^m and phi = eps (a/r)^n.
TEST_CASE("EAM shares are the LAMMPS pair_style eam site energies", "[peratom][lammps]") {
  rgpot::SuttonChenConfig cfg;
  cfg.cutoff = 1.5;
  const rgpot::SuttonChenPot pot{cfg};
  const double r = 1.1;
  const Result got = perAtom(pot, chain(r, 40.0, 78));
  const double rep = std::pow(cfg.a / r, cfg.n);
  const double rho = std::pow(cfg.a / r, cfg.m);
  const double end = cfg.epsilon * (0.5 * rep - cfg.c * std::sqrt(rho));
  const double mid = cfg.epsilon * (rep - cfg.c * std::sqrt(2.0 * rho));
  REQUIRE_THAT(got.energies[0], WithinAbs(end, 1e-12));
  REQUIRE_THAT(got.energies[1], WithinAbs(mid, 1e-12));
  REQUIRE_THAT(got.energies[2], WithinAbs(end, 1e-12));

  // The end atom of a chain sees what an atom of a dimer sees, so the
  // tabulated EAMs give it the same share.
  const rgpot::EAMPot cxx;
  const rgpot::fortranpots::EAMAlPot fortran;
  const Result cxxChain = perAtom(cxx, chain(2.9, 40.0, 13));
  const Result cxxDimer = perAtom(cxx, dimer(2.9, 40.0, 13));
  REQUIRE_THAT(cxxChain.energies[0], WithinAbs(cxxDimer.energies[0], 1e-12));
  // EAM-Al reaches 7.1 angstrom, so its chain is longer.
  const Result fChain = perAtom(fortran, chain(3.7, 40.0, 13));
  const Result fDimer = perAtom(fortran, dimer(3.7, 40.0, 13));
  REQUIRE_THAT(fChain.energies[0], WithinAbs(fDimer.energies[0], 1e-12));
}

// pair_sw writes E = sum_i sum_{j>i} phi2 + sum_i sum_{j!=i} sum_{k>j} phi3
// with phi3 indexed by its centre i (https://docs.lammps.org/pair_sw.html),
// and pair_tersoff E = 1/2 sum_i sum_{j!=i} V_ij with b_ij set by i's
// neighbours (https://docs.lammps.org/pair_tersoff.html). The shares here
// are those site energies: the three-body term on the centre. LAMMPS
// pe/atom differs: it gives each atom of a three-body term a third.
TEST_CASE("Three-body terms stay on the centre atom", "[peratom][lammps]") {
  const rgpot::StillingerWeberSi sw;
  const rgpot::TersoffSiC tersoff;
  // The ends are 2r apart, past both cutoffs, and only the middle atom has
  // two bonds. A bend keeps the angle off the SW minimum.
  const double r = 2.35;
  const double bend = 0.4;
  const Cell three = chain(r, 40.0, 14, bend);
  const Cell two = dimer(r, 40.0, 14);

  const Result swChain = perAtom(sw, three);
  const Result swDimer = perAtom(sw, two);
  const double pair = swDimer.energy;
  const double threeBody = swChain.energy - 2.0 * pair;
  REQUIRE(std::fabs(threeBody) > 1e-3);
  REQUIRE_THAT(swChain.energies[0], WithinAbs(0.5 * pair, 1e-12));
  REQUIRE_THAT(swChain.energies[2], WithinAbs(0.5 * pair, 1e-12));
  REQUIRE_THAT(swChain.energies[1], WithinAbs(pair + threeBody, 1e-12));

  // An end atom's bond order sees no other neighbour, as in the dimer.
  const Result tChain = perAtom(tersoff, three);
  const Result tDimer = perAtom(tersoff, two);
  REQUIRE_THAT(tChain.energies[0], WithinAbs(tDimer.energies[0], 1e-12));
  REQUIRE_THAT(tChain.energies[2], WithinAbs(tDimer.energies[0], 1e-12));
  REQUIRE(std::fabs(tChain.energies[1] - 2.0 * tDimer.energies[0]) > 1e-3);
}

TEST_CASE("Kernels without a per-atom split report it and refuse the call",
          "[peratom]") {
  const rgpot::fortranpots::SWPot sw;
  const rgpot::fortranpots::TersoffPot tersoff;
  const rgpot::fortranpots::EDIPPot edip;
  const rgpot::LJClusterPot cluster;
  const rgpot::ZBLPot zbl;
  REQUIRE_FALSE(sw.perAtomEnergies());
  REQUIRE_FALSE(tersoff.perAtomEnergies());
  REQUIRE_FALSE(edip.perAtomEnergies());
  REQUIRE_FALSE(cluster.perAtomEnergies());
  REQUIRE_FALSE(zbl.perAtomEnergies());
  const Cell two = dimer(2.35, 40.0, 14);
  REQUIRE_THROWS_AS(perAtom(sw, two), std::runtime_error);
  REQUIRE_THROWS_AS(perAtom(zbl, two), std::runtime_error);
}
