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

// The symmetric primitive fcc rows, every lattice vector off-diagonal.
Box fccRows(double a) {
  const double h = 0.5 * a;
  return {0.0, h, h, h, 0.0, h, h, h, 0.0};
}

Cell diamondPrimitive(double a, int type) {
  Cell cell;
  cell.box = fccRows(a);
  cell.pos = {0.0, 0.0, 0.0, 0.25 * a, 0.25 * a, 0.25 * a};
  cell.types = {type, type};
  return cell;
}

// Rows replaced by the integer combinations in `m`, of determinant one.
// The lattice and the atoms are unchanged.
Cell rebased(const Cell &s, const int m[3][3]) {
  Cell out = s;
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) {
      out.box[static_cast<std::size_t>(3 * r + c)] =
          m[r][0] * s.box[static_cast<std::size_t>(c)] +
          m[r][1] * s.box[static_cast<std::size_t>(3 + c)] +
          m[r][2] * s.box[static_cast<std::size_t>(6 + c)];
    }
  }
  return out;
}

// Fractional coordinates of `r` in the rows of `h`.
std::array<double, 3> fractional(const Box &h, const double *r) {
  const double det = h[0] * (h[4] * h[8] - h[5] * h[7]) - h[1] * (h[3] * h[8] - h[5] * h[6]) +
                     h[2] * (h[3] * h[7] - h[4] * h[6]);
  const double inv[9] = {(h[4] * h[8] - h[5] * h[7]) / det, (h[2] * h[7] - h[1] * h[8]) / det,
                         (h[1] * h[5] - h[2] * h[4]) / det, (h[5] * h[6] - h[3] * h[8]) / det,
                         (h[0] * h[8] - h[2] * h[6]) / det, (h[2] * h[3] - h[0] * h[5]) / det,
                         (h[3] * h[7] - h[4] * h[6]) / det, (h[1] * h[6] - h[0] * h[7]) / det,
                         (h[0] * h[4] - h[1] * h[3]) / det};
  std::array<double, 3> f{};
  for (int k = 0; k < 3; ++k) {
    f[static_cast<std::size_t>(k)] = r[0] * inv[k] + r[1] * inv[3 + k] + r[2] * inv[6 + k];
  }
  return f;
}

// Every atom moved into the cell along its rows, as ASE's wrap does.
Cell wrapped(const Cell &s) {
  Cell out = s;
  for (std::size_t p = 0; p < s.size(); ++p) {
    auto f = fractional(s.box, &s.pos[3 * p]);
    for (double &x : f) {
      x -= std::floor(x);
    }
    for (std::size_t c = 0; c < 3; ++c) {
      out.pos[3 * p + c] = f[0] * s.box[c] + f[1] * s.box[3 + c] + f[2] * s.box[6 + c];
    }
  }
  return out;
}

// Each atom moved by a different lattice vector, up to three cells away,
// as an MD run that never wraps leaves them.
Cell unwrapped(const Cell &s) {
  Cell out = s;
  for (std::size_t p = 0; p < s.size(); ++p) {
    const int k[3] = {static_cast<int>(p % 3) - 1, static_cast<int>(p % 7) - 3,
                      static_cast<int>(p % 2) * 2};
    for (std::size_t c = 0; c < 3; ++c) {
      out.pos[3 * p + c] += k[0] * s.box[c] + k[1] * s.box[3 + c] + k[2] * s.box[6 + c];
    }
  }
  return out;
}

// The same crystal rotated so the rows are lower triangular, as ASE's
// standard_form and LAMMPS write a cell: a along x, b in the xy plane.
Cell lowerTriangular(const Cell &s) {
  const double *a = &s.box[0];
  const double *b = &s.box[3];
  const double *c = &s.box[6];
  const auto dot = [](const double *u, const double *v) {
    return u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
  };
  const double la = std::sqrt(dot(a, a));
  const double bx = dot(a, b) / la;
  const double by = std::sqrt(dot(b, b) - bx * bx);
  const double cx = dot(a, c) / la;
  const double cy = (dot(b, c) - bx * cx) / by;
  const double cz = std::sqrt(dot(c, c) - cx * cx - cy * cy);
  Cell out = s;
  out.box = {la, 0.0, 0.0, bx, by, 0.0, cx, cy, cz};
  for (std::size_t p = 0; p < s.size(); ++p) {
    const auto f = fractional(s.box, &s.pos[3 * p]);
    for (std::size_t k = 0; k < 3; ++k) {
      out.pos[3 * p + k] = f[0] * out.box[k] + f[1] * out.box[3 + k] + f[2] * out.box[6 + k];
    }
  }
  return out;
}

// Spacing of the lattice planes parallel to rows other than `axis`.
double height(const Box &box, int axis) {
  const double *a = &box[static_cast<std::size_t>(3 * ((axis + 1) % 3))];
  const double *b = &box[static_cast<std::size_t>(3 * ((axis + 2) % 3))];
  const double *c = &box[static_cast<std::size_t>(3 * axis)];
  const double n[3] = {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
                       a[0] * b[1] - a[1] * b[0]};
  const double area = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
  return std::fabs(n[0] * c[0] + n[1] * c[1] + n[2] * c[2]) / area;
}

// `layers` copies of `cell` along its third row, then `vacuum` added to
// that row along z: periodic in-plane and open across, a (T, T, F) slab.
Cell slab(const Cell &cell, int layers, double vacuum) {
  Cell out = repeat(cell, 1, 1, layers);
  out.box[8] += vacuum;
  return out;
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

/// The shares of one call, after checking that they sum to its energy to
/// 1e-10 and that energy and forces are forceImpl's. Both sums run over
/// thousands of terms, so the bound scales with an energy past 1 eV.
template <class Pot>
Result summed(const Pot &pot, const Cell &cell) {
  const Result ref = plain(pot, cell);
  Result got = perAtom(pot, cell);
  const double tol = 1e-10 * std::max(1.0, std::fabs(ref.energy));
  CHECK(std::fabs(sum(got.energies) - got.energy) <= tol);
  CHECK_THAT(got.energy, WithinAbs(ref.energy, tol));
  for (std::size_t k = 0; k < ref.forces.size(); ++k) {
    CHECK_THAT(got.forces[k], WithinAbs(ref.forces[k], 1e-10));
  }
  return got;
}

/// Atom by atom, the shares of two calls on one crystal.
void requireSameShares(const Result &a, const Result &b, double tol) {
  REQUIRE(a.energies.size() == b.energies.size());
  for (std::size_t i = 0; i < a.energies.size(); ++i) {
    CAPTURE(i);
    CHECK_THAT(b.energies[i], WithinAbs(a.energies[i], tol));
  }
}

/// Atom i of every copy in `big` has the share `small` gives atom i.
void requireCopiesShare(const Result &small, const Result &big, double tol) {
  const std::size_t n = small.energies.size();
  REQUIRE(big.energies.size() % n == 0);
  for (std::size_t p = 0; p < big.energies.size(); ++p) {
    CAPTURE(p);
    CHECK_THAT(big.energies[p], WithinAbs(small.energies[p % n], tol));
  }
}

/// The shares of `cell` follow its atoms when the cell is written in
/// lower-triangular form, when the atoms are wrapped into it, and when
/// they are moved several cells out of it.
template <class Pot>
Result requireFormInvariant(const Pot &pot, const Cell &cell, double tol) {
  const Result base = summed(pot, cell);
  requireSameShares(base, summed(pot, lowerTriangular(cell)), tol);
  requireSameShares(base, summed(pot, wrapped(cell)), tol);
  requireSameShares(base, summed(pot, unwrapped(wrapped(cell))), tol);
  requireSameShares(base, summed(pot, unwrapped(lowerTriangular(cell))), tol);
  return base;
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
  // are neighbours.
  Cell cube = lattice(a, 1, kDiamond, {14});
  rattle(cube, 0.05);

  const rgpot::StillingerWeberSi sw;
  requireAll(sw, bulk);
  requireAll(sw, thin);
  requireAll(sw, drop);
  requireThinMatchesRepeat(sw, cube, 3, 3, 3, 1e-11);
  requireThinMatchesRepeat(sw, thin, 5, 5, 5, 1e-11);

  const rgpot::TersoffSiC tersoff;
  requireAll(tersoff, bulk);
  requireAll(tersoff, thin);
  requireAll(tersoff, drop);
  requireThinMatchesRepeat(tersoff, cube, 3, 3, 3, 1e-11);
  requireThinMatchesRepeat(tersoff, thin, 5, 5, 5, 1e-11);
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

TEST_CASE("Per-atom SW and Tersoff on diamond primitive supercells",
          "[peratom][images][triclinic]") {
  // Up to 4 x 4 x 4 the primitive supercell is thinner than twice either
  // search and walks the image table; from 5 x 5 x 5 it takes the minimum
  // image, binned by its face spacings.
  const double a = 5.431;
  const rgpot::StillingerWeberSi sw;
  const rgpot::TersoffSiC tersoff;
  const Cell cubic = lattice(a, 3, kDiamond, {14});
  const double swSite = summed(sw, cubic).energy / static_cast<double>(cubic.size());
  const double tSite = summed(tersoff, cubic).energy / static_cast<double>(cubic.size());

  const Cell perfect = diamondPrimitive(a, 14);
  Cell rattled = perfect;
  rattle(rattled, 0.05);
  for (const Cell *prim : {&perfect, static_cast<const Cell *>(&rattled)}) {
    const Result swOne = summed(sw, *prim);
    const Result tOne = summed(tersoff, *prim);
    if (prim == &perfect) {
      for (std::size_t i = 0; i < prim->size(); ++i) {
        CHECK_THAT(swOne.energies[i], WithinAbs(swSite, 1e-10));
        CHECK_THAT(tOne.energies[i], WithinAbs(tSite, 1e-10));
      }
    }
    for (int n = 1; n <= 7; ++n) {
      CAPTURE(prim == &rattled, n);
      const Cell big = repeat(*prim, n, n, n);
      requireCopiesShare(swOne, requireFormInvariant(sw, big, 1e-10), 1e-10);
      requireCopiesShare(tOne, requireFormInvariant(tersoff, big, 1e-10), 1e-10);
    }
  }

  Threads four(4);
  for (const Cell &big : {repeat(rattled, 3, 3, 3), repeat(rattled, 7, 7, 7),
                          lowerTriangular(repeat(rattled, 7, 7, 7))}) {
    requireConsistent(sw, big);
    requireConsistent(tersoff, big);
  }
}

TEST_CASE("Per-atom shares on thin cells follow the atoms in any cell form",
          "[peratom][images][thin]") {
  // Each cell is under twice its search along some face spacing, so each
  // walks the image table, and each starts off lower-triangular form.
  const int skew[3][3] = {{1, 0, 0}, {1, 1, 0}, {-1, 2, 1}};
  const rgpot::LJPot lj{rgpot::LJConfig{.u0 = 0.0104, .cutoff = 8.5, .psi = 3.40}};
  const Cell ar = rebased(thinTriclinic(5.26, false, 18, 18), skew);
  REQUIRE(height(ar.box, 0) < 2.0 * 8.5);
  requireFormInvariant(lj, ar, 1e-10);
  requireThinMatchesRepeat(lj, ar, 4, 4, 4, 1e-10);

  const rgpot::MorsePot morse{
      rgpot::MorseConfig{.De = 0.3429, .a = 1.3588, .re = 2.866, .cutoff = 6.0}};
  requireFormInvariant(morse, rebased(thinTriclinic(3.61, false, 29, 29), skew), 1e-10);

  rgpot::SuttonChenConfig scCfg;
  scCfg.cutoff = 2.3;
  scCfg.periodic = true;
  const rgpot::SuttonChenPot sc{scCfg};
  requireFormInvariant(sc, rebased(thinTriclinic(std::sqrt(2.0), false, 78, 78), skew), 1e-10);

  const rgpot::fortranpots::EAMAlPot eamAl;
  const Cell al = rebased(thinTriclinic(4.05, false, 13, 13), skew);
  requireFormInvariant(eamAl, al, 1e-10);
  requireThinMatchesRepeat(eamAl, al, 4, 4, 4, 1e-10);
  const rgpot::fortranpots::CuH2Pot cuh2;
  requireFormInvariant(cuh2, rebased(thinTriclinic(3.61, false, 29, 29), skew), 1e-10);

  const rgpot::EAMPot eam;
  Cell cube = lattice(4.05, 1, kFcc, {13});
  rattle(cube, 0.03);
  requireFormInvariant(eam, cube, 1e-10);
  requireFormInvariant(eam, al, 1e-10);
  requireThinMatchesRepeat(eam, al, 4, 4, 4, 1e-10);

  Cell si = diamondPrimitive(5.431, 14);
  rattle(si, 0.05);
  const rgpot::StillingerWeberSi sw;
  const rgpot::TersoffSiC tersoff;
  requireFormInvariant(sw, si, 1e-10);
  requireFormInvariant(tersoff, si, 1e-10);
  requireFormInvariant(sw, rebased(si, skew), 1e-10);
  requireFormInvariant(tersoff, rebased(si, skew), 1e-10);
  Cell sic = thinTriclinic(4.36, true, 14, 6);
  requireFormInvariant(tersoff, rebased(sic, skew), 1e-10);
}

TEST_CASE("Per-atom shares on (T, T, F) slabs", "[peratom][images][slab]") {
  // Periodic in-plane, thin there, with vacuum wider than twice any search
  // across. The in-plane rows are taken square and sheared.
  const int shear[3][3] = {{1, 0, 0}, {1, 1, 0}, {0, 0, 1}};
  const double vacuum = 20.0;
  auto check = [&](const auto &pot, Cell base, int layers, bool sheared) {
    Cell s = slab(base, layers, vacuum);
    rattle(s, 0.04);
    if (sheared) {
      s = rebased(s, shear);
    }
    CAPTURE(sheared, s.size());
    const Result one = requireFormInvariant(pot, s, 1e-10);
    requireCopiesShare(one, summed(pot, repeat(s, 3, 3, 1)), 1e-10);
    // The two faces differ from the middle layer.
    double lo = one.energies[0];
    double hi = one.energies[0];
    for (double e : one.energies) {
      lo = std::min(lo, e);
      hi = std::max(hi, e);
    }
    CHECK(hi - lo > 1e-3 * std::fabs(lo));
  };
  for (bool sheared : {false, true}) {
    check(rgpot::LJPot{rgpot::LJConfig{.u0 = 0.0104, .cutoff = 8.5, .psi = 3.40}},
          lattice(5.26, 1, kFcc, {18}), 4, sheared);
    rgpot::SuttonChenConfig scCfg;
    scCfg.cutoff = 2.3;
    scCfg.periodic = true;
    check(rgpot::SuttonChenPot{scCfg}, lattice(std::sqrt(2.0), 1, kFcc, {78}), 5, sheared);
    check(rgpot::fortranpots::EAMAlPot{}, lattice(4.05, 1, kFcc, {13}), 4, sheared);
    check(rgpot::fortranpots::CuH2Pot{}, lattice(3.61, 1, kFcc, {29}), 4, sheared);
    check(rgpot::EAMPot{}, lattice(4.05, 1, kFcc, {13}), 4, sheared);
    check(rgpot::StillingerWeberSi{}, lattice(5.431, 1, kDiamond, {14}), 3, sheared);
    check(rgpot::TersoffSiC{}, lattice(5.431, 1, kDiamond, {14}), 3, sheared);
  }
}

TEST_CASE("Per-atom EAM on fcc primitive supercells", "[peratom][images][triclinic][eam]") {
  // The symmetric fcc rows have zeros on the diagonal; the lower-triangular
  // form has diagonal entries that are not periodic lengths of the lattice.
  const double a = 4.05;
  Cell perfect;
  perfect.box = fccRows(a);
  perfect.pos = {0.0, 0.0, 0.0};
  perfect.types = {13};
  // Two atoms in the cell doubled along its first row, the second moved
  // off the lattice site at that row's midpoint.
  Cell pair = perfect;
  for (std::size_t k = 0; k < 3; ++k) {
    pair.pos.push_back(pair.box[k]);
    pair.box[k] *= 2.0;
  }
  pair.types.push_back(13);
  const double push[3] = {0.031, -0.022, 0.043};
  for (std::size_t k = 0; k < 3; ++k) {
    pair.pos[3 + k] += push[k];
  }
  const Cell cube = lattice(a, 6, kFcc, {13});
  const rgpot::EAMPot cxx;
  const rgpot::fortranpots::EAMAlPot fortran;
  auto run = [&](const auto &pot) {
    const double site = summed(pot, cube).energy / static_cast<double>(cube.size());
    const Result one = summed(pot, perfect);
    CHECK_THAT(one.energies[0], WithinAbs(site, 1e-10));
    const Result two = summed(pot, pair);
    for (int n = 1; n <= 7; ++n) {
      CAPTURE(n);
      requireCopiesShare(one, requireFormInvariant(pot, repeat(perfect, n, n, n), 1e-10), 1e-10);
      requireCopiesShare(two, requireFormInvariant(pot, repeat(pair, n, n, n), 1e-10), 1e-10);
    }
  };
  run(cxx);
  run(fortran);
  for (auto mode : {rgpot::nlist::Mode::Stored, rgpot::nlist::Mode::Stream}) {
    rgpot::EAMPot fixed;
    fixed.setNeighbourMode(mode);
    const Cell big = repeat(pair, 7, 7, 7);
    requireSameShares(summed(cxx, big), summed(fixed, lowerTriangular(big)), 1e-10);
    Threads four(4);
    requireConsistent(fixed, big);
    requireConsistent(fixed, unwrapped(lowerTriangular(big)));
  }
}

TEST_CASE("Per-atom LJ and EAM shares of atoms outside a wide cell",
          "[peratom][images][unwrapped]") {
  // Wide enough for five or more bins along each axis, so the fast tiles
  // add each bin's lattice shift; an atom outside the cell is binned by
  // its image inside it.
  Cell ar = lattice(5.26, 12, kFcc, {18});
  rattle(ar, 0.05);
  Cell al = lattice(4.05, 10, kFcc, {13});
  rattle(al, 0.04);
  for (int threads : {1, 4}) {
    Threads use(threads);
    for (auto mode : {rgpot::nlist::Mode::Auto, rgpot::nlist::Mode::Stored,
                      rgpot::nlist::Mode::Stream}) {
      CAPTURE(threads, static_cast<int>(mode));
      rgpot::LJPot lj{rgpot::LJConfig{.u0 = 0.0104, .cutoff = 8.5, .psi = 3.40}};
      lj.setNeighbourMode(mode);
      requireSameShares(summed(lj, ar), summed(lj, unwrapped(ar)), 1e-10);
      rgpot::EAMPot eam;
      eam.setNeighbourMode(mode);
      requireSameShares(summed(eam, al), summed(eam, unwrapped(al)), 1e-10);
    }
  }

  // The same Al and Cu on a sheared basis of the lattice, wide by its face
  // spacings, through the Fortran EAM kernels.
  const int shear[3][3] = {{1, 0, 0}, {1, 1, 0}, {0, 0, 1}};
  const rgpot::fortranpots::EAMAlPot eamAl;
  Cell alSmall = lattice(4.05, 6, kFcc, {13});
  rattle(alSmall, 0.04);
  const Result alRef = summed(eamAl, alSmall);
  requireSameShares(alRef, summed(eamAl, unwrapped(alSmall)), 1e-10);
  requireSameShares(alRef, requireFormInvariant(eamAl, rebased(alSmall, shear), 1e-10), 1e-10);
  const rgpot::fortranpots::CuH2Pot cuh2;
  Cell cu = lattice(3.61, 6, kFcc, {29});
  rattle(cu, 0.04);
  const Result cuRef = summed(cuh2, cu);
  requireSameShares(cuRef, summed(cuh2, unwrapped(cu)), 1e-10);
  requireSameShares(cuRef, requireFormInvariant(cuh2, rebased(cu, shear), 1e-10), 1e-10);
}
