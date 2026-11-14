// MIT License
// Copyright 2023--present rgpot developers
//
// Cells thinner than twice the cutoff. Each kernel must count a neighbour
// at every periodic image inside the cutoff, so a primitive cell gives the
// per-atom energy, forces and stress of an explicit supercell thick enough
// for the minimum image.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <vector>

#include <catch2/catch_all.hpp>

#include "rgpot/EAM/EAMPot.hpp"
#include "rgpot/ForceStructs.hpp"
#include "rgpot/LennardJones/LJPot.hpp"
#include "rgpot/Morse/MorsePot.hpp"
#include "rgpot/StillingerWeber/SWSi.hpp"
#include "rgpot/SuttonChen/SuttonChenPot.hpp"
#include "rgpot/Tersoff/TersoffSiC.hpp"
#include "rgpot/fortran/FortranPots.hpp"
#include "rgpot/nlist/NeighbourList.hpp"
#include "rgpot/simd/from_cache.hpp"
#include "rgpot/stress.hpp"

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
  std::array<double, 9> stress{};
  int has_stress = 0;
};

template <class Pot>
Result evaluate(const Pot &pot, const Cell &cell) {
  Result r;
  r.forces.assign(3 * cell.size(), 0.0);
  rgpot::ForceOut out{r.forces.data(), 0.0, 0.0, {}, 0};
  pot.forceImpl(rgpot::ForceInput{cell.size(), cell.pos.data(), cell.types.data(),
                                  cell.box.data()},
                &out);
  r.energy = out.energy;
  r.has_stress = out.has_stress;
  for (int k = 0; k < 9; ++k) {
    r.stress[static_cast<std::size_t>(k)] = out.stress[k];
  }
  return r;
}

// Copies are laid out copy-major, so atom i of the first copy is row i.
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

double height(const Box &box, int axis) {
  const double *a = &box[static_cast<std::size_t>(3 * ((axis + 1) % 3))];
  const double *b = &box[static_cast<std::size_t>(3 * ((axis + 2) % 3))];
  const double *c = &box[static_cast<std::size_t>(3 * axis)];
  const double n[3] = {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
                       a[0] * b[1] - a[1] * b[0]};
  const double area = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
  return std::fabs(n[0] * c[0] + n[1] * c[1] + n[2] * c[2]) / area;
}

// Primitive fcc vectors in lower-triangular form for cubic constant `a`.
Box triclinicFcc(double a) {
  const double s2 = std::sqrt(2.0);
  const double s3 = std::sqrt(3.0);
  const double s6 = std::sqrt(6.0);
  return {a / s2, 0.0, 0.0, a / (2.0 * s2), a * s3 / (2.0 * s2), 0.0,
          a / (2.0 * s2), a / (2.0 * s6), a / s3};
}

// Two atoms in a cell doubled along the first primitive vector, the second
// pushed off its lattice site so forces and stress are not zero.
Cell rattledPair(const Box &primitive, int type) {
  Cell cell;
  cell.box = primitive;
  for (int k = 0; k < 3; ++k) {
    cell.box[static_cast<std::size_t>(k)] *= 2.0;
  }
  const double push[3] = {0.031, -0.022, 0.043};
  cell.pos = {0.0, 0.0, 0.0};
  for (int k = 0; k < 3; ++k) {
    cell.pos.push_back(primitive[static_cast<std::size_t>(k)] + push[k]);
  }
  cell.types = {type, type};
  return cell;
}

// The small cell must be thinner than twice `search` and the supercell at
// least that thick, or the check proves nothing.
void requireThinAndThick(const Cell &small, const Cell &big, double search) {
  bool thin = false;
  for (int axis = 0; axis < 3; ++axis) {
    thin = thin || height(small.box, axis) < 2.0 * search;
    REQUIRE(height(big.box, axis) >= 2.0 * search);
  }
  REQUIRE(thin);
}

template <class Pot>
void requireMatchesRepeat(const Pot &pot, const Cell &small, int na, int nb, int nc,
                          double search, bool stress, double tol) {
  const Cell big = repeat(small, na, nb, nc);
  requireThinAndThick(small, big, search);
  const Result a = evaluate(pot, small);
  const Result b = evaluate(pot, big);
  const int copies = na * nb * nc;
  REQUIRE_THAT(a.energy, WithinAbs(b.energy / copies, tol * std::fabs(a.energy) + tol));
  for (std::size_t k = 0; k < a.forces.size(); ++k) {
    REQUIRE_THAT(a.forces[k], WithinAbs(b.forces[k], tol));
  }
  if (stress) {
    REQUIRE(a.has_stress == 1);
    REQUIRE(b.has_stress == 1);
    for (std::size_t k = 0; k < 9; ++k) {
      REQUIRE_THAT(a.stress[k], WithinAbs(b.stress[k], tol));
    }
  }
}

void setThreads(const char *value) {
#if defined(_WIN32)
  _putenv_s("OMP_NUM_THREADS", value);
#else
  setenv("OMP_NUM_THREADS", value, 1);
#endif
}

// One atom, every lattice vector off-diagonal.
Cell fccPrimitive(double a, int type) {
  Cell s;
  const double h = 0.5 * a;
  s.box = {0.0, h, h, h, 0.0, h, h, h, 0.0};
  s.pos = {0.0, 0.0, 0.0};
  s.types = {type};
  return s;
}

// Conventional fcc, four atoms per cube, nx x ny x nz cubes.
Cell fccCubic(double a, int nx, int ny, int nz, int type) {
  static constexpr double basis[4][3] = {
      {0.0, 0.0, 0.0}, {0.5, 0.5, 0.0}, {0.5, 0.0, 0.5}, {0.0, 0.5, 0.5}};
  Cell s;
  s.box = {nx * a, 0.0, 0.0, 0.0, ny * a, 0.0, 0.0, 0.0, nz * a};
  for (int i = 0; i < nx; ++i) {
    for (int j = 0; j < ny; ++j) {
      for (int k = 0; k < nz; ++k) {
        for (const auto &b : basis) {
          s.pos.push_back((i + b[0]) * a);
          s.pos.push_back((j + b[1]) * a);
          s.pos.push_back((k + b[2]) * a);
          s.types.push_back(type);
        }
      }
    }
  }
  return s;
}

Cell diamondCubic(double a, int n, int type) {
  Cell s = fccCubic(a, n, n, n, type);
  const std::size_t half = s.size();
  for (std::size_t i = 0; i < half; ++i) {
    for (int k = 0; k < 3; ++k) {
      s.pos.push_back(s.pos[3 * i + static_cast<std::size_t>(k)] + 0.25 * a);
    }
    s.types.push_back(type);
  }
  return s;
}

Cell diamondPrimitive(double a, int type) {
  Cell s = fccPrimitive(a, type);
  s.pos = {0.0, 0.0, 0.0, 0.25 * a, 0.25 * a, 0.25 * a};
  s.types = {type, type};
  return s;
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
    for (int c = 0; c < 3; ++c) {
      out.pos[3 * p + static_cast<std::size_t>(c)] =
          f[0] * s.box[static_cast<std::size_t>(c)] +
          f[1] * s.box[static_cast<std::size_t>(3 + c)] +
          f[2] * s.box[static_cast<std::size_t>(6 + c)];
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

void jitter(Cell &s, double amount, unsigned seed) {
  for (double &x : s.pos) {
    seed = seed * 1103515245u + 12345u;
    x += amount * (static_cast<double>((seed >> 8) & 0xffff) / 65535.0 - 0.5);
  }
}

// Cartesian strain r -> r (I + e) on positions and cell rows.
Cell strained(const Cell &s, int a, int b, double e) {
  Cell out = s;
  auto apply = [&](double *v) { v[b] += v[a] * e; };
  for (std::size_t p = 0; p < out.size(); ++p) {
    apply(&out.pos[3 * p]);
  }
  for (int r = 0; r < 3; ++r) {
    apply(&out.box[static_cast<std::size_t>(3 * r)]);
  }
  return out;
}

// Shifted 12-6 Lennard-Jones on a perfect fcc lattice, per atom.
double ljFccLatticeSum(double a, double u0, double psi, double rc) {
  const double s6 = std::pow(psi / rc, 6);
  const double shift = 4.0 * u0 * s6 * (s6 - 1.0);
  const int reach = static_cast<int>(std::ceil(2.0 * rc / a)) + 1;
  double sum = 0.0;
  for (int i = -reach; i <= reach; ++i) {
    for (int j = -reach; j <= reach; ++j) {
      for (int k = -reach; k <= reach; ++k) {
        if (((i + j + k) & 1) != 0 || (i == 0 && j == 0 && k == 0)) {
          continue;
        }
        const double r = 0.5 * a * std::sqrt(static_cast<double>(i * i + j * j + k * k));
        if (r > rc) {
          continue;
        }
        const double q = std::pow(psi / r, 6);
        sum += 4.0 * u0 * q * (q - 1.0) - shift;
      }
    }
  }
  return 0.5 * sum;
}

// Sutton-Chen on a perfect fcc lattice, per atom.
double scFccLatticeSum(double a, const rgpot::SuttonChenConfig &c) {
  const double rc = c.cutoff;
  const int reach = static_cast<int>(std::ceil(2.0 * rc / a)) + 1;
  double pair = 0.0;
  double rho = 0.0;
  for (int i = -reach; i <= reach; ++i) {
    for (int j = -reach; j <= reach; ++j) {
      for (int k = -reach; k <= reach; ++k) {
        if (((i + j + k) & 1) != 0 || (i == 0 && j == 0 && k == 0)) {
          continue;
        }
        const double r = 0.5 * a * std::sqrt(static_cast<double>(i * i + j * j + k * k));
        if (r > rc) {
          continue;
        }
        pair += std::pow(c.a / r, c.n);
        rho += std::pow(c.a / r, c.m);
      }
    }
  }
  return c.epsilon * (0.5 * pair - c.c * std::sqrt(rho));
}

rgpot::LJPot reducedLj() {
  return rgpot::LJPot(rgpot::LJConfig{.u0 = 1.0, .cutoff = 3.0, .psi = 1.0});
}

rgpot::SuttonChenConfig copperSc() {
  // Sutton and Chen, Phil. Mag. Lett. 61, 139 (1990): Cu 12-6.
  rgpot::SuttonChenConfig c;
  c.epsilon = 1.2382e-2;
  c.c = 39.432;
  c.a = 3.61;
  c.n = 12;
  c.m = 6;
  c.cutoff = 7.0;
  c.periodic = true;
  c.stress = true;
  return c;
}

const double kLjFcc = std::pow(2.0, 1.0 / 6.0) * std::sqrt(2.0);

// Energies of the jittered wide cells under the minimum-image walk of
// develop 8be4d794. A wide cell must not move.
constexpr double kWideLj = -6645.0085405826667;
constexpr double kWideSc = -189.41980600758961;
constexpr double kWideEam = -774.20652086039615;

template <typename Pot>
void requireSamePerAtom(const Pot &pot, const Cell &thin, const Cell &wide, double tol) {
  const auto a = evaluate(pot, thin);
  const auto b = evaluate(pot, wide);
  CHECK_THAT(a.energy / static_cast<double>(thin.size()),
             WithinAbs(b.energy / static_cast<double>(wide.size()), tol));
}

template <typename Pot>
void requireFiniteDifferenceForces(const Pot &pot, const Cell &s, double h, double tol) {
  const auto base = evaluate(pot, s);
  for (std::size_t k = 0; k < s.pos.size(); ++k) {
    Cell up = s;
    Cell down = s;
    up.pos[k] += h;
    down.pos[k] -= h;
    const double fd = -(evaluate(pot, up).energy - evaluate(pot, down).energy) / (2.0 * h);
    CHECK_THAT(base.forces[k], WithinAbs(fd, tol));
  }
}

// Central differences of the energy under strain against sigma V.
template <typename Pot>
void requireFiniteDifferenceStress(const Pot &pot, const Cell &s, double e, double tol) {
  const auto base = evaluate(pot, s);
  const double volume = rgpot::cellVolume(s.box.data());
  for (int a = 0; a < 3; ++a) {
    for (int b = 0; b < 3; ++b) {
      const double up = evaluate(pot, strained(s, a, b, e)).energy;
      const double down = evaluate(pot, strained(s, a, b, -e)).energy;
      const double fd = (up - down) / (2.0 * e * volume);
      CHECK_THAT(base.stress[static_cast<std::size_t>(3 * a + b)], WithinAbs(fd, tol));
    }
  }
}

// The public list against a direct sum over shifts in [-reach, reach].
void requirePublicListMatchesBrute(const Cell &s, double cutoff, int reach) {
  std::size_t brute = 0;
  double brute_sum = 0.0;
  for (std::size_t i = 0; i < s.size(); ++i) {
    for (std::size_t j = 0; j < s.size(); ++j) {
      for (int a = -reach; a <= reach; ++a) {
        for (int b = -reach; b <= reach; ++b) {
          for (int c = -reach; c <= reach; ++c) {
            if (i == j && a == 0 && b == 0 && c == 0) {
              continue;
            }
            double r2 = 0.0;
            for (std::size_t k = 0; k < 3; ++k) {
              const double d = s.pos[3 * j + k] - s.pos[3 * i + k] + a * s.box[k] +
                               b * s.box[3 + k] + c * s.box[6 + k];
              r2 += d * d;
            }
            if (r2 < cutoff * cutoff) {
              ++brute;
              brute_sum += 1.0 / r2;
            }
          }
        }
      }
    }
  }
  for (auto shell : {rgpot::nlist::Shell::Full, rgpot::nlist::Shell::Half}) {
    rgpot::nlist::Query q;
    q.cutoff = cutoff;
    q.shape = rgpot::nlist::Pairs{shell};
    rgpot::nlist::Table t;
    rgpot::nlist::build(t, s.pos.data(), s.size(), s.box.data(), q);
    const double weight = shell == rgpot::nlist::Shell::Full ? 1.0 : 2.0;
    double sum = 0.0;
    double worst = 0.0;
    for (std::size_t p = 0; p < t.size(); ++p) {
      const auto i = static_cast<std::size_t>(t.first[p]);
      const auto j = static_cast<std::size_t>(t.second[p]);
      const int *sh = &t.shift[3 * p];
      for (std::size_t k = 0; k < 3; ++k) {
        const double v = s.pos[3 * j + k] - s.pos[3 * i + k] + sh[0] * s.box[k] +
                         sh[1] * s.box[3 + k] + sh[2] * s.box[6 + k];
        worst = std::max(worst, std::abs(t.vector[3 * p + k] - v));
      }
      sum += weight / (t.distance[p] * t.distance[p]);
    }
    CHECK(worst < 1e-9);
    CHECK(static_cast<double>(t.size()) * weight == static_cast<double>(brute));
    CHECK_THAT(sum, WithinAbs(brute_sum, 1e-12 * brute_sum));
  }
}

} // namespace

TEST_CASE("LJ one-atom fcc primitive cell sums every image", "[images][lj]") {
  // Cubic constant 1.6, u0 = psi = 1, cutoff 2.5: -7.2294117627 per atom.
  const rgpot::LJPot pot{rgpot::LJConfig{.u0 = 1.0, .cutoff = 2.5, .psi = 1.0}};
  const double a = 1.6;
  Cell primitive;
  primitive.box = {0.0, a / 2, a / 2, a / 2, 0.0, a / 2, a / 2, a / 2, 0.0};
  primitive.pos = {0.0, 0.0, 0.0};
  primitive.types = {1};
  const Result one = evaluate(pot, primitive);
  REQUIRE_THAT(one.energy, WithinAbs(-7.2294117627, 1e-9));
  for (double f : one.forces) {
    REQUIRE_THAT(f, WithinAbs(0.0, 1e-12));
  }

  Cell cubic;
  cubic.box = {a, 0.0, 0.0, 0.0, a, 0.0, 0.0, 0.0, a};
  cubic.pos = {0.0, 0.0, 0.0, 0.0, a / 2, a / 2, a / 2, 0.0, a / 2, a / 2, a / 2, 0.0};
  cubic.types = {1, 1, 1, 1};
  const Cell big = repeat(cubic, 5, 5, 5);
  requireThinAndThick(primitive, big, 3.0);
  const Result many = evaluate(pot, big);
  REQUIRE_THAT(one.energy, WithinAbs(many.energy / static_cast<double>(big.size()), 1e-11));
  for (std::size_t k = 0; k < 9; ++k) {
    REQUIRE_THAT(one.stress[k], WithinAbs(many.stress[k], 1e-11));
  }
}

TEST_CASE("Triclinic primitive cells match their supercells", "[images][triclinic]") {
  const double a = 1.6;
  const Box tri = triclinicFcc(a);

  SECTION("LJ one atom in the lower-triangular fcc cell") {
    const rgpot::LJPot pot{rgpot::LJConfig{.u0 = 1.0, .cutoff = 2.5, .psi = 1.0}};
    Cell one;
    one.box = tri;
    one.pos = {0.0, 0.0, 0.0};
    one.types = {1};
    REQUIRE_THAT(evaluate(pot, one).energy, WithinAbs(-7.2294117627, 1e-9));
    requireMatchesRepeat(pot, one, 7, 7, 7, 3.0, true, 1e-10);
  }

  SECTION("LJ rattled pair") {
    const rgpot::LJPot pot{rgpot::LJConfig{.u0 = 1.0, .cutoff = 2.5, .psi = 1.0}};
    requireMatchesRepeat(pot, rattledPair(tri, 1), 4, 7, 7, 3.0, true, 1e-10);
  }

  SECTION("Morse rattled pair") {
    const rgpot::MorsePot pot{
        rgpot::MorseConfig{.De = 0.7, .a = 1.6, .re = 1.13, .cutoff = 2.5}};
    requireMatchesRepeat(pot, rattledPair(tri, 1), 4, 7, 7, 3.0, true, 1e-10);
  }
}

TEST_CASE("Sutton-Chen primitive cell matches a supercell", "[images][eam]") {
  // Nearest neighbour 1 in reduced units. The cutoff sits between the
  // fifth and sixth fcc shells, clear of the rattled distances.
  rgpot::SuttonChenConfig cfg;
  cfg.cutoff = 2.3;
  cfg.periodic = true;
  cfg.stress = true;
  const rgpot::SuttonChenPot pot{cfg};
  const Box tri = triclinicFcc(std::sqrt(2.0));
  Cell one;
  one.box = tri;
  one.pos = {0.0, 0.0, 0.0};
  one.types = {78};
  requireMatchesRepeat(pot, one, 8, 8, 8, 2.3 * 1.2, true, 1e-10);
  requireMatchesRepeat(pot, rattledPair(tri, 78), 4, 8, 8, 2.3 * 1.2, true, 1e-10);
}

TEST_CASE("EAM-Al primitive cell matches a supercell", "[images][eam]") {
  // Neither EAM-Al kernel writes a stress, so energy and forces only.
  const Box tri = triclinicFcc(4.05);
  const rgpot::fortranpots::EAMAlPot fortran;
  Cell one;
  one.box = tri;
  one.pos = {0.0, 0.0, 0.0};
  one.types = {13};
  requireMatchesRepeat(fortran, one, 8, 8, 8, 7.1 * 1.2, false, 1e-9);
  requireMatchesRepeat(fortran, rattledPair(tri, 13), 4, 8, 8, 7.1 * 1.2, false, 1e-9);

  const rgpot::EAMPot cell;
  const double a = 4.05;
  Cell cubic;
  cubic.box = {a, 0.0, 0.0, 0.0, a, 0.0, 0.0, 0.0, a};
  cubic.pos = {0.0, 0.0, 0.0, 0.03, a / 2 - 0.02, a / 2 + 0.04, a / 2, 0.0, a / 2,
               a / 2, a / 2, 0.0};
  cubic.types = {13, 13, 13, 13};
  requireMatchesRepeat(cell, cubic, 4, 4, 4, 5.555 * 1.2, false, 1e-9);
  requireMatchesRepeat(cell, one, 8, 8, 8, 5.555 * 1.2, false, 1e-9);
  requireMatchesRepeat(cell, rattledPair(tri, 13), 4, 8, 8, 5.555 * 1.2, false, 1e-9);
  // The cubic lattice energy in the primitive cell's symmetric rows.
  CHECK_THAT(evaluate(cell, fccPrimitive(a, 13)).energy,
             WithinAbs(evaluate(cell, fccCubic(a, 5, 5, 5, 13)).energy / 500.0, 1e-10));
}

TEST_CASE("one-atom and four-atom fcc Lennard-Jones match the lattice sum",
          "[images][lj]") {
  setThreads("1");
  const auto pot = reducedLj();
  const double exact = ljFccLatticeSum(kLjFcc, 1.0, 1.0, 3.0);
  REQUIRE_THAT(exact, WithinAbs(-7.762386540408, 1e-12));
  const auto wide = fccCubic(kLjFcc, 6, 6, 6, 18);
  REQUIRE(height(wide.box, 0) > 2.0 * (3.0 + 1.0));
  REQUIRE_THAT(evaluate(pot, wide).energy / static_cast<double>(wide.size()),
               WithinAbs(exact, 1e-10));

  const auto one = fccPrimitive(kLjFcc, 18);
  REQUIRE(height(one.box, 0) < 3.0);
  const auto r1 = evaluate(pot, one);
  CHECK_THAT(r1.energy, WithinAbs(exact, 1e-10));
  CHECK_THAT(r1.forces[0], WithinAbs(0.0, 1e-10));

  CHECK_THAT(evaluate(pot, fccCubic(kLjFcc, 1, 1, 1, 18)).energy / 4.0, WithinAbs(exact, 1e-10));
  const auto slab = fccCubic(kLjFcc, 1, 2, 3, 18);
  CHECK_THAT(evaluate(pot, slab).energy / static_cast<double>(slab.size()),
             WithinAbs(exact, 1e-10));
}

TEST_CASE("thin Lennard-Jones and Morse cells match their supercell with displacements",
          "[images][lj][morse]") {
  setThreads("1");
  auto thin = fccCubic(kLjFcc, 1, 1, 2, 18);
  jitter(thin, 0.08, 7u);
  const auto wide = repeat(thin, 6, 6, 3);
  const auto lj = reducedLj();
  requireSamePerAtom(lj, thin, wide, 1e-10);
  const auto a = evaluate(lj, thin);
  const auto b = evaluate(lj, wide);
  for (std::size_t k = 0; k < thin.pos.size(); ++k) {
    CHECK_THAT(a.forces[k], WithinAbs(b.forces[k], 1e-9));
  }
  for (std::size_t k = 0; k < 9; ++k) {
    CHECK_THAT(a.stress[k], WithinAbs(b.stress[k], 1e-9));
  }

  auto cu = fccCubic(3.61, 1, 1, 1, 29);
  jitter(cu, 0.1, 11u);
  const rgpot::MorsePot morse;
  requireSamePerAtom(morse, cu, repeat(cu, 7, 7, 7), 1e-10);
}

TEST_CASE("Sutton-Chen embedding density counts every image", "[images][sc]") {
  setThreads("1");
  const auto cfg = copperSc();
  const rgpot::SuttonChenPot pot(cfg);
  const double exact = scFccLatticeSum(cfg.a, cfg);
  const auto wide = fccCubic(cfg.a, 5, 5, 5, 29);
  REQUIRE(height(wide.box, 0) > 2.0 * cfg.cutoff * 1.2);
  CHECK_THAT(evaluate(pot, wide).energy / static_cast<double>(wide.size()),
             WithinAbs(exact, 1e-10));
  CHECK_THAT(evaluate(pot, fccPrimitive(cfg.a, 29)).energy, WithinAbs(exact, 1e-10));
  CHECK_THAT(evaluate(pot, fccCubic(cfg.a, 1, 1, 1, 29)).energy / 4.0, WithinAbs(exact, 1e-10));

  auto thin = fccCubic(cfg.a, 1, 1, 1, 29);
  jitter(thin, 0.1, 3u);
  const auto big = repeat(thin, 5, 5, 5);
  const auto a = evaluate(pot, thin);
  const auto b = evaluate(pot, big);
  CHECK_THAT(a.energy / 4.0, WithinAbs(b.energy / static_cast<double>(big.size()), 1e-10));
  for (std::size_t k = 0; k < thin.pos.size(); ++k) {
    CHECK_THAT(a.forces[k], WithinAbs(b.forces[k], 1e-9));
  }
  for (std::size_t k = 0; k < 9; ++k) {
    CHECK_THAT(a.stress[k], WithinAbs(b.stress[k], 1e-9));
  }
}

TEST_CASE("Sutton-Chen threaded walk counts every image", "[images][sc][threads]") {
  setThreads("3");
  const auto cfg = copperSc();
  const rgpot::SuttonChenPot pot(cfg);
  auto thin = fccCubic(cfg.a, 1, 1, 2, 29);
  jitter(thin, 0.1, 5u);
  const auto threaded = evaluate(pot, thin);
  setThreads("1");
  const auto serial = evaluate(pot, thin);
  CHECK_THAT(threaded.energy, WithinAbs(serial.energy, 1e-10));
  for (std::size_t k = 0; k < thin.pos.size(); ++k) {
    CHECK_THAT(threaded.forces[k], WithinAbs(serial.forces[k], 1e-10));
  }
  for (std::size_t k = 0; k < 9; ++k) {
    CHECK_THAT(threaded.stress[k], WithinAbs(serial.stress[k], 1e-10));
  }
  CHECK_THAT(serial.energy / 8.0,
             WithinAbs(evaluate(pot, repeat(thin, 5, 5, 3)).energy / (8.0 * 75.0), 1e-10));
}

TEST_CASE("EAM densities count every image in a thin cell", "[images][eam]") {
  setThreads("1");
  const rgpot::EAMPot pot;
  auto thin = fccCubic(4.05, 1, 1, 1, 13);
  REQUIRE(height(thin.box, 0) < 2.0 * 5.555);
  requireSamePerAtom(pot, thin, repeat(thin, 4, 4, 4), 1e-10);
  jitter(thin, 0.1, 9u);
  const auto big = repeat(thin, 4, 4, 4);
  const auto a = evaluate(pot, thin);
  const auto b = evaluate(pot, big);
  CHECK_THAT(a.energy / 4.0, WithinAbs(b.energy / static_cast<double>(big.size()), 1e-10));
  for (std::size_t k = 0; k < thin.pos.size(); ++k) {
    CHECK_THAT(a.forces[k], WithinAbs(b.forces[k], 1e-9));
  }
  for (auto mode : {rgpot::nlist::Mode::Stream, rgpot::nlist::Mode::Stored}) {
    rgpot::EAMPot fixed;
    fixed.setNeighbourMode(mode);
    CHECK_THAT(evaluate(fixed, thin).energy, WithinAbs(a.energy, 1e-10));
  }
  // One atom in a simple cubic cell: every partner is a self image.
  Cell one;
  one.box = {2.9, 0, 0, 0, 2.9, 0, 0, 0, 2.9};
  one.pos = {0.3, 0.2, 0.1};
  one.types = {13};
  requireSamePerAtom(pot, one, repeat(one, 5, 5, 5), 1e-10);
}

TEST_CASE("three-body potentials count every image", "[images][sw][tersoff]") {
  setThreads("1");
  const double a = 5.431;
  const rgpot::StillingerWeberSi sw;
  const rgpot::TersoffSiC tersoff;
  const auto wide = diamondCubic(a, 3, 14);
  const auto prim = diamondPrimitive(a, 14);
  REQUIRE(height(prim.box, 0) < 2.0 * 2.0951 * 1.8);
  requireSamePerAtom(sw, prim, wide, 1e-10);
  requireSamePerAtom(tersoff, prim, wide, 1e-10);
  auto cubic = diamondCubic(a, 1, 14);
  jitter(cubic, 0.15, 13u);
  const auto big = repeat(cubic, 3, 3, 3);
  requireSamePerAtom(sw, cubic, big, 1e-10);
  requireSamePerAtom(tersoff, cubic, big, 1e-10);
  auto skew = diamondPrimitive(a, 14);
  jitter(skew, 0.1, 17u);
  requireFiniteDifferenceForces(sw, skew, 1e-5, 1e-6);
  requireFiniteDifferenceStress(sw, skew, 1e-6, 1e-6);
  requireFiniteDifferenceForces(tersoff, skew, 1e-5, 1e-6);
}

TEST_CASE("three-body potentials on wide supercells of the diamond primitive cell",
          "[images][sw][tersoff][triclinic]") {
  setThreads("1");
  // From 5x5x5 the 60 degree cell is wide enough for the minimum-image
  // walk, which must bin it by face spacing. Only the lower triangular
  // form puts its long diagonal entries where the row lengths mislead.
  const double a = 5.431;
  const rgpot::StillingerWeberSi sw;
  const rgpot::TersoffSiC tersoff;
  const auto cubic = diamondCubic(a, 3, 14);
  const double e_sw = evaluate(sw, cubic).energy / static_cast<double>(cubic.size());
  const double e_tersoff = evaluate(tersoff, cubic).energy / static_cast<double>(cubic.size());
  for (const auto &prim : {diamondPrimitive(a, 14), lowerTriangular(diamondPrimitive(a, 14))}) {
    for (int n = 1; n <= 7; ++n) {
      CAPTURE(prim.box, n);
      const auto big = repeat(prim, n, n, n);
      const auto atoms = static_cast<double>(big.size());
      CHECK_THAT(evaluate(sw, big).energy / atoms, WithinAbs(e_sw, 1e-10));
      CHECK_THAT(evaluate(tersoff, big).energy / atoms, WithinAbs(e_tersoff, 1e-10));
    }
  }
}

TEST_CASE("a skewed triclinic basis of one lattice gives the same answer",
          "[images][triclinic]") {
  setThreads("1");
  // Wide, but the rows are long combinations of the cubic rows. Once the
  // atoms are wrapped into that cell, bins cut along the diagonal lengths
  // no longer hold every partner in the adjacent bins.
  auto cubic = fccCubic(kLjFcc, 11, 6, 6, 18);
  jitter(cubic, 0.05, 21u);
  const int m[3][3] = {{1, 0, 0}, {2, 1, 0}, {1, 1, 1}};
  const auto skew = rebased(cubic, m);
  const auto lj = reducedLj();
  const auto a = evaluate(lj, cubic);
  const auto b = evaluate(lj, skew);
  CHECK_THAT(b.energy, WithinAbs(a.energy, 1e-10 * static_cast<double>(cubic.size())));
  for (std::size_t k = 0; k < cubic.pos.size(); ++k) {
    CHECK_THAT(b.forces[k], WithinAbs(a.forces[k], 1e-9));
  }
  for (std::size_t k = 0; k < 9; ++k) {
    CHECK_THAT(b.stress[k], WithinAbs(a.stress[k], 1e-10));
  }
  const auto inside = wrapped(skew);
  const auto c = evaluate(lj, inside);
  CHECK_THAT(c.energy, WithinAbs(a.energy, 1e-10 * static_cast<double>(cubic.size())));
  for (std::size_t k = 0; k < cubic.pos.size(); ++k) {
    CHECK_THAT(c.forces[k], WithinAbs(a.forces[k], 1e-9));
  }

  // Thin and skewed: the primitive fcc cell on a sheared basis.
  const int n[3][3] = {{1, 0, 0}, {1, 1, 0}, {-1, 2, 1}};
  const auto thin = rebased(fccPrimitive(kLjFcc, 18), n);
  CHECK_THAT(evaluate(lj, thin).energy,
             WithinAbs(ljFccLatticeSum(kLjFcc, 1.0, 1.0, 3.0), 1e-10));

  const auto cfg = copperSc();
  const rgpot::SuttonChenPot sc(cfg);
  const auto sc_prim = rebased(fccPrimitive(cfg.a, 29), n);
  CHECK_THAT(evaluate(sc, sc_prim).energy, WithinAbs(scFccLatticeSum(cfg.a, cfg), 1e-10));
}

TEST_CASE("forces and stress on thin cells match finite differences", "[images][fd]") {
  setThreads("1");
  const rgpot::LJPot lj{
      rgpot::LJConfig{.u0 = 1.0, .cutoff = 3.0, .psi = 1.0, .switch_width = 0.5}};
  Cell two = repeat(fccPrimitive(kLjFcc, 18), 2, 1, 1);
  jitter(two, 0.1, 23u);
  requireFiniteDifferenceForces(lj, two, 1e-5, 1e-6);
  requireFiniteDifferenceStress(lj, two, 1e-6, 1e-6);

  auto four = fccCubic(kLjFcc, 1, 1, 1, 18);
  jitter(four, 0.1, 29u);
  requireFiniteDifferenceForces(lj, four, 1e-5, 1e-6);
  requireFiniteDifferenceStress(lj, four, 1e-6, 1e-6);

  const rgpot::SuttonChenPot sc(copperSc());
  Cell cu = repeat(fccPrimitive(3.61, 29), 1, 2, 1);
  jitter(cu, 0.1, 31u);
  requireFiniteDifferenceForces(sc, cu, 1e-5, 1e-7);
  requireFiniteDifferenceStress(sc, cu, 1e-6, 1e-8);
}

TEST_CASE("a slab thin in two directions sums the in-plane images", "[images][slab]") {
  setThreads("1");
  // The vacuum along z is wider than the cutoff, so the two faces do not
  // see each other.
  auto slab = fccCubic(kLjFcc, 1, 1, 3, 18);
  slab.box[8] = 3.0 * kLjFcc + 10.0;
  jitter(slab, 0.05, 37u);
  const auto lj = reducedLj();
  const auto wide = repeat(slab, 6, 6, 1);
  requireSamePerAtom(lj, slab, wide, 1e-10);
  const auto a = evaluate(lj, slab);
  const auto b = evaluate(lj, wide);
  for (std::size_t k = 0; k < slab.pos.size(); ++k) {
    CHECK_THAT(a.forces[k], WithinAbs(b.forces[k], 1e-9));
  }
}

TEST_CASE("the public neighbour list searches sheared and thin cells by their widths",
          "[images][nlist]") {
  setThreads("1");
  // Wide, sheared and wrapped: the rows are long combinations of a cubic
  // lattice, so Cartesian bins on the diagonal miss partners.
  auto cubic = fccCubic(kLjFcc, 11, 6, 6, 18);
  jitter(cubic, 0.05, 21u);
  const int m[3][3] = {{1, 0, 0}, {2, 1, 0}, {1, 1, 1}};
  requirePublicListMatchesBrute(wrapped(rebased(cubic, m)), 3.0, 2);
  // Thin, at the size where the direct loop hands over to the cells.
  auto thin = repeat(fccPrimitive(kLjFcc, 18), 16, 8, 4);
  jitter(thin, 0.05, 53u);
  requirePublicListMatchesBrute(wrapped(thin), 3.0, 4);
}

TEST_CASE("wide orthorhombic cells keep the minimum-image walk", "[images][fast]") {
  setThreads("1");
  auto lj_cell = fccCubic(kLjFcc, 6, 6, 6, 18);
  jitter(lj_cell, 0.05, 41u);
  auto cu = fccCubic(3.61, 5, 5, 5, 29);
  jitter(cu, 0.05, 43u);
  auto al = fccCubic(4.05, 4, 4, 4, 13);
  jitter(al, 0.05, 47u);

  const std::array<bool, 3> periodic{{true, true, true}};
  const auto images = [&](const Cell &s, double cutoff) {
    rgpot::nlist::Neighbours cells;
    rgpot::nlist::updateCells(cells, s.pos.data(), s.size(), s.box.data(), cutoff, 0.0,
                              rgpot::nlist::Image::Round, rgpot::nlist::Mode::Auto, periodic);
    return cells.images();
  };
  CHECK_FALSE(images(lj_cell, 3.0));
  CHECK_FALSE(images(cu, 7.0));
  CHECK_FALSE(images(al, 5.555));
  CHECK(images(fccCubic(kLjFcc, 1, 1, 1, 18), 3.0));
  // A sheared cell takes the table only when a face spacing, 3.61 / sqrt(3)
  // here, is under twice the search.
  CHECK_FALSE(images(fccPrimitive(3.61, 29), 1.0));
  CHECK(images(fccPrimitive(3.61, 29), 1.1));

  const double lj = evaluate(reducedLj(), lj_cell).energy;
  const double sc = evaluate(rgpot::SuttonChenPot(copperSc()), cu).energy;
  const double eam = evaluate(rgpot::EAMPot(), al).energy;
  CHECK_THAT(lj, WithinAbs(kWideLj, 1e-12 * std::abs(kWideLj)));
  CHECK_THAT(sc, WithinAbs(kWideSc, 1e-12 * std::abs(kWideSc)));
  CHECK_THAT(eam, WithinAbs(kWideEam, 1e-12 * std::abs(kWideEam)));
}

TEST_CASE("the neighbour layer honours (T, T, F) on a thin cell", "[images][slab][nlist]") {
  setThreads("1");
  // Rows not aligned with the axes, an open third row shorter than the
  // atoms' extent, and atoms outside the cell.
  Cell s;
  s.box = {1.3, 0.2, 0.0, 0.4, 1.1, 0.1, 0.3, 0.2, 1.0};
  for (int i = 0; i < 5; ++i) {
    s.pos.push_back(0.9 * i - 1.0);
    s.pos.push_back(0.37 * i * i - 0.4);
    s.pos.push_back(0.83 * i);
    s.types.push_back(1);
  }
  const std::array<bool, 3> periodic{{true, true, false}};
  const double cutoff = 3.2;
  struct Hit {
    std::int32_t i, j;
    double r2;
  };
  std::vector<Hit> brute;
  for (std::int32_t i = 0; i < 5; ++i) {
    for (std::int32_t j = 0; j < 5; ++j) {
      for (int sa = -20; sa <= 20; ++sa) {
        for (int sb = -20; sb <= 20; ++sb) {
          if (i == j && sa == 0 && sb == 0) {
            continue;
          }
          double d[3];
          for (std::size_t c = 0; c < 3; ++c) {
            d[c] = s.pos[3 * static_cast<std::size_t>(i) + c] -
                   s.pos[3 * static_cast<std::size_t>(j) + c] - sa * s.box[c] -
                   sb * s.box[3 + c];
          }
          const double r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
          if (r2 <= cutoff * cutoff) {
            brute.push_back({i, j, r2});
          }
        }
      }
    }
  }
  REQUIRE(brute.size() > 100);
  double brute_sum = 0.0;
  for (const auto &h : brute) {
    brute_sum += 1.0 / h.r2;
  }
  for (auto mode : {rgpot::nlist::Mode::Stream, rgpot::nlist::Mode::Stored}) {
    rgpot::nlist::Neighbours cells;
    rgpot::nlist::updateCells(cells, s.pos.data(), s.size(), s.box.data(), cutoff, 0.0,
                              rgpot::nlist::Image::Round, mode, periodic);
    std::size_t full = 0;
    cells.forDirected(0, 5, [&](std::int32_t, std::int32_t, double, double, double, double r2) {
      full += r2 <= cutoff * cutoff ? 1 : 0;
    });
    std::size_t half = 0;
    double half_sum = 0.0;
    cells.forHalf(0, 5, [&](std::int32_t i, std::int32_t j, double, double, double, double r2) {
      CHECK(j >= i);
      if (r2 <= cutoff * cutoff) {
        ++half;
        half_sum += 1.0 / r2;
      }
    });
    CHECK(full == brute.size());
    CHECK(2 * half == brute.size());
    CHECK_THAT(2.0 * half_sum, WithinAbs(brute_sum, 1e-12 * brute_sum));
  }
}

TEST_CASE("atoms several cells outside a thin cell see the same images",
          "[images][unwrapped]") {
  setThreads("1");
  // An MD step does not wrap, so a thin cell meets atoms far outside it.
  // Moving an atom by a lattice vector changes nothing.
  const int n[3][3] = {{1, 0, 0}, {1, 1, 0}, {-1, 2, 1}};
  auto inside = wrapped(rebased(repeat(fccPrimitive(kLjFcc, 18), 2, 1, 2), n));
  jitter(inside, 0.05, 59u);
  Cell outside = inside;
  for (std::size_t p = 0; p < outside.size(); ++p) {
    const int s[3] = {static_cast<int>(p % 3) * 3 - 3, 5 - static_cast<int>(p) * 2,
                      static_cast<int>(p % 2) * 7 - 2};
    for (std::size_t c = 0; c < 3; ++c) {
      outside.pos[3 * p + c] += s[0] * inside.box[c] + s[1] * inside.box[3 + c] +
                                s[2] * inside.box[6 + c];
    }
  }
  const auto lj = reducedLj();
  const auto a = evaluate(lj, inside);
  const auto b = evaluate(lj, outside);
  CHECK_THAT(b.energy, WithinAbs(a.energy, 1e-10));
  for (std::size_t k = 0; k < a.forces.size(); ++k) {
    CHECK_THAT(b.forces[k], WithinAbs(a.forces[k], 1e-9));
  }
  for (std::size_t k = 0; k < 9; ++k) {
    CHECK_THAT(b.stress[k], WithinAbs(a.stress[k], 1e-9));
  }
  requirePublicListMatchesBrute(outside, 3.0, 16);
  const rgpot::SuttonChenPot sc(copperSc());
  auto cu = wrapped(rebased(repeat(fccPrimitive(3.61, 29), 2, 1, 2), n));
  jitter(cu, 0.05, 63u);
  Cell cu_out = cu;
  for (std::size_t p = 0; p < cu_out.size(); ++p) {
    for (std::size_t c = 0; c < 3; ++c) {
      cu_out.pos[3 * p + c] += (static_cast<double>(p) - 1.5) * 2.0 * cu.box[3 + c];
    }
  }
  const auto e = evaluate(sc, cu);
  const auto g = evaluate(sc, cu_out);
  CHECK_THAT(g.energy, WithinAbs(e.energy, 1e-10));
  for (std::size_t k = 0; k < e.forces.size(); ++k) {
    CHECK_THAT(g.forces[k], WithinAbs(e.forces[k], 1e-9));
  }

  // Past the direct loop: a thin cell of 512 atoms, each a cell or two
  // outside.
  auto many = wrapped(repeat(fccPrimitive(kLjFcc, 18), 16, 8, 4));
  jitter(many, 0.05, 61u);
  Cell stale = many;
  for (std::size_t p = 0; p < stale.size(); ++p) {
    const int s[3] = {static_cast<int>(p % 3) - 1, static_cast<int>(p % 5) - 2,
                      static_cast<int>(p % 2)};
    for (std::size_t c = 0; c < 3; ++c) {
      stale.pos[3 * p + c] +=
          s[0] * many.box[c] + s[1] * many.box[3 + c] + s[2] * many.box[6 + c];
    }
  }
  REQUIRE(stale.size() >= 512);
  requirePublicListMatchesBrute(stale, 3.0, 6);
  CHECK_THAT(evaluate(lj, stale).energy,
             WithinAbs(evaluate(lj, many).energy, 1e-10 * static_cast<double>(many.size())));
}

TEST_CASE("a pair view keeps its list across calls", "[images][simd]") {
  setThreads("1");
  auto cu = fccCubic(3.61, 5, 5, 5, 29);
  jitter(cu, 0.05, 67u);
  rgpot::simd::PairSpec opt;
  opt.cutoff = 5.0;
  opt.skin = 1.0;
  rgpot::nlist::Neighbours reuse;
  rgpot::simd::CachedPairs pairs;
  for (int call = 0; call < 3; ++call) {
    rgpot::simd::fill_view(pairs, cu.pos.data(), cu.size(), cu.box.data(), opt,
                           rgpot::nlist::Mode::Auto, &reuse);
  }
  CHECK(reuse.builds() == 1);
  CHECK(reuse.reused());
}

TEST_CASE("each walk rebuilds its tile stencil after the atoms move", "[images][fast]") {
  // The one-thread walk and the team walk keep separate stencils. Moving
  // the atoms between them must not leave the team walk on the old one
  // when the next call reuses the bins.
  auto before = fccCubic(kLjFcc, 8, 8, 8, 18);
  jitter(before, 0.05, 79u);
  auto after = before;
  jitter(after, 0.3, 83u);
  auto lj = reducedLj();
  lj.setNeighbourMode(rgpot::nlist::Mode::Stream);
  const auto fresh = [&](const Cell &s) {
    auto pot = reducedLj();
    pot.setNeighbourMode(rgpot::nlist::Mode::Stream);
    return evaluate(pot, s);
  };
  setThreads("2");
  (void)evaluate(lj, before);
  setThreads("1");
  (void)evaluate(lj, after);
  setThreads("2");
  const auto team = evaluate(lj, after);
  const auto ref = fresh(after);
  setThreads("1");
  CHECK_THAT(team.energy, WithinAbs(ref.energy, 1e-10 * std::abs(ref.energy)));
  for (std::size_t k = 0; k < ref.forces.size(); ++k) {
    CHECK_THAT(team.forces[k], WithinAbs(ref.forces[k], 1e-9));
  }
}

TEST_CASE("atoms outside a wide cell keep their partners", "[images][unwrapped][fast]") {
  setThreads("1");
  // Enough bins that a clamped atom leaves its partners' stencil.
  auto cubic = fccCubic(kLjFcc, 14, 14, 14, 18);
  jitter(cubic, 0.05, 71u);
  const auto away = [](const Cell &s) {
    Cell out = s;
    for (std::size_t p = 0; p < out.size(); ++p) {
      const int k[3] = {static_cast<int>(p % 3) - 1, static_cast<int>(p % 5) - 2,
                        static_cast<int>(p % 2)};
      for (std::size_t c = 0; c < 3; ++c) {
        out.pos[3 * p + c] += k[0] * s.box[c] + k[1] * s.box[3 + c] + k[2] * s.box[6 + c];
      }
    }
    return out;
  };
  const auto moved = away(cubic);
  const double inside = evaluate(reducedLj(), cubic).energy;
  const double tol = 1e-10 * static_cast<double>(cubic.size());
  for (auto mode : {rgpot::nlist::Mode::Auto, rgpot::nlist::Mode::Stored,
                    rgpot::nlist::Mode::Stream}) {
    auto lj = reducedLj();
    lj.setNeighbourMode(mode);
    const auto a = evaluate(lj, moved);
    CHECK_THAT(a.energy, WithinAbs(inside, tol));
  }
  const auto ref = evaluate(reducedLj(), cubic);
  const auto out = evaluate(reducedLj(), moved);
  for (std::size_t k = 0; k < ref.forces.size(); ++k) {
    CHECK_THAT(out.forces[k], WithinAbs(ref.forces[k], 1e-9));
  }

  // The same lattice on a sheared basis, wide by its face spacings, with
  // the atoms wrapped into it and then moved out again.
  const int m[3][3] = {{1, 0, 0}, {1, 1, 0}, {0, 0, 1}};
  auto small = fccCubic(kLjFcc, 8, 8, 8, 18);
  jitter(small, 0.05, 71u);
  const double small_inside = evaluate(reducedLj(), small).energy;
  const auto skew = wrapped(rebased(small, m));
  for (int k = 0; k < 3; ++k) {
    REQUIRE(height(skew.box, k) > 2.0 * (3.0 + 1.0));
  }
  CHECK_THAT(evaluate(reducedLj(), skew).energy, WithinAbs(small_inside, 1e-10 * 2048.0));
  CHECK_THAT(evaluate(reducedLj(), away(skew)).energy, WithinAbs(small_inside, 1e-10 * 2048.0));
  const rgpot::SuttonChenPot sc(copperSc());
  auto cu = fccCubic(3.61, 8, 8, 8, 29);
  jitter(cu, 0.05, 73u);
  const double cu_inside = evaluate(sc, cu).energy;
  const auto cu_skew = wrapped(rebased(cu, m));
  CHECK_THAT(evaluate(sc, cu_skew).energy, WithinAbs(cu_inside, 1e-10 * 2048.0));
  CHECK_THAT(evaluate(sc, away(cu)).energy, WithinAbs(cu_inside, 1e-10 * 2048.0));
}
