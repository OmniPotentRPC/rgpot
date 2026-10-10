// MIT License
// Copyright 2023--present rgpot developers
//
// Pair search and pair-kernel checks: the linked-cell scan reports the
// brute-force pair set, the cached force loop agrees with a fresh scan in
// every fold mode, and the analytic forces of every pair potential equal
// -dE/dx by central differences.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

#include <catch2/catch_all.hpp>

#include "rgpot/ForceStructs.hpp"
#include "rgpot/LennardJones/LJClusterPot.hpp"
#include "rgpot/LennardJones/LJPot.hpp"
#include "rgpot/Morse/MorsePot.hpp"
#include "rgpot/ZBL/ZBLPot.hpp"
#include "rgpot/nlist/PairListCache.hpp"
#include "rgpot/nlist/cell_visit.hpp"
#include "rgpot/nlist/vesin_visit.hpp"

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;
using rgpot::nlist::CachedPairList;
using rgpot::nlist::CellGrid;
using rgpot::nlist::PairTerm;

namespace {

/// Uniform random positions in [lo, hi)^3 with a minimum separation, so
/// no pair sits at a singular distance.
std::vector<double> randomPositions(std::size_t n, double lo, double hi,
                                    double minSep, std::uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<double> u(lo, hi);
  std::vector<double> R;
  R.reserve(3 * n);
  while (R.size() < 3 * n) {
    const double x = u(rng), y = u(rng), z = u(rng);
    bool ok = true;
    for (std::size_t a = 0; ok && a < R.size() / 3; ++a) {
      const double dx = x - R[3 * a], dy = y - R[3 * a + 1],
                   dz = z - R[3 * a + 2];
      ok = dx * dx + dy * dy + dz * dz >= minSep * minSep;
    }
    if (ok) {
      R.insert(R.end(), {x, y, z});
    }
  }
  return R;
}

using PairRecord = std::tuple<int32_t, int32_t, double, double, double>;

std::vector<PairRecord> brutePairs(const std::vector<double> &R,
                                   const double w[3], const double inv[3],
                                   double rl) {
  std::vector<PairRecord> out;
  std::vector<int32_t> ij;
  vesin::cpu::brute_force_visit(
      R.data(), R.size() / 3, w, inv, rl * rl, rl * rl, ij,
      [&](int32_t i, int32_t j, double dx, double dy, double dz, double) {
        out.emplace_back(i, j, dx, dy, dz);
      });
  std::sort(out.begin(), out.end());
  return out;
}

std::vector<PairRecord> cellPairs(const std::vector<double> &R,
                                  const double w[3], const double inv[3],
                                  double rl, bool &usedGrid) {
  std::vector<PairRecord> out;
  std::vector<int32_t> ij;
  CellGrid g;
  usedGrid = g.build(R.data(), R.size() / 3, w, inv, rl);
  if (!usedGrid) {
    return out;
  }
  rgpot::nlist::cell_visit<true>(
      g, R.data(), w, inv, rl * rl, rl * rl, ij,
      [&](int32_t i, int32_t j, double dx, double dy, double dz, double) {
        out.emplace_back(i, j, dx, dy, dz);
      });
  REQUIRE(ij.size() == 2 * out.size());
  std::sort(out.begin(), out.end());
  return out;
}

/// LJ pair kernel with Ar parameters, as LJPot evaluates it unshifted.
PairTerm ljTerm(double r2) {
  const double psi2 = 3.4 * 3.4;
  const double invR2 = 1.0 / r2;
  const double sr2 = psi2 * invR2;
  const double a = sr2 * sr2 * sr2;
  const double b = 4.0 * 0.0104 * a;
  return {b * (a - 1.0), 6.0 * b * invR2 * (2.0 * a - 1.0)};
}

struct Eval {
  double energy;
  std::vector<double> F;
};

/// Reference: every pair of every image within the cutoff, no list.
Eval referenceLJ(const std::vector<double> &R, const std::array<double, 9> &box,
                 double rc) {
  const std::size_t n = R.size() / 3;
  Eval e{0.0, std::vector<double>(3 * n, 0.0)};
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = i + 1; j < n; ++j) {
      double d[3];
      for (int k = 0; k < 3; ++k) {
        const double w = box[static_cast<std::size_t>(4 * k)];
        d[k] = R[3 * i + static_cast<std::size_t>(k)] -
               R[3 * j + static_cast<std::size_t>(k)];
        d[k] -= w * std::round(d[k] / w);
      }
      const double r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
      if (r2 > rc * rc) {
        continue;
      }
      const PairTerm t = ljTerm(r2);
      e.energy += t.energy;
      for (int k = 0; k < 3; ++k) {
        e.F[3 * i + static_cast<std::size_t>(k)] += t.fscale * d[k];
        e.F[3 * j + static_cast<std::size_t>(k)] -= t.fscale * d[k];
      }
    }
  }
  return e;
}

/// Build a slot on R0 (phantom then capture), then evaluate at R.
Eval cachedLJ(const std::vector<double> &R0, const std::vector<double> &R,
              const std::array<double, 9> &box, double rc,
              CachedPairList::FoldMode &mode) {
  const std::size_t n = R.size() / 3;
  CachedPairList::Options opt;
  opt.cutoff = rc;
  CachedPairList list;
  auto ignore = [](int32_t, int32_t, double, double, double, double) {};
  list.visitOnly(R0.data(), n, box.data(), opt, ignore);
  REQUIRE(list.cacheable());
  list.rebuildFused(R0.data(), n, box.data(), opt, ignore);
  REQUIRE(list.valid(R.data(), n, box.data(), opt));
  mode = list.foldMode();
  Eval e{0.0, std::vector<double>(3 * n, 0.0)};
  e.energy =
      list.accumulate(R.data(), e.F.data(),
                      [](int32_t, int32_t, double r2) { return ljTerm(r2); });
  return e;
}

void requireSame(const Eval &a, const Eval &b) {
  REQUIRE_THAT(a.energy, WithinRel(b.energy, 1e-12));
  double fmax = 0.0;
  for (double f : b.F) {
    fmax = std::max(fmax, std::abs(f));
  }
  for (std::size_t k = 0; k < a.F.size(); ++k) {
    REQUIRE_THAT(a.F[k], WithinAbs(b.F[k], 1e-12 * fmax));
  }
}

/// Small displacement of every coordinate, well inside skin/2 = 0.5 A.
std::vector<double> jiggle(const std::vector<double> &R, std::uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<double> u(-0.15, 0.15);
  std::vector<double> out = R;
  for (double &x : out) {
    x += u(rng);
  }
  return out;
}

template <typename Pot>
void requireForcesAreMinusGradient(const Pot &pot, std::vector<double> R,
                                   const std::vector<int> &types,
                                   const std::array<double, 9> &box,
                                   double tol) {
  const std::size_t n = R.size() / 3;
  std::vector<double> F(3 * n, 0.0);
  auto energyAt = [&](const std::vector<double> &X, double *Fout) {
    std::vector<double> scratch(3 * n, 0.0);
    rgpot::ForceInput fi{.nAtoms = n,
                         .pos = X.data(),
                         .atmnrs = types.data(),
                         .box = box.data()};
    rgpot::ForceOut fo{.F = Fout ? Fout : scratch.data(),
                       .energy = 0.0,
                       .variance = 0.0,
                       .stress = {},
                       .has_stress = 0};
    pot.forceImpl(fi, &fo);
    return fo.energy;
  };
  energyAt(R, F.data());
  const double h = 1e-5;
  for (std::size_t k = 0; k < 3 * n; ++k) {
    const double x0 = R[k];
    R[k] = x0 + h;
    const double ep = energyAt(R, nullptr);
    R[k] = x0 - h;
    const double em = energyAt(R, nullptr);
    R[k] = x0;
    const double fd = -(ep - em) / (2.0 * h);
    REQUIRE_THAT(F[k], WithinAbs(fd, tol * std::max(1.0, std::abs(fd))));
  }
}

} // namespace

TEST_CASE("Linked cells report the brute-force pair set", "[PairList]") {
  const double rl = 3.0;
  SECTION("periodic box, four or more cells per axis") {
    const double side = 14.0;
    const auto R = randomPositions(600, -2.0, side + 2.0, 0.8, 11);
    const double w[3] = {side, side + 1.0, side + 2.5};
    const double inv[3] = {1.0 / w[0], 1.0 / w[1], 1.0 / w[2]};
    bool used = false;
    const auto cell = cellPairs(R, w, inv, rl, used);
    REQUIRE(used);
    REQUIRE(cell == brutePairs(R, w, inv, rl));
    REQUIRE(!cell.empty());
  }
  SECTION("free boundaries") {
    const auto R = randomPositions(600, 0.0, 20.0, 0.8, 12);
    const double w[3] = {1e6, 1e6, 1e6};
    const double inv[3] = {0.0, 0.0, 0.0};
    bool used = false;
    const auto cell = cellPairs(R, w, inv, rl, used);
    REQUIRE(used);
    REQUIRE(cell == brutePairs(R, w, inv, rl));
  }
  SECTION("mixed: slab periodic in x and y") {
    const double side = 15.0;
    const auto R = randomPositions(600, 0.0, side, 0.8, 13);
    const double w[3] = {side, side, 1e6};
    const double inv[3] = {1.0 / side, 1.0 / side, 0.0};
    bool used = false;
    const auto cell = cellPairs(R, w, inv, rl, used);
    REQUIRE(used);
    REQUIRE(cell == brutePairs(R, w, inv, rl));
  }
  SECTION("fewer than three cells along a periodic axis refuses the grid") {
    const auto R = randomPositions(100, 0.0, 8.0, 0.8, 14);
    const double w[3] = {8.0, 30.0, 30.0};
    const double inv[3] = {1.0 / 8.0, 1.0 / 30.0, 1.0 / 30.0};
    CellGrid g;
    REQUIRE_FALSE(g.build(R.data(), 100, w, inv, rl));
  }
}

TEST_CASE("Cached force loop matches the reference in every fold mode",
          "[PairList]") {
  using Mode = CachedPairList::FoldMode;
  const double rc = 8.5;
  SECTION("stored image shifts: cutoff + skin below half the box") {
    const double side = 20.0;
    const std::array<double, 9> box{side, 0, 0, 0, side, 0, 0, 0, side};
    const auto R0 = randomPositions(300, 0.0, side, 2.5, 21);
    const auto R = jiggle(R0, 22);
    Mode mode{};
    const Eval got = cachedLJ(R0, R, box, rc, mode);
    REQUIRE(mode == Mode::Coded);
    requireSame(got, referenceLJ(R, box, rc));
  }
  SECTION("per-call fold: cutoff + skin reaches past half the box") {
    const double side = 18.0;
    const std::array<double, 9> box{side, 0, 0, 0, side, 0, 0, 0, side};
    const auto R0 = randomPositions(250, 0.0, side, 2.5, 23);
    const auto R = jiggle(R0, 24);
    Mode mode{};
    const Eval got = cachedLJ(R0, R, box, rc, mode);
    REQUIRE(mode == Mode::Round);
    requireSame(got, referenceLJ(R, box, rc));
  }
  SECTION("unwrapped coordinates fall back to the per-call fold") {
    const double side = 20.0;
    const std::array<double, 9> box{side, 0, 0, 0, side, 0, 0, 0, side};
    auto R0 = randomPositions(300, 0.0, side, 2.5, 25);
    // Move every third atom three box widths away: same periodic system.
    for (std::size_t a = 0; a < R0.size() / 3; a += 3) {
      R0[3 * a] += 3.0 * side;
    }
    const auto R = jiggle(R0, 26);
    Mode mode{};
    const Eval got = cachedLJ(R0, R, box, rc, mode);
    REQUIRE(mode == Mode::Round);
    requireSame(got, referenceLJ(R, box, rc));
  }
  SECTION("large box served by the cell grid") {
    const double side = 60.0;
    const std::array<double, 9> box{side, 0, 0, 0, side, 0, 0, 0, side};
    const auto R0 = randomPositions(2500, 0.0, side, 2.5, 27);
    const auto R = jiggle(R0, 28);
    Mode mode{};
    const Eval got = cachedLJ(R0, R, box, rc, mode);
    REQUIRE(mode == Mode::Coded);
    requireSame(got, referenceLJ(R, box, rc));
  }
}

TEST_CASE("Pair potentials: warm, cold and cell-grid calls agree",
          "[PairList]") {
  // A Pt fcc block of 6^3 cells (864 atoms, 23.5 A): the Morse cutoff
  // 9.5 + skin 1.0 gives two cells per axis, so this box runs the brute
  // scan; the 11^3 block (5324 atoms, 43.1 A) has four cells per axis and
  // runs the cell grid.
  for (int m : {6, 11}) {
    const double a = 3.92;
    std::vector<double> R;
    static const double basis[4][3] = {
        {0.0, 0.0, 0.0}, {0.0, 0.5, 0.5}, {0.5, 0.0, 0.5}, {0.5, 0.5, 0.0}};
    std::mt19937_64 rng(31 + static_cast<unsigned>(m));
    std::uniform_real_distribution<double> u(-0.05, 0.05);
    for (int ix = 0; ix < m; ++ix)
      for (int iy = 0; iy < m; ++iy)
        for (int iz = 0; iz < m; ++iz)
          for (const auto &b : basis) {
            R.push_back((ix + b[0]) * a + u(rng));
            R.push_back((iy + b[1]) * a + u(rng));
            R.push_back((iz + b[2]) * a + u(rng));
          }
    const std::size_t n = R.size() / 3;
    const double side = m * a;
    const std::array<double, 9> box{side, 0, 0, 0, side, 0, 0, 0, side};
    const std::vector<int> types(n, 78);
    rgpot::MorsePot pot;
    auto call = [&](const std::vector<double> &X) {
      Eval e{0.0, std::vector<double>(3 * n, 0.0)};
      rgpot::ForceInput fi{.nAtoms = n,
                           .pos = X.data(),
                           .atmnrs = types.data(),
                           .box = box.data()};
      rgpot::ForceOut fo{.F = e.F.data(),
                         .energy = 0.0,
                         .variance = 0.0,
                         .stress = {},
                         .has_stress = 0};
      pot.forceImpl(fi, &fo);
      e.energy = fo.energy;
      return e;
    };
    const Eval cold = call(R);    // first sighting: list at the cutoff
    const Eval capture = call(R); // second sighting: list captured
    const Eval warm = call(R);    // list hit
    // Every call sums the same pairs in the same order, so the three agree
    // to the bit whichever list served them.
    REQUIRE(capture.energy == cold.energy);
    REQUIRE(warm.energy == cold.energy);
    REQUIRE(capture.F == cold.F);
    REQUIRE(warm.F == cold.F);
    REQUIRE(std::isfinite(cold.energy));
    // A nearby geometry served by the captured list matches a pool that
    // never saw this family (first sighting after the pool turns over).
    std::vector<double> near = R;
    for (std::size_t k = 0; k < near.size(); ++k) {
      near[k] += 0.05 * std::sin(0.7 * static_cast<double>(k));
    }
    const Eval nearWarm = call(near);
    for (int k = 1; k <= 9; ++k) {
      std::vector<double> away = R;
      for (std::size_t q = 0; q < away.size(); q += 3) {
        away[q] += 3.0 * k;
      }
      (void)call(away);
    }
    const Eval nearCold = call(near);
    REQUIRE(nearWarm.energy == nearCold.energy);
    REQUIRE(nearWarm.F == nearCold.F);
  }
}

TEST_CASE("Pair forces equal minus the energy gradient", "[PairList]") {
  const double side = 22.0;
  const std::array<double, 9> box{side, 0, 0, 0, side, 0, 0, 0, side};
  SECTION("LJPot, Ar") {
    const auto R = randomPositions(24, 0.0, side, 3.2, 41);
    const rgpot::LJPot pot{
        rgpot::LJConfig{.u0 = 0.0104, .cutoff = 8.5, .psi = 3.4}};
    requireForcesAreMinusGradient(pot, R, std::vector<int>(24, 18), box, 1e-6);
  }
  SECTION("LJClusterPot, Ar") {
    const auto R = randomPositions(24, 0.0, 12.0, 3.2, 42);
    const rgpot::LJClusterPot pot{
        rgpot::LJClusterConfig{.u0 = 0.0104, .cutoff = 8.5, .psi = 3.4}};
    requireForcesAreMinusGradient(pot, R, std::vector<int>(24, 18), box, 1e-6);
  }
  SECTION("MorsePot, Pt") {
    const auto R = randomPositions(24, 0.0, side, 2.4, 43);
    const rgpot::MorsePot pot;
    requireForcesAreMinusGradient(pot, R, std::vector<int>(24, 78), box, 1e-6);
  }
  SECTION("ZBLPot, Si and Au") {
    const auto R = randomPositions(16, 0.0, 6.0, 1.2, 44);
    std::vector<int> types(16, 14);
    for (std::size_t k = 1; k < types.size(); k += 2) {
      types[k] = 79;
    }
    const rgpot::ZBLPot pot;
    requireForcesAreMinusGradient(pot, R, types, box, 1e-6);
  }
}

TEST_CASE("Quintic switch takes the pair term smoothly to zero",
          "[PairList][switch]") {
  const double rc = 8.5;
  const double width = 2.0;
  const rgpot::LJConfig base{.u0 = 0.0104, .cutoff = rc, .psi = 3.4};
  rgpot::LJConfig swc = base;
  swc.switch_width = width;
  const rgpot::LJPot plain{base};
  const rgpot::LJPot switched{swc};
  const std::array<double, 9> box{40, 0, 0, 0, 40, 0, 0, 0, 40};
  const std::vector<int> types{18, 18};

  auto pair = [&](const rgpot::LJPot &pot, double r) {
    const std::vector<double> R{10.0, 10.0, 10.0, 10.0 + r, 10.0, 10.0};
    std::vector<double> F(6, 0.0);
    rgpot::ForceInput fi{.nAtoms = 2,
                         .pos = R.data(),
                         .atmnrs = types.data(),
                         .box = box.data()};
    rgpot::ForceOut fo{.F = F.data(),
                       .energy = 0.0,
                       .variance = 0.0,
                       .stress = {},
                       .has_stress = 0};
    pot.forceImpl(fi, &fo);
    return std::pair<double, double>{fo.energy, F[3]};
  };

  // Below the switch the term is the unshifted LJ.
  for (double r : {3.5, 4.0, 6.0, rc - width}) {
    const double a = std::pow(3.4 / r, 6.0);
    const double v = 4.0 * 0.0104 * a * (a - 1.0);
    REQUIRE_THAT(pair(switched, r).first, WithinRel(v, 1e-13));
  }
  // Energy and force fall to zero at the cutoff, to the first order the
  // C^2 switch promises: |E| and |F| shrink like (rc - r)^3 and (rc - r)^2.
  const auto near = pair(switched, rc - 1e-3);
  REQUIRE(std::abs(near.first) < 1e-9 * 0.0104 * 1e3);
  REQUIRE(std::abs(near.second) < 1e-6 * 0.0104 * 1e3);
  REQUIRE(pair(switched, rc + 1e-9).first == 0.0);
  // The shifted truncation keeps a force jump at the cutoff.
  REQUIRE(std::abs(pair(plain, rc - 1e-9).second) > 1e-5);

  SECTION("forces equal minus the gradient inside the switch region") {
    const double side = 22.0;
    const std::array<double, 9> cell{side, 0, 0, 0, side, 0, 0, 0, side};
    const auto R = randomPositions(24, 0.0, side, 3.2, 51);
    requireForcesAreMinusGradient(switched, R, std::vector<int>(24, 18), cell,
                                  1e-6);
    rgpot::MorseConfig mc;
    mc.switch_width = 2.5;
    const rgpot::MorsePot morse{mc};
    REQUIRE(morse.energyShift() == 0.0);
    requireForcesAreMinusGradient(morse,
                                  randomPositions(24, 0.0, side, 2.4, 52),
                                  std::vector<int>(24, 78), cell, 1e-6);
    rgpot::LJClusterConfig cc{.u0 = 0.0104, .cutoff = 8.5, .psi = 3.4};
    cc.switch_width = 1.5;
    const rgpot::LJClusterPot cluster{cc};
    requireForcesAreMinusGradient(cluster,
                                  randomPositions(24, 0.0, 12.0, 3.2, 53),
                                  std::vector<int>(24, 18), cell, 1e-6);
  }

  SECTION("configuration") {
    REQUIRE(rgpot::LJPot{base}.paramsKey() != switched.paramsKey());
    rgpot::LJConfig zero = base;
    zero.switch_width = 0.0;
    REQUIRE(rgpot::LJPot{zero}.paramsKey() == rgpot::LJPot{base}.paramsKey());
    rgpot::LJConfig bad = base;
    bad.switch_width = -1.0;
    REQUIRE_THROWS_AS(rgpot::LJPot{bad}, std::invalid_argument);
    bad.switch_width = rc + 1.0;
    REQUIRE_THROWS_AS(rgpot::LJPot{bad}, std::invalid_argument);
  }
}

namespace {
std::array<double, 3> enumeratedImage(std::array<double, 3> displacement,
                                      const std::array<double, 9> &cell,
                                      const std::array<bool, 3> &periodic) {
  auto best = displacement;
  const auto norm2 = [](const auto &v) {
    return v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
  };
  double shortest = norm2(best);
  const int nx = periodic[0] ? 64 : 0;
  const int ny = periodic[1] ? 12 : 0;
  const int nz = periodic[2] ? 12 : 0;
  for (int i = -nx; i <= nx; ++i)
    for (int j = -ny; j <= ny; ++j)
      for (int k = -nz; k <= nz; ++k) {
        std::array<double, 3> v{};
        for (int c = 0; c < 3; ++c)
          v[c] =
              displacement[c] - i * cell[c] - j * cell[3 + c] - k * cell[6 + c];
        if (norm2(v) < shortest) {
          shortest = norm2(v);
          best = v;
        }
      }
  return best;
}
} // namespace

TEST_CASE("Full-cell pair paths match independent lattice enumeration",
          "[PairList][triclinic]") {
  const std::array<std::array<double, 9>, 3> cells{
      {{4.0, 0.2, 0.1, 0.3, 5.0, -0.2, 0.1, -0.3, 6.0},
       {4.0, 0.0, 0.0, 15.6, 0.5, 0.0, 0.2, 0.3, 5.0},
       {0.1, -0.3, 6.0, 0.3, 5.0, -0.2, 4.0, 0.2, 0.1}}};
  const std::array<double, 12> positions{0.17,  -0.31, 0.23, 3.87, 0.42,  -0.39,
                                         -3.46, 0.21,  1.63, 7.31, -0.41, 2.07};
  for (const auto &cell : cells) {
    for (unsigned mask = 0; mask < 8; ++mask) {
      CAPTURE(cell, mask);
      CachedPairList::Options opt;
      opt.cutoff = 3.0;
      for (unsigned k = 0; k < 3; ++k)
        opt.periodic[k] = (mask & (1u << k)) != 0;
      std::vector<PairRecord> expected;
      std::array<double, 12> expected_force{};
      double expected_energy = 0.0;
      for (int i = 0; i < 4; ++i) {
        for (int j = i + 1; j < 4; ++j) {
          std::array<double, 3> d{};
          for (int k = 0; k < 3; ++k)
            d[k] = positions[3 * i + k] - positions[3 * j + k];
          const auto folded = enumeratedImage(d, cell, opt.periodic);
          const double r2 = folded[0] * folded[0] + folded[1] * folded[1] +
                            folded[2] * folded[2];
          if (r2 > opt.cutoff * opt.cutoff)
            continue;
          expected.emplace_back(i, j, folded[0], folded[1], folded[2]);
          expected_energy += 0.5 * r2;
          for (int k = 0; k < 3; ++k) {
            expected_force[3 * i + k] -= folded[k];
            expected_force[3 * j + k] += folded[k];
          }
        }
      }
      const auto check = [&](const std::vector<PairRecord> &got) {
        REQUIRE(got.size() == expected.size());
        for (std::size_t p = 0; p < got.size(); ++p) {
          REQUIRE(std::get<0>(got[p]) == std::get<0>(expected[p]));
          REQUIRE(std::get<1>(got[p]) == std::get<1>(expected[p]));
          REQUIRE_THAT(std::get<2>(got[p]),
                       WithinAbs(std::get<2>(expected[p]), 1e-12));
          REQUIRE_THAT(std::get<3>(got[p]),
                       WithinAbs(std::get<3>(expected[p]), 1e-12));
          REQUIRE_THAT(std::get<4>(got[p]),
                       WithinAbs(std::get<4>(expected[p]), 1e-12));
        }
      };
      std::vector<PairRecord> got;
      auto collect = [&](int32_t i, int32_t j, double dx, double dy, double dz,
                         double) { got.emplace_back(i, j, dx, dy, dz); };
      CachedPairList direct;
      direct.visitOnly(positions.data(), 4, cell.data(), opt, collect);
      check(got);
      for (int route = 0; route < 3; ++route) {
        CAPTURE(route);
        CachedPairList list;
        got.clear();
        if (route == 0) {
          list.rebuildFused(positions.data(), 4, cell.data(), opt, collect);
          check(got);
        } else if (route == 1) {
          list.buildForEval(positions.data(), 4, cell.data(), opt);
        } else {
          list.rebuild(positions.data(), 4, cell.data(), opt);
        }
        got.clear();
        list.forEach(positions.data(), collect);
        check(got);
        std::array<double, 12> force{};
        const double energy = list.accumulate(positions.data(), force.data(),
                                              [](int32_t, int32_t, double r2) {
                                                return PairTerm{0.5 * r2, -1.0};
                                              });
        REQUIRE_THAT(energy, WithinAbs(expected_energy, 1e-12));
        for (std::size_t i = 0; i < force.size(); ++i)
          REQUIRE_THAT(force[i], WithinAbs(expected_force[i], 1e-12));
      }
    }
  }
}

TEST_CASE("Full-cell image search rejects invalid and unrepresentable cells",
          "[PairList][triclinic]") {
  const std::array<bool, 3> periodic{true, true, true};
  const double singular[9] = {1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 1.0};
  REQUIRE_THROWS_AS(rgpot::nlist::MinimumImage(singular, periodic),
                    std::invalid_argument);
  auto invalid =
      std::array<double, 9>{1.0, 0.1, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
  invalid[1] = std::numeric_limits<double>::infinity();
  REQUIRE_THROWS_AS(rgpot::nlist::MinimumImage(invalid.data(), periodic),
                    std::invalid_argument);
  const double ill_conditioned[9] = {1.0, 1.0, 0.0, 1.0, 1.0 + 1e-12,
                                     0.0, 0.0, 0.0, 1.0};
  REQUIRE_THROWS_AS(rgpot::nlist::MinimumImage(ill_conditioned, periodic),
                    std::invalid_argument);
  const double cell[9] = {1.0, 0.1, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
  const rgpot::nlist::MinimumImage image(cell, periodic);
  double dx = 1e30, dy = 0.0, dz = 0.0;
  REQUIRE_THROWS_AS(image.fold(dx, dy, dz), std::overflow_error);
  dx = std::numeric_limits<double>::quiet_NaN();
  REQUIRE_THROWS_AS(image.fold(dx, dy, dz), std::invalid_argument);
}

namespace {

/// The previous long-double search, kept as the agreement oracle.
class LongDoubleMinimumImage {
public:
  LongDoubleMinimumImage(const double *cell,
                         const std::array<bool, 3> &periodic)
      : periodic_(periodic) {
    for (int k = 0; k < 9; ++k) {
      if (!std::isfinite(cell[k]))
        throw std::invalid_argument("periodic cell must be finite");
      cell_[k] = cell[k];
    }
    const auto &a = cell_;
    const long double det = a[0] * (a[4] * a[8] - a[5] * a[7]) -
                            a[1] * (a[3] * a[8] - a[5] * a[6]) +
                            a[2] * (a[3] * a[7] - a[4] * a[6]);
    if (!std::isfinite(det) || det == 0.0L)
      throw std::invalid_argument("periodic cell must be nonsingular");
    inverse_ = {
        (a[4] * a[8] - a[5] * a[7]) / det, (a[2] * a[7] - a[1] * a[8]) / det,
        (a[1] * a[5] - a[2] * a[4]) / det, (a[5] * a[6] - a[3] * a[8]) / det,
        (a[0] * a[8] - a[2] * a[6]) / det, (a[2] * a[3] - a[0] * a[5]) / det,
        (a[3] * a[7] - a[4] * a[6]) / det, (a[1] * a[6] - a[0] * a[7]) / det,
        (a[0] * a[4] - a[1] * a[3]) / det};
    long double cell_norm = 0.0L, inverse_norm = 0.0L;
    for (int row = 0; row < 3; ++row) {
      long double cell_sum = 0.0L, inverse_sum = 0.0L;
      for (int col = 0; col < 3; ++col) {
        cell_sum += std::abs(cell_[3 * row + col]);
        inverse_sum += std::abs(inverse_[3 * row + col]);
      }
      cell_norm = std::max(cell_norm, cell_sum);
      inverse_norm = std::max(inverse_norm, inverse_sum);
    }
    const long double eps = std::numeric_limits<long double>::epsilon();
    if (!(cell_norm * inverse_norm * eps <= std::sqrt(eps)))
      throw std::invalid_argument(
          "periodic cell is too ill-conditioned for image bounds");
    for (int k = 0; k < 3; ++k) {
      dual_norm_[k] =
          std::hypot(inverse_[k], inverse_[3 + k], inverse_[6 + k]);
      if (!std::isfinite(dual_norm_[k]))
        throw std::overflow_error("periodic cell inverse is not finite");
    }
  }

  void fold(double &dx, double &dy, double &dz) const {
    if (!std::isfinite(dx) || !std::isfinite(dy) || !std::isfinite(dz))
      throw std::invalid_argument("pair displacement must be finite");
    std::array<long double, 3> reduced{dx, dy, dz};
    std::array<long double, 3> initial{};
    for (int k = 0; k < 3; ++k) {
      if (periodic_[k])
        initial[k] = std::round(reduced[0] * inverse_[k] +
                                reduced[1] * inverse_[3 + k] +
                                reduced[2] * inverse_[6 + k]);
      checkedIndex(initial[k]);
    }
    for (int c = 0; c < 3; ++c)
      for (int k = 0; k < 3; ++k)
        reduced[c] -= initial[k] * cell_[3 * k + c];
    auto best = reduced;
    long double best2 =
        best[0] * best[0] + best[1] * best[1] + best[2] * best[2];
    const long double radius = std::sqrt(best2);
    std::array<std::int64_t, 3> low{}, high{};
    std::uint64_t count = 1;
    for (int k = 0; k < 3; ++k) {
      if (periodic_[k]) {
        const long double f = reduced[0] * inverse_[k] +
                              reduced[1] * inverse_[3 + k] +
                              reduced[2] * inverse_[6 + k];
        const long double extent = radius * dual_norm_[k];
        low[k] = checkedIndex(std::floor(f - extent) - 1.0L);
        high[k] = checkedIndex(std::ceil(f + extent) + 1.0L);
      }
      const auto width = static_cast<std::uint64_t>(high[k]) -
                         static_cast<std::uint64_t>(low[k]) + 1;
      if (width == 0 || count > static_cast<std::uint64_t>(
                                    std::numeric_limits<std::int64_t>::max()) /
                                    width)
        throw std::overflow_error("periodic image search count is too large");
      count *= width;
    }
    for (auto i = low[0]; i <= high[0]; ++i)
      for (auto j = low[1]; j <= high[1]; ++j)
        for (auto k = low[2]; k <= high[2]; ++k) {
          std::array<long double, 3> candidate{};
          for (int c = 0; c < 3; ++c)
            candidate[c] = reduced[c] - i * cell_[c] - j * cell_[3 + c] -
                           k * cell_[6 + c];
          const long double r2 = candidate[0] * candidate[0] +
                                 candidate[1] * candidate[1] +
                                 candidate[2] * candidate[2];
          if (r2 < best2) {
            best = candidate;
            best2 = r2;
          }
        }
    dx = static_cast<double>(best[0]);
    dy = static_cast<double>(best[1]);
    dz = static_cast<double>(best[2]);
  }

private:
  static std::int64_t checkedIndex(long double value) {
    const long double limit = std::ldexp(1.0L, 63);
    if (!std::isfinite(value) || value <= -limit + 1.0L ||
        value >= limit - 1.0L)
      throw std::overflow_error("periodic image index is out of range");
    return static_cast<std::int64_t>(value);
  }
  std::array<long double, 9> cell_{};
  std::array<long double, 9> inverse_{};
  std::array<long double, 3> dual_norm_{};
  std::array<bool, 3> periodic_{};
};

void expectSameImage(const LongDoubleMinimumImage &old_image,
                     const rgpot::nlist::MinimumImage &image, double x,
                     double y, double z, double tol) {
  double ox = x, oy = y, oz = z;
  double nx = x, ny = y, nz = z;
  old_image.fold(ox, oy, oz);
  image.fold(nx, ny, nz);
  CAPTURE(x, y, z, ox, oy, oz, nx, ny, nz);
  REQUIRE_THAT(nx, WithinAbs(ox, tol));
  REQUIRE_THAT(ny, WithinAbs(oy, tol));
  REQUIRE_THAT(nz, WithinAbs(oz, tol));
  const double ro = std::sqrt(ox * ox + oy * oy + oz * oz);
  const double rn = std::sqrt(nx * nx + ny * ny + nz * nz);
  REQUIRE(rn <= ro + tol);
}

} // namespace

TEST_CASE("Double minimum image agrees with the long-double search",
          "[PairList][triclinic]") {
  // 1e-9 is the contract stated on MinimumImage: ordinary cells, skewed
  // cells, and near-ill-conditioned cells that still pass the guard.
  constexpr double kTol = 1e-9;
  const std::array<std::array<double, 9>, 3> cells{{
      {10.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0},
      {4.0, 0.2, 0.1, 0.3, 5.0, -0.2, 0.1, -0.3, 6.0},
      {4.0, 0.0, 0.0, 15.6, 0.5, 0.0, 0.2, 0.3, 5.0},
  }};
  std::mt19937_64 rng(7);
  std::uniform_real_distribution<double> local(-12.0, 12.0);
  std::uniform_real_distribution<double> far(-1.0e8, 1.0e8);
  for (const auto &cell : cells) {
    for (unsigned mask = 0; mask < 8; ++mask) {
      std::array<bool, 3> periodic{};
      for (int k = 0; k < 3; ++k)
        periodic[k] = (mask & (1u << k)) != 0;
      CAPTURE(cell, mask);
      const LongDoubleMinimumImage old_image(cell.data(), periodic);
      const rgpot::nlist::MinimumImage image(cell.data(), periodic);
      for (int n = 0; n < 12; ++n)
        expectSameImage(old_image, image, local(rng), local(rng), local(rng),
                        kTol);
      if (mask == 7) {
        for (int n = 0; n < 8; ++n)
          expectSameImage(old_image, image, far(rng), far(rng), far(rng),
                          kTol);
      }
    }
    const std::array<bool, 3> periodic{true, true, true};
    const LongDoubleMinimumImage old_image(cell.data(), periodic);
    const rgpot::nlist::MinimumImage image(cell.data(), periodic);
    for (int ix = -2; ix <= 2; ++ix)
      for (int iy = -2; iy <= 2; ++iy)
        for (int iz = -2; iz <= 2; ++iz)
          expectSameImage(old_image, image, ix * 0.5 * cell[0],
                          iy * 0.5 * cell[4], iz * 0.5 * cell[8], kTol);
  }

  // Shear just inside the long-double guard (condition about 1e9) and
  // one still on the double path (condition about 5e7). Displacements
  // stay near the short lattice vector so the image box stays small.
  const std::array<std::array<double, 9>, 2> shears{{
      {1.0, 1.0, 0.0, 0.0, 4.0e-8, 0.0, 0.0, 0.0, 1.0},
      {1.0, 1.0, 0.0, 0.0, 2.0e-9, 0.0, 0.0, 0.0, 1.0},
  }};
  std::uniform_real_distribution<double> tiny(-1.0e-8, 1.0e-8);
  const std::array<bool, 3> periodic{true, true, true};
  for (const auto &cell : shears) {
    CAPTURE(cell);
    const LongDoubleMinimumImage old_image(cell.data(), periodic);
    const rgpot::nlist::MinimumImage image(cell.data(), periodic);
    for (int n = 0; n < 16; ++n)
      expectSameImage(old_image, image, tiny(rng), tiny(rng), tiny(rng), kTol);
  }
  const double past_guard[9] = {1.0, 1.0, 0.0, 0.0, 5.0e-10, 0.0, 0.0, 0.0,
                                1.0};
  REQUIRE_THROWS_AS(rgpot::nlist::MinimumImage(past_guard, periodic),
                    std::invalid_argument);
}

TEST_CASE("Benchmark minimum image fold and one pair-list build",
          "[.][benchmark]") {
  const double triclinic[9] = {4.0, 0.2, 0.1, 0.3, 5.0, -0.2, 0.1, -0.3, 6.0};
  const std::array<bool, 3> periodic{true, true, true};
  const LongDoubleMinimumImage old_image(triclinic, periodic);
  const rgpot::nlist::MinimumImage image(triclinic, periodic);
  std::mt19937_64 rng(1);
  std::uniform_real_distribution<double> u(-20.0, 20.0);
  constexpr int kN = 20000;
  std::vector<double> disp(3 * kN);
  for (double &v : disp)
    v = u(rng);
  BENCHMARK("long-double fold, 20000 displacements") {
    double sink = 0.0;
    for (int i = 0; i < kN; ++i) {
      double dx = disp[3 * i], dy = disp[3 * i + 1], dz = disp[3 * i + 2];
      old_image.fold(dx, dy, dz);
      sink += dx + dy + dz;
    }
    return sink;
  };
  BENCHMARK("double fold, 20000 displacements") {
    double sink = 0.0;
    for (int i = 0; i < kN; ++i) {
      double dx = disp[3 * i], dy = disp[3 * i + 1], dz = disp[3 * i + 2];
      image.fold(dx, dy, dz);
      sink += dx + dy + dz;
    }
    return sink;
  };

  constexpr int kAtoms = 180;
  std::vector<double> R(3 * kAtoms);
  std::uniform_real_distribution<double> p(0.0, 4.0);
  for (double &v : R)
    v = p(rng);
  CachedPairList::Options opt;
  opt.cutoff = 3.0;
  opt.skin = 1.0;
  opt.periodic = periodic;
  BENCHMARK("pair-list rebuild, 180 atoms, triclinic") {
    CachedPairList list;
    list.rebuild(R.data(), kAtoms, triclinic, opt);
    return list.valid(R.data(), kAtoms, triclinic, opt);
  };
}
