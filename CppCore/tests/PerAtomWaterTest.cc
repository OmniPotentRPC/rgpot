// MIT License
// Copyright 2023--present rgpot developers
//
// Per-atom energies from the water kernels. Intramolecular terms sit on
// the oxygen, intermolecular site pairs split half and half between the
// atoms that carry the sites (the TIP4P M site folds onto its molecule
// with the M-site weights), and the platinum terms split between the
// water atom and the platinum atom.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <vector>

#include <catch2/catch_all.hpp>

#include "rgpot/ForceStructs.hpp"
#include "rgpot/Water/WaterPots.hpp"

using Catch::Matchers::WithinAbs;

namespace {

struct Cell {
  std::vector<double> pos;
  std::vector<int> types;
  std::array<double, 9> box{};
  [[nodiscard]] std::size_t size() const { return types.size(); }
};

struct Result {
  double energy = 0.0;
  std::vector<double> forces;
  std::vector<double> energies;
};

std::array<double, 9> diag(double x, double y, double z) {
  return {x, 0.0, 0.0, 0.0, y, 0.0, 0.0, 0.0, z};
}

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

double sum(const std::vector<double> &v) { return std::accumulate(v.begin(), v.end(), 0.0); }

template <class Pot>
void requireConsistent(const Pot &pot, const Cell &cell) {
  REQUIRE(pot.perAtomEnergies());
  const Result a = plain(pot, cell);
  const Result b = perAtom(pot, cell);
  REQUIRE(b.energy == a.energy);
  REQUIRE(b.forces == a.forces);
  REQUIRE(a.energy != 0.0);
  REQUIRE_THAT(sum(b.energies), WithinAbs(b.energy, 1e-10 * std::abs(b.energy)));
}

Cell permute(const Cell &cell, const std::vector<std::size_t> &order) {
  Cell out = cell;
  for (std::size_t i = 0; i < order.size(); ++i) {
    out.types[i] = cell.types[order[i]];
    for (int k = 0; k < 3; ++k) {
      out.pos[3 * i + k] = cell.pos[3 * order[i] + k];
    }
  }
  return out;
}

template <class Pot>
void requireSymmetric(const Pot &pot, const Cell &cell, const std::vector<std::size_t> &order,
                      double dz) {
  const Result ref = perAtom(pot, cell);
  Cell moved = cell;
  for (std::size_t i = 0; i < cell.size(); ++i) {
    moved.pos[3 * i] += 0.37;
    moved.pos[3 * i + 1] -= 1.21;
    moved.pos[3 * i + 2] += dz;
  }
  const Result shifted = perAtom(pot, moved);
  const Result swapped = perAtom(pot, permute(cell, order));
  for (std::size_t i = 0; i < cell.size(); ++i) {
    const double tol = 1e-10 * std::max(1.0, std::abs(ref.energies[i]));
    REQUIRE_THAT(shifted.energies[i], WithinAbs(ref.energies[i], tol));
    REQUIRE_THAT(swapped.energies[i], WithinAbs(ref.energies[order[i]], tol));
  }
}

/// `copies` of a diagonal cell, each block of equal types kept together so
/// row i of block [start, end) maps to row start * total + copy * (end -
/// start) + (i - start).
Cell repeatWater(const Cell &cell, const std::array<int, 3> &copies) {
  Cell big;
  big.box = diag(copies[0] * cell.box[0], copies[1] * cell.box[4], copies[2] * cell.box[8]);
  std::size_t start = 0;
  while (start < cell.size()) {
    std::size_t end = start;
    while (end < cell.size() && cell.types[end] == cell.types[start]) {
      ++end;
    }
    for (int a = 0; a < copies[0]; ++a) {
      for (int b = 0; b < copies[1]; ++b) {
        for (int c = 0; c < copies[2]; ++c) {
          const double shift[3] = {a * cell.box[0], b * cell.box[4], c * cell.box[8]};
          for (std::size_t i = start; i < end; ++i) {
            for (int k = 0; k < 3; ++k) {
              big.pos.push_back(cell.pos[3 * i + k] + shift[k]);
            }
            big.types.push_back(cell.types[i]);
          }
        }
      }
    }
    start = end;
  }
  return big;
}

template <class Pot>
void requireThinMatchesRepeat(const Pot &pot, const Cell &cell,
                              const std::array<int, 3> &copies) {
  const int total = copies[0] * copies[1] * copies[2];
  const Cell big = repeatWater(cell, copies);
  const Result a = perAtom(pot, cell);
  const Result b = perAtom(pot, big);
  REQUIRE_THAT(a.energy, WithinAbs(b.energy / total, 1e-11));
  std::size_t start = 0;
  while (start < cell.size()) {
    std::size_t end = start;
    while (end < cell.size() && cell.types[end] == cell.types[start]) {
      ++end;
    }
    const std::size_t width = end - start;
    for (std::size_t i = start; i < end; ++i) {
      for (int copy = 0; copy < total; ++copy) {
        const std::size_t row = start * total + copy * width + (i - start);
        REQUIRE_THAT(b.energies[row], WithinAbs(a.energies[i], 1e-11));
      }
    }
    start = end;
  }
}

/// Two flexible waters, H rows first, then O.
Cell twoWaters(double x, double y, double z) {
  return {{1.8, 1.1, 1.5, 0.3, 1.05, 1.55, 4.3, 3.6, 4.9, 3.1, 4.6, 4.9, 1.0, 1.0, 1.0, 3.6,
           3.9, 4.4},
          {1, 1, 1, 1, 8, 8},
          diag(x, y, z)};
}

/// One water above a 2x2 Pt(100) layer whose top plane is the mirror z = 0.
Cell slab(double a) {
  return {{2.0, 1.1, 3.1, 0.5, 1.05, 3.15, 1.2, 1.0, 2.6, 0.0, 0.0, 0.0, a, 0.0, 0.0, 0.0, a,
           0.0, a, a, 0.0},
          {1, 1, 8, 78, 78, 78, 78},
          diag(2 * a, 2 * a, 30.0)};
}

// Swap the two molecules, and H1 with H2 inside the first.
const std::vector<std::size_t> kSwapWaters = {3, 2, 0, 1, 5, 4};

} // namespace

TEST_CASE("Per-atom water energies sum to the total", "[water][per-atom]") {
  const rgpot::TIP4PPot tip(rgpot::WaterConfig{});
  const rgpot::SPCEPot spc(rgpot::WaterConfig{});
  const Cell cluster = twoWaters(40.0, 40.0, 40.0);
  const Cell bulk = twoWaters(5.2, 6.1, 7.3);
  for (const Cell &cell : {cluster, bulk}) {
    requireConsistent(tip, cell);
    requireConsistent(spc, cell);
    requireSymmetric(tip, cell, kSwapWaters, 2.05);
    requireSymmetric(spc, cell, kSwapWaters, 2.05);
  }
  requireThinMatchesRepeat(tip, bulk, {4, 3, 3});
  requireThinMatchesRepeat(spc, bulk, {4, 3, 3});

  const rgpot::TIP4PPtPot zp(rgpot::WaterConfig{});
  const Cell surface = slab(2.77);
  requireConsistent(zp, surface);
  // The mirror plane is fixed, so the slab only moves across the surface.
  requireSymmetric(zp, surface, {1, 0, 2, 4, 3, 6, 5}, 0.0);
  requireThinMatchesRepeat(zp, surface, {4, 4, 1});
}

TEST_CASE("Water intramolecular energy sits on the oxygen", "[water][per-atom]") {
  // One molecule in a large box has no intermolecular terms.
  const Cell one{{0.80, 0.10, 0.50, -0.70, 0.05, 0.55, 0.0, 0.0, 0.0},
                 {1, 1, 8},
                 diag(40.0, 40.0, 40.0)};
  const rgpot::TIP4PPot tip(rgpot::WaterConfig{});
  const rgpot::SPCEPot spc(rgpot::WaterConfig{});
  for (const Result &r : {perAtom(tip, one), perAtom(spc, one)}) {
    REQUIRE(r.energies[0] == 0.0);
    REQUIRE(r.energies[1] == 0.0);
    REQUIRE(r.energies[2] == r.energy);
  }
}

TEST_CASE("Water intermolecular pairs split evenly between molecules",
          "[water][per-atom]") {
  // Two copies of one rigid geometry related by inversion through the
  // midpoint: every site pair is symmetric, so both molecules get equal
  // shares once their own intramolecular energies are removed.
  const double m[9] = {0.80, 0.10, 0.50, -0.70, 0.05, 0.55, 0.0, 0.0, 0.0};
  const double c = 20.0;
  const double d = 1.6;
  Cell pair{{}, {1, 1, 1, 1, 8, 8}, diag(40.0, 40.0, 40.0)};
  Cell lone{{}, {1, 1, 8}, diag(40.0, 40.0, 40.0)};
  std::array<std::array<double, 3>, 6> rows{};
  for (int i = 0; i < 2; ++i) {
    for (int k = 0; k < 3; ++k) {
      rows[i][k] = c + (k == 2 ? -d : 0.0) + m[3 * i + k];
      rows[2 + i][k] = c + (k == 2 ? d : 0.0) - m[3 * i + k];
    }
  }
  for (int k = 0; k < 3; ++k) {
    rows[4][k] = c + (k == 2 ? -d : 0.0) + m[6 + k];
    rows[5][k] = c + (k == 2 ? d : 0.0) - m[6 + k];
  }
  for (const auto &row : rows) {
    pair.pos.insert(pair.pos.end(), row.begin(), row.end());
  }
  lone.pos.assign(m, m + 9);
  const rgpot::TIP4PPot tip(rgpot::WaterConfig{});
  const rgpot::SPCEPot spc(rgpot::WaterConfig{});
  const auto check = [&](const auto &pot) {
    const Result r = perAtom(pot, pair);
    const double intra = perAtom(pot, lone).energy;
    const double first = r.energies[0] + r.energies[1] + r.energies[4] - intra;
    const double second = r.energies[2] + r.energies[3] + r.energies[5] - intra;
    REQUIRE(std::abs(first) > 1e-3);
    REQUIRE_THAT(first, WithinAbs(second, 1e-10));
    REQUIRE_THAT(first + second, WithinAbs(r.energy - 2 * intra, 1e-10));
    REQUIRE_THAT(r.energies[0], WithinAbs(r.energies[2], 1e-10));
    REQUIRE_THAT(r.energies[1], WithinAbs(r.energies[3], 1e-10));
  };
  check(tip);
  check(spc);
}
