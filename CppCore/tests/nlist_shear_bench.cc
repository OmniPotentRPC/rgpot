// MIT License
// Copyright 2023--present rgpot developers
//
// Neighbour build time on an orthorhombic Cu cell against the same atoms on
// a sheared basis of that lattice. The sheared rows are (L, 0, 0),
// (L, L, 0), (0, 0, L): the density and the pairs are those of the cube,
// only the cell form changes. scripts/bench_nlist_shear.sh runs this
// binary from two builds, interleaved, at one and eight threads.
//
//   nlist_shear_bench [--n 4000|11000] [--cell ortho|sheared]
//                     [--kind stored|stream|public] [--calls N]
//
// Without --n/--cell/--kind every row is run once. Each row prints the raw
// per-call seconds, their median, and a pair count that must not depend on
// the cell form or the build.
//
//   stored  Neighbours::update in Mode::Stored, a fresh object per call.
//   stream  Neighbours::update in Mode::Stream and one forDirected walk of
//           every atom: the stream defers its search to the walk.
//   public  rgpot::nlist::build, a strict half list.
//
// Cutoff 5.0 A and skin 0.5 A (Neighbours), cutoff 5.0 A (public list),
// a = 3.615 A. 4000 atoms are 10^3 cubic cells, 11000 are 14^3 (10976).

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "rgpot/nlist/NeighbourList.hpp"
#include "rgpot/nlist/Neighbours.hpp"

namespace {

using clock_type = std::chrono::steady_clock;

constexpr double kLattice = 3.615;
constexpr double kCutoff = 5.0;
constexpr double kSkin = 0.5;

struct Geometry {
  std::vector<double> pos;
  double box[9]{};
};

Geometry copper(int ncell, bool sheared) {
  Geometry g;
  const double basis[4][3] = {{0, 0, 0}, {0, 0.5, 0.5}, {0.5, 0, 0.5}, {0.5, 0.5, 0}};
  for (int ix = 0; ix < ncell; ++ix) {
    for (int iy = 0; iy < ncell; ++iy) {
      for (int iz = 0; iz < ncell; ++iz) {
        for (const auto &b : basis) {
          g.pos.push_back((ix + b[0]) * kLattice);
          g.pos.push_back((iy + b[1]) * kLattice);
          g.pos.push_back((iz + b[2]) * kLattice);
        }
      }
    }
  }
  // A fixed rattle so no pair sits exactly on the cutoff.
  for (std::size_t k = 0; k < g.pos.size(); ++k) {
    g.pos[k] += 0.03 * std::sin(1.7 * static_cast<double>(k) + 0.3);
  }
  const double l = ncell * kLattice;
  const double ortho[9] = {l, 0, 0, 0, l, 0, 0, 0, l};
  const double shear[9] = {l, 0, 0, l, l, 0, 0, 0, l};
  std::memcpy(g.box, sheared ? shear : ortho, sizeof(g.box));
  return g;
}

double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  const std::size_t n = v.size();
  return n % 2 == 1 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

std::uint64_t once(const Geometry &g, const std::string &kind) {
  const std::size_t n = g.pos.size() / 3;
  const std::array<bool, 3> periodic{{true, true, true}};
  if (kind == "public") {
    rgpot::nlist::Query query;
    query.cutoff = kCutoff;
    query.shape = rgpot::nlist::Pairs{rgpot::nlist::Shell::Half};
    query.cut = rgpot::nlist::Cut::Strict;
    rgpot::nlist::Table table;
    rgpot::nlist::build(table, g.pos.data(), n, g.box, query);
    return table.size();
  }
  rgpot::nlist::Neighbours cells;
  const auto mode = kind == "stored" ? rgpot::nlist::Mode::Stored : rgpot::nlist::Mode::Stream;
  rgpot::nlist::updateCells(cells, g.pos.data(), n, g.box, kCutoff, kSkin,
                            rgpot::nlist::Image::Round, mode, periodic);
  if (kind == "stored") {
    return cells.slots();
  }
  std::uint64_t count = 0;
  cells.forDirected(0, static_cast<std::int64_t>(n),
                    [&](std::int32_t, std::int32_t, double, double, double, double r2) {
                      count += r2 <= (kCutoff + kSkin) * (kCutoff + kSkin) ? 1 : 0;
                    });
  return count;
}

int row(int natoms, const std::string &cell, const std::string &kind, int calls) {
  const int ncell = natoms >= 11000 ? 14 : 10;
  const Geometry g = copper(ncell, cell == "sheared");
  const char *threads = std::getenv("OMP_NUM_THREADS");
  std::uint64_t pairs = once(g, kind);
  std::vector<double> times;
  for (int c = 0; c < calls; ++c) {
    const auto t0 = clock_type::now();
    const std::uint64_t got = once(g, kind);
    times.push_back(std::chrono::duration<double>(clock_type::now() - t0).count());
    if (got != pairs) {
      std::fprintf(stderr, "pair count changed between calls\n");
      return 1;
    }
  }
  std::printf("row natoms=%zu cell=%s kind=%s threads=%s pairs=%llu median_s=%.6e raw_s=",
              g.pos.size() / 3, cell.c_str(), kind.c_str(), threads ? threads : "unset",
              static_cast<unsigned long long>(pairs), median(times));
  for (std::size_t k = 0; k < times.size(); ++k) {
    std::printf("%s%.6e", k == 0 ? "" : ",", times[k]);
  }
  std::printf("\n");
  return 0;
}

} // namespace

int main(int argc, char **argv) {
  int natoms = 0;
  std::string cell;
  std::string kind;
  int calls = 5;
  for (int a = 1; a + 1 < argc; a += 2) {
    const std::string key = argv[a];
    const char *value = argv[a + 1];
    if (key == "--n") {
      natoms = std::atoi(value);
    } else if (key == "--cell") {
      cell = value;
    } else if (key == "--kind") {
      kind = value;
    } else if (key == "--calls") {
      calls = std::max(1, std::atoi(value));
    } else {
      std::fprintf(stderr, "unknown argument %s\n", key.c_str());
      return 2;
    }
  }
  const std::vector<int> sizes = natoms > 0 ? std::vector<int>{natoms} : std::vector<int>{4000, 11000};
  const std::vector<std::string> cells =
      cell.empty() ? std::vector<std::string>{"ortho", "sheared"} : std::vector<std::string>{cell};
  const std::vector<std::string> kinds = kind.empty()
                                             ? std::vector<std::string>{"stored", "stream", "public"}
                                             : std::vector<std::string>{kind};
  for (int n : sizes) {
    for (const auto &c : cells) {
      for (const auto &k : kinds) {
        if (row(n, c, k, calls) != 0) {
          return 1;
        }
      }
    }
  }
  return 0;
}
