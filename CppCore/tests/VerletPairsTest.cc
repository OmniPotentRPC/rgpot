// MIT License
// Copyright 2023--present rgpot developers
//
// VerletPairCache: a skin of 0.1 * cutoff returns the same pair list, bit
// for bit, as a rebuild on every call, over an MD-like walk and over the
// images of a band, while building far fewer candidate lists.

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <set>
#include <tuple>
#include <vector>

#include <catch2/catch_all.hpp>

#include "rgpot/nlist/VerletPairs.hpp"

namespace {

using rgpot::nlist::VerletPairCache;
using rgpot::nlist::VerletPairList;

struct System {
  std::vector<double> pos;
  std::array<double, 9> box;
};

// Jittered simple cubic lattice in a slightly triclinic cell, 4^3 = 64 atoms.
System lattice(std::mt19937_64 &rng) {
  System s;
  const int m = 4;
  const double a = 2.6;
  s.box = {m * a, 0.0, 0.0, 0.3, m * a, 0.0, 0.2, 0.4, m * a};
  std::uniform_real_distribution<double> u(-0.15, 0.15);
  for (int x = 0; x < m; ++x)
    for (int y = 0; y < m; ++y)
      for (int z = 0; z < m; ++z) {
        const double f[3] = {(x + 0.5) / m, (y + 0.5) / m, (z + 0.5) / m};
        for (int d = 0; d < 3; ++d)
          s.pos.push_back(f[0] * s.box[d] + f[1] * s.box[3 + d] +
                          f[2] * s.box[6 + d] + u(rng));
      }
  return s;
}

// A smooth pair model summed in list order: energy and forces, so the
// check covers what a potential does with the list.
void pairModel(const VerletPairList &l, size_t n, bool full, double cutoff,
               double &e, std::vector<double> &f) {
  e = 0.0;
  f.assign(3 * n, 0.0);
  const double w = full ? 0.5 : 1.0;
  for (size_t k = 0; k < l.size(); ++k) {
    const auto &v = l.vectors[k];
    const double r2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
    const double r = std::sqrt(r2);
    const double t = 1.0 - r / cutoff;
    const double inv6 = 1.0 / (r2 * r2 * r2);
    e += w * (inv6 * inv6 - inv6) * t * t;
    const double dedr =
        w * ((-12.0 * inv6 * inv6 + 6.0 * inv6) / r * t * t -
             (inv6 * inv6 - inv6) * 2.0 * t / cutoff);
    for (int d = 0; d < 3; ++d) {
      const double g = dedr * v[d] / r;
      f[3 * l.pairs[k][0] + d] += g;
      f[3 * l.pairs[k][1] + d] -= g;
    }
  }
}

bool sameList(const VerletPairList &a, const VerletPairList &b) {
  if (a.size() != b.size())
    return false;
  return a.pairs == b.pairs && a.shifts == b.shifts &&
         std::memcmp(a.vectors.data(), b.vectors.data(),
                     a.size() * sizeof(a.vectors[0])) == 0;
}

// Pair set from a direct vesin call at the cutoff, as (i, j, shift)
// oriented the way the cache orients half lists.
std::set<std::tuple<int64_t, int64_t, int32_t, int32_t, int32_t>>
vesinSet(const System &s, double cutoff, bool full) {
  VesinOptions o{};
  rgpot::vesin_compat::fill_neighbor_options(o, cutoff, full);
  VesinNeighborList nl{};
  const char *err = nullptr;
  bool pbc[3] = {true, true, true};
  REQUIRE(vesin_neighbors(reinterpret_cast<const double(*)[3]>(s.pos.data()),
                          s.pos.size() / 3,
                          reinterpret_cast<const double(*)[3]>(s.box.data()),
                          pbc, rgpot::vesin_compat::make_cpu_device(), o, &nl,
                          &err) == EXIT_SUCCESS);
  std::set<std::tuple<int64_t, int64_t, int32_t, int32_t, int32_t>> out;
  for (size_t k = 0; k < nl.length; ++k) {
    int64_t i = nl.pairs[k][0], j = nl.pairs[k][1];
    int32_t sh[3] = {nl.shifts[k][0], nl.shifts[k][1], nl.shifts[k][2]};
    bool pos = false;
    for (int d = 0; d < 3; ++d)
      if (sh[d] != 0) {
        pos = sh[d] > 0;
        break;
      }
    if (!full && (i > j || (i == j && !pos))) {
      std::swap(i, j);
      for (auto &x : sh)
        x = -x;
    }
    out.emplace(i, j, sh[0], sh[1], sh[2]);
  }
  vesin_free(&nl);
  return out;
}

} // namespace

TEST_CASE("Verlet skin matches a rebuild on every call over an MD walk",
          "[verlet]") {
  const double cutoff = 5.0;
  for (bool full : {false, true}) {
    std::mt19937_64 rng(20261014);
    System s = lattice(rng);
    const size_t n = s.pos.size() / 3;
    const bool pbc[3] = {true, true, true};
    VerletPairCache cached(0.1 * cutoff);
    VerletPairCache fresh(0.0);
    // Stationary step about 0.008 Angstrom, a typical 1 fs MD displacement.
    std::normal_distribution<double> g(0.0, 0.002);
    std::vector<double> vel(s.pos.size());
    for (auto &v : vel)
      v = g(rng);
    const int steps = 200;
    for (int step = 0; step < steps; ++step) {
      for (size_t k = 0; k < s.pos.size(); ++k) {
        vel[k] = 0.9 * vel[k] + g(rng);
        s.pos[k] += vel[k];
      }
      const auto a = cached.compute(s.pos.data(), n, s.box.data(), pbc,
                                    cutoff, full);
      const auto b =
          fresh.compute(s.pos.data(), n, s.box.data(), pbc, cutoff, full);
      REQUIRE(sameList(a, b));
      double ea = 0, eb = 0;
      std::vector<double> fa, fb;
      pairModel(a, n, full, cutoff, ea, fa);
      pairModel(b, n, full, cutoff, eb, fb);
      REQUIRE(std::memcmp(&ea, &eb, sizeof ea) == 0);
      REQUIRE(std::memcmp(fa.data(), fb.data(), fa.size() * 8) == 0);
      if (step % 50 == 0) {
        std::set<std::tuple<int64_t, int64_t, int32_t, int32_t, int32_t>> got;
        for (size_t k = 0; k < a.size(); ++k)
          got.emplace(a.pairs[k][0], a.pairs[k][1], a.shifts[k][0],
                      a.shifts[k][1], a.shifts[k][2]);
        REQUIRE(got == vesinSet(s, cutoff, full));
      }
    }
    REQUIRE(fresh.rebuilds() == static_cast<uint64_t>(steps));
    WARN("MD walk, full=" << full << ": " << cached.rebuilds()
                          << " rebuilds in " << steps << " calls");
    REQUIRE(cached.rebuilds() * 4 < static_cast<uint64_t>(steps));
  }
}

TEST_CASE("Per-image Verlet caches over band iterations", "[verlet]") {
  const double cutoff = 5.0;
  std::mt19937_64 rng(7);
  System a = lattice(rng);
  System b = a;
  std::normal_distribution<double> g(0.0, 0.3);
  for (auto &x : b.pos)
    x += g(rng);
  const size_t n = a.pos.size() / 3;
  const int images = 7;
  const int iters = 60;
  const bool pbc[3] = {true, true, true};
  std::vector<VerletPairCache> cached(images, VerletPairCache(0.1 * cutoff));
  std::vector<VerletPairCache> fresh(images, VerletPairCache(0.0));
  std::normal_distribution<double> relax(0.0, 0.004);
  std::vector<std::vector<double>> path(images);
  for (int im = 0; im < images; ++im) {
    const double t = (im + 1.0) / (images + 1.0);
    path[im].resize(a.pos.size());
    for (size_t k = 0; k < a.pos.size(); ++k)
      path[im][k] = (1 - t) * a.pos[k] + t * b.pos[k];
  }
  for (int it = 0; it < iters; ++it)
    for (int im = 0; im < images; ++im) {
      for (auto &x : path[im])
        x += relax(rng);
      const auto x = cached[im].compute(path[im].data(), n, a.box.data(), pbc,
                                        cutoff, true);
      const auto y = fresh[im].compute(path[im].data(), n, a.box.data(), pbc,
                                       cutoff, true);
      REQUIRE(sameList(x, y));
    }
  uint64_t total = 0;
  for (const auto &c : cached)
    total += c.rebuilds();
  WARN("band: " << total << " rebuilds in " << images * iters << " calls");
  REQUIRE(total * 4 < static_cast<uint64_t>(images * iters));
}

TEST_CASE("Cell change forces a rebuild", "[verlet]") {
  std::mt19937_64 rng(3);
  System s = lattice(rng);
  const bool pbc[3] = {true, true, true};
  VerletPairCache c(0.5);
  (void)c.compute(s.pos.data(), s.pos.size() / 3, s.box.data(), pbc, 5.0,
                  false);
  s.box[0] += 1e-6;
  (void)c.compute(s.pos.data(), s.pos.size() / 3, s.box.data(), pbc, 5.0,
                  false);
  REQUIRE(c.rebuilds() == 2);
}
