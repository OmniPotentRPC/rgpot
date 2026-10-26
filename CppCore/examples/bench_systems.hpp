#pragma once
// MIT License
// Copyright 2023--present rgpot developers
//
// Physical test systems shared by the timing examples: fcc nanoparticles of
// an exact atom count in vacuum and periodic fcc bulk.

#include <algorithm>
#include <array>
#include <cstddef>
#include <random>
#include <utility>
#include <vector>

namespace rgpot_bench {

struct System {
  std::vector<double> pos;
  std::vector<int> types;
  std::array<double, 9> box{};
};

/// fcc sites of an m x m x m block of cubic cells with lattice constant a.
inline std::vector<double> fccBlock(int m, double a) {
  static const double basis[4][3] = {
      {0.0, 0.0, 0.0}, {0.0, 0.5, 0.5}, {0.5, 0.0, 0.5}, {0.5, 0.5, 0.0}};
  std::vector<double> R;
  R.reserve(static_cast<std::size_t>(12 * m * m * m));
  for (int ix = 0; ix < m; ++ix)
    for (int iy = 0; iy < m; ++iy)
      for (int iz = 0; iz < m; ++iz)
        for (const auto &b : basis) {
          R.push_back((ix + b[0]) * a);
          R.push_back((iy + b[1]) * a);
          R.push_back((iz + b[2]) * a);
        }
  return R;
}

/// The n fcc sites nearest the centre of a large block: a near-spherical
/// nanoparticle of exactly n atoms, centred in a vacuum box whose side
/// leaves `vacuum` of empty space on every face.
inline System nanoparticle(std::size_t n, double a, double vacuum, int z) {
  int m = 2;
  while (4.0 * m * m * m < 2.0 * static_cast<double>(n) + 64.0)
    ++m;
  auto R = fccBlock(m, a);
  const double c = 0.5 * (m - 1) * a + 0.25 * a;
  std::vector<std::pair<double, std::size_t>> d;
  for (std::size_t i = 0; i < R.size() / 3; ++i) {
    const double dx = R[3 * i] - c, dy = R[3 * i + 1] - c,
                 dz = R[3 * i + 2] - c;
    d.emplace_back(dx * dx + dy * dy + dz * dz, i);
  }
  std::sort(d.begin(), d.end());
  System s;
  double lo = 1e300, hi = -1e300;
  for (std::size_t k = 0; k < n; ++k) {
    const std::size_t i = d[k].second;
    for (int q = 0; q < 3; ++q) {
      s.pos.push_back(R[3 * i + q]);
      lo = std::min(lo, R[3 * i + q]);
      hi = std::max(hi, R[3 * i + q]);
    }
  }
  const double side = (hi - lo) + 2.0 * vacuum;
  for (double &x : s.pos)
    x += vacuum - lo;
  s.types.assign(n, z);
  s.box = {side, 0.0, 0.0, 0.0, side, 0.0, 0.0, 0.0, side};
  return s;
}

inline System bulk(int m, double a, int z) {
  System s;
  s.pos = fccBlock(m, a);
  // Thermal-scale noise so no two pair distances coincide exactly.
  std::mt19937_64 rng(7);
  std::uniform_real_distribution<double> u(-0.03, 0.03);
  for (double &x : s.pos)
    x += u(rng);
  s.types.assign(s.pos.size() / 3, z);
  const double side = m * a;
  s.box = {side, 0.0, 0.0, 0.0, side, 0.0, 0.0, 0.0, side};
  return s;
}

inline int cellsFor(std::size_t n) {
  int m = 1;
  while (4.0 * (m + 1) * (m + 1) * (m + 1) <= 1.15 * static_cast<double>(n))
    ++m;
  return std::max(m, 1);
}

} // namespace rgpot_bench
