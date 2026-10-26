// MIT License
// Copyright 2023--present rgpot developers
//
// The half-shell list must emit pairs in the same order as the 27-cell
// linked walk it replaced, and a Verlet hit must keep that order.

#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>

#include <catch2/catch_all.hpp>

#include "rgpot/nlist/HalfShell.hpp"

using rgpot::nlist::NeighborList;
using rgpot::nlist::orderedPairs;

namespace {

double roundAway(double t) { return std::round(t); }

void fold(double &dx, double &dy, double &dz, double lx, double ly, double lz) {
  dx -= lx * roundAway(dx / lx);
  dy -= ly * roundAway(dy / ly);
  dz -= lz * roundAway(dz / lz);
}

double wrapped(double x, double len) {
  double f = std::fmod(x, len);
  if (f < 0.0) {
    f += len;
  }
  if (f >= len) {
    f = 0.0;
  }
  return f;
}

// Pairs the old 27-cell walk would have passed to the force loop.
std::vector<std::pair<int, int>> legacy(const std::vector<double> &pos,
                                        double lx, double ly, double lz,
                                        double cut) {
  const int n = static_cast<int>(pos.size() / 3);
  const int nx = static_cast<int>(std::floor(lx / cut));
  const int ny = static_cast<int>(std::floor(ly / cut));
  const int nz = static_cast<int>(std::floor(lz / cut));
  std::vector<std::pair<int, int>> out;
  const double cut2 = cut * cut;
  auto keep = [&](int i, int j) {
    double dx = pos[3 * j] - pos[3 * i];
    double dy = pos[3 * j + 1] - pos[3 * i + 1];
    double dz = pos[3 * j + 2] - pos[3 * i + 2];
    fold(dx, dy, dz, lx, ly, lz);
    const double r2 = dx * dx + dy * dy + dz * dz;
    if (r2 > 0.0 && r2 < cut2) {
      out.emplace_back(i, j);
    }
  };
  if (nx < 3 || ny < 3 || nz < 3) {
    for (int i = 0; i < n; ++i) {
      for (int j = i + 1; j < n; ++j) {
        keep(i, j);
      }
    }
    return out;
  }
  std::vector<int> head(static_cast<std::size_t>(nx * ny * nz), -1);
  std::vector<int> next(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i) {
    int ix = static_cast<int>(wrapped(pos[3 * i], lx) / lx * nx);
    int iy = static_cast<int>(wrapped(pos[3 * i + 1], ly) / ly * ny);
    int iz = static_cast<int>(wrapped(pos[3 * i + 2], lz) / lz * nz);
    if (ix < 0) {
      ix = 0;
    }
    if (iy < 0) {
      iy = 0;
    }
    if (iz < 0) {
      iz = 0;
    }
    if (ix >= nx) {
      ix = nx - 1;
    }
    if (iy >= ny) {
      iy = ny - 1;
    }
    if (iz >= nz) {
      iz = nz - 1;
    }
    const int c = (ix * ny + iy) * nz + iz;
    next[static_cast<std::size_t>(i)] = head[static_cast<std::size_t>(c)];
    head[static_cast<std::size_t>(c)] = i;
  }
  for (int ix = 0; ix < nx; ++ix) {
    for (int iy = 0; iy < ny; ++iy) {
      for (int iz = 0; iz < nz; ++iz) {
        const int c = (ix * ny + iy) * nz + iz;
        for (int ox = -1; ox <= 1; ++ox) {
          for (int oy = -1; oy <= 1; ++oy) {
            for (int oz = -1; oz <= 1; ++oz) {
              int jx = ix + ox;
              int jy = iy + oy;
              int jz = iz + oz;
              if (jx < 0) {
                jx += nx;
              } else if (jx >= nx) {
                jx -= nx;
              }
              if (jy < 0) {
                jy += ny;
              } else if (jy >= ny) {
                jy -= ny;
              }
              if (jz < 0) {
                jz += nz;
              } else if (jz >= nz) {
                jz -= nz;
              }
              const int cj = (jx * ny + jy) * nz + jz;
              for (int i = head[static_cast<std::size_t>(c)]; i >= 0;
                   i = next[static_cast<std::size_t>(i)]) {
                for (int j = head[static_cast<std::size_t>(cj)]; j >= 0;
                     j = next[static_cast<std::size_t>(j)]) {
                  if (j > i) {
                    keep(i, j);
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  return out;
}

void expectSame(const std::vector<double> &pos, double len, double cut) {
  const double box[9] = {len, 0, 0, 0, len, 0, 0, 0, len};
  const auto want = legacy(pos, len, len, len, cut);
  const auto &got = orderedPairs(pos.data(), box, pos.size() / 3, cut, nullptr);
  REQUIRE(got.lo.size() == want.size());
  for (std::size_t p = 0; p < want.size(); ++p) {
    REQUIRE(got.lo[p] == want[p].first);
    REQUIRE(got.hi[p] == want[p].second);
  }
  NeighborList skin;
  skin.skin = 0.5;
  const auto &wide = orderedPairs(pos.data(), box, pos.size() / 3, cut, &skin);
  std::size_t k = 0;
  const double cut2 = cut * cut;
  for (std::size_t p = 0; p < wide.lo.size(); ++p) {
    double dx = pos[3 * wide.hi[p]] - pos[3 * wide.lo[p]];
    double dy = pos[3 * wide.hi[p] + 1] - pos[3 * wide.lo[p] + 1];
    double dz = pos[3 * wide.hi[p] + 2] - pos[3 * wide.lo[p] + 2];
    fold(dx, dy, dz, len, len, len);
    if (dx * dx + dy * dy + dz * dz < cut2) {
      REQUIRE(k < want.size());
      REQUIRE(wide.lo[p] == want[k].first);
      REQUIRE(wide.hi[p] == want[k].second);
      ++k;
    }
  }
  REQUIRE(k == want.size());
  const int *data = skin.lo.data();
  orderedPairs(pos.data(), box, pos.size() / 3, cut, &skin);
  REQUIRE(skin.lo.data() == data);
}

} // namespace

TEST_CASE("half-shell order matches the 27-cell walk", "[HalfShell]") {
  const double cut = 1.8 * 2.0951;
  const double u = 1.1224620482975474;
  const double a0 = u * 2.0951 * 4.0 / std::sqrt(3.0);
  std::vector<double> pos;
  const double basis[8][3] = {{0, 0, 0},          {0.5, 0.5, 0},
                              {0.5, 0, 0.5},      {0, 0.5, 0.5},
                              {0.25, 0.25, 0.25}, {0.75, 0.75, 0.25},
                              {0.75, 0.25, 0.75}, {0.25, 0.75, 0.75}};
  for (int ix = 0; ix < 3; ++ix) {
    for (int iy = 0; iy < 3; ++iy) {
      for (int iz = 0; iz < 3; ++iz) {
        for (const auto &b : basis) {
          pos.push_back((ix + b[0]) * a0);
          pos.push_back((iy + b[1]) * a0);
          pos.push_back((iz + b[2]) * a0);
        }
      }
    }
  }
  expectSame(pos, 3 * a0, cut);
  REQUIRE(
      orderedPairs(
          pos.data(),
          std::vector<double>{3 * a0, 0, 0, 0, 3 * a0, 0, 0, 0, 3 * a0}.data(),
          pos.size() / 3, cut, nullptr)
          .lo.size() == 432);

  std::vector<double> cloud(48 * 3);
  std::uint32_t s = 1;
  const double len = 18.0;
  for (double &x : cloud) {
    s = s * 1664525u + 1013904223u;
    x = (s >> 8) * (1.0 / 16777216.0) * len;
  }
  expectSame(cloud, len, 3.5);
}

TEST_CASE("half-shell all-pairs order below three cells", "[HalfShell]") {
  std::vector<double> pos = {0.2, 0.2, 0.2, 1.4, 0.3, 0.2,
                             0.4, 1.5, 0.3, 2.2, 2.1, 0.4};
  expectSame(pos, 6.0, 3.0);
}
