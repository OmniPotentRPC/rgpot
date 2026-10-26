// MIT License
// Copyright 2023--present rgpot developers

#include "rgpot/nlist/HalfShellDetail.hpp"

#include "rgpot/nlist/MinimumImage.hpp"

#include <cmath>
#include <cstring>

namespace rgpot::nlist::detail {
// Half-away rounding. The pair walk uses the branchless form. It matches
// std::round for every pair the cell list can keep: that list is built
// only when the box is at least three cutoffs wide, so a neighbour lies
// strictly inside the half-box, where multiplying by the reciprocal
// cannot cross a half-integer boundary.
__attribute__((always_inline)) inline double roundHalf(double t) {
  return static_cast<double>(static_cast<long long>(t + std::copysign(0.5, t)));
}

__attribute__((noinline, cold)) double foldExact(double delta, double len) {
  return delta - len * std::round(delta / len);
}

constexpr int kHalf[14][3] = {
    {0, 0, 0},   {0, 0, 1},  {0, 1, -1}, {0, 1, 0},  {0, 1, 1},
    {1, -1, -1}, {1, -1, 0}, {1, -1, 1}, {1, 0, -1}, {1, 0, 0},
    {1, 0, 1},   {1, 1, -1}, {1, 1, 0},  {1, 1, 1},
};

bool orthogonal(const double *box, double &lx, double &ly, double &lz) {
  if (box[1] != 0.0 || box[2] != 0.0 || box[3] != 0.0 || box[5] != 0.0 ||
      box[6] != 0.0 || box[7] != 0.0) {
    return false;
  }
  lx = box[0];
  ly = box[4];
  lz = box[8];
  return lx > 0.0 && ly > 0.0 && lz > 0.0 && std::isfinite(lx) &&
         std::isfinite(ly) && std::isfinite(lz);
}

double folded(double x, double len) {
  if (x >= 0.0 && x < len) {
    return x;
  }
  double f = std::fmod(x, len);
  if (f < 0.0) {
    f += len;
  }
  if (f >= len) {
    f = 0.0;
  }
  return f;
}

int bin(double x, double len, int n) {
  int i = static_cast<int>(folded(x, len) / len * static_cast<double>(n));
  if (i < 0) {
    i = 0;
  }
  if (i >= n) {
    i = n - 1;
  }
  return i;
}

bool makeGrid(const double *box, double edge, Grid &g) {
  if (!orthogonal(box, g.lx, g.ly, g.lz) || !(edge > 0.0)) {
    return false;
  }
  g.nx = static_cast<int>(std::floor(g.lx / edge));
  g.ny = static_cast<int>(std::floor(g.ly / edge));
  g.nz = static_cast<int>(std::floor(g.lz / edge));
  if (g.nx < 3 || g.ny < 3 || g.nz < 3) {
    return false;
  }
  g.ncell = g.nx * g.ny * g.nz;
  return true;
}

void binAtoms(const double *pos, std::size_t n, const Grid &g,
              std::vector<int> &cell_of, std::vector<int> &rank,
              std::vector<int> &counts, std::vector<int> &start,
              std::vector<int> &atoms, bool ranks) {
  counts.assign(static_cast<std::size_t>(g.ncell), 0);
  cell_of.resize(n);
  for (std::size_t i = 0; i < n; ++i) {
    const int ix = bin(pos[3 * i], g.lx, g.nx);
    const int iy = bin(pos[3 * i + 1], g.ly, g.ny);
    const int iz = bin(pos[3 * i + 2], g.lz, g.nz);
    const int c = (ix * g.ny + iy) * g.nz + iz;
    cell_of[i] = c;
    ++counts[static_cast<std::size_t>(c)];
  }
  start.resize(static_cast<std::size_t>(g.ncell) + 1);
  start[0] = 0;
  for (int c = 0; c < g.ncell; ++c) {
    start[static_cast<std::size_t>(c) + 1] =
        start[static_cast<std::size_t>(c)] +
        counts[static_cast<std::size_t>(c)];
  }
  for (int c = 0; c < g.ncell; ++c) {
    counts[static_cast<std::size_t>(c)] = start[static_cast<std::size_t>(c)];
  }
  atoms.resize(n);
  for (std::size_t i = 0; i < n; ++i) {
    const int c = cell_of[i];
    atoms[static_cast<std::size_t>(counts[static_cast<std::size_t>(c)]++)] =
        static_cast<int>(i);
  }
  if (!ranks) {
    return;
  }
  // The linked walk inserted atoms in increasing index and iterated each
  // cell from the head, so the first atom visited is the highest index.
  rank.resize(n);
  for (int c = 0; c < g.ncell; ++c) {
    const int a = start[static_cast<std::size_t>(c)];
    const int b = start[static_cast<std::size_t>(c) + 1];
    const int count = b - a;
    for (int p = a; p < b; ++p) {
      rank[static_cast<std::size_t>(atoms[static_cast<std::size_t>(p)])] =
          count - 1 - (p - a);
    }
  }
}

} // namespace rgpot::nlist::detail
