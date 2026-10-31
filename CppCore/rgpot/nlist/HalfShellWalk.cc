// MIT License
// Copyright 2023--present rgpot developers

#include "rgpot/nlist/HalfShellDetail.hpp"

#include "rgpot/nlist/MinimumImage.hpp"

#include <cmath>
#include <cstring>

#if defined(__GNUC__) || defined(__clang__)
#define RGPOT_HS_INLINE __attribute__((always_inline)) inline
#define RGPOT_HS_COLD __attribute__((noinline, cold))
#define RGPOT_HS_UNLIKELY(cond) __builtin_expect(static_cast<bool>(cond), 0)
#elif defined(_MSC_VER)
#define RGPOT_HS_INLINE __forceinline
#define RGPOT_HS_COLD __declspec(noinline)
#define RGPOT_HS_UNLIKELY(cond) (static_cast<bool>(cond))
#else
#define RGPOT_HS_INLINE inline
#define RGPOT_HS_COLD
#define RGPOT_HS_UNLIKELY(cond) (static_cast<bool>(cond))
#endif

namespace rgpot::nlist::detail {
// Half-away rounding. The pair walk uses the branchless form. It matches
// std::round for every pair the cell list can keep: that list is built
// only when the box is at least three cutoffs wide, so a neighbour lies
// strictly inside the half-box, where multiplying by the reciprocal
// cannot cross a half-integer boundary.
RGPOT_HS_INLINE double roundHalf(double t) {
  return static_cast<double>(static_cast<long long>(t + std::copysign(0.5, t)));
}

RGPOT_HS_COLD double foldExact(double delta, double len) {
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

void foldOrtho(double &dx, double &dy, double &dz, double lx, double ly,
               double lz) {
  dx = foldExact(dx, lx);
  dy = foldExact(dy, ly);
  dz = foldExact(dz, lz);
}

bool within(const double *pos, int i, int j, double lx, double ly, double lz,
            bool ortho, MinimumImage &image, double cut2) {
  double dx = pos[3 * j] - pos[3 * i];
  double dy = pos[3 * j + 1] - pos[3 * i + 1];
  double dz = pos[3 * j + 2] - pos[3 * i + 2];
  if (ortho) {
    foldOrtho(dx, dy, dz, lx, ly, lz);
  } else {
    image.fold(dx, dy, dz);
  }
  const double r2 = dx * dx + dy * dy + dz * dz;
  return r2 > 0.0 && r2 < cut2;
}

void pushPair(Scratch &s, int a, int b, double dx, double dy, double dz) {
  if (a > b) {
    dx = -dx;
    dy = -dy;
    dz = -dz;
  }
  s.lo.push_back(a < b ? a : b);
  s.hi.push_back(a < b ? b : a);
  s.dx.push_back(dx);
  s.dy.push_back(dy);
  s.dz.push_back(dz);
}

void halfShell(const double *pos, std::size_t n, const Grid &g,
               const std::vector<int> &start, const std::vector<int> &atoms,
               double cut2, Scratch &s) {
  (void)n;
  const double lx = g.lx;
  const double ly = g.ly;
  const double lz = g.lz;
  const double invx = 1.0 / lx;
  const double invy = 1.0 / ly;
  const double invz = 1.0 / lz;
  const double hx = 0.5 * lx;
  const double hy = 0.5 * ly;
  const double hz = 0.5 * lz;
  const int *st = start.data();
  const int *at = atoms.data();
  for (int ix = 0; ix < g.nx; ++ix) {
    for (int iy = 0; iy < g.ny; ++iy) {
      for (int iz = 0; iz < g.nz; ++iz) {
        const int c = (ix * g.ny + iy) * g.nz + iz;
        const int i0 = st[c];
        const int i1 = st[c + 1];
        if (i0 == i1) {
          continue;
        }
        for (const auto &off : kHalf) {
          int jx = ix + off[0];
          int jy = iy + off[1];
          int jz = iz + off[2];
          if (jx < 0) {
            jx += g.nx;
          } else if (jx >= g.nx) {
            jx -= g.nx;
          }
          if (jy < 0) {
            jy += g.ny;
          } else if (jy >= g.ny) {
            jy -= g.ny;
          }
          if (jz < 0) {
            jz += g.nz;
          } else if (jz >= g.nz) {
            jz -= g.nz;
          }
          const int cj = (jx * g.ny + jy) * g.nz + jz;
          const int j0 = st[cj];
          const int j1 = st[cj + 1];
          const int ja0 = (off[0] | off[1] | off[2]) == 0 ? i0 : j0;
          for (int ia = i0; ia < i1; ++ia) {
            const int i = at[ia];
            const double ix0 = pos[3 * i];
            const double iy0 = pos[3 * i + 1];
            const double iz0 = pos[3 * i + 2];
            const int jb = (ja0 == i0) ? ia + 1 : ja0;
            for (int ja = jb; ja < j1; ++ja) {
              const int j = at[ja];
              double dx = pos[3 * j] - ix0;
              double dy = pos[3 * j + 1] - iy0;
              double dz = pos[3 * j + 2] - iz0;
              double fx = dx - lx * roundHalf(dx * invx);
              double fy = dy - ly * roundHalf(dy * invy);
              double fz = dz - lz * roundHalf(dz * invz);
              if (RGPOT_HS_UNLIKELY(std::fabs(fx) > hx || std::fabs(fy) > hy ||
                                    std::fabs(fz) > hz)) {
                fx = foldExact(dx, lx);
                fy = foldExact(dy, ly);
                fz = foldExact(dz, lz);
              }
              const double r2 = fx * fx + fy * fy + fz * fz;
              if (r2 > 0.0 && r2 < cut2) {
                pushPair(s, i, j, fx, fy, fz);
              }
            }
          }
        }
      }
    }
  }
}

void fillPairs(const double *positions, const double *box, std::size_t n,
               double cutoff, double search, Scratch &s) {
  const double search2 = search * search;
  s.lo.clear();
  s.hi.clear();
  s.dx.clear();
  s.dy.clear();
  s.dz.clear();
  if (s.lo.capacity() < n * 8) {
    s.lo.reserve(n * 8);
    s.hi.reserve(n * 8);
    s.dx.reserve(n * 8);
    s.dy.reserve(n * 8);
    s.dz.reserve(n * 8);
  }
  double lx = 0;
  double ly = 0;
  double lz = 0;
  const bool ortho = orthogonal(box, lx, ly, lz);
  MinimumImage image;
  if (!ortho) {
    image = MinimumImage(box, {true, true, true});
  }
  Grid order{};
  s.ordered = makeGrid(box, cutoff, order);
  if (s.ordered) {
    s.order = order;
    binAtoms(positions, n, order, s.cell_of, s.rank, s.counts, s.start, s.atoms,
             true);
  }
  Grid searched{};
  const bool cells = makeGrid(box, search, searched);
  const bool same = cells && s.ordered && searched.nx == order.nx &&
                    searched.ny == order.ny && searched.nz == order.nz;
  const int ni = static_cast<int>(n);
  if (!ortho || !cells) {
    for (int i = 0; i < ni; ++i) {
      for (int j = i + 1; j < ni; ++j) {
        if (within(positions, i, j, lx, ly, lz, ortho, image, search2)) {
          double dx = positions[3 * j] - positions[3 * i];
          double dy = positions[3 * j + 1] - positions[3 * i + 1];
          double dz = positions[3 * j + 2] - positions[3 * i + 2];
          if (ortho) {
            foldOrtho(dx, dy, dz, lx, ly, lz);
          } else {
            image.fold(dx, dy, dz);
          }
          pushPair(s, i, j, dx, dy, dz);
        }
      }
    }
  } else if (same) {
    halfShell(positions, n, order, s.start, s.atoms, search2, s);
  } else {
    binAtoms(positions, n, searched, s.scell, s.rank, s.scounts, s.sstart,
             s.satoms, false);
    halfShell(positions, n, searched, s.sstart, s.satoms, search2, s);
  }
}

} // namespace rgpot::nlist::detail
