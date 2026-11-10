#pragma once
// MIT License
// Copyright 2023--present rgpot developers
//
// Linked-cell pair search with the contract of vesin_visit.hpp's
// brute_force_visit: every unordered pair (a, b), a < b, whose nearest
// image lies within the list cutoff is reported exactly once, with the
// folded vector d = r_b - r_a. Only the order of the pairs differs from the
// brute-force scan.
//
// The box is cut into cells at least one list cutoff wide, so every partner
// of an atom lies in its own cell or one of the 26 around it. A half
// stencil (the cell itself plus 13 forward neighbours) visits each pair of
// adjacent cells once. That needs three or more cells along each periodic
// axis: with fewer, a forward and a backward neighbour are the same cell
// and pairs would repeat. Three cells per axis also put the list cutoff
// below a third of the width, so at most one image of a partner is close
// enough to count, and the minimum image fold picks it.
//
// Free (non-periodic) axes are binned over the coordinates' bounding box
// and not wrapped.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "rgpot/nlist/vesin_visit.hpp"

namespace rgpot {
namespace nlist {

/// Fewest cells worth binning for. Even with the atoms spread evenly the
/// cell scan tests about 14 n^2 / ncells pairs against n^2 / 2, so below
/// some 30 cells it cannot win; CellGrid::build then also compares the
/// pair tests the grid would make for the actual occupancy.
inline constexpr std::size_t kMinCellsForGrid = 64;

/// The half stencil: the 13 neighbours that come after a cell in (z, y, x)
/// lexicographic order, as (dx, dy, dz) offsets.
inline constexpr int kHalfStencil[13][3] = {
    {1, 0, 0},  {-1, 1, 0}, {0, 1, 0},  {1, 1, 0},  {-1, -1, 1},
    {0, -1, 1}, {1, -1, 1}, {-1, 0, 1}, {0, 0, 1},  {1, 0, 1},
    {-1, 1, 1}, {0, 1, 1},  {1, 1, 1}};

/// Cell decomposition of one configuration: atoms sorted by cell (CSR).
struct CellGrid {
  std::array<int, 3> nc{{1, 1, 1}};
  std::array<bool, 3> wrap{{false, false, false}};
  std::vector<int32_t> start; //!< ncells + 1 row starts into `atoms`.
  std::vector<int32_t> atoms; //!< Atom indices grouped by cell.

  /// Linear index of cell (cx, cy, cz) shifted by stencil offset `o`, or
  /// -1 when the shift leaves a free axis.
  [[nodiscard]] long neighbour(int cx, int cy, int cz, const int o[3]) const {
    int c[3] = {cx + o[0], cy + o[1], cz + o[2]};
    for (int k = 0; k < 3; ++k) {
      const int m = nc[static_cast<std::size_t>(k)];
      if (wrap[static_cast<std::size_t>(k)]) {
        c[k] = (c[k] + m) % m;
      } else if (c[k] < 0 || c[k] >= m) {
        return -1;
      }
    }
    return (static_cast<long>(c[2]) * nc[1] + c[1]) * nc[0] + c[0];
  }

private:
  /// With `start` still holding per-cell counts (shifted by one): true when
  /// the cell scan's pair tests plus its stencil walk stay below half the
  /// brute-force scan's.
  [[nodiscard]] bool pays(std::size_t n, std::size_t ncells) const {
    const double brute = 0.5 * static_cast<double>(n) *
                         static_cast<double>(n - 1);
    double tests = 14.0 * static_cast<double>(ncells);
    for (int cz = 0; cz < nc[2]; ++cz) {
      for (int cy = 0; cy < nc[1]; ++cy) {
        for (int cx = 0; cx < nc[0]; ++cx) {
          const long c = (static_cast<long>(cz) * nc[1] + cy) * nc[0] + cx;
          const auto here =
              static_cast<double>(start[static_cast<std::size_t>(c) + 1]);
          if (here == 0.0) {
            continue;
          }
          tests += 0.5 * here * (here - 1.0);
          for (const auto &o : kHalfStencil) {
            const long d = neighbour(cx, cy, cz, o);
            if (d >= 0) {
              tests +=
                  here * static_cast<double>(start[static_cast<std::size_t>(d) + 1]);
            }
          }
        }
      }
    }
    return tests < 0.5 * brute;
  }

public:

  /// Bin `points` into cells strictly wider than `rl`, except a direction
  /// whose span is shorter than `rl`, which gets one cell. `w[k]` is the
  /// periodic width and `inv[k]` its inverse, zero for a free direction.
  /// Returns false when a grid cannot serve (fewer than three cells along
  /// a periodic direction, or fewer than kMinCellsForGrid cells in all)
  /// or would not pay (the cell scan's pair tests for this occupancy,
  /// plus the stencil walk, above half the brute-force scan).
  bool build(const double *points, std::size_t n, const double w[3],
             const double inv[3], double rl);
};

/// Linked-cell counterpart of vesin::cpu::brute_force_visit over a built
/// grid. With `Collect` false no pairs are recorded (the eval-only scan)
/// and `pairs` is left untouched.
template <bool Collect, typename Visitor>
inline void cell_visit(const CellGrid &g, const double *points,
                       const double w[3], const double inv[3],
                       double list_cutoff2, double visit_cutoff2,
                       std::vector<int32_t> &pairs, Visitor &&visit) {
  if constexpr (Collect) {
    pairs.clear();
  }
  const double w0 = w[0], w1 = w[1], w2 = w[2];
  const double i0 = inv[0], i1 = inv[1], i2 = inv[2];
  const int nx = g.nc[0], ny = g.nc[1], nz = g.nc[2];

  auto test = [&](int32_t i, int32_t j) {
    const auto ui = static_cast<std::size_t>(i);
    const auto uj = static_cast<std::size_t>(j);
    double dx = points[3 * uj] - points[3 * ui];
    double dy = points[3 * uj + 1] - points[3 * ui + 1];
    double dz = points[3 * uj + 2] - points[3 * ui + 2];
    dx -= w0 * vesin::cpu::visit_round(dx * i0);
    dy -= w1 * vesin::cpu::visit_round(dy * i1);
    dz -= w2 * vesin::cpu::visit_round(dz * i2);
    const double r2 = dx * dx + dy * dy + dz * dz;
    if (r2 < list_cutoff2) {
      // Report (a, b) with a < b and d = r_b - r_a, as the brute scan does.
      if (i > j) {
        std::swap(i, j);
        dx = -dx;
        dy = -dy;
        dz = -dz;
      }
      if constexpr (Collect) {
        pairs.push_back(i);
        pairs.push_back(j);
      }
      if (r2 <= visit_cutoff2) {
        visit(i, j, dx, dy, dz, r2);
      }
    }
  };

  for (int cz = 0; cz < nz; ++cz) {
    for (int cy = 0; cy < ny; ++cy) {
      for (int cx = 0; cx < nx; ++cx) {
        const std::size_t c =
            (static_cast<std::size_t>(cz) * static_cast<std::size_t>(ny) +
             static_cast<std::size_t>(cy)) *
                static_cast<std::size_t>(nx) +
            static_cast<std::size_t>(cx);
        const int32_t b0 = g.start[c];
        const int32_t b1 = g.start[c + 1];
        if (b0 == b1) {
          continue;
        }
        for (int32_t p = b0; p < b1; ++p) {
          for (int32_t q = p + 1; q < b1; ++q) {
            test(g.atoms[static_cast<std::size_t>(p)],
                 g.atoms[static_cast<std::size_t>(q)]);
          }
        }
        for (const auto &o : kHalfStencil) {
          const long dl = g.neighbour(cx, cy, cz, o);
          if (dl < 0) {
            continue;
          }
          const auto d = static_cast<std::size_t>(dl);
          const int32_t e0 = g.start[d];
          const int32_t e1 = g.start[d + 1];
          for (int32_t p = b0; p < b1; ++p) {
            const int32_t i = g.atoms[static_cast<std::size_t>(p)];
            for (int32_t q = e0; q < e1; ++q) {
              test(i, g.atoms[static_cast<std::size_t>(q)]);
            }
          }
        }
      }
    }
  }
}

} // namespace nlist
} // namespace rgpot
