// MIT License
// Copyright (c) rgpot developers

// Cell counts and the bounding-box scan. Kept out of the pair-list header
// so a force loop that only reads a finished list does not compile them.

#include "rgpot/nlist/PairListCache.hpp"

#include <algorithm>
#include <cmath>

namespace rgpot {
namespace nlist {
namespace {

/// How many cells fit in `extent` at a width strictly above `rl`.
/// A span shorter than `rl`, including a zero span, gets one cell.
/// An exact multiple of `rl` does not take a cell as wide as `rl`:
/// a pair at that distance could otherwise land two cells apart.
int cellsForSpan(double extent, double rl) {
  if (!(extent > 0.0) || !(rl > 0.0)) {
    return 1;
  }
  const double ratio = std::nextafter(extent / rl, 0.0);
  int cells = 1;
  if (ratio > 0.0) {
    cells = static_cast<int>(std::floor(ratio));
  }
  if (cells < 1) {
    cells = 1;
  }
  if (cells > 1024) {
    cells = 1024;
  }
  while (cells > 1 && !(extent / static_cast<double>(cells) > rl)) {
    --cells;
  }
  return cells;
}

} // namespace

bool CellGrid::build(const double *points, std::size_t n, const double w[3],
                     const double inv[3], double rl) {
  if (!(rl > 0.0) || n < 2 || points == nullptr) {
    return false;
  }
  // One pass measures every direction. The binning below reuses it.
  std::array<double, 3> origin{};
  std::array<double, 3> span{};
  for (int k = 0; k < 3; ++k) {
    const auto uk = static_cast<std::size_t>(k);
    if (inv[k] != 0.0) {
      origin[uk] = 0.0;
      span[uk] = w[k];
      continue;
    }
    double mn = points[k];
    double mx = points[k];
    for (std::size_t a = 1; a < n; ++a) {
      const double t = points[3 * a + uk];
      mn = std::min(mn, t);
      mx = std::max(mx, t);
    }
    origin[uk] = mn;
    span[uk] = mx - mn;
  }
  std::array<double, 3> scale{};
  std::size_t ncells = 1;
  for (int k = 0; k < 3; ++k) {
    const auto uk = static_cast<std::size_t>(k);
    wrap[uk] = inv[k] != 0.0;
    const int cells = cellsForSpan(span[uk], rl);
    if (wrap[uk] && cells < 3) {
      return false;
    }
    nc[uk] = cells;
    scale[uk] =
        static_cast<double>(cells) / (span[uk] > 0.0 ? span[uk] : 1.0);
    ncells *= static_cast<std::size_t>(cells);
  }
  if (ncells < kMinCellsForGrid || ncells > 8 * n + 64) {
    return false;
  }

  std::vector<int32_t> cellOf(n);
  start.assign(ncells + 1, 0);
  for (std::size_t a = 0; a < n; ++a) {
    std::array<int, 3> c{};
    for (int k = 0; k < 3; ++k) {
      const auto uk = static_cast<std::size_t>(k);
      double t = points[3 * a + uk];
      if (wrap[uk]) {
        t *= inv[k];
        t -= std::floor(t);
        t *= static_cast<double>(nc[uk]);
      } else {
        t = (t - origin[uk]) * scale[uk];
      }
      // A NaN lands in cell 0 instead of a bad float-to-int conversion.
      const int last = nc[uk] - 1;
      c[uk] = t >= 0.0 ? (t < static_cast<double>(nc[uk])
                              ? static_cast<int>(t)
                              : last)
                       : 0;
      if (c[uk] > last) {
        c[uk] = last;
      }
    }
    const auto id = static_cast<int32_t>(
        (static_cast<std::size_t>(c[2]) * static_cast<std::size_t>(nc[1]) +
         static_cast<std::size_t>(c[1])) *
            static_cast<std::size_t>(nc[0]) +
        static_cast<std::size_t>(c[0]));
    cellOf[a] = id;
    ++start[static_cast<std::size_t>(id) + 1];
  }
  if (!pays(n, ncells)) {
    return false;
  }
  for (std::size_t c = 0; c < ncells; ++c) {
    start[c + 1] += start[c];
  }
  atoms.resize(n);
  std::vector<int32_t> fill(start.begin(), start.end() - 1);
  for (std::size_t a = 0; a < n; ++a) {
    atoms[static_cast<std::size_t>(
        fill[static_cast<std::size_t>(cellOf[a])]++)] = static_cast<int32_t>(a);
  }
  return true;
}

void CachedPairList::setup(const double *R, std::size_t n, const double *box,
                           const Options &opt) {
  const bool orthorhombic = box[1] == 0.0 && box[2] == 0.0 && box[3] == 0.0 &&
                            box[5] == 0.0 && box[6] == 0.0 && box[7] == 0.0;
  general_ = !orthorhombic &&
             (opt.periodic[0] || opt.periodic[1] || opt.periodic[2]);
  if (general_) {
    generalImage_ = MinimumImage(box, opt.periodic);
  }
  const bool fits = cellGridFits(R, n, box, opt);
  mic_ = orthorhombic && (n <= 20000 || fits);
  for (int k = 0; mic_ && k < 3; ++k) {
    if (opt.periodic[static_cast<std::size_t>(k)] &&
        opt.cutoff > 0.5 * box[4 * k]) {
      mic_ = false;
    }
  }
}

bool CachedPairList::cellGridFits(const double *R, std::size_t n,
                                  const double *box, const Options &opt) {
  const double rl = opt.cutoff + opt.skin;
  if (!(rl > 0.0) || n < 2 || R == nullptr) {
    return false;
  }
  std::size_t ncells = 1;
  for (int k = 0; k < 3; ++k) {
    const auto uk = static_cast<std::size_t>(k);
    double extent = 0.0;
    if (opt.periodic[uk]) {
      extent = box[4 * k];
    } else {
      double mn = R[k];
      double mx = R[k];
      for (std::size_t a = 1; a < n; ++a) {
        const double t = R[3 * a + uk];
        mn = std::min(mn, t);
        mx = std::max(mx, t);
      }
      extent = mx - mn;
    }
    const int cells = cellsForSpan(extent, rl);
    if (opt.periodic[uk] && cells < 3) {
      return false;
    }
    ncells *= static_cast<std::size_t>(cells);
  }
  return ncells >= kMinCellsForGrid && ncells <= 8 * n + 64;
}

} // namespace nlist
} // namespace rgpot
