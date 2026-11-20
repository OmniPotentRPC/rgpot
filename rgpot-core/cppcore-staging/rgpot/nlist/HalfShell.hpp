#pragma once
// MIT License
// Copyright 2023--present rgpot developers

/**
 * @file HalfShell.hpp
 * @brief Half-shell pair list in the order of a 27-cell linked walk.
 *
 * The cell edge is the cutoff (or the cutoff plus a Verlet skin). A
 * half-shell stencil visits each unordered pair of adjacent cells once.
 * Pairs are sorted into the order the full 27-cell walk would have
 * emitted, so replaying the list reproduces that walk's force sum.
 */

#include <cstddef>
#include <vector>

namespace rgpot::nlist {

/// Reusable pair list. A null list passed to orderedPairs rebuilds every
/// call. skin is the extra search distance in Angstrom. The stored pairs
/// are reused while every atom has moved by at most skin/2 and the cell
/// is unchanged. skin 0 reuses the list only when the coordinates match.
struct NeighborList {
  double skin = 0;
  void clear() { built = false; }

  std::vector<int> lo;
  std::vector<int> hi;
  /// Folded r_hi - r_lo from the build. Valid when use_stored is set.
  std::vector<double> dx;
  std::vector<double> dy;
  std::vector<double> dz;
  /// True when dx, dy, dz match the current coordinates.
  bool use_stored = false;
  std::vector<double> ref;
  double box[9]{};
  std::size_t n = 0;
  double cutoff = 0;
  double skin_used = 0;
  bool built = false;
};

/// Pairs with r < cutoff + skin, ordered as the 27-cell walk.
/// The reference is valid until the next call on this thread that passes
/// a null cache. On a hit, the cached pairs are returned as stored.
const NeighborList &orderedPairs(const double *positions, const double *box,
                                 std::size_t n, double cutoff,
                                 NeighborList *cache);

} // namespace rgpot::nlist
