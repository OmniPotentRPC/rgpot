#pragma once
// Minimum-image rows from linkcell's cutoff list.
//
// pairs_within reports every atom-image. A classical pair kernel in this
// library keeps one image per unordered pair, the shortest, which is the
// image a lattice enumeration selects. The shift is an integer combination
// of the caller's cell, adjusted so
// d = r_i - r_j - S H
// uses the coordinates the caller passed, including a point that sits
// outside the primary cell.

#include <cstddef>
#include <cstdint>
#include <vector>

namespace rgpot {
namespace nlist {

struct ImageRow {
  int32_t i{0};
  int32_t j{0};
  int32_t s[3]{0, 0, 0};
  double dx{0.0};
  double dy{0.0};
  double dz{0.0};
  double r2{0.0};
};

/// Half list, i < j, one shortest image each, for a fully periodic cell.
/// `box` is row-major, one lattice vector per row. `r2` is the cutoff
/// list's squared distance. Returns false when the cell cannot be
/// inverted; an empty `out` with true means no pair is inside `cutoff`.
[[nodiscard]] bool periodicImages(const double *positions, std::size_t n,
                                  const double *box, double cutoff,
                                  std::vector<ImageRow> &out);

} // namespace nlist
} // namespace rgpot
