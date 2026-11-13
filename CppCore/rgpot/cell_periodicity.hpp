#pragma once
// MIT License
// Copyright 2023--present rgpot developers

/**
 * Decide which axes of a row-major cell are periodic.
 *
 * Lattice vectors are the rows of `box`, in the same units as `pos`.
 * A singular cell is a cluster. `request` 0 forces a cluster. A positive
 * request is periodic on every axis. The default, a negative request,
 * is periodic only when the atoms span more than half of every axis, which
 * is a crystal and not the open box a cluster calculation uses. One or two
 * periodic axes are not returned: libxtb aborts on 1D and 2D cells.
 */

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace rgpot {

inline bool invertRowMajorCell(const double *box, double inv[9]) {
  if (box == nullptr) {
    return false;
  }
  const double a = box[0], b = box[1], c = box[2];
  const double d = box[3], e = box[4], f = box[5];
  const double g = box[6], h = box[7], i = box[8];
  const double det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
  if (!(std::abs(det) > 1e-12)) {
    return false;
  }
  const double s = 1.0 / det;
  inv[0] = (e * i - f * h) * s;
  inv[1] = (c * h - b * i) * s;
  inv[2] = (b * f - c * e) * s;
  inv[3] = (f * g - d * i) * s;
  inv[4] = (a * i - c * g) * s;
  inv[5] = (c * d - a * f) * s;
  inv[6] = (d * h - e * g) * s;
  inv[7] = (b * g - a * h) * s;
  inv[8] = (a * e - b * d) * s;
  return true;
}

/// `request` < 0 selects the crystal test. Returns false when a positive
/// request names a singular cell.
inline bool latticePeriodicity(const double *pos, std::size_t nAtoms,
                               const double *box, int request, bool periodic[3]) {
  periodic[0] = periodic[1] = periodic[2] = false;
  if (request == 0 || nAtoms == 0 || pos == nullptr) {
    return true;
  }
  double inv[9];
  if (!invertRowMajorCell(box, inv)) {
    return request <= 0;
  }
  if (request > 0) {
    periodic[0] = periodic[1] = periodic[2] = true;
    return true;
  }
  double lo[3] = {0, 0, 0};
  double hi[3] = {0, 0, 0};
  for (std::size_t a = 0; a < nAtoms; ++a) {
    const double x = pos[3 * a];
    const double y = pos[3 * a + 1];
    const double z = pos[3 * a + 2];
    const double frac[3] = {x * inv[0] + y * inv[1] + z * inv[2],
                            x * inv[3] + y * inv[4] + z * inv[5],
                            x * inv[6] + y * inv[7] + z * inv[8]};
    for (int k = 0; k < 3; ++k) {
      if (a == 0 || frac[k] < lo[k])
        lo[k] = frac[k];
      if (a == 0 || frac[k] > hi[k])
        hi[k] = frac[k];
    }
  }
  bool crystal = true;
  for (int k = 0; k < 3; ++k) {
    if (!(hi[k] - lo[k] > 0.5))
      crystal = false;
  }
  if (crystal) {
    periodic[0] = periodic[1] = periodic[2] = true;
  }
  return true;
}

} // namespace rgpot
