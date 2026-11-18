#pragma once
// MIT License
// Copyright 2023--present rgpot developers

#include "rgpot/stress.hpp"

#include <cmath>

namespace rgpot {

/// True when cutoff lies strictly below half of each face spacing.
inline bool cutoffWithinHalfFaceSpacing(const double *box, double cutoff) {
  const double vol = cellVolume(box);
  for (int k = 0; k < 3; ++k) {
    const double *u = box + 3 * ((k + 1) % 3);
    const double *v = box + 3 * ((k + 2) % 3);
    const double cx = u[1] * v[2] - u[2] * v[1];
    const double cy = u[2] * v[0] - u[0] * v[2];
    const double cz = u[0] * v[1] - u[1] * v[0];
    if (!(2.0 * cutoff < vol / std::sqrt(cx * cx + cy * cy + cz * cz))) {
      return false;
    }
  }
  return true;
}

} // namespace rgpot
