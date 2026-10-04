#pragma once
// MIT License
// Copyright 2023--present rgpot developers

// Bounds in units in the last place for comparing two evaluations of the same
// sum on different summation orders or hardware. A bound written as a decimal
// literal is met by one architecture's rounding and missed by another's; an
// ulp count scales with the quantity compared and stays valid on any IEEE
// double platform.

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace rgpot::testing {

/// Spacing of doubles at |x|.
inline double ulp_of(double x) {
  x = std::abs(x);
  return std::nextafter(x, std::numeric_limits<double>::infinity()) - x;
}

/// Largest |got_i - ref_i| in ulp of max(|ref_i|, max_j |ref_j|). Entries
/// that cancel to near zero carry the rounding of the terms that cancelled,
/// not of the result, so the spacing is floored at the array's largest
/// magnitude.
inline double max_deviation_ulp(const std::vector<double> &got,
                                const std::vector<double> &ref) {
  double scale = 0.0;
  for (double v : ref) {
    scale = std::max(scale, std::abs(v));
  }
  const double floor_ulp = ulp_of(scale);
  double worst = 0.0;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    const double unit = std::max(ulp_of(ref[i]), floor_ulp);
    if (unit > 0.0) {
      worst = std::max(worst, std::abs(got[i] - ref[i]) / unit);
    } else if (got[i] != ref[i]) {
      return std::numeric_limits<double>::infinity();
    }
  }
  return worst;
}

} // namespace rgpot::testing
