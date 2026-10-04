#pragma once
// MIT License
// Copyright 2023--present rgpot developers

// Bound for comparing two evaluations of the same sum on different summation
// orders or hardware: |a - b| <= k * eps * max|ref| over the array, with eps
// the machine epsilon of double. The rounding of a sum scales with the
// magnitude of what is summed, so the array's largest entry sets the scale;
// entries that cancel to near zero carry the rounding of the terms that
// formed them, which a per-element relative bound would misjudge. A decimal
// literal instead fits one architecture's rounding and misses another's.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

namespace rgpot::testing {

/// max_i |got_i - ref_i| / (eps * max_j |ref_j|); infinite when ref is
/// identically zero and got is not.
inline double deviation_in_eps(const std::vector<double> &got,
                               const std::vector<double> &ref) {
  double scale = 0.0;
  double dev = 0.0;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    scale = std::max(scale, std::abs(ref[i]));
    dev = std::max(dev, std::abs(got[i] - ref[i]));
  }
  if (scale == 0.0) {
    return dev == 0.0 ? 0.0 : std::numeric_limits<double>::infinity();
  }
  return dev / (std::numeric_limits<double>::epsilon() * scale);
}

} // namespace rgpot::testing
