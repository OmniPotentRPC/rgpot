// MIT License
// Copyright 2023--present rgpot developers

#include "rgpot/simd/exp.hpp"
#include "rgpot/simd/clones.hpp"

namespace rgpot::simd {

double exp_fixed(double x) { return detail::exp_fixed_lane(x); }

RGPOT_SIMD_CLONES void exp_fixed_n(const double *x, double *out, std::size_t n) {
  for (std::size_t i = 0; i < n; ++i) {
    out[i] = detail::exp_fixed_lane(x[i]);
  }
}

} // namespace rgpot::simd
