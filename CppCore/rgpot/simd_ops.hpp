#pragma once
// MIT License
// Copyright 2023--present rgpot developers

/**
 * @brief SIMD-friendly vectorized operations for unit conversion loops.
 *
 * These functions are thin wrappers over simple loops annotated with
 * compiler hints for auto-vectorization. With -O2 or higher, GCC/Clang
 * will emit AVX2/SSE2 instructions for these patterns.
 */

#include <cstddef>

namespace rgpot::simd {

/// dst[i] = src[i] * factor, for n elements.
inline void scale(double *__restrict__ dst, const double *__restrict__ src,
                  double factor, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    dst[i] = src[i] * factor;
  }
}

/// arr[i] *= factor, in-place, for n elements.
inline void scale_inplace(double *__restrict__ arr, double factor, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    arr[i] *= factor;
  }
}

/// dst[i] = src[i] * a + b, for n elements (fused multiply-add).
inline void fma(double *__restrict__ dst, const double *__restrict__ src,
                double a, double b, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    dst[i] = src[i] * a + b;
  }
}

} // namespace rgpot::simd
