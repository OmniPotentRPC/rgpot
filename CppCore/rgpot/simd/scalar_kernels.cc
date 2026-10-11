// MIT License
// Copyright 2023--present rgpot developers

#include "rgpot/simd/pair_loops.hpp"

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC optimize("no-tree-vectorize")
#endif

namespace rgpot::simd {

#if defined(__GNUC__) && !defined(__clang__)
#define RGPOT_SCALAR __attribute__((noinline))
#else
#define RGPOT_SCALAR
#endif

RGPOT_SCALAR double lj_scalar_entry(const PairView &view, const LjParams &params,
                                    double *F, double strain[6]) {
#if defined(__x86_64__) && defined(RGPOT_WITH_HIGHWAY)
  // The SSE2 kernel uses a hardware fma when glibc reports one and the
  // truncating product otherwise. Same bits as the scalar reference.
  return lj_sse2_entry(view, params, F, strain);
#elif defined(__x86_64__)
  if (detail::glibc_fma_active()) {
    return lj_hw(view, params, F, strain);
  }
#endif
  return lj_reference(view, params, F, strain);
}

RGPOT_SCALAR double morse_scalar_entry(const PairView &view,
                                        const MorseParams &params, double *F,
                                        double strain[6]) {
#if defined(__x86_64__) && defined(RGPOT_WITH_HIGHWAY)
  return morse_sse2_entry(view, params, F, strain);
#elif defined(__x86_64__)
  if (detail::glibc_fma_active()) {
    return morse_hw(view, params, F, strain);
  }
#endif
  return morse_reference(view, params, F, strain);
}

RGPOT_SCALAR void eam_scalar_entry(long n, const double *R,
                                     const EamAtom *atoms, const double *box,
                                     double *F, double *U) {
#if defined(__x86_64__)
  // SSE kernel. It issues a hardware fma when glibc can use one, and the
  // truncating Dekker product otherwise.
  eam_sse(n, R, atoms, box, F, U);
  return;
#endif
  eam_reference(n, R, atoms, box, F, U);
}

RGPOT_SCALAR void fma_scalar_entry(const double *a, const double *b,
                                    const double *c, double *out,
                                    std::size_t n) {
  for (std::size_t i = 0; i < n; ++i) {
    out[i] = mul_add(a[i], b[i], c[i]);
  }
}

RGPOT_SCALAR void fold_scalar_entry(double *x, double *y, double *z,
                                      std::size_t n, double w0, double w1,
                                      double w2, double i0, double i1,
                                      double i2) {
  for (std::size_t i = 0; i < n; ++i) {
    x[i] = x[i] - w0 * round_reference(x[i] * i0);
    y[i] = y[i] - w1 * round_reference(y[i] * i1);
    z[i] = z[i] - w2 * round_reference(z[i] * i2);
  }
}

} // namespace rgpot::simd
