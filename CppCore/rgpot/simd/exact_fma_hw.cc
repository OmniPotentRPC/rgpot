// MIT License
// Copyright 2023--present rgpot developers

#include "rgpot/simd/exact_fma.hpp"

namespace rgpot::simd {

#if defined(__x86_64__)
// Weak so a later link with the highway dispatch object, which defines the
// same symbol, keeps a single copy. The attribute puts vfmadd in this
// function while the caller stays on the baseline ISA.
__attribute__((weak, target("fma"), noinline)) double
exact_fma_hw(double a, double b, double c) {
  return __builtin_fma(a, b, c);
}
#endif

} // namespace rgpot::simd
