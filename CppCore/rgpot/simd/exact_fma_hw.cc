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

#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
namespace detail {
// xgetbv runs only after CPUID reports OSXSAVE, so it cannot fault.
bool cpuid_fma_usable() {
  unsigned int eax = 0;
  unsigned int ebx = 0;
  unsigned int ecx = 0;
  unsigned int edx = 0;
  __asm__("cpuid" : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx) : "a"(0u), "c"(0u));
  if (eax < 1u) {
    return false;
  }
  __asm__("cpuid" : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx) : "a"(1u), "c"(0u));
  constexpr unsigned int kFma = 1u << 12;
  constexpr unsigned int kOsxsave = 1u << 27;
  constexpr unsigned int kAvx = 1u << 28;
  if ((ecx & (kFma | kOsxsave | kAvx)) != (kFma | kOsxsave | kAvx)) {
    return false;
  }
  unsigned int xlo = 0;
  unsigned int xhi = 0;
  __asm__("xgetbv" : "=a"(xlo), "=d"(xhi) : "c"(0u));
  (void)xhi;
  return (xlo & 6u) == 6u;
}
} // namespace detail
#endif

} // namespace rgpot::simd
