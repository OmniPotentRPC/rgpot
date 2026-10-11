// MIT License
// Copyright 2023--present rgpot developers

#include "rgpot/simd/exp.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>

namespace rgpot::simd {

#if defined(__GNUC__) && !defined(__clang__)
__attribute__((noinline, optimize("no-tree-vectorize")))
#elif defined(__GNUC__)
__attribute__((noinline))
#endif
double exp(double x) {
  std::uint64_t ux = 0;
  std::memcpy(&ux, &x, sizeof(ux));
  std::uint32_t hx = static_cast<std::uint32_t>(ux >> 32);
  const int xsb = static_cast<int>((hx >> 31) & 1u);
  hx &= 0x7fffffffu;
  const std::uint32_t lx = static_cast<std::uint32_t>(ux);

  if (hx >= 0x40862e42u) {
    if (hx >= 0x7ff00000u) {
      if (((hx & 0xfffffu) | lx) != 0u) {
        return x + x;
      }
      return xsb == 0 ? x : 0.0;
    }
    if (x > kOverflow) {
      const double huge = 1.0e300;
      return huge * huge;
    }
    if (x < kUnderflow) {
      const double tiny = 9.33263618503218878990e-302;
      return tiny * tiny;
    }
  }

  int k = 0;
  double hi = 0.0;
  double lo = 0.0;
  if (hx > 0x3fd62e42u) {
    if (hx < 0x3ff0a2b2u) {
      const double ln2_hi = xsb ? -kLn2Hi : kLn2Hi;
      const double ln2_lo = xsb ? -kLn2Lo : kLn2Lo;
      hi = x - ln2_hi;
      lo = ln2_lo;
      k = 1 - xsb - xsb;
    } else {
      const double half = xsb ? -0.5 : 0.5;
      const double scaled = kInvLn2 * x + half;
      k = static_cast<int>(scaled);
      const double tk = static_cast<double>(k);
      const double t_hi = tk * kLn2Hi;
      const double t_lo = tk * kLn2Lo;
      hi = x - t_hi;
      lo = t_lo;
    }
    x = hi - lo;
  } else if (hx < 0x3e300000u) {
    const double huge = 1.0e300;
    if (huge + x > 1.0) {
      return 1.0 + x;
    }
  }

  const double t = x * x;
  const double tp5 = t * kP5;
  const double p4 = kP4 + tp5;
  const double tp4 = t * p4;
  const double p3 = kP3 + tp4;
  const double tp3 = t * p3;
  const double p2 = kP2 + tp3;
  const double tp2 = t * p2;
  const double p1 = kP1 + tp2;
  const double poly = t * p1;
  const double c = x - poly;
  if (k == 0) {
    const double xc = x * c;
    const double den = c - 2.0;
    const double quot = xc / den;
    const double inner = quot - x;
    return 1.0 - inner;
  }
  const double xc = x * c;
  const double den = 2.0 - c;
  const double quot = xc / den;
  const double loq = lo - quot;
  const double hiq = loq - hi;
  const double y = 1.0 - hiq;
  // ldexp by an integer power of two is exact when the result is finite.
  // It is not the libm exp ifunc, which picks an FMA variant at run time.
  return std::ldexp(y, k);
}

} // namespace rgpot::simd
