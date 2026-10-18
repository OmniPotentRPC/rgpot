#pragma once
// MIT License
// Copyright 2023--present rgpot developers

#include "rgpot/simd/exact_fma.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace rgpot::simd {

// fdlibm e_exp constants. The scalar body and every vector lane use these
// values in the same order.
inline constexpr double kLn2Hi = 6.93147180369123816490e-01;
inline constexpr double kLn2Lo = 1.90821492927058770002e-10;
inline constexpr double kInvLn2 = 1.44269504088896338700e+00;
inline constexpr double kP1 = 1.66666666666666019037e-01;
inline constexpr double kP2 = -2.77777777770155933842e-03;
inline constexpr double kP3 = 6.61375632143793436117e-05;
inline constexpr double kP4 = -1.65339022054652515390e-06;
inline constexpr double kP5 = 4.13813679705723846039e-08;
inline constexpr double kOverflow = 7.09782712893383973096e+02;
inline constexpr double kUnderflow = -7.45133219101941108420e+02;

template <bool Hw> inline double fad(double a, double b, double c) {
  if constexpr (Hw) {
    // MSVC has no __builtin_fma. The hardware bodies that pass Hw are
    // GCC and Clang, compiled with target("fma").
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_fma(a, b, c);
#else
    return mul_add(a, b, c);
#endif
  } else {
    return fma_quick(a, b, c);
  }
}

template <bool Hw> inline double r2_fast(double dx, double dy, double dz) {
  return fad<Hw>(dz, dz, fad<Hw>(dy, dy, dx * dx));
}

// Inlined exp so a hardware-FMA caller fuses the polynomial instead of
// calling libm once per product-sum. Hw selects the instruction; the
// operations match exp().
template <bool Hw> inline double exp_fast(double x) {
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
      double scaled = 0;
      if constexpr (Hw) {
#if defined(__GNUC__) || defined(__clang__)
        scaled = __builtin_fma(kInvLn2, x, half);
#else
        scaled = fma_trunc(kInvLn2, x, half);
#endif
      } else {
        scaled = fma_trunc(kInvLn2, x, half);
      }
      k = static_cast<int>(scaled);
      const double tk = static_cast<double>(k);
      hi = x - tk * kLn2Hi;
      lo = tk * kLn2Lo;
    }
    x = hi - lo;
  } else if (hx < 0x3e300000u) {
    const double huge = 1.0e300;
    if (huge + x > 1.0) {
      return 1.0 + x;
    }
  }

  const double t = x * x;
  double p4 = 0;
  double p3 = 0;
  double p2 = 0;
  double p1 = 0;
  if constexpr (Hw) {
#if defined(__GNUC__) || defined(__clang__)
    p4 = __builtin_fma(t, kP5, kP4);
    p3 = __builtin_fma(t, p4, kP3);
    p2 = __builtin_fma(t, p3, kP2);
    p1 = __builtin_fma(t, p2, kP1);
#else
    p4 = fma_trunc(t, kP5, kP4);
    p3 = fma_trunc(t, p4, kP3);
    p2 = fma_trunc(t, p3, kP2);
    p1 = fma_trunc(t, p2, kP1);
#endif
  } else {
    p4 = fma_trunc(t, kP5, kP4);
    p3 = fma_trunc(t, p4, kP3);
    p2 = fma_trunc(t, p3, kP2);
    p1 = fma_trunc(t, p2, kP1);
  }
  const double c = x - t * p1;
  if (k == 0) {
    const double xc = x * c;
    const double den = c - 2.0;
    const double quot = xc / den;
    return 1.0 - (quot - x);
  }
  const double xc = x * c;
  const double den = 2.0 - c;
  const double quot = xc / den;
  const double y = 1.0 - ((lo - quot) - hi);
  std::uint64_t yb = 0;
  std::memcpy(&yb, &y, sizeof(yb));
  const int ye = static_cast<int>((yb >> 52) & 0x7ff);
  const int ne = ye + k;
  yb = (yb & 0x800fffffffffffffULL) | (static_cast<std::uint64_t>(ne) << 52);
  double out = 0;
  std::memcpy(&out, &yb, sizeof(out));
  return out;
}

/// Scalar exp. Cody-Waite reduction by a split ln(2), then the fdlibm
/// degree-5 polynomial. The large-argument scale and each polynomial
/// product-sum are one rounding. Overflow returns +inf, underflow returns 0,
/// a NaN returns a NaN, and exp(-0) is +1. Vector targets replay these
/// operations per lane.
double exp(double x);

#if defined(__x86_64__)
/// Same bits as exp(). The body is compiled with a hardware FMA.
void exp_hw_n(const double *in, double *out, std::size_t n);

struct PairView;
struct LjParams;
struct MorseParams;
struct EamAtom;
double lj_sse2_entry(const PairView &view, const LjParams &params, double *F,
                     double strain[6]);
double morse_sse2_entry(const PairView &view, const MorseParams &params, double *F,
                        double strain[6]);
void eam_sse2_entry(long n, const double *R, const EamAtom *atoms, const double *box,
                    double *F, double *U);
/// EAM with an SSE2 Dekker product. Same bits as the scalar kernel.
void eam_sse(long n, const double *R, const EamAtom *atoms, const double *box, double *F,
             double *U);
#endif

} // namespace rgpot::simd
