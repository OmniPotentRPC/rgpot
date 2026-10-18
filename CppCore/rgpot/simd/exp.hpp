#pragma once
// MIT License
// Copyright 2023--present rgpot developers
//
// e^x from a fixed sequence of IEEE additions, multiplications and bit
// operations. Every target computes the same bits: there is no table, no
// fused multiply-add and no libm call, so scalar code, SSE2, AVX2, AVX-512
// and NEON lanes agree, and the result does not depend on the C library.
// The translation units that include this header build with contraction
// off (-ffp-contract=off); clang also gets a pragma in the body. With
// -fno-trapping-math the clamps if-convert and loops over the body vectorise.

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace rgpot::simd {

namespace detail {

// Cody and Waite reduction x = k ln2 + r. kLn2Hi has 32 trailing zero bits,
// so k * kLn2Hi is exact for |k| < 2^21 (fdlibm's split).
inline constexpr double kExpInvLn2 = 0x1.71547652b82fep0;
inline constexpr double kExpLn2Hi = 0x1.62e42feep-1;
inline constexpr double kExpLn2Lo = 0x1.a39ef35793c76p-33;
// Adding 1.5 * 2^52 rounds a double below 2^51 in magnitude to an integer
// held in the low mantissa bits.
inline constexpr double kExpShift = 0x1.8p52;
// Inputs are clamped to [kExpMin, kExpMax]. exp(kExpMax) overflows and
// exp(kExpMin) rounds to zero, so the clamp keeps both limits.
inline constexpr double kExpMax = 710.0;
inline constexpr double kExpMin = -746.0;

// exp(r) ~ 1 + r + r^2 (c2 + c3 r + ... + c11 r^9) on |r| <= 0.3466.
// Sollya fpminimax with double coefficients and the constrained part 1 + r;
// the relative approximation error is below 3.8e-18 (2^-57.8), from
// sollya's supnorm. See docs/orgmode/reference/simd_exp.org.
inline constexpr double kExpC2 = 0x1.000000000000ap-1;
inline constexpr double kExpC3 = 0x1.55555555554fep-3;
inline constexpr double kExpC4 = 0x1.55555555506bcp-5;
inline constexpr double kExpC5 = 0x1.1111111126b36p-7;
inline constexpr double kExpC6 = 0x1.6c16c18477083p-10;
inline constexpr double kExpC7 = 0x1.a01a012fee99cp-13;
inline constexpr double kExpC8 = 0x1.a019995efb00fp-16;
inline constexpr double kExpC9 = 0x1.71df18ef76144p-19;
inline constexpr double kExpC10 = 0x1.28adfccb26cd8p-22;
inline constexpr double kExpC11 = 0x1.ad92e899366cbp-26;

inline std::uint64_t exp_bits(double x) {
  std::uint64_t u = 0;
  std::memcpy(&u, &x, sizeof(u));
  return u;
}

inline double exp_from_bits(std::uint64_t u) {
  double x = 0;
  std::memcpy(&x, &u, sizeof(x));
  return x;
}

/// 2^m for an integer-valued m in [-1022, 1023], built in the exponent
/// field. m + 1023 lands in the low bits of the shifted sum.
inline double exp_pow2(double m) {
  return exp_from_bits(exp_bits(m + (kExpShift + 1023.0)) << 52);
}

/// The body every exp in rgpot runs, scalar or as a vector lane. Branch
/// free: GCC and clang vectorise a loop over it. NaN propagates, +inf gives
/// +inf and -inf gives +0.
inline double exp_fixed_lane(double x) {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
  x = x < kExpMin ? kExpMin : x;
  x = x > kExpMax ? kExpMax : x;
  const double kd = (x * kExpInvLn2 + kExpShift) - kExpShift;
  // x - kd * kExpLn2Hi is exact (Sterbenz) and |r| <= 0.3466.
  const double r = (x - kd * kExpLn2Hi) - kd * kExpLn2Lo;
  double q = kExpC11;
  q = q * r + kExpC10;
  q = q * r + kExpC9;
  q = q * r + kExpC8;
  q = q * r + kExpC7;
  q = q * r + kExpC6;
  q = q * r + kExpC5;
  q = q * r + kExpC4;
  q = q * r + kExpC3;
  q = q * r + kExpC2;
  const double p = 1.0 + (r + (r * r) * q);
  // 2^k as two factors: both stay normal for k in [-1076, 1024], so only
  // the last product rounds, into the subnormal range or to infinity.
  const double k1 = (kd * 0.5 + kExpShift) - kExpShift;
  const double k2 = kd - k1;
  return (p * exp_pow2(k1)) * exp_pow2(k2);
}

} // namespace detail

/// e^x with the same bits on every instruction set and C library. Within
/// 1.24 ulp for normal results (see docs/orgmode/reference/simd_exp.org).
double exp_fixed(double x);

/// out[i] = exp_fixed(x[i]) for i < n. Vectorised and dispatched on the CPU
/// at run time; every variant returns the bits of exp_fixed.
void exp_fixed_n(const double *x, double *out, std::size_t n);

} // namespace rgpot::simd
