#pragma once
// MIT License
// Copyright 2023--present rgpot developers

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

namespace rgpot::simd {

#if defined(__GNUC__)
#define RGPOT_NO_FP_CONTRACT __attribute__((optimize("-ffp-contract=off")))
#else
#define RGPOT_NO_FP_CONTRACT
#endif

namespace detail {

// Veltkamp split at 2^27. The product of the two 26-bit halves fits in a
// double, which is what makes TwoProd exact for finite normals.
inline constexpr double kFmaSplit = 134217729.0;

RGPOT_NO_FP_CONTRACT inline void two_sum(double a, double b, double &s, double &e) {
  s = a + b;
  const double z = s - a;
  e = (a - (s - z)) + (b - z);
}

RGPOT_NO_FP_CONTRACT inline void two_prod(double a, double b, double &p, double &e) {
  p = a * b;
  const double ca = kFmaSplit * a;
  const double ah = ca - (ca - a);
  const double al = a - ah;
  const double cb = kFmaSplit * b;
  const double bh = cb - (cb - b);
  const double bl = b - bh;
  e = ((ah * bh - p) + ah * bl + al * bh) + al * bl;
}

// True when TwoProd and the following sums stay inside the normal range,
// so the split factor cannot overflow and the product cannot underflow.
inline bool fma_dekker_domain(double a, double b, double c) {
  if (!std::isfinite(a) || !std::isfinite(b) || !std::isfinite(c)) {
    return false;
  }
  const double aa = std::fabs(a);
  const double bb = std::fabs(b);
  const double cc = std::fabs(c);
  if (aa > 0x1p480 || bb > 0x1p480 || cc > 0x1p900) {
    return false;
  }
  if ((aa != 0.0 && aa < 0x1p-480) || (bb != 0.0 && bb < 0x1p-480) ||
      (cc != 0.0 && cc < 0x1p-900)) {
    return false;
  }
  return true;
}

// Next representable value from pred toward the sign of sh.
RGPOT_NO_FP_CONTRACT inline double next_toward(double pred, double sh) {
  std::uint64_t ub = 0;
  std::memcpy(&ub, &pred, sizeof(ub));
  if (std::signbit(sh) == std::signbit(pred)) {
    ub += 1;
  } else {
    ub -= 1;
  }
  double out = 0;
  std::memcpy(&out, &ub, sizeof(out));
  return out;
}

// pred + (sh + sl) rounded to nearest, ties to even. pred is the high part.
RGPOT_NO_FP_CONTRACT inline double round_residual(double pred, double sh, double sl) {
  if (sh == 0.0 || pred == 0.0 || !std::isfinite(pred)) {
    return pred + (sh + sl);
  }
  std::uint64_t ub = 0;
  std::memcpy(&ub, &pred, sizeof(ub));
  const int be = static_cast<int>((ub >> 52) & 0x7ff);
  if (be == 0 || be == 0x7ff) {
    return pred + (sh + sl);
  }
  const bool pow2 = (ub & 0xfffffffffffffULL) == 0;
  const bool down = std::signbit(sh) != std::signbit(pred);
  const int ulp_exp = be - 1023 - 52 - ((pow2 && down) ? 1 : 0);
  const int mid_be = ulp_exp - 1 + 1023;
  double mid = 0;
  if (mid_be <= 0 || mid_be >= 0x7ff) {
    mid = std::copysign(std::ldexp(1.0, ulp_exp - 1), sh);
  } else {
    std::uint64_t mb = static_cast<std::uint64_t>(mid_be) << 52;
    if (std::signbit(sh)) {
      mb |= 0x8000000000000000ULL;
    }
    std::memcpy(&mid, &mb, sizeof(mid));
  }
  const double diff = sh - mid;
  double d2 = 0;
  double e2 = 0;
  two_sum(diff, sl, d2, e2);
  if (d2 == 0.0 && e2 == 0.0) {
    return pred + mid;
  }
  const bool up = (d2 > 0.0 && mid > 0.0) || (d2 < 0.0 && mid < 0.0) ||
                  (d2 == 0.0 && ((e2 > 0.0 && mid > 0.0) || (e2 < 0.0 && mid < 0.0)));
  if (!up) {
    return pred;
  }
  return next_toward(pred, sh);
}

// Dekker TwoProd plus a TwoSum chain. The residual's sticky bit breaks
// ties, including the tie just below a power of two. The caller rejects
// a zero, a subnormal, or a non-finite result and uses std::fma there.
RGPOT_NO_FP_CONTRACT inline double fma_dekker(double a, double b, double c) {
  double p = 0;
  double pe = 0;
  two_prod(a, b, p, pe);
  double s = 0;
  double t = 0;
  two_sum(p, c, s, t);
  double v = 0;
  double z = 0;
  two_sum(t, pe, v, z);
  double r = 0;
  double h = 0;
  two_sum(s, v, r, h);
  double rem = 0;
  double u = 0;
  two_sum(h, z, rem, u);
  if (u == 0.0) {
    return r + rem;
  }
  double pred = 0;
  double err = 0;
  two_sum(r, rem, pred, err);
  double sh = 0;
  double sl = 0;
  two_sum(err, u, sh, sl);
  return round_residual(pred, sh, sl);
}

// True when rem is exactly half an ulp of pred, the case where r+rem can
// round the tie the wrong way and the sticky bit has to decide.
RGPOT_NO_FP_CONTRACT inline bool rem_is_half_ulp(double pred, double rem) {
  std::uint64_t ub = 0;
  std::memcpy(&ub, &pred, sizeof(ub));
  const int be = static_cast<int>((ub >> 52) & 0x7ff);
  if (be == 0 || be == 0x7ff) {
    return true;
  }
  const bool pow2 = (ub & 0xfffffffffffffULL) == 0;
  const bool down = std::signbit(rem) != std::signbit(pred);
  const int mid_be = be - 53 - ((pow2 && down) ? 1 : 0);
  if (mid_be <= 0 || mid_be >= 0x7ff) {
    return true;
  }
  std::uint64_t mb = static_cast<std::uint64_t>(mid_be) << 52;
  if (std::signbit(rem)) {
    mb |= 0x8000000000000000ULL;
  }
  double half = 0;
  std::memcpy(&half, &mb, sizeof(half));
  return std::fabs(rem) == std::fabs(half);
}

// glibc's usable-FMA bit follows the hwcaps tunable. CPUID stays set when
// that tunable masks FMA, so this is the bit that selects the software path.
// The index is x86_cpu_FMA from <sys/platform/x86.h> (leaf 1, ECX, bit 12).
// The reader is weak: a sysroot whose libc stub lacks GLIBC_2.33 still links,
// and a runtime older than 2.33 leaves the symbol null.
#if defined(__x86_64__) && defined(__GLIBC__)
struct glibc_cpuid_leaf {
  unsigned int cpuid_array[4];
  unsigned int active_array[4];
};
extern "C" const glibc_cpuid_leaf *__x86_get_cpuid_feature_leaf(unsigned int)
    __attribute__((weak, pure));
#endif

inline bool glibc_fma_active() {
#if defined(__x86_64__) && defined(__GLIBC__)
  static const bool active = [] {
    if (__x86_get_cpuid_feature_leaf == nullptr) {
      return false;
    }
    constexpr unsigned int kFma =
        2u * 8u * static_cast<unsigned int>(sizeof(unsigned int)) + 12u;
    constexpr unsigned int kLeafSpan =
        8u * static_cast<unsigned int>(sizeof(unsigned int)) * 4u;
    const glibc_cpuid_leaf *leaf = __x86_get_cpuid_feature_leaf(kFma / kLeafSpan);
    const unsigned int reg = kFma & (kLeafSpan - 1u);
    const unsigned int bit = reg & (8u * sizeof(unsigned int) - 1u);
    const unsigned int word = reg / (8u * sizeof(unsigned int));
    return (leaf->active_array[word] & (1u << bit)) != 0;
  }();
  return active;
#else
  return false;
#endif
}

} // namespace detail

#if defined(__x86_64__)
double exact_fma_hw(double a, double b, double c);
double r2_hw(double dx, double dy, double dz);
void r2_hw_n(const double *dx, const double *dy, const double *dz, double *out,
             std::size_t n);
#endif

/// a*b+c with one rounding, the same bits as a hardware fused multiply-add.
/// Normals in range use a Dekker product and a correctly rounded sum of the
/// three exact parts. Zeros, subnormals, overflows, and non-finite values
/// use std::fma.
RGPOT_NO_FP_CONTRACT inline double exact_fma(double a, double b, double c) {
  if (!detail::fma_dekker_domain(a, b, c)) {
    return std::fma(a, b, c);
  }
  const double y = detail::fma_dekker(a, b, c);
  if (y == 0.0 || !std::isfinite(y) || std::fabs(y) < 0x1p-1022) {
    return std::fma(a, b, c);
  }
  return y;
}

// r+rem. This is a hardware fma on normal inputs except an exact half-ulp
// tie, a zero, a subnormal, or a non-finite result. Callers that need those
// cases use fma_quick. The exp polynomial and a squared distance do not.
RGPOT_NO_FP_CONTRACT inline double fma_trunc(double a, double b, double c) {
  double p = 0;
  double pe = 0;
  detail::two_prod(a, b, p, pe);
  double s = 0;
  double t = 0;
  detail::two_sum(p, c, s, t);
  double v = 0;
  double z = 0;
  detail::two_sum(t, pe, v, z);
  double r = 0;
  double h = 0;
  detail::two_sum(s, v, r, h);
  double rem = 0;
  double u = 0;
  detail::two_sum(h, z, rem, u);
  (void)u;
  return r + rem;
}

// Same bits as exact_fma. The rounded sum r+rem matches a hardware fma except
// when the residual is exactly half an ulp (a tie) or the result is zero,
// subnormal, or non-finite. Those lanes use exact_fma.
RGPOT_NO_FP_CONTRACT inline double fma_quick(double a, double b, double c) {
  if (!detail::fma_dekker_domain(a, b, c)) {
    return exact_fma(a, b, c);
  }
  double p = 0;
  double pe = 0;
  detail::two_prod(a, b, p, pe);
  double s = 0;
  double t = 0;
  detail::two_sum(p, c, s, t);
  double v = 0;
  double z = 0;
  detail::two_sum(t, pe, v, z);
  double r = 0;
  double h = 0;
  detail::two_sum(s, v, r, h);
  double rem = 0;
  double u = 0;
  detail::two_sum(h, z, rem, u);
  (void)u;
  const double y = r + rem;
  if (y == 0.0 || !std::isfinite(y) || std::fabs(y) < 0x1p-1022 ||
      detail::rem_is_half_ulp(r, rem)) {
    return exact_fma(a, b, c);
  }
  return y;
}

/// a*b+c, one rounding. A hardware instruction when this translation unit
/// has one, or when glibc reports FMA usable. Otherwise fma_quick.
RGPOT_NO_FP_CONTRACT inline double mul_add(double a, double b, double c) {
#if defined(__FMA__) || defined(__ARM_FEATURE_FMA)
  // The instruction, not std::fma: a glibc ifunc follows the hwcaps tunable
  // and would run the software path while this unit can still emit FMA.
  return __builtin_fma(a, b, c);
#elif defined(__x86_64__)
  if (detail::glibc_fma_active()) {
    return exact_fma_hw(a, b, c);
  }
  return fma_quick(a, b, c);
#else
  return fma_quick(a, b, c);
#endif
}

/// dx^2 + dy^2 + dz^2. The first square is a multiply. The two sums fuse.
RGPOT_NO_FP_CONTRACT inline double r2_of(double dx, double dy, double dz) {
#if defined(__x86_64__) && !defined(__FMA__) && !defined(__ARM_FEATURE_FMA)
  if (detail::glibc_fma_active()) {
    return r2_hw(dx, dy, dz);
  }
#endif
  return mul_add(dz, dz, mul_add(dy, dy, dx * dx));
}

#undef RGPOT_NO_FP_CONTRACT

} // namespace rgpot::simd
