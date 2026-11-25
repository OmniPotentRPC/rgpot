#pragma once
// MIT License
// Copyright 2023--present rgpot developers

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>

namespace rgpot::simd {

// GCC will not inline across a differing optimize attribute, which turned
// every software fma into an out-of-line call. An x86-64 unit without
// __FMA__ has no fused instruction to contract into, so the attribute is
// only needed where contraction is possible.
#if defined(__GNUC__) && !defined(__clang__) &&                                 \
    (!defined(__x86_64__) || defined(__FMA__))
#define RGPOT_NO_FP_CONTRACT __attribute__((optimize("-ffp-contract=off")))
#define RGPOT_FMA_INLINE
#elif defined(__GNUC__)
#define RGPOT_NO_FP_CONTRACT
#define RGPOT_FMA_INLINE __attribute__((always_inline))
#else
#define RGPOT_NO_FP_CONTRACT
#define RGPOT_FMA_INLINE
#endif
#if defined(__GNUC__)
#define RGPOT_UNLIKELY(cond) __builtin_expect(static_cast<bool>(cond), 0)
#else
#define RGPOT_UNLIKELY(cond) (static_cast<bool>(cond))
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

// glibc's usable-FMA bit follows the hwcaps tunable. CPUID stays set when
// that tunable masks FMA, so where glibc offers the bit it has the last word.
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

#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
/// FMA3 from CPUID leaf 1 (ECX bit 12), with the OS saving the YMM state
/// (OSXSAVE and AVX set, XCR0 bits 1 and 2), the conditions Intel's manual
/// gives for executing VEX-encoded FMA. Needs no C library support. Out of
/// line in exact_fma_hw.cc, so xgetbv sits in one allowlisted function.
bool cpuid_fma_usable();

/// glibc's verdict on FMA where the C library has one (2.33 and newer), so
/// GLIBC_TUNABLES=glibc.cpu.hwcaps=-FMA also selects the software path.
/// True when there is no such reader.
inline bool glibc_allows_fma() {
#if defined(__GLIBC__)
  if (__x86_get_cpuid_feature_leaf == nullptr) {
    return true;
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
#else
  return true;
#endif
}

/// RGPOT_FMA=software in the environment selects the software path.
inline bool fma_env_software() {
  const char *v = std::getenv("RGPOT_FMA");
  return v != nullptr && std::strcmp(v, "software") == 0;
}
#endif

/// True when the hardware fused multiply-add may run: the CPU has FMA3 and
/// the OS saves its registers (CPUID), glibc 2.33 or newer does not mask it,
/// and RGPOT_FMA is not ``software``. Read once per process. The hardware
/// instruction and the software path return the same bits, so this decides
/// speed only.
inline bool fma_hw_active() {
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
  static const bool active =
      !fma_env_software() && cpuid_fma_usable() && glibc_allows_fma();
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

namespace detail {

inline std::uint64_t bits_of(double x) {
  std::uint64_t u = 0;
  std::memcpy(&u, &x, sizeof(u));
  return u;
}

inline double from_bits(std::uint64_t u) {
  double x = 0;
  std::memcpy(&x, &u, sizeof(x));
  return x;
}

// Biased exponent in [lo, hi], or a signed zero.
inline bool exp_in_band(std::uint64_t u, unsigned lo, unsigned hi) {
  const unsigned e = static_cast<unsigned>(u >> 52) & 0x7ffu;
  return ((u << 1) == 0) | (e - lo <= hi - lo);
}

// True when the Dekker product and the sums after it stay inside the normal
// range, so the split factor cannot overflow and the product cannot
// underflow. Integer compares on the exponent bits only.
inline bool fma_ro_domain(double a, double b, double c) {
  return exp_in_band(bits_of(a), 1023u - 479u, 1023u + 479u) &
         exp_in_band(bits_of(b), 1023u - 479u, 1023u + 479u) &
         exp_in_band(bits_of(c), 1023u - 899u, 1023u + 899u);
}

// Boldo and Melquiond, "Emulation of FMA and correctly rounded sums: proved
// algorithms using rounding to odd" (IEEE TC 57(4), 2008), Algorithm 5.4.
// The exact product p+pe comes from Dekker, then TwoSum(c, p) gives s+t.
// t+pe is rounded to odd from its TwoSum error: an inexact even result
// steps one ulp toward the error. A single round to nearest of s plus that
// odd value is the correctly rounded a*b+c, ties included, with no branch
// on the tie. Inside fma_ro_domain nothing overflows and every partial is
// normal, so only a zero or subnormal result needs another path.
RGPOT_NO_FP_CONTRACT RGPOT_FMA_INLINE inline double fma_ro(double a, double b, double c,
                                          bool &exact_ok) {
  const double p = a * b;
  const double ca = kFmaSplit * a;
  const double ah = ca - (ca - a);
  const double al = a - ah;
  const double cb = kFmaSplit * b;
  const double bh = cb - (cb - b);
  const double bl = b - bh;
  const double pe = ((ah * bh - p) + ah * bl + al * bh) + al * bl;
  const double s = c + p;
  const double sz = s - c;
  const double t = (c - (s - sz)) + (p - sz);
  const double v = t + pe;
  const double vz = v - t;
  const double ve = (t - (v - vz)) + (pe - vz);
  std::uint64_t uv = bits_of(v);
  const bool inexact_even = (ve != 0.0) & ((uv & 1u) == 0u);
  const std::uint64_t toward = ((uv ^ bits_of(ve)) >> 63) != 0 ? ~0ULL : 1ULL;
  uv += inexact_even ? toward : 0ULL;
  const double y = s + from_bits(uv);
  const unsigned ey = static_cast<unsigned>(bits_of(y) >> 52) & 0x7ffu;
  exact_ok = ey != 0u;
  // A zero here is an exact zero: c is exactly -a*b, or a*b and c are both
  // zeros. Then p is exact and p+c carries the IEEE sign of the zero.
  if (y == 0.0) {
    exact_ok = true;
    return p + c;
  }
  return y;
}

} // namespace detail

/// a*b+c with one rounding, the same bits as a hardware fused multiply-add.
/// Normals in range use a Dekker product and a round-to-odd sum
/// (Boldo and Melquiond). Inputs outside that range and subnormal results
/// use std::fma.
RGPOT_NO_FP_CONTRACT RGPOT_FMA_INLINE inline double exact_fma(double a, double b, double c) {
  if (RGPOT_UNLIKELY(!detail::fma_ro_domain(a, b, c))) {
    return std::fma(a, b, c);
  }
  bool ok = true;
  const double y = detail::fma_ro(a, b, c, ok);
  if (RGPOT_UNLIKELY(!ok)) {
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

// Same bits as exact_fma. Kept as a name for callers of the earlier
// two-stage path; the round-to-odd sum has no tie case left to filter.
RGPOT_NO_FP_CONTRACT RGPOT_FMA_INLINE inline double fma_quick(double a, double b, double c) {
  return exact_fma(a, b, c);
}

#if defined(__GNUC__)
/// Two doubles in one register, for the software path.
typedef double f64x2 __attribute__((vector_size(16)));

namespace detail {
typedef long long i64x2 __attribute__((vector_size(16)));
typedef unsigned long long u64x2 __attribute__((vector_size(16)));

// |x| in [lo, hi) or x a signed zero, from packed double compares. SSE2 has
// these natively, unlike 64-bit integer compares.
RGPOT_FMA_INLINE inline i64x2 lane_in_band(f64x2 x, double lo, double hi) {
  const f64x2 ax = reinterpret_cast<f64x2>(reinterpret_cast<u64x2>(x) &
                                           0x7fffffffffffffffULL);
  return ((ax >= lo) & (ax < hi)) | (x == 0.0);
}
} // namespace detail

/// exact_fma on both lanes. The lanes run the same round-to-odd sequence
/// as exact_fma, element by element, so each lane has the bits of std::fma.
/// When either lane leaves the domain or lands on a zero or a subnormal,
/// both lanes go through the scalar exact_fma.
RGPOT_NO_FP_CONTRACT RGPOT_FMA_INLINE inline f64x2 exact_fma_x2(f64x2 a,
                                                                 f64x2 b,
                                                                 f64x2 c) {
  using detail::i64x2;
  using detail::u64x2;
  const i64x2 dom = detail::lane_in_band(a, 0x1p-479, 0x1p480) &
                    detail::lane_in_band(b, 0x1p-479, 0x1p480) &
                    detail::lane_in_band(c, 0x1p-899, 0x1p900);
  const f64x2 split = {detail::kFmaSplit, detail::kFmaSplit};
  const f64x2 p = a * b;
  const f64x2 ca = split * a;
  const f64x2 ah = ca - (ca - a);
  const f64x2 al = a - ah;
  const f64x2 cb = split * b;
  const f64x2 bh = cb - (cb - b);
  const f64x2 bl = b - bh;
  const f64x2 pe = ((ah * bh - p) + ah * bl + al * bh) + al * bl;
  const f64x2 s = c + p;
  const f64x2 sz = s - c;
  const f64x2 t = (c - (s - sz)) + (p - sz);
  const f64x2 v = t + pe;
  const f64x2 vz = v - t;
  const f64x2 ve = (t - (v - vz)) + (pe - vz);
  const i64x2 iv = reinterpret_cast<i64x2>(v);
  const i64x2 ie = reinterpret_cast<i64x2>(ve);
  // Round to odd without 64-bit compares or arithmetic shifts, which SSE2
  // lacks: (iv & 1) - 1 is all ones for an even v, and the sign of iv ^ ie
  // picks a step of +1 or -1 on the bit pattern.
  const i64x2 even = (iv & 1) - 1;
  const i64x2 flip = reinterpret_cast<i64x2>(reinterpret_cast<u64x2>(iv ^ ie) >> 63);
  const i64x2 toward = 1 - (flip << 1);
  const i64x2 inexact = (ve != 0.0);
  const f64x2 vo = reinterpret_cast<f64x2>(iv + (inexact & even & toward));
  const f64x2 y = s + vo;
  const f64x2 ay = reinterpret_cast<f64x2>(reinterpret_cast<u64x2>(y) &
                                           0x7fffffffffffffffULL);
  const i64x2 ok = dom & (ay >= 0x1p-1022);
  if (RGPOT_UNLIKELY((ok[0] & ok[1]) == 0)) {
    return f64x2{exact_fma(a[0], b[0], c[0]), exact_fma(a[1], b[1], c[1])};
  }
  return y;
}
#endif

/// a*b+c, one rounding. A hardware instruction when this translation unit
/// has one, or when fma_hw_active(). Otherwise fma_quick.
RGPOT_NO_FP_CONTRACT RGPOT_FMA_INLINE inline double mul_add(double a, double b, double c) {
#if defined(__FMA__) || defined(__ARM_FEATURE_FMA)
  // The instruction, not std::fma: a glibc ifunc follows the hwcaps tunable
  // and would run the software path while this unit can still emit FMA.
  return __builtin_fma(a, b, c);
#elif defined(__x86_64__)
  if (detail::fma_hw_active()) {
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
  if (detail::fma_hw_active()) {
    return r2_hw(dx, dy, dz);
  }
#endif
  return mul_add(dz, dz, mul_add(dy, dy, dx * dx));
}

#undef RGPOT_NO_FP_CONTRACT
#undef RGPOT_FMA_INLINE
#undef RGPOT_UNLIKELY

} // namespace rgpot::simd
