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

// x * 2^n with one rounding, after musl's scalbn. A subnormal result is
// reached in a step that leaves n below -53, so it rounds only once.
inline double scale2(double x, int n) {
  if (n > 1023) {
    x *= 0x1p1023;
    n -= 1023;
    if (n > 1023) {
      x *= 0x1p1023;
      n -= 1023;
      if (n > 1023) {
        n = 1023;
      }
    }
  } else if (n < -1022) {
    x *= 0x1p-1022 * 0x1p53;
    n += 1022 - 53;
    if (n < -1022) {
      x *= 0x1p-1022 * 0x1p53;
      n += 1022 - 53;
      if (n < -1022) {
        n = -1022;
      }
    }
  }
  return x * from_bits(static_cast<std::uint64_t>(0x3ff + n) << 52);
}

inline int clz64(std::uint64_t x) {
#if defined(__GNUC__) || defined(__clang__)
  return __builtin_clzll(x);
#else
  int n = 0;
  for (std::uint64_t bit = 1ULL << 63; (x & bit) == 0; bit >>= 1) {
    ++n;
  }
  return n;
#endif
}

struct fma_num {
  std::uint64_t m;
  int e;
  bool neg;
};

// Significand shifted so its top ten bits and last bit are clear, and
// x == m * 2^e. Zero gets an exponent above every finite one, infinity
// and NaN exactly kFmaZeroInfNan.
inline constexpr int kFmaZeroInfNan = 0x7ff - 0x3ff - 52 - 1;
inline fma_num fma_normalize(double x) {
  std::uint64_t ix = bits_of(x);
  int e = static_cast<int>(ix >> 52);
  const bool neg = (e & 0x800) != 0;
  e &= 0x7ff;
  if (e == 0) {
    ix = bits_of(x * 0x1p63);
    e = static_cast<int>(ix >> 52) & 0x7ff;
    e = e != 0 ? e - 63 : 0x800;
  }
  ix &= (1ULL << 52) - 1;
  ix |= 1ULL << 52;
  ix <<= 1;
  e -= 0x3ff + 52 + 1;
  return fma_num{ix, e, neg};
}

inline void mul64(std::uint64_t x, std::uint64_t y, std::uint64_t &hi,
                  std::uint64_t &lo) {
  const std::uint64_t xlo = static_cast<std::uint32_t>(x);
  const std::uint64_t xhi = x >> 32;
  const std::uint64_t ylo = static_cast<std::uint32_t>(y);
  const std::uint64_t yhi = y >> 32;
  const std::uint64_t t1 = xlo * ylo;
  const std::uint64_t t2 = xlo * yhi + xhi * ylo;
  const std::uint64_t t3 = xhi * yhi;
  lo = t1 + (t2 << 32);
  hi = t3 + (t2 >> 32) + (t1 > lo ? 1u : 0u);
}

/// a*b+c with one rounding in round-to-nearest, from integer arithmetic
/// only: musl's fma (Szabolcs Nagy, MIT). A NaN input comes back quiet, as
/// the x86 instruction returns it; the default NaN appears only when no
/// input is a NaN. Used where the C library's fma is not fused.
RGPOT_NO_FP_CONTRACT inline double fma_soft(double x, double y, double z) {
  if (std::isnan(z) && !std::isnan(x) && !std::isnan(y)) {
    return z + z;
  }
  const fma_num nx = fma_normalize(x);
  const fma_num ny = fma_normalize(y);
  const fma_num nz = fma_normalize(z);
  if (nx.e >= kFmaZeroInfNan || ny.e >= kFmaZeroInfNan) {
    return x * y + z;
  }
  if (nz.e >= kFmaZeroInfNan) {
    // A zero z: x*y is exact and nonzero, so its one rounding is the
    // answer. Adding z would turn a product that underflows to -0 into +0.
    if (nz.e > kFmaZeroInfNan) {
      return x * y;
    }
    return z;
  }

  std::uint64_t rhi = 0;
  std::uint64_t rlo = 0;
  std::uint64_t zhi = 0;
  std::uint64_t zlo = 0;
  mul64(nx.m, ny.m, rhi, rlo);
  int e = nx.e + ny.e;
  int d = nz.e - e;
  if (d > 0) {
    if (d < 64) {
      zlo = nz.m << d;
      zhi = nz.m >> (64 - d);
    } else {
      zlo = 0;
      zhi = nz.m;
      e = nz.e - 64;
      d -= 64;
      if (d == 0) {
      } else if (d < 64) {
        rlo = (rhi << (64 - d)) | (rlo >> d) | ((rlo << (64 - d)) != 0 ? 1u : 0u);
        rhi = rhi >> d;
      } else {
        rlo = 1;
        rhi = 0;
      }
    }
  } else {
    zhi = 0;
    d = -d;
    if (d == 0) {
      zlo = nz.m;
    } else if (d < 64) {
      zlo = (nz.m >> d) | ((nz.m << (64 - d)) != 0 ? 1u : 0u);
    } else {
      zlo = 1;
    }
  }

  bool neg = nx.neg != ny.neg;
  const bool samesign = neg == nz.neg;
  bool nonzero = true;
  if (samesign) {
    rlo += zlo;
    rhi += zhi + (rlo < zlo ? 1u : 0u);
  } else {
    const std::uint64_t t = rlo;
    rlo -= zlo;
    rhi = rhi - zhi - (t < rlo ? 1u : 0u);
    if ((rhi >> 63) != 0) {
      rlo = 0 - rlo;
      rhi = 0 - rhi - (rlo != 0 ? 1u : 0u);
      neg = !neg;
    }
    nonzero = rhi != 0;
  }

  if (nonzero) {
    e += 64;
    d = clz64(rhi) - 1;
    rhi = (rhi << d) | (rlo >> (64 - d)) | ((rlo << d) != 0 ? 1u : 0u);
  } else if (rlo != 0) {
    d = clz64(rlo) - 1;
    if (d < 0) {
      rhi = (rlo >> 1) | (rlo & 1u);
    } else {
      rhi = rlo << d;
    }
  } else {
    return x * y + z;
  }
  e -= d;

  std::int64_t i = static_cast<std::int64_t>(rhi);
  if (neg) {
    i = -i;
  }
  double r = static_cast<double>(i);
  if (e < -1022 - 62) {
    if (e == -1022 - 63) {
      const double c = neg ? -0x1p63 : 0x1p63;
      if (r == c) {
        return neg ? -0x1p-1022 : 0x1p-1022;
      }
      // One bit goes when scaled; a set top bit keeps the conversion to
      // the only rounding when the value is inexact.
      if ((rhi << 53) != 0) {
        i = static_cast<std::int64_t>((rhi >> 1) | (rhi & 1u) | (1ULL << 62));
        if (neg) {
          i = -i;
        }
        r = static_cast<double>(i);
        r = 2 * r - c;
      }
    } else {
      d = 10;
      i = static_cast<std::int64_t>(((rhi >> d) | ((rhi << (64 - d)) != 0 ? 1u : 0u))
                                    << d);
      if (neg) {
        i = -i;
      }
      r = static_cast<double>(i);
    }
  }
  return scale2(r, e);
}

/// The fallback for inputs the round-to-odd path does not take.
/// mingw-w64's fma (libmingwex) is not fused: it returns the rounded a*b+c
/// when that is infinite or NaN and sums separately rounded partial products
/// otherwise.
RGPOT_NO_FP_CONTRACT inline double fma_fallback(double a, double b, double c) {
#if defined(__MINGW32__) && defined(__x86_64__)
  return fma_soft(a, b, c);
#else
  return std::fma(a, b, c);
#endif
}

#if defined(__GNUC__) || defined(__clang__)
/// The instruction, for hardware bodies inlined into a target("fma")
/// function. A copy left out of line calls the C library's fma instead,
/// and at -O0 every copy is, since flatten does nothing there.
inline double fma_insn(double a, double b, double c) {
#if defined(__MINGW32__) && defined(__x86_64__) && !defined(__FMA__) &&         \
    !defined(__OPTIMIZE__)
  return fma_soft(a, b, c);
#else
  return __builtin_fma(a, b, c);
#endif
}
#endif

} // namespace detail

/// a*b+c with one rounding, the same bits as a hardware fused multiply-add.
/// Normals in range use a Dekker product and a round-to-odd sum
/// (Boldo and Melquiond). Inputs outside that range and subnormal results
/// use the C library's fma, or fma_soft where that is not fused.
RGPOT_NO_FP_CONTRACT RGPOT_FMA_INLINE inline double exact_fma(double a, double b, double c) {
  if (RGPOT_UNLIKELY(!detail::fma_ro_domain(a, b, c))) {
    return detail::fma_fallback(a, b, c);
  }
  bool ok = true;
  const double y = detail::fma_ro(a, b, c, ok);
  if (RGPOT_UNLIKELY(!ok)) {
    return detail::fma_fallback(a, b, c);
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
