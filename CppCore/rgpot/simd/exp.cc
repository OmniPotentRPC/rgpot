// MIT License
// Copyright 2023--present rgpot developers

#include "rgpot/simd/exp.hpp"
#include "rgpot/simd/pair_loops.hpp"

#include <cstdint>
#include <cstring>

#if defined(__x86_64__)
#include <immintrin.h>
#endif

namespace rgpot::simd {

#if defined(__GNUC__) && !defined(__clang__)
__attribute__((noinline, optimize("no-tree-vectorize")))
#elif defined(__GNUC__)
__attribute__((noinline))
#endif
double
exp(double x) {
#if defined(__FMA__) || defined(__ARM_FEATURE_FMA)
  return exp_fast<true>(x);
#elif defined(__x86_64__)
  if (detail::glibc_fma_active()) {
    double y = 0;
    exp_hw_n(&x, &y, 1);
    return y;
  }
  return exp_fast<false>(x);
#else
  return exp_fast<false>(x);
#endif
}

#if defined(__x86_64__)

__attribute__((target("fma"), flatten, noinline)) double lj_hw(const PairView &v,
                                                      const LjParams &p,
                                                      double *F,
                                                      double strain[6]) {
  return lj_scalar<true>(v, p, F, strain);
}

__attribute__((target("fma"), flatten, noinline)) double morse_hw(const PairView &v,
                                                         const MorseParams &p,
                                                         double *F,
                                                         double strain[6]) {
  return morse_scalar<true>(v, p, F, strain);
}

__attribute__((target("fma"), flatten, noinline)) void eam_hw(long n, const double *R,
                                                     const EamAtom *atoms,
                                                     const double *box,
                                                     double *F, double *U) {
  eam_scalar<true>(n, R, atoms, box, F, U);
}

__attribute__((target("fma"), noinline)) double exact_fma_hw(double a, double b,
                                                             double c) {
  return __builtin_fma(a, b, c);
}

__attribute__((target("fma"), noinline)) double r2_hw(double dx, double dy,
                                                      double dz) {
  return __builtin_fma(dz, dz, __builtin_fma(dy, dy, dx * dx));
}

__attribute__((target("fma"), noinline)) void r2_hw_n(const double *dx,
                                                      const double *dy,
                                                      const double *dz,
                                                      double *out,
                                                      std::size_t n) {
  std::size_t i = 0;
  for (; i + 2 <= n; i += 2) {
    const __m128d x = _mm_loadu_pd(dx + i);
    const __m128d y = _mm_loadu_pd(dy + i);
    const __m128d z = _mm_loadu_pd(dz + i);
    const __m128d r2 =
        _mm_fmadd_pd(z, z, _mm_fmadd_pd(y, y, _mm_mul_pd(x, x)));
    _mm_storeu_pd(out + i, r2);
  }
  for (; i < n; ++i) {
    out[i] = __builtin_fma(dz[i], dz[i], __builtin_fma(dy[i], dy[i], dx[i] * dx[i]));
  }
}

// One pass: the squared distance is fused while dx, dy, dz are still in
// registers. Storing every candidate and calling r2_hw_n reloads them.
__attribute__((target("fma"), noinline)) std::size_t
accept_row_hw(const PairView &v, std::size_t i, double *dxb, double *dyb,
              double *dzb, double *r2b, std::int32_t *jb) {
  const std::int32_t p0 = v.rows[i];
  const std::int32_t p1 = v.rows[i + 1];
  const double xi = v.R[3 * i];
  const double yi = v.R[3 * i + 1];
  const double zi = v.R[3 * i + 2];
  const int fold = v.fold;
  std::size_t m = 0;
  for (std::int32_t p = p0; p < p1; ++p) {
    const auto up = static_cast<std::size_t>(p);
    const auto j = static_cast<std::size_t>(v.nbr[up]);
    double dx = xi - v.R[3 * j];
    double dy = yi - v.R[3 * j + 1];
    double dz = zi - v.R[3 * j + 2];
    if (fold == 1) {
      dx = dx - v.w[0] * visit_round(dx * v.inv[0]);
      dy = dy - v.w[1] * visit_round(dy * v.inv[1]);
      dz = dz - v.w[2] * visit_round(dz * v.inv[2]);
    } else if (fold == 2) {
      const double *sh = v.shift + 3 * static_cast<std::size_t>(v.code[up]);
      dx = dx - sh[0];
      dy = dy - sh[1];
      dz = dz - sh[2];
    }
    const double r2 = __builtin_fma(dz, dz, __builtin_fma(dy, dy, dx * dx));
    if (!(r2 <= v.cutoff2)) {
      continue;
    }
    dxb[m] = dx;
    dyb[m] = dy;
    dzb[m] = dz;
    r2b[m] = r2;
    jb[m] = static_cast<std::int32_t>(j);
    ++m;
  }
  return m;
}

#include "rgpot/simd/sse_soft.hpp"

__attribute__((target("fma"), always_inline)) static inline __m128d exp2_hw_finite(__m128d x_in) {
  const __m128i sign = _mm_and_si128(_mm_castpd_si128(x_in),
                                     _mm_set1_epi64x(static_cast<long long>(0x8000000000000000ULL)));
  const __m128d half = _mm_or_pd(_mm_set1_pd(0.5), _mm_castsi128_pd(sign));
  const __m128d scaled = _mm_fmadd_pd(_mm_set1_pd(kInvLn2), x_in, half);
  const __m128i k32 = _mm_cvttpd_epi32(scaled);
  const __m128d tk = _mm_cvtepi32_pd(k32);
  const __m128i k64 = _mm_unpacklo_epi32(k32, _mm_srai_epi32(k32, 31));
  const __m128d hi = _mm_sub_pd(x_in, _mm_mul_pd(tk, _mm_set1_pd(kLn2Hi)));
  const __m128d lo = _mm_mul_pd(tk, _mm_set1_pd(kLn2Lo));
  const __m128d x = _mm_sub_pd(hi, lo);
  const __m128d t = _mm_mul_pd(x, x);
  __m128d p = _mm_fmadd_pd(t, _mm_set1_pd(kP5), _mm_set1_pd(kP4));
  p = _mm_fmadd_pd(t, p, _mm_set1_pd(kP3));
  p = _mm_fmadd_pd(t, p, _mm_set1_pd(kP2));
  p = _mm_fmadd_pd(t, p, _mm_set1_pd(kP1));
  const __m128d c = _mm_sub_pd(x, _mm_mul_pd(t, p));
  const __m128d quot = _mm_div_pd(_mm_mul_pd(x, c), _mm_sub_pd(_mm_set1_pd(2.0), c));
  const __m128d y = _mm_sub_pd(_mm_set1_pd(1.0), _mm_sub_pd(_mm_sub_pd(lo, quot), hi));
  const __m128i yb = _mm_castpd_si128(y);
  const __m128i yexp = _mm_and_si128(_mm_srli_epi64(yb, 52), _mm_set1_epi64x(0x7ff));
  const __m128i ne = _mm_add_epi64(yexp, k64);
  const __m128i packed =
      _mm_or_si128(_mm_and_si128(yb, _mm_set1_epi64x(static_cast<long long>(0x800fffffffffffffULL))),
                   _mm_slli_epi64(ne, 52));
  return _mm_castsi128_pd(packed);
}

__attribute__((target("fma"), noinline)) __m128d exp2_hw(__m128d x_in) {
  const __m128d ax = _mm_andnot_pd(_mm_set1_pd(-0.0), x_in);
  // hx >= 0x40862e42, the fdlibm overflow/underflow/non-finite bin.
  const __m128d lim = _mm_castsi128_pd(_mm_set1_epi64x(0x40862e4200000000LL));
  const __m128d bad = _mm_or_pd(_mm_cmpge_pd(ax, lim), _mm_cmpunord_pd(x_in, x_in));
  if (_mm_movemask_pd(bad) == 0) {
    return exp2_hw_finite(x_in);
  }
  alignas(16) double in[2], out[2];
  _mm_store_pd(in, x_in);
  out[0] = exp(in[0]);
  out[1] = exp(in[1]);
  const int mask = _mm_movemask_pd(bad);
  if (mask != 3) {
    const __m128d fast = exp2_hw_finite(x_in);
    alignas(16) double ff[2];
    _mm_store_pd(ff, fast);
    if ((mask & 1) == 0) {
      out[0] = ff[0];
    }
    if ((mask & 2) == 0) {
      out[1] = ff[1];
    }
  }
  return _mm_load_pd(out);
}


__attribute__((target("fma"), always_inline)) static inline void exp8_hw(const double *in, double *out) {
  __m128d x0 = _mm_loadu_pd(in);
  __m128d x1 = _mm_loadu_pd(in + 2);
  __m128d x2 = _mm_loadu_pd(in + 4);
  __m128d x3 = _mm_loadu_pd(in + 6);
  const __m128d lim = _mm_castsi128_pd(_mm_set1_epi64x(0x40862e4200000000LL));
  auto bad_of = [&](__m128d x) {
    const __m128d ax = _mm_andnot_pd(_mm_set1_pd(-0.0), x);
    return _mm_or_pd(_mm_cmpge_pd(ax, lim), _mm_cmpunord_pd(x, x));
  };
  if ((_mm_movemask_pd(bad_of(x0)) | _mm_movemask_pd(bad_of(x1)) |
       _mm_movemask_pd(bad_of(x2)) | _mm_movemask_pd(bad_of(x3))) != 0) {
    for (int i = 0; i < 8; ++i) {
      out[i] = exp(in[i]);
    }
    return;
  }
  auto half_of = [](__m128d x) {
    const __m128i sign = _mm_and_si128(
        _mm_castpd_si128(x), _mm_set1_epi64x(static_cast<long long>(0x8000000000000000ULL)));
    return _mm_or_pd(_mm_set1_pd(0.5), _mm_castsi128_pd(sign));
  };
  const __m128d inv = _mm_set1_pd(kInvLn2);
  __m128d s0 = _mm_fmadd_pd(inv, x0, half_of(x0));
  __m128d s1 = _mm_fmadd_pd(inv, x1, half_of(x1));
  __m128d s2 = _mm_fmadd_pd(inv, x2, half_of(x2));
  __m128d s3 = _mm_fmadd_pd(inv, x3, half_of(x3));
  __m128i k32_0 = _mm_cvttpd_epi32(s0);
  __m128i k32_1 = _mm_cvttpd_epi32(s1);
  __m128i k32_2 = _mm_cvttpd_epi32(s2);
  __m128i k32_3 = _mm_cvttpd_epi32(s3);
  __m128d tk0 = _mm_cvtepi32_pd(k32_0);
  __m128d tk1 = _mm_cvtepi32_pd(k32_1);
  __m128d tk2 = _mm_cvtepi32_pd(k32_2);
  __m128d tk3 = _mm_cvtepi32_pd(k32_3);
  __m128i k0 = _mm_unpacklo_epi32(k32_0, _mm_srai_epi32(k32_0, 31));
  __m128i k1 = _mm_unpacklo_epi32(k32_1, _mm_srai_epi32(k32_1, 31));
  __m128i k2 = _mm_unpacklo_epi32(k32_2, _mm_srai_epi32(k32_2, 31));
  __m128i k3 = _mm_unpacklo_epi32(k32_3, _mm_srai_epi32(k32_3, 31));
  const __m128d ln_hi = _mm_set1_pd(kLn2Hi);
  const __m128d ln_lo = _mm_set1_pd(kLn2Lo);
  __m128d hi0 = _mm_sub_pd(x0, _mm_mul_pd(tk0, ln_hi));
  __m128d hi1 = _mm_sub_pd(x1, _mm_mul_pd(tk1, ln_hi));
  __m128d hi2 = _mm_sub_pd(x2, _mm_mul_pd(tk2, ln_hi));
  __m128d hi3 = _mm_sub_pd(x3, _mm_mul_pd(tk3, ln_hi));
  __m128d lo0 = _mm_mul_pd(tk0, ln_lo);
  __m128d lo1 = _mm_mul_pd(tk1, ln_lo);
  __m128d lo2 = _mm_mul_pd(tk2, ln_lo);
  __m128d lo3 = _mm_mul_pd(tk3, ln_lo);
  __m128d r0 = _mm_sub_pd(hi0, lo0);
  __m128d r1 = _mm_sub_pd(hi1, lo1);
  __m128d r2 = _mm_sub_pd(hi2, lo2);
  __m128d r3 = _mm_sub_pd(hi3, lo3);
  __m128d t0 = _mm_mul_pd(r0, r0);
  __m128d t1 = _mm_mul_pd(r1, r1);
  __m128d t2 = _mm_mul_pd(r2, r2);
  __m128d t3 = _mm_mul_pd(r3, r3);
  const __m128d p5 = _mm_set1_pd(kP5);
  const __m128d p4c = _mm_set1_pd(kP4);
  const __m128d p3c = _mm_set1_pd(kP3);
  const __m128d p2c = _mm_set1_pd(kP2);
  const __m128d p1c = _mm_set1_pd(kP1);
  __m128d p0 = _mm_fmadd_pd(t0, p5, p4c);
  __m128d p1 = _mm_fmadd_pd(t1, p5, p4c);
  __m128d p2 = _mm_fmadd_pd(t2, p5, p4c);
  __m128d p3 = _mm_fmadd_pd(t3, p5, p4c);
  p0 = _mm_fmadd_pd(t0, p0, p3c);
  p1 = _mm_fmadd_pd(t1, p1, p3c);
  p2 = _mm_fmadd_pd(t2, p2, p3c);
  p3 = _mm_fmadd_pd(t3, p3, p3c);
  p0 = _mm_fmadd_pd(t0, p0, p2c);
  p1 = _mm_fmadd_pd(t1, p1, p2c);
  p2 = _mm_fmadd_pd(t2, p2, p2c);
  p3 = _mm_fmadd_pd(t3, p3, p2c);
  p0 = _mm_fmadd_pd(t0, p0, p1c);
  p1 = _mm_fmadd_pd(t1, p1, p1c);
  p2 = _mm_fmadd_pd(t2, p2, p1c);
  p3 = _mm_fmadd_pd(t3, p3, p1c);
  auto finish = [](__m128d x, __m128d t, __m128d p, __m128d hi, __m128d lo, __m128i k) {
    const __m128d c = _mm_sub_pd(x, _mm_mul_pd(t, p));
    const __m128d quot = _mm_div_pd(_mm_mul_pd(x, c), _mm_sub_pd(_mm_set1_pd(2.0), c));
    const __m128d y = _mm_sub_pd(_mm_set1_pd(1.0), _mm_sub_pd(_mm_sub_pd(lo, quot), hi));
    const __m128i yb = _mm_castpd_si128(y);
    const __m128i yexp = _mm_and_si128(_mm_srli_epi64(yb, 52), _mm_set1_epi64x(0x7ff));
    const __m128i ne = _mm_add_epi64(yexp, k);
    const __m128i packed = _mm_or_si128(
        _mm_and_si128(yb, _mm_set1_epi64x(static_cast<long long>(0x800fffffffffffffULL))),
        _mm_slli_epi64(ne, 52));
    return _mm_castsi128_pd(packed);
  };
  _mm_storeu_pd(out, finish(r0, t0, p0, hi0, lo0, k0));
  _mm_storeu_pd(out + 2, finish(r1, t1, p1, hi1, lo1, k1));
  _mm_storeu_pd(out + 4, finish(r2, t2, p2, hi2, lo2, k2));
  _mm_storeu_pd(out + 6, finish(r3, t3, p3, hi3, lo3, k3));
}

__attribute__((target("fma"), noinline)) void morse8_hw(const double *r2,
                                                       const MorseParams &p,
                                                       double *energy,
                                                       double *fscale) {
  if (p.use_switch != 0) {
    for (int i = 0; i < 8; ++i) {
      morse_term<true>(r2[i], p, energy[i], fscale[i]);
    }
    return;
  }
  alignas(16) double arg[8], ex[8];
  alignas(16) __m128d rr[4];
  const __m128d req = _mm_set1_pd(p.r_eq);
  const __m128d nrange = _mm_set1_pd(-p.range);
  for (int i = 0; i < 4; ++i) {
    const int off = i * 2;
    rr[i] = _mm_sqrt_pd(_mm_loadu_pd(r2 + off));
    const __m128d dr = _mm_sub_pd(rr[i], req);
    _mm_store_pd(arg + off, _mm_mul_pd(nrange, dr));
  }
  exp8_hw(arg, ex);
  const __m128d one = _mm_set1_pd(1.0);
  const __m128d depth = _mm_set1_pd(p.depth);
  const __m128d ndepth = _mm_set1_pd(-p.depth);
  const __m128d shift = _mm_set1_pd(p.shift);
  const __m128d two = _mm_set1_pd(p.two_de_a);
  for (int i = 0; i < 4; ++i) {
    const int off = i * 2;
    const __m128d exv = _mm_load_pd(ex + off);
    const __m128d md = _mm_sub_pd(one, exv);
    const __m128d dd = _mm_mul_pd(depth, md);
    const __m128d v = _mm_fmadd_pd(dd, md, ndepth);
    _mm_store_pd(energy + off, _mm_sub_pd(v, shift));
    const __m128d d_m1 = _mm_sub_pd(md, one);
    const __m128d num = _mm_mul_pd(_mm_mul_pd(two, md), d_m1);
    _mm_store_pd(fscale + off, _mm_div_pd(num, rr[i]));
  }
}

__attribute__((target("sse2"), noinline)) void morse8_soft(const double *r2,
                                                        const MorseParams &p,
                                                        double *energy,
                                                        double *fscale) {
  if (p.use_switch != 0) {
    for (int i = 0; i < 8; ++i) {
      morse_term<false>(r2[i], p, energy[i], fscale[i]);
    }
    return;
  }
  alignas(16) double arg[8], ex[8];
  alignas(16) __m128d rr[4];
  const __m128d req = _mm_set1_pd(p.r_eq);
  const __m128d nrange = _mm_set1_pd(-p.range);
  for (int i = 0; i < 4; ++i) {
    const int off = i * 2;
    rr[i] = _mm_sqrt_pd(_mm_loadu_pd(r2 + off));
    const __m128d dr = _mm_sub_pd(rr[i], req);
    _mm_store_pd(arg + off, _mm_mul_pd(nrange, dr));
  }
  exp8(arg, ex);
  const __m128d one = _mm_set1_pd(1.0);
  const __m128d depth = _mm_set1_pd(p.depth);
  const __m128d ndepth = _mm_set1_pd(-p.depth);
  const __m128d shift = _mm_set1_pd(p.shift);
  const __m128d two = _mm_set1_pd(p.two_de_a);
  for (int i = 0; i < 4; ++i) {
    const int off = i * 2;
    const __m128d exv = _mm_load_pd(ex + off);
    const __m128d md = _mm_sub_pd(one, exv);
    const __m128d dd = _mm_mul_pd(depth, md);
    const __m128d v = fma2_trunc(dd, md, ndepth);
    _mm_store_pd(energy + off, _mm_sub_pd(v, shift));
    const __m128d d_m1 = _mm_sub_pd(md, one);
    const __m128d num = _mm_mul_pd(_mm_mul_pd(two, md), d_m1);
    _mm_store_pd(fscale + off, _mm_div_pd(num, rr[i]));
  }
}

__attribute__((target("fma"), noinline)) void eam_sse(long n, const double *R,
                                                       const EamAtom *atoms,
                                                       const double *box, double *F,
                                                       double *U) {
  const double b0 = box[0];
  const double b1 = box[1];
  const double b2 = box[2];
  const double h0 = b0 * 0.5;
  const double h1 = b1 * 0.5;
  const double h2 = b2 * 0.5;
  const bool hw = detail::glibc_fma_active();
  struct Scratch {
    std::vector<double> drho;
    std::vector<double> dxs;
    std::vector<double> dys;
    std::vector<double> dzs;
    std::vector<double> rs;
    std::vector<double> r2s;
    std::vector<long> js;
  };
  thread_local Scratch scratch;
  const auto nn = static_cast<std::size_t>(n);
  if (scratch.drho.size() != nn * 3) {
    scratch.drho.assign(nn * 3, 0.0);
  }
  if (scratch.dxs.size() < nn) {
    scratch.dxs.resize(nn);
    scratch.dys.resize(nn);
    scratch.dzs.resize(nn);
    scratch.rs.resize(nn);
    scratch.r2s.resize(nn);
    scratch.js.resize(nn);
  }
  for (long i = 0; i < n; ++i) {
    const EamAtom &a = atoms[i];
    Part8 dens_part{};
    Part8 pair_part{};
    int nd = 0;
    int npair = 0;
    long ns = 0;
    const double xi = R[3 * i];
    const double yi = R[3 * i + 1];
    const double zi = R[3 * i + 2];
    auto consider = [&](long j, double dx, double dy, double dz, double r2) {
      if (j == i || !(r2 <= a.cutoff2)) {
        return;
      }
      const auto us = static_cast<std::size_t>(ns);
      scratch.dxs[us] = dx;
      scratch.dys[us] = dy;
      scratch.dzs[us] = dz;
      scratch.rs[us] = std::sqrt(r2);
      scratch.r2s[us] = r2;
      scratch.js[us] = j;
      ++ns;
    };
    long j = 0;
    for (; j + 1 < n; j += 2) {
      double dx0 = 0, dy0 = 0, dz0 = 0, dx1 = 0, dy1 = 0, dz1 = 0;
      if (j != i) {
        dx0 = xi - R[3 * j];
        dy0 = yi - R[3 * j + 1];
        dz0 = zi - R[3 * j + 2];
        half_box(dx0, dy0, dz0, b0, b1, b2, h0, h1, h2);
      }
      if (j + 1 != i) {
        dx1 = xi - R[3 * (j + 1)];
        dy1 = yi - R[3 * (j + 1) + 1];
        dz1 = zi - R[3 * (j + 1) + 2];
        half_box(dx1, dy1, dz1, b0, b1, b2, h0, h1, h2);
      }
      const __m128d dxv = _mm_set_pd(dx1, dx0);
      const __m128d dyv = _mm_set_pd(dy1, dy0);
      const __m128d dzv = _mm_set_pd(dz1, dz0);
      const __m128d r2v =
          hw ? _mm_fmadd_pd(dzv, dzv, _mm_fmadd_pd(dyv, dyv, _mm_mul_pd(dxv, dxv)))
             : fma2_trunc(dzv, dzv, fma2_trunc(dyv, dyv, _mm_mul_pd(dxv, dxv)));
      alignas(16) double r2a[2];
      _mm_store_pd(r2a, r2v);
      consider(j, dx0, dy0, dz0, r2a[0]);
      consider(j + 1, dx1, dy1, dz1, r2a[1]);
    }
    if (j < n && j != i) {
      double dx = xi - R[3 * j];
      double dy = yi - R[3 * j + 1];
      double dz = zi - R[3 * j + 2];
      half_box(dx, dy, dz, b0, b1, b2, h0, h1, h2);
      const double r2 = hw ? r2_fast<true>(dx, dy, dz)
                           : fma_trunc(dz, dz, fma_trunc(dy, dy, dx * dx));
      consider(j, dx, dy, dz, r2);
    }
    const double two_b2 = -2.0 * a.beta2;
    auto consume = [&](long nb, double dx, double dy, double dz, double r, double r2,
                       double eb1, double eb2, double ex) {
      const double r2sq = r2 * r2;
      const double r6 = r2sq * r2;
      const double r4 = r2sq;
      const double r5 = r4 * r;
      // 512 is a power of two, so the scale is exact and the add is the fma.
      const double rho_pair = 512.0 * eb2 + eb1;
      const double dens_term = r6 * rho_pair;
      const double t_a = (6.0 * r5) * rho_pair;
      const double neg_b1 = -a.beta1 * eb1;
      const double b2_term = (1024.0 * a.beta2) * eb2;
      const double inner = neg_b1 - b2_term;
      const double mag = hw ? __builtin_fma(r6, inner, t_a) : fma_trunc(r6, inner, t_a);
      const double inv_r = 1.0 / r;
      const double fscale = -mag * inv_r;
      const double fx = fscale * dx;
      const double fy = fscale * dy;
      const double fz = fscale * dz;
      const int kd = nd & 7;
      dens_part.e[kd] += dens_term;
      dens_part.fx[kd] += fx;
      dens_part.fy[kd] += fy;
      dens_part.fz[kd] += fz;
      ++nd;
      scratch.drho[static_cast<std::size_t>(3 * nb)] -= fx;
      scratch.drho[static_cast<std::size_t>(3 * nb + 1)] -= fy;
      scratch.drho[static_cast<std::size_t>(3 * nb + 2)] -= fz;
      if (nb > i) {
        const double md = 1.0 - ex;
        const double dd = a.d_m * md;
        const double phi = hw ? __builtin_fma(dd, md, -a.d_m) : fma_trunc(dd, md, -a.d_m);
        const double mag_f = ((2.0 * a.alpha_m) * a.d_m) * md * (md - 1.0);
        const double fcomp = mag_f * inv_r;
        const double pfx = fcomp * dx;
        const double pfy = fcomp * dy;
        const double pfz = fcomp * dz;
        const int kp = npair & 7;
        pair_part.e[kp] += phi;
        pair_part.fx[kp] += pfx;
        pair_part.fy[kp] += pfy;
        pair_part.fz[kp] += pfz;
        ++npair;
        F[3 * nb] += pfx;
        F[3 * nb + 1] += pfy;
        F[3 * nb + 2] += pfz;
      }
    };
    long t = 0;
    for (; t + 8 <= ns; t += 8) {
      alignas(16) double a1[8], a2[8], a3[8], e1[8], e2[8], e3[8];
      for (int k = 0; k < 8; ++k) {
        const double r = scratch.rs[static_cast<std::size_t>(t + k)];
        a1[k] = -a.beta1 * r;
        a2[k] = two_b2 * r;
        a3[k] = -a.alpha_m * (r - a.r_m);
      }
      if (hw) {
        exp8_hw(a1, e1);
        exp8_hw(a2, e2);
        exp8_hw(a3, e3);
      } else {
        exp8(a1, e1);
        exp8(a2, e2);
        exp8(a3, e3);
      }
      for (int k = 0; k < 8; ++k) {
        const auto us = static_cast<std::size_t>(t + k);
        consume(scratch.js[us], scratch.dxs[us], scratch.dys[us], scratch.dzs[us],
                scratch.rs[us], scratch.r2s[us], e1[k], e2[k], e3[k]);
      }
    }
    for (; t < ns; ++t) {
      const auto us = static_cast<std::size_t>(t);
      const double r = scratch.rs[us];
      const double eb1 = exp(-a.beta1 * r);
      const double eb2 = exp(two_b2 * r);
      const double ex = exp(-a.alpha_m * (r - a.r_m));
      consume(scratch.js[us], scratch.dxs[us], scratch.dys[us], scratch.dzs[us], r,
              scratch.r2s[us], eb1, eb2, ex);
    }
    const double dens = combine8(dens_part.e);
    scratch.drho[static_cast<std::size_t>(3 * i)] += combine8(dens_part.fx);
    scratch.drho[static_cast<std::size_t>(3 * i + 1)] += combine8(dens_part.fy);
    scratch.drho[static_cast<std::size_t>(3 * i + 2)] += combine8(dens_part.fz);
    *U += combine8(pair_part.e);
    *U += embed_horner<false>(a.coeff, dens);
    F[3 * i] -= combine8(pair_part.fx);
    F[3 * i + 1] -= combine8(pair_part.fy);
    F[3 * i + 2] -= combine8(pair_part.fz);
    const double dF = embed_force<false>(a.coeff, dens);
    auto apply_rho = [&](long atom) {
      const std::size_t base = static_cast<std::size_t>(3 * atom);
      F[base] -= dF * scratch.drho[base];
      F[base + 1] -= dF * scratch.drho[base + 1];
      F[base + 2] -= dF * scratch.drho[base + 2];
      scratch.drho[base] = 0.0;
      scratch.drho[base + 1] = 0.0;
      scratch.drho[base + 2] = 0.0;
    };
    apply_rho(i);
    for (long u = 0; u < ns; ++u) {
      apply_rho(scratch.js[static_cast<std::size_t>(u)]);
    }
  }
}

__attribute__((target("fma"), noinline)) void exp_hw_n(const double *in,
                                                       double *out,
                                                       std::size_t n) {
  std::size_t i = 0;
  for (; i + 2 <= n; i += 2) {
    const double a = exp_fast<true>(in[i]);
    const double b = exp_fast<true>(in[i + 1]);
    out[i] = a;
    out[i + 1] = b;
  }
  for (; i < n; ++i) {
    out[i] = exp_fast<true>(in[i]);
  }
}

#endif

} // namespace rgpot::simd
