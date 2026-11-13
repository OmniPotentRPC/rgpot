// Included from the SSE2 and SSE4 Highway passes. No include guard: each
// pass needs its own static copy under that pass's target pragma.
// r+rem matches a hardware fma on normals; a half-ulp tie, a zero, a
// subnormal, or a non-finite lane is repaired with exact_fma.

#include <emmintrin.h>

__attribute__((noinline)) static __m128d fma2_repair(__m128d a, __m128d b, __m128d c,
                                                        __m128d y, int mask) {
  alignas(16) double aa[2], bb[2], cc[2], yy[2];
  _mm_store_pd(aa, a);
  _mm_store_pd(bb, b);
  _mm_store_pd(cc, c);
  _mm_store_pd(yy, y);
  if ((mask & 1) != 0) {
    yy[0] = exact_fma(aa[0], bb[0], cc[0]);
  }
  if ((mask & 2) != 0) {
    yy[1] = exact_fma(aa[1], bb[1], cc[1]);
  }
  return _mm_load_pd(yy);
}

__attribute__((always_inline)) static inline __m128d fma2(__m128d a, __m128d b,
                                                          __m128d c) {
  const __m128d split = _mm_set1_pd(detail::kFmaSplit);
  const __m128d p = _mm_mul_pd(a, b);
  const __m128d ca = _mm_mul_pd(split, a);
  const __m128d ah = _mm_sub_pd(ca, _mm_sub_pd(ca, a));
  const __m128d al = _mm_sub_pd(a, ah);
  const __m128d cb = _mm_mul_pd(split, b);
  const __m128d bh = _mm_sub_pd(cb, _mm_sub_pd(cb, b));
  const __m128d bl = _mm_sub_pd(b, bh);
  const __m128d t1 = _mm_add_pd(_mm_sub_pd(_mm_mul_pd(ah, bh), p), _mm_mul_pd(ah, bl));
  const __m128d pe = _mm_add_pd(_mm_add_pd(t1, _mm_mul_pd(al, bh)), _mm_mul_pd(al, bl));

  auto two_sum = [](__m128d x, __m128d y, __m128d &s, __m128d &e) {
    s = _mm_add_pd(x, y);
    const __m128d z = _mm_sub_pd(s, x);
    e = _mm_add_pd(_mm_sub_pd(x, _mm_sub_pd(s, z)), _mm_sub_pd(y, z));
  };
  __m128d s, t, v, z, r, h, rem, u;
  two_sum(p, c, s, t);
  two_sum(t, pe, v, z);
  two_sum(s, v, r, h);
  two_sum(h, z, rem, u);
  (void)u;
  const __m128d y = _mm_add_pd(r, rem);

  const __m128i sign = _mm_set1_epi64x(static_cast<long long>(0x8000000000000000ULL));
  const __m128i rb = _mm_castpd_si128(r);
  const __m128i be = _mm_and_si128(_mm_srli_epi64(rb, 52), _mm_set1_epi64x(0x7ff));
  const __m128i mant = _mm_and_si128(rb, _mm_set1_epi64x(0xfffffffffffffLL));
  const __m128i mz = _mm_cmpeq_epi32(mant, _mm_setzero_si128());
  const __m128i pow2 = _mm_and_si128(mz, _mm_shuffle_epi32(mz, 0xB1));
  const __m128i remb = _mm_castpd_si128(rem);
  const __m128i same32 =
      _mm_cmpeq_epi32(_mm_xor_si128(_mm_and_si128(remb, sign), _mm_and_si128(rb, sign)),
                      _mm_setzero_si128());
  const __m128i same = _mm_and_si128(same32, _mm_shuffle_epi32(same32, 0xB1));
  const __m128i adj = _mm_andnot_si128(same, pow2);
  __m128i mid_be = _mm_sub_epi64(be, _mm_set1_epi64x(53));
  mid_be = _mm_sub_epi64(mid_be, _mm_and_si128(adj, _mm_set1_epi64x(1)));
  const __m128i mb = _mm_or_si128(_mm_slli_epi64(mid_be, 52), _mm_and_si128(remb, sign));
  const __m128d half = _mm_castsi128_pd(mb);
  const __m128d abs_mask = _mm_castsi128_pd(_mm_set1_epi64x(0x7fffffffffffffffLL));
  const __m128d tie = _mm_cmpeq_pd(_mm_and_pd(rem, abs_mask), _mm_and_pd(half, abs_mask));
  const __m128i yb = _mm_castpd_si128(y);
  const __m128i ye = _mm_and_si128(_mm_srli_epi64(yb, 52), _mm_set1_epi64x(0x7ff));
  const __m128i ye0 = _mm_cmpeq_epi32(ye, _mm_setzero_si128());
  const __m128i ye_bad32 = _mm_or_si128(ye0, _mm_cmpeq_epi32(ye, _mm_set1_epi64x(0x7ff)));
  const __m128i ye_bad = _mm_and_si128(ye_bad32, _mm_shuffle_epi32(ye_bad32, 0xB1));
  const __m128d bad = _mm_or_pd(tie, _mm_castsi128_pd(ye_bad));
  const int mask = _mm_movemask_pd(bad);
  if (mask == 0) {
    return y;
  }
  return fma2_repair(a, b, c, y, mask);
}

// r+rem only. The exp reduction and polynomial do not land on a half-ulp tie,
// so the repair in fma2 is not on this path.
__attribute__((always_inline)) static inline __m128d fma2_trunc(__m128d a, __m128d b,
                                                                __m128d c) {
  const __m128d split = _mm_set1_pd(detail::kFmaSplit);
  const __m128d p = _mm_mul_pd(a, b);
  const __m128d ca = _mm_mul_pd(split, a);
  const __m128d ah = _mm_sub_pd(ca, _mm_sub_pd(ca, a));
  const __m128d al = _mm_sub_pd(a, ah);
  const __m128d cb = _mm_mul_pd(split, b);
  const __m128d bh = _mm_sub_pd(cb, _mm_sub_pd(cb, b));
  const __m128d bl = _mm_sub_pd(b, bh);
  const __m128d t1 = _mm_add_pd(_mm_sub_pd(_mm_mul_pd(ah, bh), p), _mm_mul_pd(ah, bl));
  const __m128d pe = _mm_add_pd(_mm_add_pd(t1, _mm_mul_pd(al, bh)), _mm_mul_pd(al, bl));
  auto two_sum = [](__m128d x, __m128d y, __m128d &s, __m128d &e) {
    s = _mm_add_pd(x, y);
    const __m128d z = _mm_sub_pd(s, x);
    e = _mm_add_pd(_mm_sub_pd(x, _mm_sub_pd(s, z)), _mm_sub_pd(y, z));
  };
  __m128d s, t, v, z, r, h, rem, u;
  two_sum(p, c, s, t);
  two_sum(t, pe, v, z);
  two_sum(s, v, r, h);
  two_sum(h, z, rem, u);
  (void)u;
  return _mm_add_pd(r, rem);
}

// Eight squared distances. The four Dekker chains overlap; a squared
// distance does not land on the half-ulp tie that fma2 repairs.
__attribute__((always_inline)) static inline void r2_8_trunc(const double *dx,
                                                            const double *dy,
                                                            const double *dz,
                                                            double *out) {
  const __m128d x0 = _mm_loadu_pd(dx);
  const __m128d x1 = _mm_loadu_pd(dx + 2);
  const __m128d x2 = _mm_loadu_pd(dx + 4);
  const __m128d x3 = _mm_loadu_pd(dx + 6);
  const __m128d y0 = _mm_loadu_pd(dy);
  const __m128d y1 = _mm_loadu_pd(dy + 2);
  const __m128d y2 = _mm_loadu_pd(dy + 4);
  const __m128d y3 = _mm_loadu_pd(dy + 6);
  const __m128d z0 = _mm_loadu_pd(dz);
  const __m128d z1 = _mm_loadu_pd(dz + 2);
  const __m128d z2 = _mm_loadu_pd(dz + 4);
  const __m128d z3 = _mm_loadu_pd(dz + 6);
  const __m128d r0 = fma2_trunc(z0, z0, fma2_trunc(y0, y0, _mm_mul_pd(x0, x0)));
  const __m128d r1 = fma2_trunc(z1, z1, fma2_trunc(y1, y1, _mm_mul_pd(x1, x1)));
  const __m128d r2 = fma2_trunc(z2, z2, fma2_trunc(y2, y2, _mm_mul_pd(x2, x2)));
  const __m128d r3 = fma2_trunc(z3, z3, fma2_trunc(y3, y3, _mm_mul_pd(x3, x3)));
  _mm_storeu_pd(out, r0);
  _mm_storeu_pd(out + 2, r1);
  _mm_storeu_pd(out + 4, r2);
  _mm_storeu_pd(out + 6, r3);
}

// fdlibm exp for |x| below the overflow threshold. The large reduction
// matches exp() on that range, including the small-argument bins.
__attribute__((always_inline)) static inline __m128d exp2_finite(__m128d x_in) {
  const __m128i sign = _mm_and_si128(_mm_castpd_si128(x_in),
                                     _mm_set1_epi64x(static_cast<long long>(0x8000000000000000ULL)));
  const __m128d half = _mm_or_pd(_mm_set1_pd(0.5), _mm_castsi128_pd(sign));
  const __m128d scaled = fma2_trunc(_mm_set1_pd(kInvLn2), x_in, half);
  const __m128i k32 = _mm_cvttpd_epi32(scaled);
  const __m128d tk = _mm_cvtepi32_pd(k32);
  const __m128i k64 = _mm_unpacklo_epi32(k32, _mm_srai_epi32(k32, 31));
  const __m128d hi = _mm_sub_pd(x_in, _mm_mul_pd(tk, _mm_set1_pd(kLn2Hi)));
  const __m128d lo = _mm_mul_pd(tk, _mm_set1_pd(kLn2Lo));
  const __m128d x = _mm_sub_pd(hi, lo);
  const __m128d t = _mm_mul_pd(x, x);
  __m128d p = fma2_trunc(t, _mm_set1_pd(kP5), _mm_set1_pd(kP4));
  p = fma2_trunc(t, p, _mm_set1_pd(kP3));
  p = fma2_trunc(t, p, _mm_set1_pd(kP2));
  p = fma2_trunc(t, p, _mm_set1_pd(kP1));
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

__attribute__((always_inline)) static inline __m128d exp2(__m128d x_in) {
  const __m128d ax = _mm_andnot_pd(_mm_set1_pd(-0.0), x_in);
  // hx >= 0x40862e42, the fdlibm overflow/underflow/non-finite bin.
  const __m128d lim = _mm_castsi128_pd(_mm_set1_epi64x(0x40862e4200000000LL));
  const __m128d bad = _mm_or_pd(_mm_cmpge_pd(ax, lim), _mm_cmpunord_pd(x_in, x_in));
  if (_mm_movemask_pd(bad) == 0) {
    return exp2_finite(x_in);
  }
  alignas(16) double in[2], out[2];
  _mm_store_pd(in, x_in);
  out[0] = exp(in[0]);
  out[1] = exp(in[1]);
  const int mask = _mm_movemask_pd(bad);
  if (mask != 3) {
    const __m128d fast = exp2_finite(x_in);
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

// Eight independent exps. The Horner steps of the four vectors are issued
// together so the Dekker latency overlaps.
__attribute__((always_inline)) static inline void exp8(const double *in, double *out) {
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
  __m128d s0 = fma2_trunc(inv, x0, half_of(x0));
  __m128d s1 = fma2_trunc(inv, x1, half_of(x1));
  __m128d s2 = fma2_trunc(inv, x2, half_of(x2));
  __m128d s3 = fma2_trunc(inv, x3, half_of(x3));
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
  __m128d p0 = fma2_trunc(t0, p5, p4c);
  __m128d p1 = fma2_trunc(t1, p5, p4c);
  __m128d p2 = fma2_trunc(t2, p5, p4c);
  __m128d p3 = fma2_trunc(t3, p5, p4c);
  p0 = fma2_trunc(t0, p0, p3c);
  p1 = fma2_trunc(t1, p1, p3c);
  p2 = fma2_trunc(t2, p2, p3c);
  p3 = fma2_trunc(t3, p3, p3c);
  p0 = fma2_trunc(t0, p0, p2c);
  p1 = fma2_trunc(t1, p1, p2c);
  p2 = fma2_trunc(t2, p2, p2c);
  p3 = fma2_trunc(t3, p3, p2c);
  p0 = fma2_trunc(t0, p0, p1c);
  p1 = fma2_trunc(t1, p1, p1c);
  p2 = fma2_trunc(t2, p2, p1c);
  p3 = fma2_trunc(t3, p3, p1c);
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
