// MIT License
// Copyright 2023--present rgpot developers

#include "rgpot/simd/hwy_disabled.h"

#include "rgpot/simd/exp.hpp"
#include "rgpot/simd/pair_loops.hpp"

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "rgpot/simd/hwy_kernels.cc"
#include "hwy/foreach_target.h" // IWYU pragma: keep
#include "hwy/highway.h"

#include <cmath>
#include <cstdint>
#include <vector>

#if defined(__x86_64__)
namespace rgpot::simd {
__m128d exp2_hw(__m128d x);
void morse8_hw(const double *r2, const MorseParams &params, double *energy,
               double *fscale);
void morse8_soft(const double *r2, const MorseParams &params, double *energy,
                 double *fscale);
std::size_t accept_row_hw(const PairView &v, std::size_t i, double *dxb,
                          double *dyb, double *dzb, double *r2b,
                          std::int32_t *jb);
}
#endif

HWY_BEFORE_NAMESPACE();
namespace rgpot::simd {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

#if defined(__x86_64__) && (HWY_TARGET == HWY_SSE2 || HWY_TARGET == HWY_SSE4)
#include "rgpot/simd/sse_soft.hpp"
#endif

// Hardware fused multiply-add where the target has one. Otherwise the
// same Dekker product and rounded sum as exact_fma, on the whole vector.
// A lane outside that domain, or a tie the fast sum cannot close, drops
// to exact_fma so the bits stay those of the hardware instruction.
#if !HWY_NATIVE_FMA
template <class V>
HWY_INLINE void two_sum_lanes(V a, V b, V &s, V &e) {
  s = hn::Add(a, b);
  const auto z = hn::Sub(s, a);
  e = hn::Add(hn::Sub(a, hn::Sub(s, z)), hn::Sub(b, z));
}

template <class D>
HWY_INLINE void two_prod_lanes(D d, hn::Vec<D> a, hn::Vec<D> b, hn::Vec<D> &p,
                               hn::Vec<D> &e) {
  const auto split = hn::Set(d, detail::kFmaSplit);
  p = hn::Mul(a, b);
  const auto ca = hn::Mul(split, a);
  const auto ah = hn::Sub(ca, hn::Sub(ca, a));
  const auto al = hn::Sub(a, ah);
  const auto cb = hn::Mul(split, b);
  const auto bh = hn::Sub(cb, hn::Sub(cb, b));
  const auto bl = hn::Sub(b, bh);
  const auto t1 = hn::Add(hn::Sub(hn::Mul(ah, bh), p), hn::Mul(ah, bl));
  e = hn::Add(hn::Add(t1, hn::Mul(al, bh)), hn::Mul(al, bl));
}

template <class D>
HWY_INLINE bool lanes_in_fma_domain(D d, hn::Vec<D> a, hn::Vec<D> b,
                                    hn::Vec<D> c) {
  const auto finite = hn::And(hn::IsFinite(a), hn::And(hn::IsFinite(b), hn::IsFinite(c)));
  auto in_mag = [&](hn::Vec<D> v, double lo, double hi) {
    const auto ax = hn::Abs(v);
    const auto zero = hn::Eq(ax, hn::Zero(d));
    return hn::And(hn::Le(ax, hn::Set(d, hi)),
                   hn::Or(zero, hn::Ge(ax, hn::Set(d, lo))));
  };
  const auto mag = hn::And(in_mag(a, 0x1p-480, 0x1p480),
                           hn::And(in_mag(b, 0x1p-480, 0x1p480),
                                   in_mag(c, 0x1p-900, 0x1p900)));
  return hn::AllTrue(d, hn::And(finite, mag));
}
#endif

#if !HWY_NATIVE_FMA
// Branchless Dekker product and correctly rounded sum. Inputs are finite
// normals inside fma_dekker_domain. A zero, a subnormal, or a non-finite
// lane is not meaningful here; FmaArray repairs those.
template <class D>
HWY_INLINE hn::Vec<D> fma_rounded(D d, hn::Vec<D> a, hn::Vec<D> b, hn::Vec<D> c) {
  using V = hn::Vec<D>;
  const hn::RebindToUnsigned<D> du;
  V pe, t, z, h, u, err, sl, e2;
  V p, s, v, r, rem, pred, sh, d2;
  two_prod_lanes(d, a, b, p, pe);
  two_sum_lanes(p, c, s, t);
  two_sum_lanes(t, pe, v, z);
  two_sum_lanes(s, v, r, h);
  two_sum_lanes(h, z, rem, u);
  two_sum_lanes(r, rem, pred, err);
  two_sum_lanes(err, u, sh, sl);

  const auto ub = hn::BitCast(du, pred);
  const auto be = hn::And(hn::ShiftRight<52>(ub), hn::Set(du, std::uint64_t{0x7ff}));
  const auto mant = hn::And(ub, hn::Set(du, std::uint64_t{0xfffffffffffffULL}));
  const auto pow2 = hn::Eq(mant, hn::Zero(du));
  const auto sign = hn::Set(du, std::uint64_t{0x8000000000000000ULL});
  const auto sh_bits = hn::BitCast(du, sh);
  const auto same = hn::Eq(hn::Xor(hn::And(sh_bits, sign), hn::And(ub, sign)),
                           hn::Zero(du));
  const auto adj = hn::And(pow2, hn::Not(same));
  auto mid_be = hn::Sub(be, hn::Set(du, std::uint64_t{53}));
  mid_be = hn::Sub(mid_be, hn::IfThenElseZero(adj, hn::Set(du, std::uint64_t{1})));
  auto mb = hn::ShiftLeft<52>(mid_be);
  mb = hn::Or(mb, hn::And(sh_bits, sign));
  const auto mid = hn::BitCast(d, mb);

  const auto diff = hn::Sub(sh, mid);
  two_sum_lanes(diff, sl, d2, e2);
  const auto zero_d = hn::Zero(d);
  const auto d2_0 = hn::Eq(d2, zero_d);
  const auto e2_0 = hn::Eq(e2, zero_d);
  const auto tie = hn::And(d2_0, e2_0);
  const auto d2_bits = hn::BitCast(du, d2);
  const auto e2_bits = hn::BitCast(du, e2);
  const auto d2_same =
      hn::Eq(hn::Xor(hn::And(d2_bits, sign), hn::And(mb, sign)), hn::Zero(du));
  const auto e2_same =
      hn::Eq(hn::Xor(hn::And(e2_bits, sign), hn::And(mb, sign)), hn::Zero(du));
  const auto d2_same_d = hn::RebindMask(d, d2_same);
  const auto e2_same_d = hn::RebindMask(d, e2_same);
  const auto up_d = hn::And(d2_same_d, hn::Not(d2_0));
  const auto up_e = hn::And(d2_0, hn::And(e2_same_d, hn::Not(e2_0)));
  const auto up = hn::And(hn::Not(tie), hn::Or(up_d, up_e));

  const auto nb_up = hn::Add(ub, hn::Set(du, std::uint64_t{1}));
  const auto nb_dn = hn::Sub(ub, hn::Set(du, std::uint64_t{1}));
  const auto nxt = hn::BitCast(d, hn::IfThenElse(same, nb_up, nb_dn));

  V result = pred;
  result = hn::IfThenElse(hn::RebindMask(d, tie), hn::Add(pred, mid), result);
  result = hn::IfThenElse(hn::RebindMask(d, up), nxt, result);
  const auto sh0 = hn::Eq(sh, zero_d);
  const auto pred0 = hn::Eq(pred, zero_d);
  const auto bad = hn::Or(hn::Or(sh0, pred0), hn::Or(hn::IsNaN(pred), hn::IsNaN(sh)));
  const auto simple = hn::Add(pred, hn::Add(sh, sl));
  return hn::IfThenElse(bad, simple, result);
}
#endif

template <class D>
HWY_INLINE hn::Vec<D> fma_lanes(D d, hn::Vec<D> a, hn::Vec<D> b, hn::Vec<D> c) {
#if HWY_NATIVE_FMA
  (void)d;
  return hn::MulAdd(a, b, c);
#elif defined(__x86_64__) && (HWY_TARGET == HWY_SSE2 || HWY_TARGET == HWY_SSE4)
  (void)d;
  hn::Vec<D> out;
  out.raw = fma2(a.raw, b.raw, c.raw);
  return out;
#else
  return fma_rounded(d, a, b, c);
#endif
}

// __FMA__ follows the command line, not a Highway target pragma. On an AVX
// clone the pragma still enables the instruction, so the builtin is the
// instruction. A glibc ifunc would follow the hwcaps mask.
HWY_INLINE double r2_scalar(double dx, double dy, double dz) {
#if HWY_NATIVE_FMA
  return __builtin_fma(dz, dz, __builtin_fma(dy, dy, dx * dx));
#elif defined(__x86_64__)
  if (detail::glibc_fma_active()) {
    return r2_hw(dx, dy, dz);
  }
  return fma_trunc(dz, dz, fma_trunc(dy, dy, dx * dx));
#else
  return r2_of(dx, dy, dz);
#endif
}

template <class D>
HWY_INLINE hn::Vec<D> r2_vec(D d, hn::Vec<D> dx, hn::Vec<D> dy, hn::Vec<D> dz) {
#if !HWY_NATIVE_FMA
  if (detail::glibc_fma_active()) {
    alignas(64) double dxa[8], dya[8], dza[8], out[8];
    hn::Store(dx, d, dxa);
    hn::Store(dy, d, dya);
    hn::Store(dz, d, dza);
    const std::size_t lanes = hn::Lanes(d);
    r2_hw_n(dxa, dya, dza, out, lanes);
    return hn::Load(d, out);
  }
#endif
  return fma_lanes(d, dz, dz, fma_lanes(d, dy, dy, hn::Mul(dx, dx)));
}

template <class D> HWY_INLINE hn::Vec<D> round_finite(D d, hn::Vec<D> t) {
  const auto half = hn::CopySign(hn::Set(d, 0.5), t);
  const auto shifted = hn::Add(t, half);
  const hn::Rebind<std::int64_t, D> di;
  const auto as_i = hn::ConvertTo(di, shifted);
  const auto back = hn::ConvertTo(d, as_i);
  const auto bad = hn::Or(hn::IsNaN(t), hn::IsInf(t));
  const auto huge = hn::Ge(hn::Abs(shifted), hn::Set(d, 9223372036854775808.0));
  return hn::IfThenElse(hn::Or(bad, huge), hn::Zero(d), back);
}

template <class D>
HWY_INLINE hn::Vec<D> fold_axis(D d, hn::Vec<D> delta, double width,
                                double inv) {
  const auto scaled = hn::Mul(delta, hn::Set(d, inv));
  const auto nimg = round_finite(d, scaled);
  const auto shift = hn::Mul(hn::Set(d, width), nimg);
  return hn::Sub(delta, shift);
}

// fdlibm e_exp, one lane-wise replay of exp() in exp.cc.
// Product-sums are fused. Special lanes are selected after the
// polynomial so a quiet NaN, an infinity, and an overflow keep the scalar bits.
template <class D> HWY_INLINE hn::Vec<D> exp_large(D d, hn::Vec<D> x_in) {
  // Every lane has already been shown to take fdlibm's large reduction
  // and the k != 0 reconstruction. Same operations as exp().
  const hn::Rebind<std::int64_t, D> di;
  const hn::RebindToUnsigned<D> du;
  const auto half = hn::CopySign(hn::Set(d, 0.5), x_in);
  const auto scaled = fma_lanes(d, hn::Set(d, kInvLn2), x_in, half);
  const auto k = hn::ConvertTo(di, scaled);
  const auto tk = hn::ConvertTo(d, k);
  const auto t_hi = hn::Mul(tk, hn::Set(d, kLn2Hi));
  const auto t_lo = hn::Mul(tk, hn::Set(d, kLn2Lo));
  const auto hi = hn::Sub(x_in, t_hi);
  const auto lo = t_lo;
  const auto x = hn::Sub(hi, lo);
  const auto t = hn::Mul(x, x);
  const auto p4 = fma_lanes(d, t, hn::Set(d, kP5), hn::Set(d, kP4));
  const auto p3 = fma_lanes(d, t, p4, hn::Set(d, kP3));
  const auto p2 = fma_lanes(d, t, p3, hn::Set(d, kP2));
  const auto p1 = fma_lanes(d, t, p2, hn::Set(d, kP1));
  const auto poly = hn::Mul(t, p1);
  const auto c = hn::Sub(x, poly);
  const auto xc = hn::Mul(x, c);
  const auto den = hn::Sub(hn::Set(d, 2.0), c);
  const auto quot = hn::Div(xc, den);
  const auto loq = hn::Sub(lo, quot);
  const auto hiq = hn::Sub(loq, hi);
  const auto y = hn::Sub(hn::Set(d, 1.0), hiq);
  const auto ybits = hn::BitCast(du, y);
  const auto yexp = hn::And(hn::ShiftRight<52>(ybits), hn::Set(du, 0x7ff));
  const auto new_exp = hn::Add(hn::BitCast(di, yexp), k);
  const auto mant_y = hn::And(ybits, hn::Set(du, 0x800fffffffffffffULL));
  const auto packed =
      hn::Or(mant_y, hn::ShiftLeft<52>(hn::BitCast(du, new_exp)));
  return hn::BitCast(d, packed);
}

template <class D> HWY_INLINE hn::Vec<D> exp_lanes(D d, hn::Vec<D> x_in) {
#if !HWY_NATIVE_FMA
  // One call holds every fused product of this exp, so the SSE kernel does
  // not pay a libcall per Horner step when the processor's FMA is usable.
  if (detail::glibc_fma_active()) {
#if defined(__x86_64__) && (HWY_TARGET == HWY_SSE2 || HWY_TARGET == HWY_SSE4)
    hn::Vec<D> fast;
    fast.raw = exp2_hw(x_in.raw);
    return fast;
#else
    alignas(64) double in[8], out[8];
    hn::Store(x_in, d, in);
    const std::size_t lanes = hn::Lanes(d);
    exp_hw_n(in, out, lanes);
    return hn::Load(d, out);
#endif
  }
#if defined(__x86_64__) && (HWY_TARGET == HWY_SSE2 || HWY_TARGET == HWY_SSE4)
  hn::Vec<D> fast;
  fast.raw = exp2(x_in.raw);
  return fast;
#endif
#endif
  const hn::RebindToUnsigned<D> du;
  const hn::Rebind<std::int64_t, D> di;
  const auto bits = hn::BitCast(du, x_in);
  const auto hx = hn::And(hn::ShiftRight<32>(bits), hn::Set(du, 0x7fffffffu));
  const auto lx = hn::And(bits, hn::Set(du, 0xffffffffu));
  const auto neg = hn::Ne(hn::ShiftRight<63>(bits), hn::Zero(du));
  const auto inf_or_nan = hn::Ge(hx, hn::Set(du, 0x7ff00000u));
  const auto mant = hn::Or(hn::And(hx, hn::Set(du, 0xfffffu)), lx);
  const auto is_nan = hn::And(inf_or_nan, hn::Ne(mant, hn::Zero(du)));
  const auto is_inf = hn::And(inf_or_nan, hn::Eq(mant, hn::Zero(du)));
  const auto big = hn::Ge(hx, hn::Set(du, 0x40862e42u));
  const auto finite_big = hn::And(big, hn::Not(inf_or_nan));
  const auto overflow = hn::And(
      finite_big, hn::RebindMask(du, hn::Gt(x_in, hn::Set(d, kOverflow))));
  const auto underflow = hn::And(
      finite_big, hn::RebindMask(du, hn::Lt(x_in, hn::Set(d, kUnderflow))));
  const auto special =
      hn::Or(hn::Or(is_nan, is_inf), hn::Or(overflow, underflow));
  const auto x_safe =
      hn::IfThenElse(hn::RebindMask(d, special), hn::Zero(d), x_in);

  const auto do_small = hn::And(hn::Gt(hx, hn::Set(du, 0x3fd62e42u)),
                                hn::Lt(hx, hn::Set(du, 0x3ff0a2b2u)));
  const auto do_large = hn::Ge(hx, hn::Set(du, 0x3ff0a2b2u));
  // Pair arguments are usually past one ln(2). Skip the k == 0 division
  // when every lane takes the large reduction. A mixed chunk uses both.
  if (hn::AllTrue(du, do_large) && hn::AllFalse(du, special)) {
    return exp_large(d, x_in);
  }

  const auto ln2_hi = hn::IfThenElse(
      hn::RebindMask(d, neg), hn::Neg(hn::Set(d, kLn2Hi)), hn::Set(d, kLn2Hi));
  const auto ln2_lo = hn::IfThenElse(
      hn::RebindMask(d, neg), hn::Neg(hn::Set(d, kLn2Lo)), hn::Set(d, kLn2Lo));
  const auto hi_s = hn::Sub(x_in, ln2_hi);
  const auto lo_s = ln2_lo;
  const auto k_s =
      hn::IfThenElse(hn::RebindMask(di, neg), hn::Set(di, std::int64_t{-1}),
                     hn::Set(di, std::int64_t{1}));

  const auto half = hn::CopySign(hn::Set(d, 0.5), x_safe);
  const auto scaled = fma_lanes(d, hn::Set(d, kInvLn2), x_safe, half);
  const auto k_l = hn::ConvertTo(di, scaled);
  const auto tk = hn::ConvertTo(d, k_l);
  const auto t_hi = hn::Mul(tk, hn::Set(d, kLn2Hi));
  const auto t_lo = hn::Mul(tk, hn::Set(d, kLn2Lo));
  const auto hi_l = hn::Sub(x_safe, t_hi);
  const auto lo_l = t_lo;

  auto hi = hn::Zero(d);
  auto lo = hn::Zero(d);
  auto k = hn::Zero(di);
  hi = hn::IfThenElse(hn::RebindMask(d, do_small), hi_s, hi);
  lo = hn::IfThenElse(hn::RebindMask(d, do_small), lo_s, lo);
  k = hn::IfThenElse(hn::RebindMask(di, do_small), k_s, k);
  hi = hn::IfThenElse(hn::RebindMask(d, do_large), hi_l, hi);
  lo = hn::IfThenElse(hn::RebindMask(d, do_large), lo_l, lo);
  k = hn::IfThenElse(hn::RebindMask(di, do_large), k_l, k);
  const auto reduced = hn::Or(do_small, do_large);
  const auto x =
      hn::IfThenElse(hn::RebindMask(d, reduced), hn::Sub(hi, lo), x_in);

  const auto t = hn::Mul(x, x);
  const auto p4 = fma_lanes(d, t, hn::Set(d, kP5), hn::Set(d, kP4));
  const auto p3 = fma_lanes(d, t, p4, hn::Set(d, kP3));
  const auto p2 = fma_lanes(d, t, p3, hn::Set(d, kP2));
  const auto p1 = fma_lanes(d, t, p2, hn::Set(d, kP1));
  const auto poly = hn::Mul(t, p1);
  const auto c = hn::Sub(x, poly);

  const auto xc = hn::Mul(x, c);
  const auto den0 = hn::Sub(c, hn::Set(d, 2.0));
  const auto quot0 = hn::Div(xc, den0);
  const auto inner0 = hn::Sub(quot0, x);
  const auto res0 = hn::Sub(hn::Set(d, 1.0), inner0);

  const auto den1 = hn::Sub(hn::Set(d, 2.0), c);
  const auto quot1 = hn::Div(xc, den1);
  const auto loq = hn::Sub(lo, quot1);
  const auto hiq = hn::Sub(loq, hi);
  const auto y = hn::Sub(hn::Set(d, 1.0), hiq);
  const auto ybits = hn::BitCast(du, y);
  const auto yexp = hn::And(hn::ShiftRight<52>(ybits), hn::Set(du, 0x7ff));
  const auto new_exp = hn::Add(hn::BitCast(di, yexp), k);
  const auto mant_y = hn::And(ybits, hn::Set(du, 0x800fffffffffffffULL));
  const auto packed =
      hn::Or(mant_y, hn::ShiftLeft<52>(hn::BitCast(du, new_exp)));
  const auto res1 = hn::BitCast(d, packed);

  const auto k0 = hn::Eq(k, hn::Zero(di));
  auto result = hn::IfThenElse(hn::RebindMask(d, k0), res0, res1);

  const auto tiny_hx = hn::Lt(hx, hn::Set(du, 0x3e300000u));
  const auto huge = hn::Set(d, 1.0e300);
  const auto tiny_ok =
      hn::RebindMask(du, hn::Gt(hn::Add(huge, x_in), hn::Set(d, 1.0)));
  const auto use_tiny = hn::And(tiny_hx, tiny_ok);
  result = hn::IfThenElse(hn::RebindMask(d, use_tiny),
                          hn::Add(hn::Set(d, 1.0), x_in), result);

  const auto tiny = hn::Set(d, 9.33263618503218878990e-302);
  result =
      hn::IfThenElse(hn::RebindMask(d, underflow), hn::Mul(tiny, tiny), result);
  result =
      hn::IfThenElse(hn::RebindMask(d, overflow), hn::Mul(huge, huge), result);
  const auto neg_inf = hn::And(is_inf, neg);
  const auto pos_inf = hn::And(is_inf, hn::Not(neg));
  result = hn::IfThenElse(hn::RebindMask(d, neg_inf), hn::Zero(d), result);
  result = hn::IfThenElse(hn::RebindMask(d, pos_inf), x_in, result);
  result =
      hn::IfThenElse(hn::RebindMask(d, is_nan), hn::Add(x_in, x_in), result);
  return result;
}

template <class D>
HWY_INLINE void gather_xyz(D d, const double *base,
                           hn::Vec<hn::Rebind<std::int64_t, D>> ix,
                           hn::Vec<D> &x, hn::Vec<D> &y, hn::Vec<D> &z) {
  const hn::Rebind<std::int64_t, D> di;
  const auto one = hn::Set(di, std::int64_t{1});
  x = hn::GatherIndex(d, base, ix);
  y = hn::GatherIndex(d, base, hn::Add(ix, one));
  z = hn::GatherIndex(d, base, hn::Add(ix, hn::Set(di, std::int64_t{2})));
}

template <class D>
HWY_INLINE void lj_vec(D d, hn::Vec<D> r2, const LjParams &p,
                       hn::Vec<D> &energy, hn::Vec<D> &fscale) {
  const auto one = hn::Set(d, 1.0);
  const auto inv_r2 = hn::Div(one, r2);
  const auto sr2 = hn::Mul(hn::Set(d, p.psi2), inv_r2);
  const auto sr4 = hn::Mul(sr2, sr2);
  const auto a = hn::Mul(sr4, sr2);
  const auto b = hn::Mul(hn::Set(d, p.four_u0), a);
  const auto v = hn::Mul(b, hn::Sub(a, one));
  const auto two_a = hn::Mul(hn::Set(d, 2.0), a);
  const auto two_a_m1 = hn::Sub(two_a, one);
  const auto six_b = hn::Mul(hn::Set(d, 6.0), b);
  const auto t1 = hn::Mul(six_b, inv_r2);
  const auto f_un = hn::Mul(t1, two_a_m1);
  if (p.use_switch == 0) {
    energy = hn::Sub(v, hn::Set(d, p.shift));
    fscale = f_un;
    return;
  }
  const auto r = hn::Sqrt(r2);
  const auto t_raw =
      hn::Mul(hn::Sub(r, hn::Set(d, p.r_on)), hn::Set(d, p.inv_width));
  const auto t = hn::Min(hn::Max(t_raw, hn::Zero(d)), one);
  const auto t2 = hn::Mul(t, t);
  const auto t3 = hn::Mul(t2, t);
  const auto six_t = hn::Mul(hn::Set(d, 6.0), t);
  const auto inner = hn::Sub(hn::Set(d, 15.0), six_t);
  const auto poly = fma_lanes(d, t, inner, hn::Set(d, -10.0));
  const auto s = fma_lanes(d, t3, poly, one);
  const auto omt = hn::Sub(one, t);
  const auto m30 = hn::Mul(hn::Set(d, -30.0), t2);
  const auto ds1 = hn::Mul(m30, omt);
  const auto ds2 = hn::Mul(ds1, omt);
  const auto dsdr = hn::Mul(ds2, hn::Set(d, p.inv_width));
  energy = hn::Mul(v, s);
  const auto switched = hn::Mul(f_un, s);
  const auto slope = hn::Div(hn::Mul(v, dsdr), r);
  fscale = hn::Sub(switched, slope);
}

template <class D>
HWY_INLINE void morse_vec(D d, hn::Vec<D> r2, const MorseParams &p,
                          hn::Vec<D> &energy, hn::Vec<D> &fscale) {
  const auto one = hn::Set(d, 1.0);
  const auto r = hn::Sqrt(r2);
  const auto dr = hn::Sub(r, hn::Set(d, p.r_eq));
  const auto arg = hn::Mul(hn::Neg(hn::Set(d, p.range)), dr);
  const auto ex = exp_lanes(d, arg);
  const auto md = hn::Sub(one, ex);
  const auto dd = hn::Mul(hn::Set(d, p.depth), md);
  const auto v = fma_lanes(d, dd, md, hn::Neg(hn::Set(d, p.depth)));
  const auto d_m1 = hn::Sub(md, one);
  const auto num = hn::Mul(hn::Mul(hn::Set(d, p.two_de_a), md), d_m1);
  if (p.use_switch == 0) {
    energy = hn::Sub(v, hn::Set(d, p.shift));
    fscale = hn::Div(num, r);
    return;
  }
  const auto t_raw =
      hn::Mul(hn::Sub(r, hn::Set(d, p.r_on)), hn::Set(d, p.inv_width));
  const auto t = hn::Min(hn::Max(t_raw, hn::Zero(d)), one);
  const auto t2 = hn::Mul(t, t);
  const auto t3 = hn::Mul(t2, t);
  const auto six_t = hn::Mul(hn::Set(d, 6.0), t);
  const auto inner = hn::Sub(hn::Set(d, 15.0), six_t);
  const auto poly = fma_lanes(d, t, inner, hn::Set(d, -10.0));
  const auto s = fma_lanes(d, t3, poly, one);
  const auto omt = hn::Sub(one, t);
  const auto m30 = hn::Mul(hn::Set(d, -30.0), t2);
  const auto ds1 = hn::Mul(m30, omt);
  const auto ds2 = hn::Mul(ds1, omt);
  const auto dsdr = hn::Mul(ds2, hn::Set(d, p.inv_width));
  energy = hn::Mul(v, s);
  const auto switched = hn::Mul(num, s);
  const auto slope = hn::Mul(v, dsdr);
  fscale = hn::Div(hn::Sub(switched, slope), r);
}

// One scalar pass writes accepted pairs into contiguous buffers. The vector
// body then sees only those pairs, in the same order, so the width-8 slots
// stay aligned with the lane index and a vector add matches deposit.
template <class D, bool kMorse, int Fold>
HWY_INLINE double pair_run(D d, const PairView &v, double *F, double *strain,
                           const LjParams *lj, const MorseParams *mp) {
  const std::size_t L = hn::Lanes(d);
  struct Scratch {
    std::vector<double> dxb, dyb, dzb, r2b;
    std::vector<std::int32_t> jb;
  };
  thread_local Scratch scratch;
  if (scratch.dxb.size() < v.n) {
    scratch.dxb.resize(v.n);
    scratch.dyb.resize(v.n);
    scratch.dzb.resize(v.n);
    scratch.r2b.resize(v.n);
    scratch.jb.resize(v.n);
  }
  double *dxb = scratch.dxb.data();
  double *dyb = scratch.dyb.data();
  double *dzb = scratch.dzb.data();
  double *r2b = scratch.r2b.data();
  std::int32_t *jb = scratch.jb.data();
  double energy = 0.0;
  for (std::size_t i = 0; i < v.n; ++i) {
    Part8 part{};
    int nacc = 0;
    const std::int32_t p0 = v.rows[i];
    const std::int32_t p1 = v.rows[i + 1];
    const double xi = v.R[3 * i];
    const double yi = v.R[3 * i + 1];
    const double zi = v.R[3 * i + 2];
    std::size_t m = 0;
    if constexpr (Fold != 3) {
#if !HWY_NATIVE_FMA && defined(__x86_64__)
      if (detail::glibc_fma_active()) {
        m = accept_row_hw(v, i, dxb, dyb, dzb, r2b, jb);
      } else
#endif
#if !HWY_NATIVE_FMA
      {
      std::size_t raw = 0;
      for (std::int32_t p = p0; p < p1; ++p) {
        const auto up = static_cast<std::size_t>(p);
        const auto j = static_cast<std::size_t>(v.nbr[up]);
        double dx = xi - v.R[3 * j];
        double dy = yi - v.R[3 * j + 1];
        double dz = zi - v.R[3 * j + 2];
        if constexpr (Fold == 1) {
          dx = dx - v.w[0] * visit_round(dx * v.inv[0]);
          dy = dy - v.w[1] * visit_round(dy * v.inv[1]);
          dz = dz - v.w[2] * visit_round(dz * v.inv[2]);
        } else if constexpr (Fold == 2) {
          const double *sh = v.shift + 3 * static_cast<std::size_t>(v.code[up]);
          dx = dx - sh[0];
          dy = dy - sh[1];
          dz = dz - sh[2];
        }
        dxb[raw] = dx;
        dyb[raw] = dy;
        dzb[raw] = dz;
        jb[raw] = static_cast<std::int32_t>(j);
        ++raw;
      }
#if defined(__x86_64__) && (HWY_TARGET == HWY_SSE2 || HWY_TARGET == HWY_SSE4)
      std::size_t t = 0;
      for (; t + 8 <= raw; t += 8) {
        r2_8_trunc(dxb + t, dyb + t, dzb + t, r2b + t);
      }
      for (; t + 2 <= raw; t += 2) {
        const __m128d x = _mm_loadu_pd(dxb + t);
        const __m128d y = _mm_loadu_pd(dyb + t);
        const __m128d z = _mm_loadu_pd(dzb + t);
        const __m128d r2 = fma2_trunc(z, z, fma2_trunc(y, y, _mm_mul_pd(x, x)));
        _mm_storeu_pd(r2b + t, r2);
      }
      for (; t < raw; ++t) {
        r2b[t] = fma_trunc(dzb[t], dzb[t], fma_trunc(dyb[t], dyb[t], dxb[t] * dxb[t]));
      }
#else
      for (std::size_t t = 0; t < raw; ++t) {
        r2b[t] = fma_trunc(dzb[t], dzb[t], fma_trunc(dyb[t], dyb[t], dxb[t] * dxb[t]));
      }
#endif
      for (std::size_t t = 0; t < raw; ++t) {
        if (!(r2b[t] <= v.cutoff2)) {
          continue;
        }
        if (m != t) {
          dxb[m] = dxb[t];
          dyb[m] = dyb[t];
          dzb[m] = dzb[t];
          r2b[m] = r2b[t];
          jb[m] = jb[t];
        }
        ++m;
      }
      }
#else
      for (std::int32_t p = p0; p < p1; ++p) {
        const auto up = static_cast<std::size_t>(p);
        const auto j = static_cast<std::size_t>(v.nbr[up]);
        double dx = xi - v.R[3 * j];
        double dy = yi - v.R[3 * j + 1];
        double dz = zi - v.R[3 * j + 2];
        if constexpr (Fold == 1) {
          dx = dx - v.w[0] * visit_round(dx * v.inv[0]);
          dy = dy - v.w[1] * visit_round(dy * v.inv[1]);
          dz = dz - v.w[2] * visit_round(dz * v.inv[2]);
        } else if constexpr (Fold == 2) {
          const double *sh = v.shift + 3 * static_cast<std::size_t>(v.code[up]);
          dx = dx - sh[0];
          dy = dy - sh[1];
          dz = dz - sh[2];
        }
        const double r2 = r2_scalar(dx, dy, dz);
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
#endif
      // Deposit one evaluated vector into the width-8 tree at nacc, then
      // advance. Slots are independent, so a vector add matches deposit.
      auto deposit_vec = [&](std::size_t base, hn::Vec<D> dx, hn::Vec<D> dy,
                             hn::Vec<D> dz, hn::Vec<D> ev, hn::Vec<D> fsv) {
        const auto fx = hn::Mul(fsv, dx);
        const auto fy = hn::Mul(fsv, dy);
        const auto fz = hn::Mul(fsv, dz);
        const auto neg = hn::Neg(fsv);
        const auto cx = hn::Mul(neg, dx);
        const auto cy = hn::Mul(neg, dy);
        const auto cz = hn::Mul(neg, dz);
        const int off = nacc & 7;
        auto add_slot = [&](double *slot, hn::Vec<D> val) {
          hn::StoreU(hn::Add(hn::LoadU(d, slot + off), val), d, slot + off);
        };
        add_slot(part.e, ev);
        add_slot(part.fx, fx);
        add_slot(part.fy, fy);
        add_slot(part.fz, fz);
        add_slot(part.s[0], hn::Mul(cx, dx));
        add_slot(part.s[1], hn::Mul(cy, dy));
        add_slot(part.s[2], hn::Mul(cz, dz));
        add_slot(part.s[3], hn::Mul(cy, dz));
        add_slot(part.s[4], hn::Mul(cx, dz));
        add_slot(part.s[5], hn::Mul(cx, dy));
        alignas(64) double fxa[8], fya[8], fza[8];
        hn::Store(fx, d, fxa);
        hn::Store(fy, d, fya);
        hn::Store(fz, d, fza);
        for (std::size_t lane = 0; lane < L; ++lane) {
          const auto j = static_cast<std::size_t>(jb[base + lane]);
          F[3 * j] -= fxa[lane];
          F[3 * j + 1] -= fya[lane];
          F[3 * j + 2] -= fza[lane];
        }
        nacc += static_cast<int>(L);
      };
      auto eval_vec = [&](std::size_t base, hn::Vec<D> &dx, hn::Vec<D> &dy,
                          hn::Vec<D> &dz, hn::Vec<D> &ev, hn::Vec<D> &fsv) {
        dx = hn::LoadU(d, dxb + base);
        dy = hn::LoadU(d, dyb + base);
        dz = hn::LoadU(d, dzb + base);
        const auto r2 = hn::LoadU(d, r2b + base);
        if constexpr (kMorse) {
          morse_vec(d, r2, *mp, ev, fsv);
        } else {
          lj_vec(d, r2, *lj, ev, fsv);
        }
      };
      std::size_t k = 0;
      // A lane's exp and polynomial do not read the lane beside it.
      // SSE evaluates four vectors, AVX2 two, so eight lanes deposit in
      // neighbour order before the next group starts.
      if constexpr (kMorse) {
        if (L == 2) {
#if defined(__x86_64__)
          const bool hw_fma = detail::glibc_fma_active();
          for (; k + 8 <= m; k += 8) {
            alignas(16) double r2s[8], es[8], fs[8];
            for (int t = 0; t < 8; ++t) {
              r2s[t] = r2b[k + static_cast<std::size_t>(t)];
            }
            if (hw_fma) {
              morse8_hw(r2s, *mp, es, fs);
            } else {
              morse8_soft(r2s, *mp, es, fs);
            }
            auto take = [&](std::size_t base) {
              hn::Vec<D> dx = hn::LoadU(d, dxb + base);
              hn::Vec<D> dy = hn::LoadU(d, dyb + base);
              hn::Vec<D> dz = hn::LoadU(d, dzb + base);
              hn::Vec<D> ev = hn::LoadU(d, es + (base - k));
              hn::Vec<D> fsv = hn::LoadU(d, fs + (base - k));
              deposit_vec(base, dx, dy, dz, ev, fsv);
            };
            take(k);
            take(k + 2);
            take(k + 4);
            take(k + 6);
          }
#else
          for (; k + 8 <= m; k += 8) {
            hn::Vec<D> dx0, dy0, dz0, ev0, fs0;
            hn::Vec<D> dx1, dy1, dz1, ev1, fs1;
            hn::Vec<D> dx2, dy2, dz2, ev2, fs2;
            hn::Vec<D> dx3, dy3, dz3, ev3, fs3;
            eval_vec(k, dx0, dy0, dz0, ev0, fs0);
            eval_vec(k + 2, dx1, dy1, dz1, ev1, fs1);
            eval_vec(k + 4, dx2, dy2, dz2, ev2, fs2);
            eval_vec(k + 6, dx3, dy3, dz3, ev3, fs3);
            deposit_vec(k, dx0, dy0, dz0, ev0, fs0);
            deposit_vec(k + 2, dx1, dy1, dz1, ev1, fs1);
            deposit_vec(k + 4, dx2, dy2, dz2, ev2, fs2);
            deposit_vec(k + 6, dx3, dy3, dz3, ev3, fs3);
          }
#endif
        } else if (L < 8) {
          for (; k + 2 * L <= m; k += 2 * L) {
            hn::Vec<D> dx0, dy0, dz0, ev0, fs0;
            hn::Vec<D> dx1, dy1, dz1, ev1, fs1;
            eval_vec(k, dx0, dy0, dz0, ev0, fs0);
            eval_vec(k + L, dx1, dy1, dz1, ev1, fs1);
            deposit_vec(k, dx0, dy0, dz0, ev0, fs0);
            deposit_vec(k + L, dx1, dy1, dz1, ev1, fs1);
          }
        }
      }
      for (; k + L <= m; k += L) {
        hn::Vec<D> dx, dy, dz, ev, fsv;
        eval_vec(k, dx, dy, dz, ev, fsv);
        deposit_vec(k, dx, dy, dz, ev, fsv);
      }
      for (; k < m; ++k) {
        const double dx = dxb[k];
        const double dy = dyb[k];
        const double dz = dzb[k];
        const double r2 = r2b[k];
        double e = 0, fs = 0;
        if constexpr (kMorse) {
          morse_term<false>(r2, *mp, e, fs);
        } else {
          lj_term<false>(r2, *lj, e, fs);
        }
        const double fx = fs * dx;
        const double fy = fs * dy;
        const double fz = fs * dz;
        double sxx, syy, szz, syz, sxz, sxy;
        strain_of(fs, dx, dy, dz, sxx, syy, szz, syz, sxz, sxy);
        deposit(part, nacc, e, fx, fy, fz, sxx, syy, szz, syz, sxz, sxy);
        const auto j = static_cast<std::size_t>(jb[k]);
        F[3 * j] -= fx;
        F[3 * j + 1] -= fy;
        F[3 * j + 2] -= fz;
        ++nacc;
      }
    } else {
      for (std::int32_t pr = p0; pr < p1; ++pr) {
        const auto up = static_cast<std::size_t>(pr);
        const auto j = static_cast<std::size_t>(v.nbr[up]);
        double dx = 0, dy = 0, dz = 0;
        fold_one(v, i, j, up, dx, dy, dz);
        const double r2 = r2_scalar(dx, dy, dz);
        if (!(r2 <= v.cutoff2)) {
          continue;
        }
        double e = 0, fs = 0;
        if constexpr (kMorse) {
          morse_term<false>(r2, *mp, e, fs);
        } else {
          lj_term<false>(r2, *lj, e, fs);
        }
        const double fx = fs * dx;
        const double fy = fs * dy;
        const double fz = fs * dz;
        double sxx, syy, szz, syz, sxz, sxy;
        strain_of(fs, dx, dy, dz, sxx, syy, szz, syz, sxz, sxy);
        deposit(part, nacc, e, fx, fy, fz, sxx, syy, szz, syz, sxz, sxy);
        F[3 * j] -= fx;
        F[3 * j + 1] -= fy;
        F[3 * j + 2] -= fz;
        ++nacc;
      }
    }
    energy += combine8(part.e);
    F[3 * i] += combine8(part.fx);
    F[3 * i + 1] += combine8(part.fy);
    F[3 * i + 2] += combine8(part.fz);
    for (int c = 0; c < 6; ++c) {
      strain[c] += combine8(part.s[c]);
    }
  }
  return energy;
}

template <class D, bool kMorse>
HWY_INLINE double pair_dispatch(D d, const PairView &v, double *F,
                                double *strain, const LjParams *lj,
                                const MorseParams *mp) {
  switch (v.fold) {
  case 0:
    return pair_run<D, kMorse, 0>(d, v, F, strain, lj, mp);
  case 1:
    return pair_run<D, kMorse, 1>(d, v, F, strain, lj, mp);
  case 2:
    return pair_run<D, kMorse, 2>(d, v, F, strain, lj, mp);
  default:
    return pair_run<D, kMorse, 3>(d, v, F, strain, lj, mp);
  }
}

void FoldOrtho(double *x, double *y, double *z, std::size_t n, double w0,
               double w1, double w2, double i0, double i1, double i2) {
  const hn::CappedTag<double, 8> d;
  const std::size_t L = hn::Lanes(d);
  std::size_t i = 0;
  for (; i + L <= n; i += L) {
    auto vx = hn::LoadU(d, x + i);
    auto vy = hn::LoadU(d, y + i);
    auto vz = hn::LoadU(d, z + i);
    vx = fold_axis(d, vx, w0, i0);
    vy = fold_axis(d, vy, w1, i1);
    vz = fold_axis(d, vz, w2, i2);
    hn::StoreU(vx, d, x + i);
    hn::StoreU(vy, d, y + i);
    hn::StoreU(vz, d, z + i);
  }
  for (; i < n; ++i) {
    x[i] = x[i] - w0 * round_reference(x[i] * i0);
    y[i] = y[i] - w1 * round_reference(y[i] * i1);
    z[i] = z[i] - w2 * round_reference(z[i] * i2);
  }
}

double LjRun(const PairView *view, const LjParams *params, double *F,
             double *strain) {
  // Four lanes. Eight-wide LJ uses 512-bit vectors and is slower on this
  // kernel: the coordinate gather is scalar, and the extra width does not
  // hide it. Morse and EAM stay at eight because exp does.
  const hn::CappedTag<double, 4> d;
  return pair_dispatch<decltype(d), false>(d, *view, F, strain, params,
                                           nullptr);
}

double MorseRun(const PairView *view, const MorseParams *params, double *F,
                double *strain) {
  const hn::CappedTag<double, 8> d;
  return pair_dispatch<decltype(d), true>(d, *view, F, strain, nullptr, params);
}

void EamRun(long n, const double *R, const EamAtom *atoms, const double *box,
            double *F, double *U) {
#if !HWY_NATIVE_FMA && defined(__x86_64__) &&                                          \
    (HWY_TARGET == HWY_SSE2 || HWY_TARGET == HWY_SSE4)
  eam_sse(n, R, atoms, box, F, U);
  return;
#endif
#if !HWY_NATIVE_FMA
  if (detail::glibc_fma_active()) {
    eam_hw(n, R, atoms, box, F, U);
    return;
  }
#endif
  const hn::CappedTag<double, 8> d;
  const hn::Rebind<std::int64_t, decltype(d)> di;
  const std::size_t L = hn::Lanes(d);
  const double b0 = box[0];
  const double b1 = box[1];
  const double b2 = box[2];
  const double h0 = b0 * 0.5;
  const double h1 = b1 * 0.5;
  const double h2 = b2 * 0.5;
  // Scratch lives for the thread. The timed calls after the first do not
  // allocate, and drho stays zero except on atoms this atom actually touches.
  struct Scratch {
    std::vector<double> drho;
    std::vector<double> dxs;
    std::vector<double> dys;
    std::vector<double> dzs;
    std::vector<double> r2s;
    std::vector<long> js;
  };
  thread_local Scratch scratch;
  const auto nn = static_cast<std::size_t>(n);
  if (scratch.drho.size() != nn * 3) {
    scratch.drho.assign(nn * 3, 0.0);
  }
  if (scratch.dxs.size() != nn) {
    scratch.dxs.resize(nn);
    scratch.dys.resize(nn);
    scratch.dzs.resize(nn);
    scratch.r2s.resize(nn);
    scratch.js.resize(nn);
  }
  std::vector<double> &drho = scratch.drho;
  std::vector<double> &dxs = scratch.dxs;
  std::vector<double> &dys = scratch.dys;
  std::vector<double> &dzs = scratch.dzs;
  std::vector<double> &r2s = scratch.r2s;
  std::vector<long> &js = scratch.js;
  const auto hb0 = hn::Set(d, b0);
  const auto hb1 = hn::Set(d, b1);
  const auto hb2 = hn::Set(d, b2);
  const auto hh0 = hn::Set(d, h0);
  const auto hh1 = hn::Set(d, h1);
  const auto hh2 = hn::Set(d, h2);
  const auto nh0 = hn::Neg(hh0);
  const auto nh1 = hn::Neg(hh1);
  const auto nh2 = hn::Neg(hh2);

  for (long i = 0; i < n; ++i) {
    const EamAtom &a = atoms[i];
    Part8 dens_part{};
    int nd = 0;
    Part8 pair_part{};
    int npair = 0;
    const double xi = R[3 * i];
    const double yi = R[3 * i + 1];
    const double zi = R[3 * i + 2];
    const auto vxi = hn::Set(d, xi);
    const auto vyi = hn::Set(d, yi);
    const auto vzi = hn::Set(d, zi);
    long ns = 0;
    for (long j = 0; j + static_cast<long>(L) <= n; j += static_cast<long>(L)) {
      const auto jv = hn::Iota(di, j);
      const auto ix = hn::Mul(jv, hn::Set(di, std::int64_t{3}));
      hn::Vec<decltype(d)> x, y, z;
      gather_xyz(d, R, ix, x, y, z);
      auto dx = hn::Sub(vxi, x);
      auto dy = hn::Sub(vyi, y);
      auto dz = hn::Sub(vzi, z);
      dx = hn::IfThenElse(hn::Gt(dx, hh0), hn::Sub(dx, hb0), dx);
      dx = hn::IfThenElse(hn::Lt(dx, nh0), hn::Add(dx, hb0), dx);
      dy = hn::IfThenElse(hn::Gt(dy, hh1), hn::Sub(dy, hb1), dy);
      dy = hn::IfThenElse(hn::Lt(dy, nh1), hn::Add(dy, hb1), dy);
      dz = hn::IfThenElse(hn::Gt(dz, hh2), hn::Sub(dz, hb2), dz);
      dz = hn::IfThenElse(hn::Lt(dz, nh2), hn::Add(dz, hb2), dz);
      const auto r2 = r2_vec(d, dx, dy, dz);
      alignas(64) std::int64_t jva[8];
      alignas(64) double r2a[8], dxa[8], dya[8], dza[8];
      hn::Store(jv, di, jva);
      hn::Store(r2, d, r2a);
      hn::Store(dx, d, dxa);
      hn::Store(dy, d, dya);
      hn::Store(dz, d, dza);
      for (std::size_t lane = 0; lane < L; ++lane) {
        if (jva[lane] == i) {
          continue;
        }
        if (!(r2a[lane] <= a.cutoff2)) {
          continue;
        }
        dxs[static_cast<std::size_t>(ns)] = dxa[lane];
        dys[static_cast<std::size_t>(ns)] = dya[lane];
        dzs[static_cast<std::size_t>(ns)] = dza[lane];
        r2s[static_cast<std::size_t>(ns)] = r2a[lane];
        js[static_cast<std::size_t>(ns)] = j + static_cast<long>(lane);
        ++ns;
      }
    }
    for (long j = n - (n % static_cast<long>(L)); j < n; ++j) {
      if (j == i) {
        continue;
      }
      double dx = xi - R[3 * j];
      double dy = yi - R[3 * j + 1];
      double dz = zi - R[3 * j + 2];
      half_box(dx, dy, dz, b0, b1, b2, h0, h1, h2);
      const double r2 = r2_scalar(dx, dy, dz);
      if (!(r2 <= a.cutoff2)) {
        continue;
      }
      dxs[static_cast<std::size_t>(ns)] = dx;
      dys[static_cast<std::size_t>(ns)] = dy;
      dzs[static_cast<std::size_t>(ns)] = dz;
      r2s[static_cast<std::size_t>(ns)] = r2;
      js[static_cast<std::size_t>(ns)] = j;
      ++ns;
    }

    const double two_b2 = -2.0 * a.beta2;
    const auto beta1 = hn::Set(d, a.beta1);
    const auto beta2 = hn::Set(d, a.beta2);
    const auto vb2 = hn::Set(d, two_b2);
    const auto alpha = hn::Set(d, a.alpha_m);
    const auto rm = hn::Set(d, a.r_m);
    const auto dm = hn::Set(d, a.d_m);
    const auto c512 = hn::Set(d, 512.0);
    const auto c1024 = hn::Set(d, 1024.0);
    const auto six = hn::Set(d, 6.0);
    const auto one = hn::Set(d, 1.0);
    const auto two = hn::Set(d, 2.0);
    using V = hn::Vec<decltype(d)>;
    // Locals, not members: SVE and RVV vectors are sizeless.
    auto eval = [&](std::size_t us, V & dx, V & dy, V & dz, V & dens, V & fx,
                    V & fy, V & fz, V & phi, V & pfx, V & pfy, V & pfz)
        __attribute__((always_inline)) {
      dx = hn::LoadU(d, dxs.data() + us);
      dy = hn::LoadU(d, dys.data() + us);
      dz = hn::LoadU(d, dzs.data() + us);
      const auto r2 = hn::LoadU(d, r2s.data() + us);
      const auto r = hn::Sqrt(r2);
      const auto r2sq = hn::Mul(r2, r2);
      const auto r6 = hn::Mul(r2sq, r2);
      const auto r5 = hn::Mul(r2sq, r);
      const auto eb1 = exp_lanes(d, hn::Mul(hn::Neg(beta1), r));
      const auto eb2 = exp_lanes(d, hn::Mul(vb2, r));
      const auto rho_pair = fma_lanes(d, c512, eb2, eb1);
      dens = hn::Mul(r6, rho_pair);
      const auto t_a = hn::Mul(hn::Mul(six, r5), rho_pair);
      const auto neg_b1 = hn::Mul(hn::Neg(beta1), eb1);
      const auto b2_term = hn::Mul(hn::Mul(c1024, beta2), eb2);
      const auto inner = hn::Sub(neg_b1, b2_term);
      const auto mag = fma_lanes(d, r6, inner, t_a);
      const auto inv_r = hn::Div(one, r);
      const auto fscale = hn::Mul(hn::Neg(mag), inv_r);
      fx = hn::Mul(fscale, dx);
      fy = hn::Mul(fscale, dy);
      fz = hn::Mul(fscale, dz);
      const auto arg = hn::Mul(hn::Neg(alpha), hn::Sub(r, rm));
      const auto ex = exp_lanes(d, arg);
      const auto md = hn::Sub(one, ex);
      const auto dd = hn::Mul(dm, md);
      phi = fma_lanes(d, dd, md, hn::Neg(dm));
      const auto mag_f = hn::Mul(hn::Mul(hn::Mul(hn::Mul(two, alpha), dm), md),
                                 hn::Sub(md, one));
      const auto fcomp = hn::Mul(mag_f, inv_r);
      pfx = hn::Mul(fcomp, dx);
      pfy = hn::Mul(fcomp, dy);
      pfz = hn::Mul(fcomp, dz);
    };
    // Lane order matches the scalar tail: density on every neighbour, the
    // pair term only when j > i, each into the width-8 tree.
    auto consume = [&](std::size_t us, V dens, V fx, V fy, V fz, V phi, V pfx,
                       V pfy, V pfz) __attribute__((always_inline)) {
      alignas(64) double dea[8], dfx[8], dfy[8], dfz[8];
      alignas(64) double pea[8], pfxa[8], pfya[8], pfza[8];
      hn::Store(dens, d, dea);
      hn::Store(fx, d, dfx);
      hn::Store(fy, d, dfy);
      hn::Store(fz, d, dfz);
      hn::Store(phi, d, pea);
      hn::Store(pfx, d, pfxa);
      hn::Store(pfy, d, pfya);
      hn::Store(pfz, d, pfza);
      for (std::size_t lane = 0; lane < L; ++lane) {
        const long j = js[us + lane];
        const int kd = nd & 7;
        dens_part.e[kd] += dea[lane];
        dens_part.fx[kd] += dfx[lane];
        dens_part.fy[kd] += dfy[lane];
        dens_part.fz[kd] += dfz[lane];
        ++nd;
        drho[static_cast<std::size_t>(3 * j)] -= dfx[lane];
        drho[static_cast<std::size_t>(3 * j + 1)] -= dfy[lane];
        drho[static_cast<std::size_t>(3 * j + 2)] -= dfz[lane];
        if (j > i) {
          const int kp = npair & 7;
          pair_part.e[kp] += pea[lane];
          pair_part.fx[kp] += pfxa[lane];
          pair_part.fy[kp] += pfya[lane];
          pair_part.fz[kp] += pfza[lane];
          ++npair;
          F[3 * j] += pfxa[lane];
          F[3 * j + 1] += pfya[lane];
          F[3 * j + 2] += pfza[lane];
        }
      }
    };
    long t = 0;
    // SSE is two doubles wide, so four vectors fill the width-8 tree.
    // AVX2 is four wide, so two vectors fill it.
    if (L == 2) {
      for (; t + 8 <= ns; t += 8) {
        const auto us = static_cast<std::size_t>(t);
        V dx0, dy0, dz0, dens0, fx0, fy0, fz0, phi0, pfx0, pfy0, pfz0;
        V dx1, dy1, dz1, dens1, fx1, fy1, fz1, phi1, pfx1, pfy1, pfz1;
        V dx2, dy2, dz2, dens2, fx2, fy2, fz2, phi2, pfx2, pfy2, pfz2;
        V dx3, dy3, dz3, dens3, fx3, fy3, fz3, phi3, pfx3, pfy3, pfz3;
        eval(us, dx0, dy0, dz0, dens0, fx0, fy0, fz0, phi0, pfx0, pfy0, pfz0);
        eval(us + 2, dx1, dy1, dz1, dens1, fx1, fy1, fz1, phi1, pfx1, pfy1,
             pfz1);
        eval(us + 4, dx2, dy2, dz2, dens2, fx2, fy2, fz2, phi2, pfx2, pfy2,
             pfz2);
        eval(us + 6, dx3, dy3, dz3, dens3, fx3, fy3, fz3, phi3, pfx3, pfy3,
             pfz3);
        consume(us, dens0, fx0, fy0, fz0, phi0, pfx0, pfy0, pfz0);
        consume(us + 2, dens1, fx1, fy1, fz1, phi1, pfx1, pfy1, pfz1);
        consume(us + 4, dens2, fx2, fy2, fz2, phi2, pfx2, pfy2, pfz2);
        consume(us + 6, dens3, fx3, fy3, fz3, phi3, pfx3, pfy3, pfz3);
      }
    } else if (L < 8) {
      const long step = 2 * static_cast<long>(L);
      for (; t + step <= ns; t += step) {
        const auto us = static_cast<std::size_t>(t);
        V dx0, dy0, dz0, dens0, fx0, fy0, fz0, phi0, pfx0, pfy0, pfz0;
        V dx1, dy1, dz1, dens1, fx1, fy1, fz1, phi1, pfx1, pfy1, pfz1;
        eval(us, dx0, dy0, dz0, dens0, fx0, fy0, fz0, phi0, pfx0, pfy0, pfz0);
        eval(us + L, dx1, dy1, dz1, dens1, fx1, fy1, fz1, phi1, pfx1, pfy1,
             pfz1);
        consume(us, dens0, fx0, fy0, fz0, phi0, pfx0, pfy0, pfz0);
        consume(us + L, dens1, fx1, fy1, fz1, phi1, pfx1, pfy1, pfz1);
      }
    }
    for (; t + static_cast<long>(L) <= ns; t += static_cast<long>(L)) {
      const auto us = static_cast<std::size_t>(t);
      V dx, dy, dz, dens, fx, fy, fz, phi, pfx, pfy, pfz;
      eval(us, dx, dy, dz, dens, fx, fy, fz, phi, pfx, pfy, pfz);
      consume(us, dens, fx, fy, fz, phi, pfx, pfy, pfz);
    }
    for (; t < ns; ++t) {
      const auto us = static_cast<std::size_t>(t);
      const double dx = dxs[us];
      const double dy = dys[us];
      const double dz = dzs[us];
      const double r2 = r2s[us];
      const long j = js[us];
      const double r = std::sqrt(r2);
      const double r2sq = r2 * r2;
      const double r6 = r2sq * r2;
      const double r4 = r2sq;
      const double r5 = r4 * r;
      const double eb1 = exp(-a.beta1 * r);
      const double eb2 = exp(two_b2 * r);
      const double rho_pair = mul_add(512.0, eb2, eb1);
      const double dens_term = r6 * rho_pair;
      const double t_a = (6.0 * r5) * rho_pair;
      const double neg_b1 = -a.beta1 * eb1;
      const double b2_term = (1024.0 * a.beta2) * eb2;
      const double inner = neg_b1 - b2_term;
      const double mag = mul_add(r6, inner, t_a);
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
      drho[static_cast<std::size_t>(3 * j)] -= fx;
      drho[static_cast<std::size_t>(3 * j + 1)] -= fy;
      drho[static_cast<std::size_t>(3 * j + 2)] -= fz;
      if (j > i) {
        const double arg = -a.alpha_m * (r - a.r_m);
        const double ex = exp(arg);
        const double md = 1.0 - ex;
        const double dd = a.d_m * md;
        const double phi = mul_add(dd, md, -a.d_m);
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
        F[3 * j] += pfx;
        F[3 * j + 1] += pfy;
        F[3 * j + 2] += pfz;
      }
    }

    const double dens = combine8(dens_part.e);
    drho[static_cast<std::size_t>(3 * i)] += combine8(dens_part.fx);
    drho[static_cast<std::size_t>(3 * i + 1)] += combine8(dens_part.fy);
    drho[static_cast<std::size_t>(3 * i + 2)] += combine8(dens_part.fz);
    *U += combine8(pair_part.e);
    *U += embed_horner<false>(a.coeff, dens);
    F[3 * i] -= combine8(pair_part.fx);
    F[3 * i + 1] -= combine8(pair_part.fy);
    F[3 * i + 2] -= combine8(pair_part.fz);
    const double dF = embed_force<false>(a.coeff, dens);
    // Each component is independent, so touching only i and the neighbours
    // matches subtracting dF times a zero density derivative everywhere else.
    auto apply_rho = [&](long atom) {
      const std::size_t base = static_cast<std::size_t>(3 * atom);
      F[base] -= dF * drho[base];
      F[base + 1] -= dF * drho[base + 1];
      F[base + 2] -= dF * drho[base + 2];
      drho[base] = 0.0;
      drho[base + 1] = 0.0;
      drho[base + 2] = 0.0;
    };
    apply_rho(i);
    for (long t = 0; t < ns; ++t) {
      apply_rho(js[static_cast<std::size_t>(t)]);
    }
  }
}

void FmaArray(const double *a, const double *b, const double *c, double *out,
              std::size_t n) {
  const hn::CappedTag<double, 8> d;
  const std::size_t L = hn::Lanes(d);
  std::size_t i = 0;
  for (; i + L <= n; i += L) {
    const auto va = hn::LoadU(d, a + i);
    const auto vb = hn::LoadU(d, b + i);
    const auto vc = hn::LoadU(d, c + i);
#if HWY_NATIVE_FMA
    hn::StoreU(hn::MulAdd(va, vb, vc), d, out + i);
#else
    if (!lanes_in_fma_domain(d, va, vb, vc)) {
      for (std::size_t lane = 0; lane < L; ++lane) {
        out[i + lane] = exact_fma(a[i + lane], b[i + lane], c[i + lane]);
      }
      continue;
    }
    const auto y = fma_lanes(d, va, vb, vc);
    alignas(64) double ys[8];
    hn::Store(y, d, ys);
    for (std::size_t lane = 0; lane < L; ++lane) {
      const double yi = ys[lane];
      if (yi == 0.0 || !std::isfinite(yi) || std::fabs(yi) < 0x1p-1022) {
        out[i + lane] = exact_fma(a[i + lane], b[i + lane], c[i + lane]);
      } else {
        out[i + lane] = yi;
      }
    }
#endif
  }
  for (; i < n; ++i) {
    out[i] = exact_fma(a[i], b[i], c[i]);
  }
}

std::int64_t ActiveBit() { return HWY_TARGET; }

} // namespace HWY_NAMESPACE
} // namespace rgpot::simd
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace rgpot::simd {
static_assert(HWY_DISABLED_TARGETS ==
                  (HWY_AVX3_SPR | HWY_AVX3_DL | HWY_SSSE3 | HWY_SVE2_128 |
                   HWY_NEON_BF16 | HWY_NEON_WITHOUT_AES | HWY_PPC10 | HWY_PPC9 |
                   HWY_PPC8 | HWY_Z15 | HWY_Z14 | HWY_WASM_EMU256 | HWY_WASM),
              "disabled target mask");

HWY_EXPORT(FoldOrtho);
HWY_EXPORT(LjRun);
HWY_EXPORT(MorseRun);
HWY_EXPORT(EamRun);
HWY_EXPORT(FmaArray);
HWY_EXPORT(ActiveBit);

using LjFn = double (*)(const PairView *, const LjParams *, double *, double *);
using MorseFn = double (*)(const PairView *, const MorseParams *, double *,
                           double *);
using EamFn = void (*)(long, const double *, const EamAtom *, const double *,
                       double *, double *);
using FmaFn = void (*)(const double *, const double *, const double *, double *,
                       std::size_t);
using FoldFn = void (*)(double *, double *, double *, std::size_t, double,
                        double, double, double, double, double);

LjFn g_lj = nullptr;
MorseFn g_morse = nullptr;
EamFn g_eam = nullptr;
FmaFn g_fma = nullptr;
FoldFn g_fold = nullptr;
std::int64_t (*g_active)() = nullptr;

// Same index Highway would store in ChosenTarget, computed from a mask
// this process owns. The global ChosenTarget is left alone.
size_t local_dispatch_index(std::int64_t targets) {
  const auto shifted = static_cast<std::uint64_t>(
      HWY_CHOSEN_TARGET_SHIFT(targets) | HWY_CHOSEN_TARGET_MASK_SCALAR);
  const auto masked =
      shifted & static_cast<std::uint64_t>(HWY_CHOSEN_TARGET_MASK_TARGETS);
  return hwy::Num0BitsBelowLS1Bit_Nonzero64(masked);
}

bool bind_hwy_pointers(std::int64_t targets) {
  if (targets == 0) {
    targets = hwy::SupportedTargets() & static_cast<std::int64_t>(HWY_TARGETS);
  }
  const size_t index = local_dispatch_index(targets);
  g_lj = HWY_DISPATCH_TABLE(LjRun)[index];
  g_morse = HWY_DISPATCH_TABLE(MorseRun)[index];
  g_eam = HWY_DISPATCH_TABLE(EamRun)[index];
  g_fma = HWY_DISPATCH_TABLE(FmaArray)[index];
  g_fold = HWY_DISPATCH_TABLE(FoldOrtho)[index];
  g_active = HWY_DISPATCH_TABLE(ActiveBit)[index];
  return g_lj != nullptr && g_morse != nullptr && g_eam != nullptr &&
         g_fma != nullptr && g_fold != nullptr && g_active != nullptr;
}

void fma_hwy_entry(const double *a, const double *b, const double *c,
                   double *out, std::size_t n) {
  g_fma(a, b, c, out, n);
}

double lj_hwy_entry(const PairView &view, const LjParams &params, double *F,
                    double strain[6]) {
  return g_lj(&view, &params, F, strain);
}

double morse_hwy_entry(const PairView &view, const MorseParams &params,
                       double *F, double strain[6]) {
  return g_morse(&view, &params, F, strain);
}

void eam_hwy_entry(long n, const double *R, const EamAtom *atoms,
                   const double *box, double *F, double *U) {
  g_eam(n, R, atoms, box, F, U);
}

#if defined(__x86_64__)
double lj_sse2_entry(const PairView &view, const LjParams &params, double *F,
                     double strain[6]) {
  return HWY_DISPATCH_TABLE(LjRun)[local_dispatch_index(HWY_SSE2)](&view, &params, F,
                                                                   strain);
}

double morse_sse2_entry(const PairView &view, const MorseParams &params, double *F,
                        double strain[6]) {
  return HWY_DISPATCH_TABLE(MorseRun)[local_dispatch_index(HWY_SSE2)](
      &view, &params, F, strain);
}

void eam_sse2_entry(long n, const double *R, const EamAtom *atoms, const double *box,
                    double *F, double *U) {
  HWY_DISPATCH_TABLE(EamRun)[local_dispatch_index(HWY_SSE2)](n, R, atoms, box, F, U);
}
#endif

void fold_hwy_entry(double *x, double *y, double *z, std::size_t n, double w0,
                    double w1, double w2, double i0, double i1, double i2) {
  g_fold(x, y, z, n, w0, w1, w2, i0, i1, i2);
}

std::int64_t active_hwy_bit() { return g_active(); }

} // namespace rgpot::simd
#endif
