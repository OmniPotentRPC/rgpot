// MIT License
// Copyright 2023--present rgpot developers

#include "rgpot/simd/hwy_disabled.h"

#include "rgpot/simd/pair_loops.hpp"

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "rgpot/simd/hwy_kernels.cc"
#include "hwy/foreach_target.h" // IWYU pragma: keep
#include "hwy/highway.h"

namespace rgpot::simd {
double morse_scalar_entry(const PairView &view, const MorseParams &params,
                          double *F, double strain[6]);
void eam_scalar_entry(long n, const double *R, const EamAtom *atoms,
                      const double *box, double *F, double *U);
} // namespace rgpot::simd

HWY_BEFORE_NAMESPACE();
namespace rgpot::simd {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

template <class D>
HWY_INLINE hn::Vec<D> round_finite(D d, hn::Vec<D> t) {
  const auto half = hn::CopySign(hn::Set(d, 0.5), t);
  const auto shifted = hn::Add(t, half);
  const hn::Rebind<std::int64_t, D> di;
  const auto as_i = hn::ConvertTo(di, shifted);
  const auto back = hn::ConvertTo(d, as_i);
  const auto bad = hn::Or(hn::IsNaN(t), hn::IsInf(t));
  const auto huge =
      hn::Ge(hn::Abs(shifted), hn::Set(d, 9223372036854775808.0));
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

template <class D>
HWY_INLINE void lj_vec(D d, hn::Vec<D> r2, const LjParams &p, hn::Vec<D> &energy,
                       hn::Vec<D> &fscale) {
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
  const auto poly = hn::Add(hn::Set(d, -10.0), hn::Mul(t, inner));
  const auto s = hn::Add(one, hn::Mul(t3, poly));
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

double LjRun(const PairView *view, const LjParams *params, double *F,
             double *strain) {
  const PairView &v = *view;
  const LjParams &p = *params;
  const hn::CappedTag<double, 8> d;
  const std::size_t L = hn::Lanes(d);
  double energy = 0.0;
  for (std::size_t i = 0; i < v.n; ++i) {
    Part8 part{};
    int nacc = 0;
    const std::int32_t p0 = v.rows[i];
    const std::int32_t p1 = v.rows[i + 1];
    std::int32_t cursor = p0;
    // Round and coded folds both feed lj_vec. Coded is the usual list:
    // cutoff + skin fits in half a box, and each pair stores its image.
    if (v.fold == 1 || v.fold == 2) {
      while (cursor + static_cast<std::int32_t>(L) <= p1) {
        alignas(64) double dxa[8], dya[8], dza[8], r2a[8];
        alignas(64) double ea[8], fsa[8];
        alignas(64) double sxa[8], sya[8], sza[8];
        for (std::size_t lane = 0; lane < L; ++lane) {
          const auto up = static_cast<std::size_t>(cursor) + lane;
          const auto j = static_cast<std::size_t>(v.nbr[up]);
          dxa[lane] = v.R[3 * i] - v.R[3 * j];
          dya[lane] = v.R[3 * i + 1] - v.R[3 * j + 1];
          dza[lane] = v.R[3 * i + 2] - v.R[3 * j + 2];
          if (v.fold == 2) {
            const double *sh = v.shift + 3 * static_cast<std::size_t>(v.code[up]);
            sxa[lane] = sh[0];
            sya[lane] = sh[1];
            sza[lane] = sh[2];
          }
        }
        auto vx = hn::Load(d, dxa);
        auto vy = hn::Load(d, dya);
        auto vz = hn::Load(d, dza);
        if (v.fold == 1) {
          vx = fold_axis(d, vx, v.w[0], v.inv[0]);
          vy = fold_axis(d, vy, v.w[1], v.inv[1]);
          vz = fold_axis(d, vz, v.w[2], v.inv[2]);
        } else {
          vx = hn::Sub(vx, hn::Load(d, sxa));
          vy = hn::Sub(vy, hn::Load(d, sya));
          vz = hn::Sub(vz, hn::Load(d, sza));
        }
        const auto r2 = hn::Add(hn::Add(hn::Mul(vx, vx), hn::Mul(vy, vy)),
                                hn::Mul(vz, vz));
        hn::Vec<decltype(d)> ev, fsv;
        lj_vec(d, r2, p, ev, fsv);
        hn::Store(vx, d, dxa);
        hn::Store(vy, d, dya);
        hn::Store(vz, d, dza);
        hn::Store(r2, d, r2a);
        hn::Store(ev, d, ea);
        hn::Store(fsv, d, fsa);
        for (std::size_t lane = 0; lane < L; ++lane) {
          if (!(r2a[lane] <= v.cutoff2)) {
            continue;
          }
          const auto up = static_cast<std::size_t>(cursor) + lane;
          const auto j = static_cast<std::size_t>(v.nbr[up]);
          const double fx = fsa[lane] * dxa[lane];
          const double fy = fsa[lane] * dya[lane];
          const double fz = fsa[lane] * dza[lane];
          double sxx, syy, szz, syz, sxz, sxy;
          strain_of(fsa[lane], dxa[lane], dya[lane], dza[lane], sxx, syy, szz,
                    syz, sxz, sxy);
          deposit(part, nacc, ea[lane], fx, fy, fz, sxx, syy, szz, syz, sxz,
                  sxy);
          F[3 * j] -= fx;
          F[3 * j + 1] -= fy;
          F[3 * j + 2] -= fz;
          ++nacc;
        }
        cursor += static_cast<std::int32_t>(L);
      }
    }
    for (std::int32_t pr = cursor; pr < p1; ++pr) {
      const auto up = static_cast<std::size_t>(pr);
      const auto j = static_cast<std::size_t>(v.nbr[up]);
      double dx = 0, dy = 0, dz = 0;
      fold_one(v, i, j, up, dx, dy, dz);
      const double r2 = (dx * dx) + (dy * dy) + (dz * dz);
      if (!(r2 <= v.cutoff2)) {
        continue;
      }
      double e = 0, fs = 0;
      lj_term(r2, p, e, fs);
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

double MorseRun(const PairView *view, const MorseParams *params, double *F,
                double *strain) {
  // Same out-of-line body as the scalar path, so exp and the W=8 tree are
  // one function. The LJ round path and FoldOrtho carry the vector code.
  return morse_scalar_entry(*view, *params, F, strain);
}

void EamRun(long n, const double *R, const EamAtom *atoms, const double *box,
            double *F, double *U) {
  eam_scalar_entry(n, R, atoms, box, F, U);
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
HWY_EXPORT(ActiveBit);

double lj_hwy_entry(const PairView &view, const LjParams &params, double *F,
                    double strain[6]) {
  return HWY_DYNAMIC_DISPATCH(LjRun)(&view, &params, F, strain);
}

double morse_hwy_entry(const PairView &view, const MorseParams &params,
                       double *F, double strain[6]) {
  return HWY_DYNAMIC_DISPATCH(MorseRun)(&view, &params, F, strain);
}

void eam_hwy_entry(long n, const double *R, const EamAtom *atoms,
                   const double *box, double *F, double *U) {
  HWY_DYNAMIC_DISPATCH(EamRun)(n, R, atoms, box, F, U);
}

void fold_hwy_entry(double *x, double *y, double *z, std::size_t n, double w0,
                    double w1, double w2, double i0, double i1, double i2) {
  HWY_DYNAMIC_DISPATCH(FoldOrtho)(x, y, z, n, w0, w1, w2, i0, i1, i2);
}

std::int64_t active_hwy_bit() { return HWY_DYNAMIC_DISPATCH(ActiveBit)(); }

} // namespace rgpot::simd
#endif
