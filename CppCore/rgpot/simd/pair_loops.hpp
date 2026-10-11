#pragma once
// MIT License
// Copyright 2023--present rgpot developers

#include "rgpot/nlist/MinimumImage.hpp"
#include "rgpot/simd/exp.hpp"
#include "rgpot/simd/pods.hpp"
#include "rgpot/simd/reduce.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace rgpot::simd {
namespace {

inline double visit_round(double t) {
  // Same truncation as vesin::cpu::visit_round for finite in-range t.
  // Non-finite inputs return 0 so the cast is not undefined.
  if (!std::isfinite(t)) {
    return 0.0;
  }
  const double shifted = t + std::copysign(0.5, t);
  if (!std::isfinite(shifted) || shifted >= 9223372036854775808.0 ||
      shifted <= -9223372036854775808.0) {
    return 0.0;
  }
  return static_cast<double>(static_cast<long long>(shifted));
}

inline void fold_one(const PairView &v, std::size_t i, std::size_t j,
                     std::size_t p, double &dx, double &dy, double &dz) {
  dx = v.R[3 * i] - v.R[3 * j];
  dy = v.R[3 * i + 1] - v.R[3 * j + 1];
  dz = v.R[3 * i + 2] - v.R[3 * j + 2];
  if (v.fold == 1) {
    dx = dx - v.w[0] * visit_round(dx * v.inv[0]);
    dy = dy - v.w[1] * visit_round(dy * v.inv[1]);
    dz = dz - v.w[2] * visit_round(dz * v.inv[2]);
  } else if (v.fold == 2) {
    const double *sh = v.shift + 3 * static_cast<std::size_t>(v.code[p]);
    dx = dx - sh[0];
    dy = dy - sh[1];
    dz = dz - sh[2];
  } else if (v.fold == 3 && v.general != nullptr) {
    const auto *image = static_cast<const nlist::MinimumImage *>(v.general);
    image->fold(dx, dy, dz);
  }
}

inline void lj_term(double r2, const LjParams &p, double &energy,
                    double &fscale) {
  const double inv_r2 = 1.0 / r2;
  const double sr2 = p.psi2 * inv_r2;
  const double sr4 = sr2 * sr2;
  const double a = sr4 * sr2;
  const double b = p.four_u0 * a;
  const double a_m1 = a - 1.0;
  const double v = b * a_m1;
  const double two_a = 2.0 * a;
  const double two_a_m1 = two_a - 1.0;
  const double six_b = 6.0 * b;
  const double t1 = six_b * inv_r2;
  const double f_un = t1 * two_a_m1;
  if (p.use_switch == 0) {
    energy = v - p.shift;
    fscale = f_un;
    return;
  }
  const double r = std::sqrt(r2);
  const double t_raw = (r - p.r_on) * p.inv_width;
  const double t = t_raw < 0.0 ? 0.0 : (t_raw > 1.0 ? 1.0 : t_raw);
  const double t2 = t * t;
  const double t3 = t2 * t;
  const double six_t = 6.0 * t;
  const double inner = 15.0 - six_t;
  const double poly = -10.0 + t * inner;
  const double s = 1.0 + t3 * poly;
  const double omt = 1.0 - t;
  const double m30 = -30.0 * t2;
  const double ds1 = m30 * omt;
  const double ds2 = ds1 * omt;
  const double dsdr = ds2 * p.inv_width;
  energy = v * s;
  const double switched = f_un * s;
  const double slope = (v * dsdr) / r;
  fscale = switched - slope;
}

inline void morse_term(double r2, const MorseParams &p, double &energy,
                       double &fscale) {
  const double r = std::sqrt(r2);
  const double dr = r - p.r_eq;
  const double arg = -p.range * dr;
  const double ex = exp(arg);
  const double d = 1.0 - ex;
  const double dd = p.depth * d;
  const double v = dd * d - p.depth;
  const double d_m1 = d - 1.0;
  const double num = (p.two_de_a * d) * d_m1;
  if (p.use_switch == 0) {
    energy = v - p.shift;
    fscale = num / r;
    return;
  }
  const double t_raw = (r - p.r_on) * p.inv_width;
  const double t = t_raw < 0.0 ? 0.0 : (t_raw > 1.0 ? 1.0 : t_raw);
  const double t2 = t * t;
  const double t3 = t2 * t;
  const double six_t = 6.0 * t;
  const double inner = 15.0 - six_t;
  const double poly = -10.0 + t * inner;
  const double s = 1.0 + t3 * poly;
  const double omt = 1.0 - t;
  const double m30 = -30.0 * t2;
  const double ds1 = m30 * omt;
  const double ds2 = ds1 * omt;
  const double dsdr = ds2 * p.inv_width;
  energy = v * s;
  const double switched = num * s;
  const double slope = v * dsdr;
  fscale = (switched - slope) / r;
}

inline double embed_horner(const double *coeff, double rho) {
  double result = coeff[8];
  for (int i = 7; i >= 0; --i) {
    const double prod = result * rho;
    result = prod + coeff[i];
  }
  return result;
}

inline double embed_force(const double *coeff, double rho) {
  double result = coeff[8] * 8.0;
  for (int i = 7; i >= 1; --i) {
    const double prod = result * rho;
    const double term = static_cast<double>(i) * coeff[i];
    result = prod + term;
  }
  return -result;
}

inline void half_box(double &dx, double &dy, double &dz, double b0, double b1,
                     double b2, double h0, double h1, double h2) {
  if (dx > h0) {
    dx = dx - b0;
  } else if (dx < -h0) {
    dx = dx + b0;
  }
  if (dy > h1) {
    dy = dy - b1;
  } else if (dy < -h1) {
    dy = dy + b1;
  }
  if (dz > h2) {
    dz = dz - b2;
  } else if (dz < -h2) {
    dz = dz + b2;
  }
}

template <typename Term>
double accumulate_pairs(const PairView &v, double *F, double strain[6],
                        Term &&term) {
  double energy = 0.0;
  for (std::size_t i = 0; i < v.n; ++i) {
    Part8 part{};
    int nacc = 0;
    const std::int32_t p0 = v.rows[i];
    const std::int32_t p1 = v.rows[i + 1];
    for (std::int32_t p = p0; p < p1; ++p) {
      const auto up = static_cast<std::size_t>(p);
      const auto j = static_cast<std::size_t>(v.nbr[up]);
      double dx = 0, dy = 0, dz = 0;
      fold_one(v, i, j, up, dx, dy, dz);
      const double r2 = (dx * dx) + (dy * dy) + (dz * dz);
      if (!(r2 <= v.cutoff2)) {
        continue;
      }
      double e = 0, fs = 0;
      term(r2, e, fs);
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

inline double lj_scalar(const PairView &v, const LjParams &p, double *F,
                        double strain[6]) {
  return accumulate_pairs(v, F, strain, [&](double r2, double &e, double &fs) {
    lj_term(r2, p, e, fs);
  });
}

inline double morse_scalar(const PairView &v, const MorseParams &p, double *F,
                           double strain[6]) {
  return accumulate_pairs(
      v, F, strain, [&](double r2, double &e, double &fs) {
        morse_term(r2, p, e, fs);
      });
}

inline void eam_scalar(long n, const double *R, const EamAtom *atoms,
                       const double *box, double *F, double *U) {
  const double b0 = box[0];
  const double b1 = box[1];
  const double b2 = box[2];
  const double h0 = b0 * 0.5;
  const double h1 = b1 * 0.5;
  const double h2 = b2 * 0.5;
  std::vector<double> drho(static_cast<std::size_t>(3 * n), 0.0);
  for (long i = 0; i < n; ++i) {
    const EamAtom &a = atoms[i];
    std::fill(drho.begin(), drho.end(), 0.0);
    Part8 dens_part{};
    int nd = 0;
    Part8 pair_part{};
    int npair = 0;
    const double xi = R[3 * i];
    const double yi = R[3 * i + 1];
    const double zi = R[3 * i + 2];
    for (long j = 0; j < n; ++j) {
      if (j == i) {
        continue;
      }
      double dx = xi - R[3 * j];
      double dy = yi - R[3 * j + 1];
      double dz = zi - R[3 * j + 2];
      half_box(dx, dy, dz, b0, b1, b2, h0, h1, h2);
      const double r2 = (dx * dx) + (dy * dy) + (dz * dz);
      if (!(r2 <= a.cutoff2)) {
        continue;
      }
      const double r = std::sqrt(r2);
      const double r2sq = r2 * r2;
      const double r6 = r2sq * r2;
      const double r4 = r2sq;
      const double r5 = r4 * r;
      const double eb1 = exp(-a.beta1 * r);
      const double two_b2 = -2.0 * a.beta2;
      const double eb2 = exp(two_b2 * r);
      const double rho_pair = eb1 + 512.0 * eb2;
      const double dens_term = r6 * rho_pair;
      const double t_a = (6.0 * r5) * rho_pair;
      const double neg_b1 = -a.beta1 * eb1;
      const double b2_term = (1024.0 * a.beta2) * eb2;
      const double inner = neg_b1 - b2_term;
      const double t_b = r6 * inner;
      const double mag = t_a + t_b;
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
        const double d = 1.0 - ex;
        const double dd = a.d_m * d;
        const double phi = dd * d - a.d_m;
        const double mag_f = ((2.0 * a.alpha_m) * a.d_m) * d * (d - 1.0);
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
    *U += embed_horner(a.coeff, dens);
    F[3 * i] -= combine8(pair_part.fx);
    F[3 * i + 1] -= combine8(pair_part.fy);
    F[3 * i + 2] -= combine8(pair_part.fz);
    const double dF = embed_force(a.coeff, dens);
    for (long k = 0; k < 3 * n; ++k) {
      const double prod = dF * drho[static_cast<std::size_t>(k)];
      F[k] -= prod;
    }
  }
}

} // namespace

inline double lj_reference(const PairView &v, const LjParams &p, double *F,
                           double strain[6]) {
  return lj_scalar(v, p, F, strain);
}

inline double morse_reference(const PairView &v, const MorseParams &p,
                              double *F, double strain[6]) {
  return morse_scalar(v, p, F, strain);
}

inline void eam_reference(long n, const double *R, const EamAtom *atoms,
                          const double *box, double *F, double *U) {
  eam_scalar(n, R, atoms, box, F, U);
}

inline double round_reference(double t) { return visit_round(t); }

} // namespace rgpot::simd
