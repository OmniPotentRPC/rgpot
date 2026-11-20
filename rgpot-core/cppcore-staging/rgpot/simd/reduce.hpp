#pragma once
// MIT License
// Copyright 2023--present rgpot developers

// Fixed logical width W = 8. Eight partial sums, then the ndarray 0.17
// unrolled_dot tree: (p0+p4), (p1+p5), (p2+p6), (p3+p7), then those
// pairs, then the two results. Every path calls this function.

namespace rgpot::simd {

inline double combine8(const double p[8]) {
  const double s0 = p[0] + p[4];
  const double s1 = p[1] + p[5];
  const double s2 = p[2] + p[6];
  const double s3 = p[3] + p[7];
  const double t0 = s0 + s2;
  const double t1 = s1 + s3;
  return t0 + t1;
}

struct Part8 {
  double e[8];
  double fx[8];
  double fy[8];
  double fz[8];
  double s[6][8];
};

inline void deposit(Part8 &a, int nacc, double e, double fx, double fy,
                    double fz, double sxx, double syy, double szz, double syz,
                    double sxz, double sxy) {
  const int k = nacc & 7;
  a.e[k] += e;
  a.fx[k] += fx;
  a.fy[k] += fy;
  a.fz[k] += fz;
  a.s[0][k] += sxx;
  a.s[1][k] += syy;
  a.s[2][k] += szz;
  a.s[3][k] += syz;
  a.s[4][k] += sxz;
  a.s[5][k] += sxy;
}

inline void strain_of(double fscale, double dx, double dy, double dz,
                      double &sxx, double &syy, double &szz, double &syz,
                      double &sxz, double &sxy) {
  const double c = -fscale;
  const double cx = c * dx;
  const double cy = c * dy;
  sxx = cx * dx;
  syy = cy * dy;
  szz = (c * dz) * dz;
  syz = cy * dz;
  sxz = cx * dz;
  sxy = cx * dy;
}

} // namespace rgpot::simd
