#pragma once
// MIT License
// Copyright 2023--present rgpot developers
//
// Cubic Hermite tables for the analytic embedded-atom pair. The
// stored list evaluates std::exp on every warm call. The stream
// evaluates these tables. The node spacing keeps a pair inside
// 1e-12 relative of that exponential. Distances below the first
// node use the exponential.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace rgpot::eam {

struct Spline1D {
  double origin = 0;
  double inv_h = 0;
  double h = 0;
  int nseg = 0;
  std::vector<double> c;

  [[nodiscard]] double eval(double r) const {
    const double x = (r - origin) * inv_h;
    int k = static_cast<int>(x);
    if (k < 0) {
      k = 0;
    } else if (k >= nseg) {
      k = nseg - 1;
    }
    const double u = x - static_cast<double>(k);
    const double *p = c.data() + static_cast<std::size_t>(k) * 4;
    return ((p[3] * u + p[2]) * u + p[1]) * u + p[0];
  }
};

inline double central(double (*fn)(const double *, double), const double *p, double r) {
  constexpr double h = 1e-6;
  return (fn(p, r + h) - fn(p, r - h)) / (2.0 * h);
}

inline Spline1D buildSpline(double (*fn)(const double *, double), const double *p, double origin,
                            double end, double h) {
  Spline1D t;
  t.origin = origin;
  t.h = h;
  t.inv_h = 1.0 / h;
  t.nseg = static_cast<int>(std::ceil((end - origin) / h));
  if (t.nseg < 1) {
    t.nseg = 1;
  }
  t.c.resize(static_cast<std::size_t>(t.nseg) * 4);
  for (int k = 0; k < t.nseg; ++k) {
    const double x0 = origin + static_cast<double>(k) * h;
    const double x1 = x0 + h;
    const double f0 = fn(p, x0);
    const double f1 = fn(p, x1);
    const double hd0 = h * central(fn, p, x0);
    const double hd1 = h * central(fn, p, x1);
    double *s = t.c.data() + static_cast<std::size_t>(k) * 4;
    s[0] = f0;
    s[1] = hd0;
    s[2] = -3.0 * f0 - 2.0 * hd0 + 3.0 * f1 - hd1;
    s[3] = 2.0 * f0 + hd0 - 2.0 * f1 + hd1;
  }
  return t;
}

// p: Dm, alpha, Rm, beta1, beta2
inline double rhoAt(const double *p, double r) {
  const double r2 = r * r;
  const double r6 = r2 * r2 * r2;
  const double e1 = std::exp(-p[3] * r);
  const double e2 = std::exp(-2.0 * p[4] * r);
  return r6 * (e1 + 512.0 * e2);
}

inline double fscaleAt(const double *p, double r) {
  const double r2 = r * r;
  const double r6 = r2 * r2 * r2;
  const double r5 = r2 * r2 * r;
  const double e1 = std::exp(-p[3] * r);
  const double e2 = std::exp(-2.0 * p[4] * r);
  const double pair = e1 + 512.0 * e2;
  const double mag = 6.0 * r5 * pair + r6 * (-p[3] * e1 - 1024.0 * p[4] * e2);
  return -mag / r;
}

inline double phiAt(const double *p, double r) {
  const double d = 1.0 - std::exp(-p[1] * (r - p[2]));
  return p[0] * d * d - p[0];
}

inline double fcompAt(const double *p, double r) {
  const double d = 1.0 - std::exp(-p[1] * (r - p[2]));
  return 2.0 * p[1] * p[0] * d * (d - 1.0) / r;
}

/// One element's density, density force, pair energy and pair force.
struct ElementSpline {
  static constexpr double kOrigin = 0.05;
  static constexpr double kStep = 5e-5;
  int Z = 0;
  double cutoff2 = 0;
  const double *embed = nullptr;
  Spline1D rho;
  Spline1D fscale;
  Spline1D phi;
  Spline1D fcomp;

  [[nodiscard]] double density(double r) const {
    return r < kOrigin ? rhoAt(packed, r) : rho.eval(r);
  }
  [[nodiscard]] double densityForce(double r) const {
    return r < kOrigin ? fscaleAt(packed, r) : fscale.eval(r);
  }
  [[nodiscard]] double pairEnergy(double r) const {
    return r < kOrigin ? phiAt(packed, r) : phi.eval(r);
  }
  [[nodiscard]] double pairForce(double r) const {
    return r < kOrigin ? fcompAt(packed, r) : fcomp.eval(r);
  }

  double packed[5]{};
};

inline ElementSpline makeElementSpline(int Z, double Dm, double alpha, double Rm, double beta1,
                                       double beta2, double cutoff, const double *embed) {
  ElementSpline s;
  s.Z = Z;
  s.cutoff2 = cutoff * cutoff;
  s.embed = embed;
  s.packed[0] = Dm;
  s.packed[1] = alpha;
  s.packed[2] = Rm;
  s.packed[3] = beta1;
  s.packed[4] = beta2;
  const double end = cutoff + ElementSpline::kStep;
  s.rho = buildSpline(rhoAt, s.packed, ElementSpline::kOrigin, end, ElementSpline::kStep);
  s.fscale = buildSpline(fscaleAt, s.packed, ElementSpline::kOrigin, end, ElementSpline::kStep);
  s.phi = buildSpline(phiAt, s.packed, ElementSpline::kOrigin, end, ElementSpline::kStep);
  s.fcomp = buildSpline(fcompAt, s.packed, ElementSpline::kOrigin, end, ElementSpline::kStep);
  return s;
}

} // namespace rgpot::eam
