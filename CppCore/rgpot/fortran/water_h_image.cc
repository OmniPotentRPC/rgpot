// MIT License
// Copyright 2023--present rgpot developers

// One 26-image scan for the H-in-water kernel and its tests. A Fortran
// matmul and a C++ copy of the same cofactors are not the same rounding
// on every compiler, and the periodic energy is then an ulp away from
// the cluster folded by the other copy. This file is built with
// contraction off, so the products stay separate multiplies and adds.

#include <cmath>

namespace {

double anint_away(double x) {
  const double ax = std::abs(x);
  const double whole = std::trunc(ax);
  const double frac = ax - whole;
  const double mag = frac < 0.5 ? whole : whole + 1.0;
  return std::copysign(mag, x);
}

double at(const double *cell, int row, int col) { return cell[col * 3 + row]; }

double mul(double a, double b) { return a * b; }

} // namespace

extern "C" void rgpot_water_h_image(const double *cell, const double *d, double *out) {
  double inv[9];
  inv[0] = mul(at(cell, 1, 1), at(cell, 2, 2)) - mul(at(cell, 1, 2), at(cell, 2, 1));
  inv[1] = mul(at(cell, 1, 2), at(cell, 2, 0)) - mul(at(cell, 1, 0), at(cell, 2, 2));
  inv[2] = mul(at(cell, 1, 0), at(cell, 2, 1)) - mul(at(cell, 1, 1), at(cell, 2, 0));
  inv[3] = mul(at(cell, 0, 2), at(cell, 2, 1)) - mul(at(cell, 0, 1), at(cell, 2, 2));
  inv[4] = mul(at(cell, 0, 0), at(cell, 2, 2)) - mul(at(cell, 0, 2), at(cell, 2, 0));
  inv[5] = mul(at(cell, 0, 1), at(cell, 2, 0)) - mul(at(cell, 0, 0), at(cell, 2, 1));
  inv[6] = mul(at(cell, 0, 1), at(cell, 1, 2)) - mul(at(cell, 0, 2), at(cell, 1, 1));
  inv[7] = mul(at(cell, 0, 2), at(cell, 1, 0)) - mul(at(cell, 0, 0), at(cell, 1, 2));
  inv[8] = mul(at(cell, 0, 0), at(cell, 1, 1)) - mul(at(cell, 0, 1), at(cell, 1, 0));
  const double det = mul(at(cell, 0, 0), inv[0]) + mul(at(cell, 0, 1), inv[1]) +
                     mul(at(cell, 0, 2), inv[2]);
  for (double &v : inv) {
    v /= det;
  }
  double s[3];
  for (int row = 0; row < 3; ++row) {
    const double t0 = mul(inv[row], d[0]);
    const double t1 = mul(inv[3 + row], d[1]);
    const double t2 = mul(inv[6 + row], d[2]);
    s[row] = (t0 + t1) + t2;
    s[row] -= anint_away(s[row]);
  }
  auto apply = [&](int a, int b, int c, double dest[3]) {
    const double f0 = s[0] + static_cast<double>(a);
    const double f1 = s[1] + static_cast<double>(b);
    const double f2 = s[2] + static_cast<double>(c);
    for (int row = 0; row < 3; ++row) {
      const double p0 = mul(at(cell, row, 0), f0);
      const double p1 = mul(at(cell, row, 1), f1);
      const double p2 = mul(at(cell, row, 2), f2);
      dest[row] = (p0 + p1) + p2;
    }
  };
  apply(0, 0, 0, out);
  double best = (mul(out[0], out[0]) + mul(out[1], out[1])) + mul(out[2], out[2]);
  for (int a = -1; a <= 1; ++a) {
    for (int b = -1; b <= 1; ++b) {
      for (int c = -1; c <= 1; ++c) {
        double trial[3];
        apply(a, b, c, trial);
        const double span =
            (mul(trial[0], trial[0]) + mul(trial[1], trial[1])) + mul(trial[2], trial[2]);
        if (span < best) {
          best = span;
          out[0] = trial[0];
          out[1] = trial[1];
          out[2] = trial[2];
        }
      }
    }
  }
}
