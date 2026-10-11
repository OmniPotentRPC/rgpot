// MIT License
// Copyright 2023--present rgpot developers

// One accumulation site from the fused Lennard-Jones term and one from
// the embedded-atom Horner, repeated n times and reduced by the fixed
// width-8 tree. Verificarlo instruments this translation unit.

#include "rgpot/simd/exp.hpp"
#include "rgpot/simd/reduce.hpp"

#include <cstdio>
#include <cstdlib>
#include <vector>

double lj_energy(double r2) {
  const double inv_r2 = 1.0 / r2;
  const double sr2 = inv_r2;
  const double sr4 = sr2 * sr2;
  const double a = sr4 * sr2;
  const double b = 4.0 * a;
  return b * (a - 1.0);
}

constexpr double kC[9] = {-0.0847305, 17.0887,  60.7909, -75.8412, 39.9948,
                          -6.21563,   -3.07993, 1.51365, -0.196034};

double embed(double rho) {
  double result = kC[8];
  for (int i = 7; i >= 0; --i) {
    result = rgpot::simd::mul_add(result, rho, kC[i]);
  }
  return result;
}

double embed_force(double rho) {
  double result = kC[8] * 8.0;
  for (int i = 7; i >= 1; --i) {
    const double term = static_cast<double>(i) * kC[i];
    result = rgpot::simd::mul_add(result, rho, term);
  }
  return -result;
}

int main(int argc, char **argv) {
  const int n = argc > 1 ? std::atoi(argv[1]) : 4000;
  rgpot::simd::Part8 part{};
  double eam = 0.0;
  double eam_force = 0.0;
  int nacc = 0;
  for (int i = 0; i < n; ++i) {
    const double r2 = 2.0 + 0.01 * static_cast<double>(i % 50);
    const double e = lj_energy(r2);
    const double fx = e * 0.1;
    double sxx = 0, syy = 0, szz = 0, syz = 0, sxz = 0, sxy = 0;
    rgpot::simd::strain_of(fx, 0.1, 0.0, 0.0, sxx, syy, szz, syz, sxz, sxy);
    rgpot::simd::deposit(part, nacc, e, fx, 0.0, 0.0, sxx, syy, szz, syz, sxz,
                         sxy);
    ++nacc;
    const double rho = 0.2 + 0.001 * static_cast<double>(i % 50);
    eam += embed(rho);
    eam_force += embed_force(rho);
  }
  const double lj = rgpot::simd::combine8(part.e);
  const double lj_force = rgpot::simd::combine8(part.fx);
  const double virial = rgpot::simd::combine8(part.s[0]);
  std::printf("%.17g %.17g %.17g %.17g %.17g\n", lj, lj_force, virial, eam,
              eam_force);
  return 0;
}
