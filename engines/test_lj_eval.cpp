// MIT License
// Copyright 2023--present rgpot developers
//
// Shifted 12-6 Lennard-Jones for one pair, compared with the shared
// evaluator the engines call.

#include "lj_eval.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace {

double shifted(double r, double u0, double cutoff, double psi) {
  const double ar = std::pow(psi / r, 6.0);
  const double ac = std::pow(psi / cutoff, 6.0);
  const double pair = 4.0 * u0 * ar * (ar - 1.0);
  const double shift = 4.0 * u0 * ac * (ac - 1.0);
  return pair - shift;
}

int fail(const char *sentence) {
  std::fprintf(stderr, "%s\n", sentence);
  return 1;
}

} // namespace

int main() {
  const double u0 = 1.0;
  const double cutoff = 8.0;
  const double psi = 1.0;
  const double r = 1.5;
  const double xyz[6] = {0.0, 0.0, 0.0, r, 0.0, 0.0};
  const int z[2] = {1, 1};
  const double box[9] = {20, 0, 0, 0, 20, 0, 0, 0, 20};
  double forces[6];
  double energy = 0.0;
  int has_stress = 0;
  double stress[9];
  if (rgpot_lj_eval(xyz, z, 2, box, u0, cutoff, psi, forces, &energy, &has_stress,
                    stress) != 0) {
    return fail("rgpot_lj_eval rejected the dimer");
  }
  const double expect = shifted(r, u0, cutoff, psi);
  if (std::abs(energy - expect) > 1e-12) {
    std::fprintf(stderr, "energy %.16e expect %.16e\n", energy, expect);
    return 1;
  }
  if (!has_stress) {
    return fail("Lennard-Jones did not report stress");
  }
  const double eps = 1e-6;
  double xyz_plus[6] = {0.0, 0.0, 0.0, r + eps, 0.0, 0.0};
  double xyz_minus[6] = {0.0, 0.0, 0.0, r - eps, 0.0, 0.0};
  double unused[6];
  double e_plus = 0.0;
  double e_minus = 0.0;
  int ignored = 0;
  double ignored_stress[9];
  rgpot_lj_eval(xyz_plus, z, 2, box, u0, cutoff, psi, unused, &e_plus, &ignored,
                ignored_stress);
  rgpot_lj_eval(xyz_minus, z, 2, box, u0, cutoff, psi, unused, &e_minus, &ignored,
                ignored_stress);
  const double dEdr = (e_plus - e_minus) / (2.0 * eps);
  // Force on atom 1 is -dE/dr along x. Force on atom 0 is the opposite.
  if (std::abs(forces[3] + dEdr) > 1e-7 || std::abs(forces[0] - dEdr) > 1e-7) {
    std::fprintf(stderr, "force %g %g dEdr %g\n", forces[0], forces[3], dEdr);
    return 1;
  }
  if (rgpot_lj_eval(nullptr, z, 2, box, u0, cutoff, psi, forces, &energy, &has_stress,
                    stress) == 0) {
    return fail("a null coordinate pointer was accepted");
  }
  return 0;
}
