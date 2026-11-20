#pragma once
// MIT License
// Copyright 2023--present rgpot developers

#include <cstddef>
#include <cstdint>

namespace rgpot::simd {

/// CSR half-list plus the fold the pair kernels apply. fold is
/// 0 none, 1 orthorhombic round, 2 coded image, 3 general MinimumImage.
struct PairView {
  const double *R;
  const std::int32_t *rows;
  const std::int32_t *nbr;
  std::size_t n;
  double cutoff2;
  int fold;
  double w[3];
  double inv[3];
  const std::uint8_t *code;
  const double *shift;
  const void *general;
};

struct LjParams {
  double psi2;
  double four_u0;
  double shift;
  double r_on;
  double inv_width;
  int use_switch;
};

struct MorseParams {
  double depth;
  double range;
  double r_eq;
  double two_de_a;
  double shift;
  double r_on;
  double inv_width;
  int use_switch;
};

/// One atom's analytical EAM parameters (Zhou form, not a spline table).
struct EamAtom {
  double beta1;
  double beta2;
  double alpha_m;
  double r_m;
  double d_m;
  double cutoff2;
  double coeff[9];
};

} // namespace rgpot::simd
