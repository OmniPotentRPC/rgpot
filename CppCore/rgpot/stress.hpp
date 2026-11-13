#pragma once
// MIT License
// Copyright 2023--present rgpot developers

/**
 * Cauchy stress from a pair force.
 *
 * sigma = (1/V) dE/dε for the right strain that sends a row of Cartesian
 * coordinates r to r (I+ε). A pair whose force on atom i is fscale * d,
 * with d = r_i - r_j, contributes -fscale * d ⊗ d to dE/dε.
 *
 * The virial is W = sum r_ij ⊗ f_ij in energy units, with r_ij = r_i - r_j
 * the folded (minimum-image) vector and f_ij the force on i from that
 * interaction. ASE's Cauchy stress is then
 *
 *   sigma = -W / V = (1/V) dE/dε
 *
 * (Larsen et al., J. Phys.: Condens. Matter 29, 273002 (2017),
 * doi:10.1088/1361-648X/aa680e; the same derivative
 * ase.calculators.fd.calculate_numerical_stress forms by central
 * differences). A positive hydrostatic pressure is a negative diagonal
 * stress. has_virial and has_stress stay 0 when that tensor was not
 * computed. A nonpositive volume leaves the stress unavailable and can
 * still publish W.
 *
 * A many-body term that depends on folded bond vectors d_k contributes
 * the sum over k of d_k ⊗ dE/dd_k (Thompson, Plimpton and Mattson, J. Chem.
 * Phys. 131, 154107 (2009), doi:10.1063/1.3245303), which is -r ⊗ f for
 * f = -dE/dr on the atom at the end of the bond.
 */

#include "rgpot/ForceStructs.hpp"

#include <algorithm>
#include <cmath>

namespace rgpot {

/// Absolute cell volume. The 3×3 box is row-major, index = 3 * row + column.
inline double cellVolume(const double *box) {
  if (box == nullptr) {
    return 0.0;
  }
  const double m00 = box[0];
  const double m01 = box[1];
  const double m02 = box[2];
  const double m10 = box[3];
  const double m11 = box[4];
  const double m12 = box[5];
  const double m20 = box[6];
  const double m21 = box[7];
  const double m22 = box[8];
  return std::abs(m00 * (m11 * m22 - m12 * m21) -
                  m01 * (m10 * m22 - m12 * m20) +
                  m02 * (m10 * m21 - m11 * m20));
}

/// Add one pair to the six Voigt components of dE/dε: xx, yy, zz, yz, xz, xy.
inline void accumulatePairStrain(double acc[6], double fscale, double dx,
                                 double dy, double dz) {
  const double c = -fscale;
  acc[0] += c * dx * dx;
  acc[1] += c * dy * dy;
  acc[2] += c * dz * dz;
  acc[3] += c * dy * dz;
  acc[4] += c * dx * dz;
  acc[5] += c * dx * dy;
}

/// Add r ⊗ f to the row-major virial W. r is the folded bond vector from
/// the atom the force acts on, or r_ij = r_i - r_j when f is the force on i.
inline void accumulateVirial(double w[9], double rx, double ry, double rz,
                             double fx, double fy, double fz) {
  w[0] += rx * fx;
  w[1] += rx * fy;
  w[2] += rx * fz;
  w[3] += ry * fx;
  w[4] += ry * fy;
  w[5] += ry * fz;
  w[6] += rz * fx;
  w[7] += rz * fy;
  w[8] += rz * fz;
}

/// Add d ⊗ g to the six Voigt components of dE/dε, where d is a bond vector
/// and g is dE/dd. Off-diagonal slots take the upper triangle, d_a g_b, a < b.
inline void accumulateBondStrain(double acc[6], double dx, double dy, double dz,
                                 double gx, double gy, double gz) {
  acc[0] += dx * gx;
  acc[1] += dy * gy;
  acc[2] += dz * gz;
  acc[3] += dy * gz;
  acc[4] += dx * gz;
  acc[5] += dx * gy;
}

/// Write acc/volume into the row-major ForceOut stress. A nonpositive
/// volume leaves has_stress clear.
inline void publishCauchyStress(ForceOut *out, const double acc[6],
                                double volume) {
  std::fill(out->stress, out->stress + 9, 0.0);
  if (!(volume > 0.0)) {
    out->has_stress = 0;
    return;
  }
  const double s = 1.0 / volume;
  const double xx = acc[0] * s;
  const double yy = acc[1] * s;
  const double zz = acc[2] * s;
  const double yz = acc[3] * s;
  const double xz = acc[4] * s;
  const double xy = acc[5] * s;
  out->stress[0] = xx;
  out->stress[1] = xy;
  out->stress[2] = xz;
  out->stress[3] = xy;
  out->stress[4] = yy;
  out->stress[5] = yz;
  out->stress[6] = xz;
  out->stress[7] = yz;
  out->stress[8] = zz;
  out->has_stress = 1;
}

/// Write W and, when volume > 0, sigma = -W/V. A nonpositive volume
/// leaves has_stress clear. has_virial is set either way: W does not
/// need a cell.
inline void publishVirialStress(ForceOut *out, const double virial[9],
                                double volume) {
  std::copy(virial, virial + 9, out->virial);
  out->has_virial = 1;
  std::fill(out->stress, out->stress + 9, 0.0);
  if (!(volume > 0.0)) {
    out->has_stress = 0;
    return;
  }
  const double scale = -1.0 / volume;
  for (int i = 0; i < 9; ++i) {
    out->stress[i] = virial[i] * scale;
  }
  out->has_stress = 1;
}

} // namespace rgpot
