#pragma once
// MIT License
// Copyright 2023--present rgpot developers

/**
 * Opt-in Cauchy stress from pair and many-body forces.
 *
 * sigma = (1/V) dE/dε for the right strain that sends a row of Cartesian
 * coordinates r to r (I+ε). A pair whose force on atom i is fscale * d,
 * with d = r_i - r_j, contributes -fscale * d ⊗ d to dE/dε. Energy is
 * measured in eV and length in Angstrom, so published stress is eV/Angstrom^3.
 * The volume is the absolute determinant of the row-major cell. A singular
 * or missing cell cannot produce stress.
 *
 * A many-body term that depends on neighbour-list image bond vectors d_k
 * contributes the sum over k of d_k ⊗ dE/dd_k. The image vector supplied by
 * the neighbour list is authoritative and must not be rewrapped (Thompson,
 * Plimpton and Mattson, J. Chem. Phys. 131, 154107 (2009),
 * doi:10.1063/1.3245303).
 */

#include "rgpot/ForceStructs.hpp"

#include <algorithm>
#include <cmath>

namespace rgpot {

/**
 * Absolute cell volume. The 3×3 box is row-major, index = 3 * row + column.
 *
 * Zero is the error value, and every caller tests `volume > 0.0`. A null
 * box, an empty (singular) box, a box with a non finite entry and a box
 * whose volume overflows all give exactly 0.0, never NaN or infinity.
 */
inline double cellVolume(const double *box) {
  if (box == nullptr) {
    return 0.0;
  }
  for (int i = 0; i < 9; ++i) {
    if (!std::isfinite(box[i])) {
      return 0.0;
    }
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
  const double volume = std::abs(m00 * (m11 * m22 - m12 * m21) -
                                 m01 * (m10 * m22 - m12 * m20) +
                                 m02 * (m10 * m21 - m11 * m20));
  return std::isfinite(volume) ? volume : 0.0;
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

/**
 * Shared opt-in virial accumulator.
 *
 * Construct one at the start of a supporting force kernel. `data()` is null
 * unless the caller set `ForceInput::compute_stress`, allowing existing pair
 * and many-body loops to skip every stress allocation and operation on the
 * force-only path. `publish()` is the only operation that sets has_stress.
 *
 * A force-only call clears the ten stress fields of the output, so a caller
 * that reuses a ForceOut never reads a stale tensor, and does nothing else:
 * the strain accumulator is left alone and every accumulating member returns
 * at the `enabled_` test.
 */
class StressAccumulator {
public:
  StressAccumulator(const ForceInput &in, ForceOut *out,
                    bool compatible_request = false) noexcept
      : out_(out), enabled_(in.compute_stress || compatible_request) {
    if (out_ != nullptr) {
      out_->has_stress = 0;
      std::fill(out_->stress, out_->stress + 9, 0.0);
    }
    if (enabled_) {
      std::fill(strain_, strain_ + 6, 0.0);
    }
  }

  [[nodiscard]] bool enabled() const noexcept { return enabled_; }

  /// Six Voigt components xx, yy, zz, yz, xz, xy, or null when disabled.
  [[nodiscard]] double *data() noexcept { return enabled_ ? strain_ : nullptr; }
  [[nodiscard]] const double *data() const noexcept {
    return enabled_ ? strain_ : nullptr;
  }

  void pair(double fscale, double dx, double dy, double dz) noexcept {
    if (enabled_) {
      accumulatePairStrain(strain_, fscale, dx, dy, dz);
    }
  }

  void bond(double dx, double dy, double dz, double gx, double gy,
            double gz) noexcept {
    if (enabled_) {
      accumulateBondStrain(strain_, dx, dy, dz, gx, gy, gz);
    }
  }

  /// Accumulate -sum_i r_i tensor F_i for a non-periodic configuration.
  /// This is translation invariant when the net force is zero.
  void cartesian(const double *positions, const double *forces,
                 std::size_t n) noexcept {
    if (!enabled_) {
      return;
    }
    for (std::size_t i = 0; i < n; ++i) {
      const double x = positions[3 * i];
      const double y = positions[3 * i + 1];
      const double z = positions[3 * i + 2];
      const double fx = forces[3 * i];
      const double fy = forces[3 * i + 1];
      const double fz = forces[3 * i + 2];
      strain_[0] -= x * fx;
      strain_[1] -= y * fy;
      strain_[2] -= z * fz;
      strain_[3] -= y * fz;
      strain_[4] -= x * fz;
      strain_[5] -= x * fy;
    }
  }

  /// Publish only when requested and the cell has positive finite volume.
  void publish(const double *box) noexcept {
    if (enabled_ && out_ != nullptr) {
      publishCauchyStress(out_, strain_, cellVolume(box));
    }
  }

private:
  ForceOut *out_;
  bool enabled_;
  double strain_[6];
};

} // namespace rgpot
