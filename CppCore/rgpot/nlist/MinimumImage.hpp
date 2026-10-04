#pragma once
// MIT License
// Copyright 2023--present rgpot developers

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace rgpot::nlist {

/// Nearest periodic image for row-major lattice vectors: d - n H.
/// The dual-vector bound includes every image shorter than the initial
/// rounded fractional image. No lattice reduction or fixed image radius
/// is assumed. Singular or ill-conditioned cells and unrepresentable
/// search bounds throw.
class MinimumImage {
public:
  MinimumImage() = default;
  MinimumImage(const double *cell, const std::array<bool, 3> &periodic)
      : periodic_(periodic) {
    for (int k = 0; k < 9; ++k) {
      if (!std::isfinite(cell[k]))
        throw std::invalid_argument("periodic cell must be finite");
      cell_[k] = cell[k];
    }
    const auto &a = cell_;
    const long double det = a[0] * (a[4] * a[8] - a[5] * a[7]) -
                            a[1] * (a[3] * a[8] - a[5] * a[6]) +
                            a[2] * (a[3] * a[7] - a[4] * a[6]);
    if (!std::isfinite(det) || det == 0.0L)
      throw std::invalid_argument("periodic cell must be nonsingular");
    inverse_ = {
        (a[4] * a[8] - a[5] * a[7]) / det, (a[2] * a[7] - a[1] * a[8]) / det,
        (a[1] * a[5] - a[2] * a[4]) / det, (a[5] * a[6] - a[3] * a[8]) / det,
        (a[0] * a[8] - a[2] * a[6]) / det, (a[2] * a[3] - a[0] * a[5]) / det,
        (a[3] * a[7] - a[4] * a[6]) / det, (a[1] * a[6] - a[0] * a[7]) / det,
        (a[0] * a[4] - a[1] * a[3]) / det};
    long double cell_norm = 0.0L, inverse_norm = 0.0L;
    for (int row = 0; row < 3; ++row) {
      long double cell_sum = 0.0L, inverse_sum = 0.0L;
      for (int col = 0; col < 3; ++col) {
        cell_sum += std::abs(cell_[3 * row + col]);
        inverse_sum += std::abs(inverse_[3 * row + col]);
      }
      cell_norm = std::max(cell_norm, cell_sum);
      inverse_norm = std::max(inverse_norm, inverse_sum);
    }
    const long double eps = std::numeric_limits<long double>::epsilon();
    // Refuse cells whose condition estimate consumes over half the
    // working precision; finite image bounds alone do not protect the inverse.
    if (!(cell_norm * inverse_norm * eps <= std::sqrt(eps)))
      throw std::invalid_argument(
          "periodic cell is too ill-conditioned for image bounds");
    for (int k = 0; k < 3; ++k) {
      dual_norm_[k] = std::hypot(inverse_[k], inverse_[3 + k], inverse_[6 + k]);
      if (!std::isfinite(dual_norm_[k]))
        throw std::overflow_error("periodic cell inverse is not finite");
    }
  }

  void fold(double &dx, double &dy, double &dz) const {
    if (!std::isfinite(dx) || !std::isfinite(dy) || !std::isfinite(dz))
      throw std::invalid_argument("pair displacement must be finite");
    std::array<long double, 3> reduced{dx, dy, dz};
    std::array<long double, 3> initial{};
    for (int k = 0; k < 3; ++k) {
      if (periodic_[k])
        initial[k] =
            std::round(reduced[0] * inverse_[k] + reduced[1] * inverse_[3 + k] +
                       reduced[2] * inverse_[6 + k]);
      checkedIndex(initial[k]);
    }
    for (int c = 0; c < 3; ++c)
      for (int k = 0; k < 3; ++k)
        reduced[c] -= initial[k] * cell_[3 * k + c];
    auto best = reduced;
    long double best2 = squaredNorm(best);
    const long double radius = std::sqrt(best2);
    std::array<std::int64_t, 3> low{}, high{};
    std::uint64_t count = 1;
    for (int k = 0; k < 3; ++k) {
      if (periodic_[k]) {
        const long double f = reduced[0] * inverse_[k] +
                              reduced[1] * inverse_[3 + k] +
                              reduced[2] * inverse_[6 + k];
        const long double extent = radius * dual_norm_[k];
        // Outward integer rounding adds one lattice plane of numerical
        // margin around the real-arithmetic image bounds.
        low[k] = checkedIndex(std::floor(f - extent) - 1.0L);
        high[k] = checkedIndex(std::ceil(f + extent) + 1.0L);
      }
      const auto width = static_cast<std::uint64_t>(high[k]) -
                         static_cast<std::uint64_t>(low[k]) + 1;
      if (width == 0 || count > static_cast<std::uint64_t>(
                                    std::numeric_limits<std::int64_t>::max()) /
                                    width)
        throw std::overflow_error("periodic image search count is too large");
      count *= width;
    }
    for (auto i = low[0]; i <= high[0]; ++i)
      for (auto j = low[1]; j <= high[1]; ++j)
        for (auto k = low[2]; k <= high[2]; ++k) {
          std::array<long double, 3> candidate{};
          for (int c = 0; c < 3; ++c)
            candidate[c] =
                reduced[c] - i * cell_[c] - j * cell_[3 + c] - k * cell_[6 + c];
          const long double r2 = squaredNorm(candidate);
          if (r2 < best2) {
            best = candidate;
            best2 = r2;
          }
        }
    dx = static_cast<double>(best[0]);
    dy = static_cast<double>(best[1]);
    dz = static_cast<double>(best[2]);
  }

private:
  static long double squaredNorm(const std::array<long double, 3> &v) {
    return v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
  }
  static std::int64_t checkedIndex(long double value) {
    const long double limit = std::ldexp(1.0L, 63);
    if (!std::isfinite(value) || value <= -limit + 1.0L ||
        value >= limit - 1.0L)
      throw std::overflow_error("periodic image index is out of range");
    return static_cast<std::int64_t>(value);
  }
  std::array<long double, 9> cell_{};
  std::array<long double, 9> inverse_{};
  std::array<long double, 3> dual_norm_{};
  std::array<bool, 3> periodic_{};
};

} // namespace rgpot::nlist
