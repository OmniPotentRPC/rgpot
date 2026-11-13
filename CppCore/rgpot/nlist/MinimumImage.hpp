#pragma once
// MIT License
// Copyright 2023--present rgpot developers

#include "rgpot/simd/exact_fma.hpp"

#include <array>
#include <cstdint>

namespace rgpot::nlist {

/// Nearest periodic image for row-major lattice vectors: d - n H.
///
/// The working type is double. Each product-sum is rgpot::simd::mul_add:
/// a hardware fused multiply-add when the processor has one, and the
/// exactly rounded TwoProduct/TwoSum emulation otherwise. This translation
/// unit is compiled with contraction off, so the folded bits do not depend
/// on the instruction set. A cell whose infinity-norm
/// condition estimate would spend more than half the double significand is
/// Minkowski-reduced (Nguyen and Stehlé, ACM Trans. Algorithms 5, 46
/// (2009), doi:10.1145/1597036.1597050) and the image is taken in that
/// basis. A reduced basis that is still too ill-conditioned searches the
/// 27 images of coefficients in {-1, 0, 1}. Singular or non-finite cells
/// throw. An index that does not fit in an int64 throws.
class MinimumImage {
public:
  MinimumImage() = default;
  MinimumImage(const double *cell, const std::array<bool, 3> &periodic);
  void fold(double &dx, double &dy, double &dz) const;

  /// Rounded fractional coefficients of `fold`. True when `fold` leaves
  /// that rounded image in place, so `subtractRounded` reproduces its bits.
  /// Rounding is not the minimum image on every triclinic cell; a false
  /// return means the search replaced it and the coefficients cannot.
  bool tryRounded(double dx, double dy, double dz, std::int32_t n[3]) const;

  /// Lower bound on the shortest nonzero periodic lattice vector of the
  /// basis `fold` uses. Zero when that bound is not available.
  [[nodiscard]] double shortestPeriodic() const;

  /// True when every Cartesian perturbation of Euclidean length at most
  /// the skin reach still folds to `subtractRounded` of this `n`.
  /// `shortest` is a lower bound from `shortestPeriodic`. The skin reach
  /// is the same one the orthorhombic index uses: each atom may move by
  /// half the skin, so the pair difference moves by at most a skin.
  bool roundedHolds(double dx, double dy, double dz, const std::int32_t n[3],
                    double skin, double shortest) const;

  /// Subtract `n` in the basis `fold` uses, with the same product-sum as
  /// the rounded step of `fold`.
  void subtractRounded(std::int32_t nx, std::int32_t ny, std::int32_t nz,
                       double &dx, double &dy, double &dz) const noexcept;

private:
  std::array<double, 9> cell_{};
  std::array<double, 9> inverse_{};
  std::array<double, 3> dual_norm_{};
  std::array<bool, 3> periodic_{};
  bool narrow_{false};
};

inline void MinimumImage::subtractRounded(std::int32_t nx, std::int32_t ny,
                                          std::int32_t nz, double &dx,
                                          double &dy,
                                          double &dz) const noexcept {
  const double i = static_cast<double>(nx);
  const double j = static_cast<double>(ny);
  const double k = static_cast<double>(nz);
  const double *c = cell_.data();
  // k, then j, then i. This is the product-sum inside sub_image.
  dx = rgpot::simd::mul_add(
      -i, c[0],
      rgpot::simd::mul_add(-j, c[3], rgpot::simd::mul_add(-k, c[6], dx)));
  dy = rgpot::simd::mul_add(
      -i, c[1],
      rgpot::simd::mul_add(-j, c[4], rgpot::simd::mul_add(-k, c[7], dy)));
  dz = rgpot::simd::mul_add(
      -i, c[2],
      rgpot::simd::mul_add(-j, c[5], rgpot::simd::mul_add(-k, c[8], dz)));
}

} // namespace rgpot::nlist
