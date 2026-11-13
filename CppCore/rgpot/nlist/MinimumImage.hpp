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
/// basis. A reduced basis that is still too ill-conditioned keeps the
/// original rows and enumerates a reduced basis of those rows. Singular or
/// non-finite cells
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
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
  // One call, nine fused products. The inline path would call out for
  // each product on a unit that was not itself built with FMA.
  void subtractRoundedHw(std::int32_t nx, std::int32_t ny, std::int32_t nz,
                         double &dx, double &dy, double &dz) const noexcept;
#endif
  std::array<double, 9> cell_{};
  std::array<double, 9> inverse_{};
  std::array<double, 3> dual_norm_{};
  std::array<bool, 3> periodic_{};
  bool narrow_{false};
  // The Minkowski basis stayed ill-conditioned. Fold enumerates the
  // original rows instead of searching that basis.
  bool enumerate_{false};
};

inline void MinimumImage::subtractRounded(std::int32_t nx, std::int32_t ny,
                                          std::int32_t nz, double &dx,
                                          double &dy,
                                          double &dz) const noexcept {
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
  if (rgpot::simd::detail::fma_hw_active()) {
    subtractRoundedHw(nx, ny, nz, dx, dy, dz);
    return;
  }
#endif
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
