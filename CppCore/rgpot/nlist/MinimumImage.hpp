#pragma once
// MIT License
// Copyright 2023--present rgpot developers

#include <array>
#include <cstdint>

namespace rgpot::nlist {

/// Nearest periodic image for row-major lattice vectors: d - n H.
///
/// The working type is double. A cell whose infinity-norm condition
/// estimate would spend more than half the double significand is
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

private:
  std::array<double, 9> cell_{};
  std::array<double, 9> inverse_{};
  std::array<double, 3> dual_norm_{};
  std::array<bool, 3> periodic_{};
  bool narrow_{false};
};

} // namespace rgpot::nlist
