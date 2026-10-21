// MIT License
// Copyright 2023--present rgpot developers

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

#include <catch2/catch_all.hpp>

#include "rgpot/nlist/MinimumImage.hpp"

using Catch::Matchers::WithinAbs;

namespace {

/// The previous long-double search, kept as the agreement oracle.
class LongDoubleMinimumImage {
public:
  LongDoubleMinimumImage(const double *cell,
                         const std::array<bool, 3> &periodic)
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
    if (!(cell_norm * inverse_norm * eps <= std::sqrt(eps)))
      throw std::invalid_argument(
          "periodic cell is too ill-conditioned for image bounds");
    for (int k = 0; k < 3; ++k) {
      dual_norm_[k] =
          std::hypot(inverse_[k], inverse_[3 + k], inverse_[6 + k]);
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
        initial[k] = std::round(reduced[0] * inverse_[k] +
                                reduced[1] * inverse_[3 + k] +
                                reduced[2] * inverse_[6 + k]);
      checkedIndex(initial[k]);
    }
    for (int c = 0; c < 3; ++c)
      for (int k = 0; k < 3; ++k)
        reduced[c] -= initial[k] * cell_[3 * k + c];
    auto best = reduced;
    long double best2 =
        best[0] * best[0] + best[1] * best[1] + best[2] * best[2];
    const long double radius = std::sqrt(best2);
    std::array<std::int64_t, 3> low{}, high{};
    std::uint64_t count = 1;
    for (int k = 0; k < 3; ++k) {
      if (periodic_[k]) {
        const long double f = reduced[0] * inverse_[k] +
                              reduced[1] * inverse_[3 + k] +
                              reduced[2] * inverse_[6 + k];
        const long double extent = radius * dual_norm_[k];
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
            candidate[c] = reduced[c] - i * cell_[c] - j * cell_[3 + c] -
                           k * cell_[6 + c];
          const long double r2 = candidate[0] * candidate[0] +
                                 candidate[1] * candidate[1] +
                                 candidate[2] * candidate[2];
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

void expectSameImage(const LongDoubleMinimumImage &old_image,
                     const rgpot::nlist::MinimumImage &image, double x,
                     double y, double z, double tol) {
  double ox = x, oy = y, oz = z;
  double nx = x, ny = y, nz = z;
  old_image.fold(ox, oy, oz);
  image.fold(nx, ny, nz);
  CAPTURE(x, y, z, ox, oy, oz, nx, ny, nz);
  REQUIRE_THAT(nx, WithinAbs(ox, tol));
  REQUIRE_THAT(ny, WithinAbs(oy, tol));
  REQUIRE_THAT(nz, WithinAbs(oz, tol));
  const double ro = std::sqrt(ox * ox + oy * oy + oz * oz);
  const double rn = std::sqrt(nx * nx + ny * ny + nz * nz);
  REQUIRE(rn <= ro + tol);
}

} // namespace

TEST_CASE("Double minimum image agrees with the long-double search",
          "[PairList][triclinic]") {
  // 1e-9 is the contract stated on MinimumImage: ordinary cells, skewed
  // cells, and near-ill-conditioned cells that still pass the guard.
  constexpr double kTol = 1e-9;
  const std::array<std::array<double, 9>, 3> cells{{
      {10.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0},
      {4.0, 0.2, 0.1, 0.3, 5.0, -0.2, 0.1, -0.3, 6.0},
      {4.0, 0.0, 0.0, 15.6, 0.5, 0.0, 0.2, 0.3, 5.0},
  }};
  std::mt19937_64 rng(7);
  std::uniform_real_distribution<double> local(-12.0, 12.0);
  std::uniform_real_distribution<double> far(-1.0e8, 1.0e8);
  for (const auto &cell : cells) {
    for (unsigned mask = 0; mask < 8; ++mask) {
      std::array<bool, 3> periodic{};
      for (int k = 0; k < 3; ++k)
        periodic[k] = (mask & (1u << k)) != 0;
      CAPTURE(cell, mask);
      const LongDoubleMinimumImage old_image(cell.data(), periodic);
      const rgpot::nlist::MinimumImage image(cell.data(), periodic);
      for (int n = 0; n < 12; ++n)
        expectSameImage(old_image, image, local(rng), local(rng), local(rng),
                        kTol);
      if (mask == 7) {
        for (int n = 0; n < 8; ++n)
          expectSameImage(old_image, image, far(rng), far(rng), far(rng),
                          kTol);
      }
    }
    const std::array<bool, 3> periodic{true, true, true};
    const LongDoubleMinimumImage old_image(cell.data(), periodic);
    const rgpot::nlist::MinimumImage image(cell.data(), periodic);
    for (int ix = -2; ix <= 2; ++ix)
      for (int iy = -2; iy <= 2; ++iy)
        for (int iz = -2; iz <= 2; ++iz)
          expectSameImage(old_image, image, ix * 0.5 * cell[0],
                          iy * 0.5 * cell[4], iz * 0.5 * cell[8], kTol);
  }

  // One shear stays on the double path (condition about 5e7). Where
  // long double is wider than double, a second shear sits just inside
  // that guard (condition about 1e9) and takes the double-double search.
  // Where long double is double, the guard is the double guard, so the
  // second shear sits just inside it (condition about 6.7e7).
  // Displacements stay near the short lattice vector so the image box
  // stays small.
  const double near_guard =
      std::numeric_limits<long double>::digits >
              std::numeric_limits<double>::digits
          ? 2.0e-9
          : 3.0e-8;
  const std::array<std::array<double, 9>, 2> shears{{
      {1.0, 1.0, 0.0, 0.0, 4.0e-8, 0.0, 0.0, 0.0, 1.0},
      {1.0, 1.0, 0.0, 0.0, near_guard, 0.0, 0.0, 0.0, 1.0},
  }};
  std::uniform_real_distribution<double> tiny(-1.0e-8, 1.0e-8);
  const std::array<bool, 3> periodic{true, true, true};
  for (const auto &cell : shears) {
    CAPTURE(cell);
    const LongDoubleMinimumImage old_image(cell.data(), periodic);
    const rgpot::nlist::MinimumImage image(cell.data(), periodic);
    for (int n = 0; n < 16; ++n)
      expectSameImage(old_image, image, tiny(rng), tiny(rng), tiny(rng), kTol);
  }
  const double past_guard[9] = {1.0, 1.0, 0.0, 0.0, 5.0e-10, 0.0, 0.0, 0.0,
                                1.0};
  REQUIRE_THROWS_AS(rgpot::nlist::MinimumImage(past_guard, periodic),
                    std::invalid_argument);
}

TEST_CASE("Benchmark minimum image fold", "[.][benchmark]") {
  const double triclinic[9] = {4.0, 0.2, 0.1, 0.3, 5.0, -0.2, 0.1, -0.3, 6.0};
  const std::array<bool, 3> periodic{true, true, true};
  const LongDoubleMinimumImage old_image(triclinic, periodic);
  const rgpot::nlist::MinimumImage image(triclinic, periodic);
  std::mt19937_64 rng(1);
  std::uniform_real_distribution<double> u(-20.0, 20.0);
  constexpr int kN = 20000;
  std::vector<double> disp(3 * kN);
  for (double &v : disp)
    v = u(rng);
  BENCHMARK("long-double fold, 20000 displacements") {
    double sink = 0.0;
    for (int i = 0; i < kN; ++i) {
      double dx = disp[3 * i], dy = disp[3 * i + 1], dz = disp[3 * i + 2];
      old_image.fold(dx, dy, dz);
      sink += dx + dy + dz;
    }
    return sink;
  };
  BENCHMARK("double fold, 20000 displacements") {
    double sink = 0.0;
    for (int i = 0; i < kN; ++i) {
      double dx = disp[3 * i], dy = disp[3 * i + 1], dz = disp[3 * i + 2];
      image.fold(dx, dy, dz);
      sink += dx + dy + dz;
    }
    return sink;
  };

}
