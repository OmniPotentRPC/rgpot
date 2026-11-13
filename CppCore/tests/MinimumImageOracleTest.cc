// MIT License
// Copyright 2023--present rgpot developers
//
// Minimum-image folds compared with a brute-force search over a wide
// integer box. The box is fixed from the displacement, not from the
// dual-norm window inside MinimumImage, so a short search cannot confirm
// itself. Non-periodic axes stay at coefficient 0.

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <random>

#include <catch2/catch_all.hpp>

#include "rgpot/nlist/MinimumImage.hpp"

namespace {

constexpr int kWide = 24;
constexpr double kPi = 3.14159265358979323846;

double square(double x, double y, double z) {
  return std::fma(x, x, std::fma(y, y, z * z));
}

bool invert_cell(const double cell[9], double inv[9]) {
  const double m0 = std::fma(-cell[5], cell[7], cell[4] * cell[8]);
  const double m1 = std::fma(-cell[5], cell[6], cell[3] * cell[8]);
  const double m2 = std::fma(-cell[4], cell[6], cell[3] * cell[7]);
  const double det = std::fma(cell[0], m0, std::fma(-cell[1], m1, cell[2] * m2));
  if (!std::isfinite(det) || det == 0.0) {
    return false;
  }
  inv[0] = m0 / det;
  inv[1] = std::fma(-cell[1], cell[8], cell[2] * cell[7]) / det;
  inv[2] = std::fma(-cell[2], cell[4], cell[1] * cell[5]) / det;
  inv[3] = -m1 / det;
  inv[4] = std::fma(-cell[2], cell[6], cell[0] * cell[8]) / det;
  inv[5] = std::fma(-cell[0], cell[5], cell[2] * cell[3]) / det;
  inv[6] = m2 / det;
  inv[7] = std::fma(-cell[0], cell[7], cell[1] * cell[6]) / det;
  inv[8] = std::fma(-cell[1], cell[3], cell[0] * cell[4]) / det;
  for (int k = 0; k < 9; ++k) {
    if (!std::isfinite(inv[k])) {
      return false;
    }
  }
  return true;
}

// Half of 1/||H^{-1}||_F, pulled in by 2^-20. This is the residual ball
// inside which the rounded image is returned without a search.
bool residual_limit2(const double cell[9], double &limit2) {
  double inv[9];
  if (!invert_cell(cell, inv)) {
    return false;
  }
  double fro2 = 0.0;
  for (int k = 0; k < 3; ++k) {
    const double dual = std::hypot(inv[k], inv[3 + k], inv[6 + k]);
    fro2 = std::fma(dual, dual, fro2);
  }
  if (!(fro2 > 0.0) || !std::isfinite(fro2)) {
    return false;
  }
  const double shortest = (1.0 / std::sqrt(fro2)) * (1.0 - 0x1p-20);
  const double limit = 0.5 * shortest;
  limit2 = limit * limit;
  return std::isfinite(limit2) && limit2 > 0.0;
}

double fractional(const double inv[9], int axis, double x, double y, double z) {
  return std::fma(x, inv[axis], std::fma(y, inv[3 + axis], z * inv[6 + axis]));
}

void at_fraction(const double cell[9], double fi, double fj, double fk,
                 double d[3]) {
  for (int c = 0; c < 3; ++c) {
    d[c] = fi * cell[c] + fj * cell[3 + c] + fk * cell[6 + c];
  }
}

void monoclinic(double degrees, double cell[9]) {
  const double rad = degrees * kPi / 180.0;
  const double L = 10.0;
  cell[0] = L;
  cell[1] = 0.0;
  cell[2] = 0.0;
  cell[3] = 0.0;
  cell[4] = L;
  cell[5] = 0.0;
  cell[6] = L * std::cos(rad);
  cell[7] = 0.0;
  cell[8] = L * std::sin(rad);
}

// The minimum image of `d` has every periodic coefficient inside `span`.
// Callers choose displacements for which that is true on these cells.
void expect_minimum(const double cell[9], const std::array<bool, 3> &periodic,
                    double dx, double dy, double dz, int span, bool unique) {
  const rgpot::nlist::MinimumImage image(cell, periodic);
  double x = dx;
  double y = dy;
  double z = dz;
  image.fold(x, y, z);
  INFO("d=" << dx << "," << dy << "," << dz);
  INFO("fold=" << x << "," << y << "," << z);
  REQUIRE(std::isfinite(x));
  REQUIRE(std::isfinite(y));
  REQUIRE(std::isfinite(z));

  const int si = periodic[0] ? span : 0;
  const int sj = periodic[1] ? span : 0;
  const int sk = periodic[2] ? span : 0;
  long double best_r2 = std::numeric_limits<long double>::infinity();
  long double second_r2 = best_r2;
  long double best_dist = best_r2;
  for (int i = -si; i <= si; ++i) {
    for (int j = -sj; j <= sj; ++j) {
      for (int k = -sk; k <= sk; ++k) {
        const long double cx =
            static_cast<long double>(dx) -
            static_cast<long double>(i) * cell[0] -
            static_cast<long double>(j) * cell[3] -
            static_cast<long double>(k) * cell[6];
        const long double cy =
            static_cast<long double>(dy) -
            static_cast<long double>(i) * cell[1] -
            static_cast<long double>(j) * cell[4] -
            static_cast<long double>(k) * cell[7];
        const long double cz =
            static_cast<long double>(dz) -
            static_cast<long double>(i) * cell[2] -
            static_cast<long double>(j) * cell[5] -
            static_cast<long double>(k) * cell[8];
        const long double r2 = cx * cx + cy * cy + cz * cz;
        if (r2 < best_r2) {
          second_r2 = best_r2;
          best_r2 = r2;
        } else if (r2 < second_r2) {
          second_r2 = r2;
        }
        const long double dist = std::abs(cx - x) + std::abs(cy - y) +
                                 std::abs(cz - z);
        best_dist = std::min(best_dist, dist);
      }
    }
  }

  const double scale =
      std::max({1.0, std::abs(dx), std::abs(dy), std::abs(dz)});
  const long double dist_tol =
      static_cast<long double>(256.0 * std::numeric_limits<double>::epsilon() *
                               scale);
  const long double g2 = static_cast<long double>(x) * x +
                         static_cast<long double>(y) * y +
                         static_cast<long double>(z) * z;
  INFO("best_r2=" << static_cast<double>(best_r2)
                  << " g2=" << static_cast<double>(g2)
                  << " dist=" << static_cast<double>(best_dist));
  REQUIRE(best_dist <= dist_tol);
  REQUIRE(g2 <= best_r2 * (1.0L + 1e-9L) + 1e-12L);
  REQUIRE(best_r2 <= g2 * (1.0L + 1e-9L) + 1e-12L);
  if (unique) {
    REQUIRE(second_r2 > best_r2);
  }
}

void expect_ball(const double cell[9], const std::array<bool, 3> &periodic,
                 int axis) {
  double limit2 = 0.0;
  REQUIRE(residual_limit2(cell, limit2));
  double inv[9];
  REQUIRE(invert_cell(cell, inv));
  const double len = std::hypot(cell[3 * axis], cell[3 * axis + 1],
                                cell[3 * axis + 2]);
  REQUIRE(len > 0.0);
  const double u[3] = {cell[3 * axis] / len, cell[3 * axis + 1] / len,
                       cell[3 * axis + 2] / len};
  const auto placed = [&](double s, double r[3]) {
    r[0] = u[0] * s;
    r[1] = u[1] * s;
    r[2] = u[2] * s;
  };
  const auto r2_of = [&](double s) {
    double r[3];
    placed(s, r);
    return square(r[0], r[1], r[2]);
  };
  double s_in = std::sqrt(limit2);
  double s_out = s_in;
  for (int n = 0; n < 64 && !(r2_of(s_in) < limit2); ++n) {
    s_in = std::nextafter(s_in, 0.0);
  }
  for (int n = 0; n < 64 && !(r2_of(s_out) >= limit2); ++n) {
    s_out = std::nextafter(s_out, std::numeric_limits<double>::infinity());
  }
  REQUIRE(r2_of(s_in) < limit2);
  REQUIRE(r2_of(s_out) >= limit2);
  REQUIRE(s_out > s_in);
  double inside[3];
  double outside[3];
  placed(s_in, inside);
  placed(s_out, outside);
  for (const double *r : {inside, outside}) {
    for (int k = 0; k < 3; ++k) {
      if (!periodic[k]) {
        continue;
      }
      const double f = fractional(inv, k, r[0], r[1], r[2]);
      // The residual is this vector: rounding leaves it unmoved, so the
      // ball test sees this length and not a wrapped one.
      REQUIRE(std::abs(f) < 0.5);
      REQUIRE(std::round(f) == 0.0);
    }
  }
  expect_minimum(cell, periodic, inside[0], inside[1], inside[2], kWide, true);
  expect_minimum(cell, periodic, outside[0], outside[1], outside[2], kWide,
                 false);
}

void expect_grid(const double cell[9], const std::array<bool, 3> &periodic,
                 const double *frac, int nfrac) {
  for (int i = 0; i < nfrac; ++i) {
    for (int j = 0; j < nfrac; ++j) {
      for (int k = 0; k < nfrac; ++k) {
        double d[3];
        at_fraction(cell, frac[i], frac[j], frac[k], d);
        expect_minimum(cell, periodic, d[0], d[1], d[2], kWide, false);
      }
    }
  }
}

} // namespace

TEST_CASE("Non-orthogonal minimum images match a wide search",
          "[PairList][triclinic][oracle]") {
  double mono[9];
  monoclinic(100.0, mono);
  const double triclinic[9] = {4.0, 0.2, 0.1, 0.3, 5.0, -0.2, 0.1, -0.3, 6.0};
  const double skewed[9] = {6.0, 1.2, 0.3, 0.8, 5.5, 1.1, 0.4, 0.6, 7.0};
  const double *cells[3] = {mono, triclinic, skewed};
  const std::array<bool, 3> periodic{true, true, true};
  const double frac[] = {-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0};
  for (const double *cell : cells) {
    expect_grid(cell, periodic, frac, 7);
    double edge[3];
    at_fraction(cell, 0.49, -0.51, 1.49, edge);
    expect_minimum(cell, periodic, edge[0], edge[1], edge[2], kWide, false);
    at_fraction(cell, 0.5, 0.5, -0.5, edge);
    expect_minimum(cell, periodic, edge[0], edge[1], edge[2], kWide, false);
  }

  std::mt19937_64 rng(11);
  std::uniform_real_distribution<double> local(-25.0, 25.0);
  for (const double *cell : cells) {
    for (int n = 0; n < 24; ++n) {
      expect_minimum(cell, periodic, local(rng), local(rng), local(rng), kWide,
                     false);
    }
  }

  // Ill-conditioned and fully periodic. The original rows are kept, and
  // the image is a vector of that lattice inside this box.
  const double thin[9] = {1.0, 1.0, 0.0, 0.0, 5.0e-10, 0.0, 0.0, 0.0, 1.0};
  expect_minimum(thin, periodic, 1.0e-12, 1.0e-12, 1.0e-12, kWide, false);
  expect_minimum(thin, periodic, 0.2, 1.0e-12, -0.15, kWide, false);
}

TEST_CASE("Partial periodicity matches a wide search",
          "[PairList][triclinic][oracle]") {
  double mono[9];
  monoclinic(100.0, mono);
  const double triclinic[9] = {4.0, 0.2, 0.1, 0.3, 5.0, -0.2, 0.1, -0.3, 6.0};
  const double ortho[9] = {10.0, 0.0, 0.0, 0.0, 12.0, 0.0, 0.0, 0.0, 8.0};
  const double *cells[3] = {mono, triclinic, ortho};
  // One periodic axis, then two. The open axes are not wrapped.
  const int masks[] = {1, 2, 4, 3, 5, 6};
  const double frac[] = {-2.0, -0.5, 0.0, 0.5, 2.0};
  for (const double *cell : cells) {
    for (int mask : masks) {
      const std::array<bool, 3> periodic{(mask & 1) != 0, (mask & 2) != 0,
                                         (mask & 4) != 0};
      expect_grid(cell, periodic, frac, 5);
      double d[3];
      at_fraction(cell, periodic[0] ? 0.3 : 2.5, periodic[1] ? -0.4 : 1.5,
                  periodic[2] ? 0.2 : -3.0, d);
      expect_minimum(cell, periodic, d[0], d[1], d[2], kWide, false);
    }
  }

  // A long open component makes the dual-norm box enormous, so the fold
  // goes through the reduced periodic rows. The integer box still covers
  // the orthogonal image.
  const std::array<bool, 3> one{true, false, false};
  const std::array<bool, 3> two{true, true, false};
  expect_minimum(ortho, one, 0.4, 0.0, 3.0e4, kWide, false);
  expect_minimum(ortho, two, 0.2, -0.3, 2.0e4, kWide, false);
  expect_minimum(ortho, one, 0.0, 1.0e3, -4.0e3, kWide, false);

  // The thin row is open. A full-cell reduction must not fold that
  // direction as if it were periodic.
  const double thin[9] = {1.0, 1.0, 0.0, 0.0, 5.0e-10, 0.0, 0.0, 0.0, 1.0};
  const std::array<bool, 3> long_axes{true, false, true};
  expect_minimum(thin, long_axes, 0.2, 1.0e-12, -0.15, kWide, false);
  expect_minimum(thin, long_axes, 1.0e-11, 2.0e-12, -1.0e-11, kWide, false);
  const std::array<bool, 3> thin_only{false, true, false};
  expect_minimum(thin, thin_only, 1.0e-12, 2.0e-12, -1.0e-12, kWide, false);
  const std::array<bool, 3> one_long{true, false, false};
  expect_minimum(thin, one_long, 0.25, 1.0e-12, 0.1, kWide, false);
}

TEST_CASE("Residual ball agrees on both sides of its bound",
          "[PairList][triclinic][oracle]") {
  const double cube[9] = {10, 0, 0, 0, 10, 0, 0, 0, 10};
  const double flat[9] = {1, 0, 0, 0, 1.0e4, 0, 0, 0, 1.0e4};
  const double tri[9] = {4.0, 0.2, 0.1, 0.3, 5.0, -0.2, 0.1, -0.3, 6.0};
  double mono[9];
  monoclinic(100.0, mono);
  const std::array<bool, 3> all{true, true, true};
  const std::array<bool, 3> short_only{true, false, false};
  const std::array<bool, 3> yz{false, true, true};
  const std::array<bool, 3> xy{true, true, false};
  expect_ball(cube, all, 0);
  expect_ball(flat, all, 0);
  expect_ball(flat, short_only, 0);
  expect_ball(tri, all, 0);
  expect_ball(tri, yz, 1);
  expect_ball(mono, all, 2);
  expect_ball(mono, xy, 0);
}
