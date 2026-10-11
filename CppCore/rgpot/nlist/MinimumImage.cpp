// MIT License
// Copyright 2023--present rgpot developers

#include "rgpot/nlist/MinimumImage.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace rgpot::nlist {
namespace {

constexpr double kTol = 1e-12;
constexpr int kMaxIt = 10000;

// a*b + c*d + e*f. The last product rounds on its own; each earlier
// product is fused into the sum. std::fma is correctly rounded, so this
// matches a hardware fused multiply-add on SSE, AVX2, AVX-512 and NEON.
double fdot3(double a, double b, double c, double d, double e, double f) {
  return std::fma(a, b, std::fma(c, d, e * f));
}

double frac_of(double x, double y, double z, const double inv[9], int k) {
  return fdot3(x, inv[k], y, inv[3 + k], z, inv[6 + k]);
}

// reduced - (i*c0 + j*c1 + k*c2), each product fused into the sum.
double sub_image(double reduced, double i, double c0, double j, double c1,
                 double k, double c2) {
  return std::fma(-i, c0, std::fma(-j, c1, std::fma(-k, c2, reduced)));
}

double n2(const double v[3]) {
  return fdot3(v[0], v[0], v[1], v[1], v[2], v[2]);
}

double vdot(const double u[3], const double v[3]) {
  return fdot3(u[0], v[0], u[1], v[1], u[2], v[2]);
}

void lin3(const double c[3], const double a[3], const double b[3],
          const double d[3], double out[3]) {
  for (int i = 0; i < 3; ++i)
    out[i] = fdot3(c[0], a[i], c[1], b[i], c[2], d[i]);
}

void sort_by_length(double basis[3][3]) {
  auto swap_row = [&](int i, int j) {
    for (int c = 0; c < 3; ++c) {
      const double t = basis[i][c];
      basis[i][c] = basis[j][c];
      basis[j][c] = t;
    }
  };
  if (n2(basis[0]) > n2(basis[1])) {
    swap_row(0, 1);
  }
  if (n2(basis[1]) > n2(basis[2])) {
    swap_row(1, 2);
  }
  if (n2(basis[0]) > n2(basis[1])) {
    swap_row(0, 1);
  }
}

void gauss_pair(double u[3], double v[3]) {
  for (int it = 0; it < kMaxIt; ++it) {
    if (n2(v) + kTol < n2(u)) {
      for (int c = 0; c < 3; ++c) {
        const double t = u[c];
        u[c] = v[c];
        v[c] = t;
      }
    }
    const double uu = n2(u);
    if (uu < kTol) {
      return;
    }
    const double n = std::round(vdot(u, v) / uu);
    if (n == 0.0) {
      return;
    }
    for (int c = 0; c < 3; ++c)
      v[c] = std::fma(-n, u[c], v[c]);
  }
}

void subtract_plane(double basis[3][3]) {
  const double *u = basis[0];
  const double *v = basis[1];
  double w[3] = {basis[2][0], basis[2][1], basis[2][2]};
  const double uu = n2(u);
  const double vv = n2(v);
  if (uu < kTol || vv < kTol) {
    return;
  }
  const double nu = std::round(vdot(w, u) / uu);
  double w1[3];
  for (int c = 0; c < 3; ++c)
    w1[c] = std::fma(-nu, u[c], w[c]);
  const double nv = std::round(vdot(w1, v) / vv);
  for (int c = 0; c < 3; ++c)
    w[c] = std::fma(-nv, v[c], w1[c]);
  double best[3] = {w[0], w[1], w[2]};
  double best2 = n2(w);
  for (int i = -1; i <= 1; ++i) {
    for (int j = -1; j <= 1; ++j) {
      if (i == 0 && j == 0) {
        continue;
      }
      double t[3];
      for (int c = 0; c < 3; ++c)
        t[c] = std::fma(static_cast<double>(i), u[c],
                        std::fma(static_cast<double>(j), v[c], w[c]));
      const double t2 = n2(t);
      if (t2 < best2) {
        best2 = t2;
        best[0] = t[0];
        best[1] = t[1];
        best[2] = t[2];
      }
    }
  }
  basis[2][0] = best[0];
  basis[2][1] = best[1];
  basis[2][2] = best[2];
}

bool is_minkowski_reduced(const double basis[3][3]) {
  const double na = n2(basis[0]);
  const double nb = n2(basis[1]);
  const double nc = n2(basis[2]);
  if (!(na - nb <= kTol && nb - nc <= kTol)) {
    return false;
  }
  const int rhs_idx[12] = {0, 1, 1, 2, 2, 1, 2, 2, 2, 2, 2, 2};
  const double rows[12][3] = {
      {0, 1, 0}, {0, 0, 1}, {1, 1, 0}, {1, 0, 1}, {0, 1, 1}, {1, -1, 0},
      {1, 0, -1}, {0, 1, -1}, {1, 1, 1}, {1, -1, 1}, {1, 1, -1}, {1, -1, -1},
  };
  const double norms[3] = {std::sqrt(na), std::sqrt(nb), std::sqrt(nc)};
  for (int i = 0; i < 12; ++i) {
    double image[3];
    lin3(rows[i], basis[0], basis[1], basis[2], image);
    if (std::sqrt(n2(image)) + kTol < norms[rhs_idx[i]]) {
      return false;
    }
  }
  return true;
}

void minkowski_reduce(double basis[3][3]) {
  if (is_minkowski_reduced(basis)) {
    return;
  }
  for (int it = 0; it < kMaxIt; ++it) {
    sort_by_length(basis);
    gauss_pair(basis[0], basis[1]);
    subtract_plane(basis);
    sort_by_length(basis);
    if (is_minkowski_reduced(basis)) {
      return;
    }
  }
}

double det_of(const double a[9]) {
  const double m0 = std::fma(-a[5], a[7], a[4] * a[8]);
  const double m1 = std::fma(-a[5], a[6], a[3] * a[8]);
  const double m2 = std::fma(-a[4], a[6], a[3] * a[7]);
  return fdot3(a[0], m0, -a[1], m1, a[2], m2);
}

bool invert(const double a[9], double inv[9]) {
  const double det = det_of(a);
  if (!std::isfinite(det) || det == 0.0) {
    return false;
  }
  inv[0] = std::fma(-a[5], a[7], a[4] * a[8]) / det;
  inv[1] = std::fma(-a[1], a[8], a[2] * a[7]) / det;
  inv[2] = std::fma(-a[2], a[4], a[1] * a[5]) / det;
  inv[3] = std::fma(-a[3], a[8], a[5] * a[6]) / det;
  inv[4] = std::fma(-a[2], a[6], a[0] * a[8]) / det;
  inv[5] = std::fma(-a[0], a[5], a[2] * a[3]) / det;
  inv[6] = std::fma(-a[4], a[6], a[3] * a[7]) / det;
  inv[7] = std::fma(-a[0], a[7], a[1] * a[6]) / det;
  inv[8] = std::fma(-a[1], a[3], a[0] * a[4]) / det;
  for (int k = 0; k < 9; ++k) {
    if (!std::isfinite(inv[k])) {
      return false;
    }
  }
  return true;
}

double inf_norm(const double m[9]) {
  double norm = 0.0;
  for (int row = 0; row < 3; ++row) {
    const double a = std::abs(m[3 * row]);
    const double b = std::abs(m[3 * row + 1]);
    const double c = std::abs(m[3 * row + 2]);
    const double sum = (a + b) + c;
    norm = std::max(norm, sum);
  }
  return norm;
}

bool well_conditioned(const double cell[9], const double inv[9]) {
  const double eps = std::numeric_limits<double>::epsilon();
  const double cond = (inf_norm(cell) * inf_norm(inv)) * eps;
  return cond <= std::sqrt(eps);
}

std::int64_t checked_index(double value) {
  const double limit = std::ldexp(1.0, 63);
  if (!std::isfinite(value) || value <= -limit || value >= limit) {
    throw std::overflow_error("periodic image index is out of range");
  }
  return static_cast<std::int64_t>(value);
}

double squared(double x, double y, double z) {
  return fdot3(x, x, y, y, z, z);
}

void search_bounds(const double cell[9], const double inv[9],
                   const double dual[3], const bool periodic[3], double &dx,
                   double &dy, double &dz) {
  double reduced[3] = {dx, dy, dz};
  double initial[3] = {};
  for (int k = 0; k < 3; ++k) {
    if (!periodic[k]) {
      continue;
    }
    initial[k] = std::round(
        frac_of(reduced[0], reduced[1], reduced[2], inv, k));
    checked_index(initial[k]);
  }
  for (int c = 0; c < 3; ++c) {
    reduced[c] = sub_image(reduced[c], initial[0], cell[c], initial[1],
                           cell[3 + c], initial[2], cell[6 + c]);
  }
  double best[3] = {reduced[0], reduced[1], reduced[2]};
  double best2 = squared(best[0], best[1], best[2]);
  const double radius = std::sqrt(best2);
  std::int64_t low[3] = {};
  std::int64_t high[3] = {};
  std::uint64_t count = 1;
  for (int k = 0; k < 3; ++k) {
    if (periodic[k]) {
      const double f = frac_of(reduced[0], reduced[1], reduced[2], inv, k);
      const double extent = radius * dual[k];
      low[k] = checked_index(std::floor(f - extent) - 1.0);
      high[k] = checked_index(std::ceil(f + extent) + 1.0);
    }
    const auto width = static_cast<std::uint64_t>(high[k]) -
                       static_cast<std::uint64_t>(low[k]) + 1;
    if (width == 0 ||
        count > static_cast<std::uint64_t>(
                    std::numeric_limits<std::int64_t>::max()) /
                    width) {
      throw std::overflow_error("periodic image search count is too large");
    }
    count *= width;
  }
  for (auto i = low[0]; i <= high[0]; ++i) {
    for (auto j = low[1]; j <= high[1]; ++j) {
      for (auto k = low[2]; k <= high[2]; ++k) {
        double cand[3];
        for (int c = 0; c < 3; ++c) {
          cand[c] = sub_image(reduced[c], static_cast<double>(i), cell[c],
                              static_cast<double>(j), cell[3 + c],
                              static_cast<double>(k), cell[6 + c]);
        }
        const double r2 = squared(cand[0], cand[1], cand[2]);
        if (r2 < best2) {
          best[0] = cand[0];
          best[1] = cand[1];
          best[2] = cand[2];
          best2 = r2;
        }
      }
    }
  }
  dx = best[0];
  dy = best[1];
  dz = best[2];
}

void search_narrow(const double cell[9], const double inv[9],
                   const bool periodic[3], double &dx, double &dy,
                   double &dz) {
  int base[3] = {};
  for (int k = 0; k < 3; ++k) {
    if (!periodic[k]) {
      continue;
    }
    const double f = frac_of(dx, dy, dz, inv, k);
    if (!std::isfinite(f)) {
      throw std::overflow_error("periodic image index is out of range");
    }
    base[k] = static_cast<int>(checked_index(std::round(f)));
  }
  double reduced[3] = {dx, dy, dz};
  for (int c = 0; c < 3; ++c) {
    reduced[c] =
        sub_image(reduced[c], static_cast<double>(base[0]), cell[c],
                  static_cast<double>(base[1]), cell[3 + c],
                  static_cast<double>(base[2]), cell[6 + c]);
  }
  double best[3] = {reduced[0], reduced[1], reduced[2]};
  double best2 = squared(best[0], best[1], best[2]);
  for (int i = -1; i <= 1; ++i) {
    if (!periodic[0] && i != 0) {
      continue;
    }
    for (int j = -1; j <= 1; ++j) {
      if (!periodic[1] && j != 0) {
        continue;
      }
      for (int k = -1; k <= 1; ++k) {
        if (!periodic[2] && k != 0) {
          continue;
        }
        double cand[3];
        for (int c = 0; c < 3; ++c) {
          cand[c] = sub_image(reduced[c], static_cast<double>(i), cell[c],
                              static_cast<double>(j), cell[3 + c],
                              static_cast<double>(k), cell[6 + c]);
        }
        const double r2 = squared(cand[0], cand[1], cand[2]);
        if (r2 < best2) {
          best2 = r2;
          best[0] = cand[0];
          best[1] = cand[1];
          best[2] = cand[2];
        }
      }
    }
  }
  dx = best[0];
  dy = best[1];
  dz = best[2];
}

void dual_of(const double inv[9], double dual[3]) {
  for (int k = 0; k < 3; ++k) {
    dual[k] = std::hypot(inv[k], inv[3 + k], inv[6 + k]);
    if (!std::isfinite(dual[k])) {
      throw std::overflow_error("periodic cell inverse is not finite");
    }
  }
}

} // namespace

MinimumImage::MinimumImage(const double *cell,
                           const std::array<bool, 3> &periodic)
    : periodic_(periodic) {
  for (int k = 0; k < 9; ++k) {
    if (!std::isfinite(cell[k])) {
      throw std::invalid_argument("periodic cell must be finite");
    }
    cell_[static_cast<std::size_t>(k)] = cell[k];
  }
  double raw[9];
  for (int k = 0; k < 9; ++k) {
    raw[k] = cell_[static_cast<std::size_t>(k)];
  }
  double inv[9];
  if (!invert(raw, inv)) {
    throw std::invalid_argument("periodic cell must be nonsingular");
  }
  double basis[3][3] = {
      {raw[0], raw[1], raw[2]},
      {raw[3], raw[4], raw[5]},
      {raw[6], raw[7], raw[8]},
  };
  bool narrow = !well_conditioned(raw, inv);
  if (narrow) {
    minkowski_reduce(basis);
    double reduced[9];
    for (int row = 0; row < 3; ++row) {
      for (int col = 0; col < 3; ++col) {
        reduced[3 * row + col] = basis[row][col];
      }
    }
    double rinv[9];
    if (!invert(reduced, rinv)) {
      throw std::invalid_argument("periodic cell must be nonsingular");
    }
    for (int k = 0; k < 9; ++k) {
      cell_[static_cast<std::size_t>(k)] = reduced[k];
      inverse_[static_cast<std::size_t>(k)] = rinv[k];
    }
    narrow = !well_conditioned(reduced, rinv);
    if (!narrow) {
      dual_of(rinv, dual_norm_.data());
    }
  } else {
    for (int k = 0; k < 9; ++k) {
      inverse_[static_cast<std::size_t>(k)] = inv[k];
    }
    dual_of(inv, dual_norm_.data());
  }
  narrow_ = narrow;
}

void MinimumImage::fold(double &dx, double &dy, double &dz) const {
  if (!std::isfinite(dx) || !std::isfinite(dy) || !std::isfinite(dz)) {
    throw std::invalid_argument("pair displacement must be finite");
  }
  bool periodic[3] = {periodic_[0], periodic_[1], periodic_[2]};
  double cell[9];
  double inv[9];
  for (int k = 0; k < 9; ++k) {
    cell[k] = cell_[static_cast<std::size_t>(k)];
    inv[k] = inverse_[static_cast<std::size_t>(k)];
  }
  if (narrow_) {
    search_narrow(cell, inv, periodic, dx, dy, dz);
    return;
  }
  double dual[3] = {dual_norm_[0], dual_norm_[1], dual_norm_[2]};
  search_bounds(cell, inv, dual, periodic, dx, dy, dz);
}

} // namespace rgpot::nlist
