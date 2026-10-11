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

double dmul(double a, double b) { return a * b; }

double dadd(double a, double b) { return a + b; }

double dsub(double a, double b) { return a - b; }

double n2(const double v[3]) {
  const double x = dmul(v[0], v[0]);
  const double y = dmul(v[1], v[1]);
  const double z = dmul(v[2], v[2]);
  return dadd(dadd(x, y), z);
}

void vadd(const double u[3], const double v[3], double out[3]) {
  out[0] = dadd(u[0], v[0]);
  out[1] = dadd(u[1], v[1]);
  out[2] = dadd(u[2], v[2]);
}

void vscale(double s, const double v[3], double out[3]) {
  out[0] = dmul(s, v[0]);
  out[1] = dmul(s, v[1]);
  out[2] = dmul(s, v[2]);
}

double vdot(const double u[3], const double v[3]) {
  const double x = dmul(u[0], v[0]);
  const double y = dmul(u[1], v[1]);
  const double z = dmul(u[2], v[2]);
  return dadd(dadd(x, y), z);
}

void lin3(const double c[3], const double a[3], const double b[3],
          const double d[3], double out[3]) {
  double t0[3], t1[3], t2[3], s[3];
  vscale(c[0], a, t0);
  vscale(c[1], b, t1);
  vscale(c[2], d, t2);
  vadd(t0, t1, s);
  vadd(s, t2, out);
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
    if (dadd(n2(v), kTol) < n2(u)) {
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
    double step[3];
    vscale(-n, u, step);
    double next[3];
    vadd(v, step, next);
    v[0] = next[0];
    v[1] = next[1];
    v[2] = next[2];
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
  double su[3], w1[3];
  vscale(-nu, u, su);
  vadd(w, su, w1);
  const double nv = std::round(vdot(w1, v) / vv);
  double sv[3];
  vscale(-nv, v, sv);
  vadd(w1, sv, w);
  double best[3] = {w[0], w[1], w[2]};
  double best2 = n2(w);
  for (int i = -1; i <= 1; ++i) {
    for (int j = -1; j <= 1; ++j) {
      if (i == 0 && j == 0) {
        continue;
      }
      double si[3], sj[3], t0[3], t[3];
      vscale(static_cast<double>(i), u, si);
      vscale(static_cast<double>(j), v, sj);
      vadd(w, si, t0);
      vadd(t0, sj, t);
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
  if (!(dadd(na, -nb) <= kTol && dadd(nb, -nc) <= kTol)) {
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
    if (dadd(std::sqrt(n2(image)), kTol) < norms[rhs_idx[i]]) {
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
  const double a48 = dmul(a[4], a[8]);
  const double a57 = dmul(a[5], a[7]);
  const double m0 = dsub(a48, a57);
  const double a38 = dmul(a[3], a[8]);
  const double a56 = dmul(a[5], a[6]);
  const double m1 = dsub(a38, a56);
  const double a37 = dmul(a[3], a[7]);
  const double a46 = dmul(a[4], a[6]);
  const double m2 = dsub(a37, a46);
  const double t0 = dmul(a[0], m0);
  const double t1 = dmul(a[1], m1);
  const double t2 = dmul(a[2], m2);
  return dadd(dsub(t0, t1), t2);
}

bool invert(const double a[9], double inv[9]) {
  const double det = det_of(a);
  if (!std::isfinite(det) || det == 0.0) {
    return false;
  }
  const double a48 = dmul(a[4], a[8]);
  const double a57 = dmul(a[5], a[7]);
  const double a27 = dmul(a[2], a[7]);
  const double a18 = dmul(a[1], a[8]);
  const double a15 = dmul(a[1], a[5]);
  const double a24 = dmul(a[2], a[4]);
  const double a56 = dmul(a[5], a[6]);
  const double a38 = dmul(a[3], a[8]);
  const double a08 = dmul(a[0], a[8]);
  const double a26 = dmul(a[2], a[6]);
  const double a23 = dmul(a[2], a[3]);
  const double a05 = dmul(a[0], a[5]);
  const double a37 = dmul(a[3], a[7]);
  const double a46 = dmul(a[4], a[6]);
  const double a16 = dmul(a[1], a[6]);
  const double a07 = dmul(a[0], a[7]);
  const double a04 = dmul(a[0], a[4]);
  const double a13 = dmul(a[1], a[3]);
  inv[0] = dsub(a48, a57) / det;
  inv[1] = dsub(a27, a18) / det;
  inv[2] = dsub(a15, a24) / det;
  inv[3] = dsub(a56, a38) / det;
  inv[4] = dsub(a08, a26) / det;
  inv[5] = dsub(a23, a05) / det;
  inv[6] = dsub(a37, a46) / det;
  inv[7] = dsub(a16, a07) / det;
  inv[8] = dsub(a04, a13) / det;
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
    const double sum = dadd(dadd(a, b), c);
    norm = std::max(norm, sum);
  }
  return norm;
}

bool well_conditioned(const double cell[9], const double inv[9]) {
  const double eps = std::numeric_limits<double>::epsilon();
  const double cond = dmul(dmul(inf_norm(cell), inf_norm(inv)), eps);
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
  const double xx = dmul(x, x);
  const double yy = dmul(y, y);
  const double zz = dmul(z, z);
  return dadd(dadd(xx, yy), zz);
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
    const double t0 = dmul(reduced[0], inv[k]);
    const double t1 = dmul(reduced[1], inv[3 + k]);
    const double t2 = dmul(reduced[2], inv[6 + k]);
    initial[k] = std::round(dadd(dadd(t0, t1), t2));
    checked_index(initial[k]);
  }
  for (int c = 0; c < 3; ++c) {
    const double n0 = dmul(initial[0], cell[c]);
    const double n1 = dmul(initial[1], cell[3 + c]);
    const double n2 = dmul(initial[2], cell[6 + c]);
    reduced[c] = dsub(reduced[c], dadd(dadd(n0, n1), n2));
  }
  double best[3] = {reduced[0], reduced[1], reduced[2]};
  double best2 = squared(best[0], best[1], best[2]);
  const double radius = std::sqrt(best2);
  std::int64_t low[3] = {};
  std::int64_t high[3] = {};
  std::uint64_t count = 1;
  for (int k = 0; k < 3; ++k) {
    if (periodic[k]) {
      const double t0 = dmul(reduced[0], inv[k]);
      const double t1 = dmul(reduced[1], inv[3 + k]);
      const double t2 = dmul(reduced[2], inv[6 + k]);
      const double f = dadd(dadd(t0, t1), t2);
      const double extent = dmul(radius, dual[k]);
      low[k] = checked_index(std::floor(dsub(f, extent)) - 1.0);
      high[k] = checked_index(std::ceil(dadd(f, extent)) + 1.0);
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
          const double ni = dmul(static_cast<double>(i), cell[c]);
          const double nj = dmul(static_cast<double>(j), cell[3 + c]);
          const double nk = dmul(static_cast<double>(k), cell[6 + c]);
          cand[c] = dsub(reduced[c], dadd(dadd(ni, nj), nk));
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
    const double t0 = dmul(dx, inv[k]);
    const double t1 = dmul(dy, inv[3 + k]);
    const double t2 = dmul(dz, inv[6 + k]);
    const double f = dadd(dadd(t0, t1), t2);
    if (!std::isfinite(f)) {
      throw std::overflow_error("periodic image index is out of range");
    }
    base[k] = static_cast<int>(checked_index(std::round(f)));
  }
  double reduced[3] = {dx, dy, dz};
  for (int c = 0; c < 3; ++c) {
    const double n0 = dmul(static_cast<double>(base[0]), cell[c]);
    const double n1 = dmul(static_cast<double>(base[1]), cell[3 + c]);
    const double n2 = dmul(static_cast<double>(base[2]), cell[6 + c]);
    reduced[c] = dsub(reduced[c], dadd(dadd(n0, n1), n2));
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
          const double ni = dmul(static_cast<double>(i), cell[c]);
          const double nj = dmul(static_cast<double>(j), cell[3 + c]);
          const double nk = dmul(static_cast<double>(k), cell[6 + c]);
          cand[c] = dsub(reduced[c], dadd(dadd(ni, nj), nk));
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
