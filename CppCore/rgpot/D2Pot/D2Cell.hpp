#pragma once
// MIT License
// Copyright 2023--present rgpot developers

// Cell geometry shared by the D2 kernels: rows of h are the lattice
// vectors, in the kernel's length unit.

#include <array>
#include <cmath>
#include <stdexcept>

namespace rgpot::d2::detail {

struct Cell {
  bool periodic = false;
  std::array<double, 9> h{};    // row k is lattice vector k
  std::array<double, 9> hinv{}; // h^{-1}: fractional s = r h^{-1}
  double volume = 0.0;

  /// @p box in Angstrom, row-major; @p bohr_to_ang converts it to bohr.
  static Cell from_box(const double *box, double bohr_to_ang) {
    Cell c;
    if (box == nullptr) {
      return c;
    }
    bool zero = true;
    for (int k = 0; k < 9; ++k) {
      c.h[k] = box[k] / bohr_to_ang;
      zero = zero && box[k] == 0.0;
    }
    if (zero) {
      return c;
    }
    const auto &h = c.h;
    const double det = h[0] * (h[4] * h[8] - h[5] * h[7]) - h[1] * (h[3] * h[8] - h[5] * h[6]) +
                       h[2] * (h[3] * h[7] - h[4] * h[6]);
    if (!(std::abs(det) > 0.0) || !std::isfinite(det)) {
      throw std::invalid_argument("the cell is singular; pass an all-zero box for no periodicity");
    }
    c.periodic = true;
    c.volume = std::abs(det);
    const double id = 1.0 / det;
    c.hinv = {(h[4] * h[8] - h[5] * h[7]) * id, (h[2] * h[7] - h[1] * h[8]) * id,
              (h[1] * h[5] - h[2] * h[4]) * id, (h[5] * h[6] - h[3] * h[8]) * id,
              (h[0] * h[8] - h[2] * h[6]) * id, (h[2] * h[3] - h[0] * h[5]) * id,
              (h[3] * h[7] - h[4] * h[6]) * id, (h[1] * h[6] - h[0] * h[7]) * id,
              (h[0] * h[4] - h[1] * h[3]) * id};
    return c;
  }

  /// Fractional coordinates of @p r: s_k = sum_a r_a hinv[a][k].
  void fractional(const double *r, double *s) const {
    for (int k = 0; k < 3; ++k) {
      s[k] = r[0] * hinv[k] + r[1] * hinv[3 + k] + r[2] * hinv[6 + k];
    }
  }

  void cartesian(const double *s, double *r) const {
    for (int a = 0; a < 3; ++a) {
      r[a] = s[0] * h[a] + s[1] * h[3 + a] + s[2] * h[6 + a];
    }
  }

  /// Moves @p d to fractional coordinates in [-1/2, 1/2].
  void wrap(double *d) const {
    double s[3];
    fractional(d, s);
    for (double &v : s) {
      v -= std::nearbyint(v);
    }
    cartesian(s, d);
  }

  /// Image counts per lattice vector that reach every image of a wrapped
  /// difference vector within @p rc.
  [[nodiscard]] std::array<int, 3> repetitions(double rc) const {
    std::array<int, 3> rep{};
    for (int k = 0; k < 3; ++k) {
      const int i = (k + 1) % 3;
      const int j = (k + 2) % 3;
      const double cx = h[3 * i + 1] * h[3 * j + 2] - h[3 * i + 2] * h[3 * j + 1];
      const double cy = h[3 * i + 2] * h[3 * j + 0] - h[3 * i + 0] * h[3 * j + 2];
      const double cz = h[3 * i + 0] * h[3 * j + 1] - h[3 * i + 1] * h[3 * j + 0];
      const double spacing = volume / std::sqrt(cx * cx + cy * cy + cz * cz);
      rep[k] = static_cast<int>(std::ceil(rc / spacing + 0.5));
    }
    return rep;
  }
};

} // namespace rgpot::d2::detail
