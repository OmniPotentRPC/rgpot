#pragma once
// MIT License
// Copyright 2023--present rgpot developers

/**
 * @brief Every periodic image inside a cutoff.
 *
 * The stored neighbour list keeps one image of each partner, the minimum
 * image. That is the whole shell only while the cutoff sits under half of
 * every periodic face. Past that, a pair can have two or more images inside
 * the cutoff, and an atom can see its own lattice translations. This walk
 * yields each of those images once, as an unordered pair.
 *
 * The box is row-major. Its rows are the lattice vectors. `dx, dy, dz` is
 * `r_i - r_j` of the image, the same sense as Neighbours::forHalf.
 */

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace rgpot::nlist {

/// Perpendicular spacing of the face opposite lattice axis `axis`.
inline double faceSpacing(const double *box, int axis) noexcept {
  const double *u = box + 3 * ((axis + 1) % 3);
  const double *v = box + 3 * ((axis + 2) % 3);
  const double cx = u[1] * v[2] - u[2] * v[1];
  const double cy = u[2] * v[0] - u[0] * v[2];
  const double cz = u[0] * v[1] - u[1] * v[0];
  const double area = std::sqrt(cx * cx + cy * cy + cz * cz);
  const double m00 = box[0], m01 = box[1], m02 = box[2];
  const double m10 = box[3], m11 = box[4], m12 = box[5];
  const double m20 = box[6], m21 = box[7], m22 = box[8];
  const double vol = std::abs(m00 * (m11 * m22 - m12 * m21) -
                              m01 * (m10 * m22 - m12 * m20) +
                              m02 * (m10 * m21 - m11 * m20));
  if (!(area > 0.0) || !(vol > 0.0)) {
    return 0.0;
  }
  return vol / area;
}

/// True when one minimum image is not the whole shell on a periodic side.
inline bool cutoffExceedsHalfFace(const double *box, double cutoff,
                                  const std::array<bool, 3> &periodic) noexcept {
  for (int axis = 0; axis < 3; ++axis) {
    if (!periodic[static_cast<std::size_t>(axis)]) {
      continue;
    }
    const double spacing = faceSpacing(box, axis);
    if (!(2.0 * cutoff < spacing)) {
      return true;
    }
  }
  return false;
}

namespace detail {

inline bool invertColumns(const double m[9], double inv[9]) noexcept {
  const double det = m[0] * (m[4] * m[8] - m[5] * m[7]) -
                     m[3] * (m[1] * m[8] - m[2] * m[7]) +
                     m[6] * (m[1] * m[5] - m[2] * m[4]);
  if (!(std::abs(det) > 0.0) || !std::isfinite(det)) {
    return false;
  }
  const double id = 1.0 / det;
  inv[0] = (m[4] * m[8] - m[5] * m[7]) * id;
  inv[1] = (m[2] * m[7] - m[1] * m[8]) * id;
  inv[2] = (m[1] * m[5] - m[2] * m[4]) * id;
  inv[3] = (m[5] * m[6] - m[3] * m[8]) * id;
  inv[4] = (m[0] * m[8] - m[2] * m[6]) * id;
  inv[5] = (m[2] * m[3] - m[0] * m[5]) * id;
  inv[6] = (m[3] * m[7] - m[4] * m[6]) * id;
  inv[7] = (m[1] * m[6] - m[0] * m[7]) * id;
  inv[8] = (m[0] * m[4] - m[1] * m[3]) * id;
  return true;
}

} // namespace detail

/// Unordered pairs `(i, j)` with `j >= i`, one call per image inside the
/// cutoff. A translation of an atom onto itself is omitted. `fn` is
/// `fn(i, j, dx, dy, dz, r2)`.
template <typename Fn>
void visitHalfImages(const double *pos, std::size_t n, const double *box,
                     const std::array<bool, 3> &periodic, double cutoff, Fn &&fn) {
  if (n < 1 || !(cutoff > 0.0) || !std::isfinite(cutoff)) {
    return;
  }
  // Columns of `columns` are the lattice vectors, so r = columns * fractional.
  const double columns[9] = {box[0], box[1], box[2], box[3], box[4],
                             box[5], box[6], box[7], box[8]};
  double inverse[9];
  if (!detail::invertColumns(columns, inverse)) {
    return;
  }
  int extent[3];
  for (int axis = 0; axis < 3; ++axis) {
    if (!periodic[static_cast<std::size_t>(axis)]) {
      extent[axis] = 0;
      continue;
    }
    const double spacing = faceSpacing(box, axis);
    if (!(spacing > 0.0)) {
      return;
    }
    // One past the shell, so a face that rounds onto either image is covered.
    const double shells = std::ceil(cutoff / spacing);
    extent[axis] = static_cast<int>(shells) + 1;
  }
  const double cut2 = cutoff * cutoff;
  for (std::size_t i = 0; i < n; ++i) {
    const double xi = pos[3 * i];
    const double yi = pos[3 * i + 1];
    const double zi = pos[3 * i + 2];
    for (std::size_t j = i; j < n; ++j) {
      const double dx0 = xi - pos[3 * j];
      const double dy0 = yi - pos[3 * j + 1];
      const double dz0 = zi - pos[3 * j + 2];
      const double f0 = inverse[0] * dx0 + inverse[3] * dy0 + inverse[6] * dz0;
      const double f1 = inverse[1] * dx0 + inverse[4] * dy0 + inverse[7] * dz0;
      const double f2 = inverse[2] * dx0 + inverse[5] * dy0 + inverse[8] * dz0;
      const int c0 = static_cast<int>(std::lround(f0));
      const int c1 = static_cast<int>(std::lround(f1));
      const int c2 = static_cast<int>(std::lround(f2));
      for (int h = c0 - extent[0]; h <= c0 + extent[0]; ++h) {
        for (int k = c1 - extent[1]; k <= c1 + extent[1]; ++k) {
          for (int l = c2 - extent[2]; l <= c2 + extent[2]; ++l) {
            if (!periodic[0] && h != 0) {
              continue;
            }
            if (!periodic[1] && k != 0) {
              continue;
            }
            if (!periodic[2] && l != 0) {
              continue;
            }
            const double sx = h * box[0] + k * box[3] + l * box[6];
            const double sy = h * box[1] + k * box[4] + l * box[7];
            const double sz = h * box[2] + k * box[5] + l * box[8];
            const double dx = dx0 - sx;
            const double dy = dy0 - sy;
            const double dz = dz0 - sz;
            const double r2 = dx * dx + dy * dy + dz * dz;
            if (!(r2 > 0.0) || !(r2 < cut2)) {
              continue;
            }
            fn(static_cast<std::int32_t>(i), static_cast<std::int32_t>(j), dx, dy, dz,
               r2);
          }
        }
      }
    }
  }
}

} // namespace rgpot::nlist
