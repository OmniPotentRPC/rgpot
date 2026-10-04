#include "rgpot/nlist/periodic_images.hpp"

#include <linkcell.hpp>

#include <algorithm>
#include <cmath>
#include <limits>

namespace rgpot {
namespace nlist {
namespace {

bool foldIndex(const double *box, const double *r, int out[3]) {
  // Columns of M are the lattice vectors stored as rows of `box`.
  const double m00 = box[0], m01 = box[3], m02 = box[6];
  const double m10 = box[1], m11 = box[4], m12 = box[7];
  const double m20 = box[2], m21 = box[5], m22 = box[8];
  const double det = m00 * (m11 * m22 - m12 * m21) -
                     m01 * (m10 * m22 - m12 * m20) +
                     m02 * (m10 * m21 - m11 * m20);
  if (!(std::fabs(det) > 1e-18) || !std::isfinite(det)) {
    return false;
  }
  const double inv = 1.0 / det;
  const double s0 =
      inv * ((m11 * m22 - m12 * m21) * r[0] + (m02 * m21 - m01 * m22) * r[1] +
             (m01 * m12 - m02 * m11) * r[2]);
  const double s1 =
      inv * ((m12 * m20 - m10 * m22) * r[0] + (m00 * m22 - m02 * m20) * r[1] +
             (m02 * m10 - m00 * m12) * r[2]);
  const double s2 =
      inv * ((m10 * m21 - m11 * m20) * r[0] + (m01 * m20 - m00 * m21) * r[1] +
             (m00 * m11 - m01 * m10) * r[2]);
  if (!std::isfinite(s0) || !std::isfinite(s1) || !std::isfinite(s2)) {
    return false;
  }
  out[0] = static_cast<int>(std::floor(s0));
  out[1] = static_cast<int>(std::floor(s1));
  out[2] = static_cast<int>(std::floor(s2));
  return true;
}

} // namespace

bool periodicImages(const double *positions, std::size_t n, const double *box,
                    double cutoff, std::vector<ImageRow> &out) {
  out.clear();
  if (positions == nullptr || box == nullptr || n < 2 || !(cutoff > 0.0) ||
      !std::isfinite(cutoff)) {
    return false;
  }
  const linkcell::Cell cell = linkcell::Cell::from_vectors(
      {box[0], box[1], box[2]}, {box[3], box[4], box[5]},
      {box[6], box[7], box[8]});
  const std::vector<linkcell::ShiftedPair> rows = linkcell::pairs_within(
      positions, n, cell, cutoff, nullptr, 0.0, true);

  std::vector<int> folded(n * 3, 0);
  for (std::size_t a = 0; a < n; ++a) {
    if (!foldIndex(box, positions + 3 * a, folded.data() + 3 * a)) {
      out.clear();
      return false;
    }
  }

  std::sort(rows.begin(), rows.end(),
            [](const linkcell::ShiftedPair &a, const linkcell::ShiftedPair &b) {
              if (a.i != b.i) {
                return a.i < b.i;
              }
              if (a.j != b.j) {
                return a.j < b.j;
              }
              return a.dist2 < b.dist2;
            });

  for (std::size_t t = 0; t < rows.size();) {
    const int i = rows[t].i;
    const int j = rows[t].j;
    std::size_t u = t + 1;
    while (u < rows.size() && rows[u].i == i && rows[u].j == j) {
      ++u;
    }
    // Shortest image of this unordered pair. Self-images are not a pair.
    if (i != j && i >= 0 && j >= 0 && static_cast<std::size_t>(i) < n &&
        static_cast<std::size_t>(j) < n) {
      const linkcell::ShiftedPair &row = rows[t];
      ImageRow item;
      item.i = i;
      item.j = j;
      for (int k = 0; k < 3; ++k) {
        item.s[k] = row.shift[static_cast<std::size_t>(k)] +
                    folded[3 * static_cast<std::size_t>(i) + k] -
                    folded[3 * static_cast<std::size_t>(j) + k];
      }
      const double shx = item.s[0] * box[0] + item.s[1] * box[3] + item.s[2] * box[6];
      const double shy = item.s[0] * box[1] + item.s[1] * box[4] + item.s[2] * box[7];
      const double shz = item.s[0] * box[2] + item.s[1] * box[5] + item.s[2] * box[8];
      item.dx = positions[3 * i] - positions[3 * j] - shx;
      item.dy = positions[3 * i + 1] - positions[3 * j + 1] - shy;
      item.dz = positions[3 * i + 2] - positions[3 * j + 2] - shz;
      item.r2 = row.dist2;
      out.push_back(item);
    }
    t = u;
  }
  return true;
}

} // namespace nlist
} // namespace rgpot
