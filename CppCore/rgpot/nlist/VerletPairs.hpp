#pragma once
// MIT License
// Copyright 2023--present rgpot developers
//
// Verlet-skin pair cache for the machine-learned potentials (Metatomic,
// UMA), which consume vesin pair lists with shifts and vectors.
//
// The cache asks vesin for the pairs within cutoff + skin and keeps them,
// with the positions and cell they were built at. A later call reuses them
// while every atom stays within skin/2 of its build position and the cell
// is unchanged; any pair inside the cutoff is then among the candidates
// (Verlet, Phys. Rev. 159, 98-103 (1967), doi:10.1103/PhysRev.159.98).
//
// Output does not depend on whether a call reused the candidates:
// - every vector is recomputed here as r_j - r_i + S H from the current
//   positions, with one fixed operation order;
// - the cutoff test is d^2 < cutoff^2 on that vector;
// - half lists are oriented so that i < j, or i == j with the first
//   nonzero shift positive;
// - pairs are sorted by (i, j, shift).
// A skin of zero rebuilds on every call and yields the same list a skin
// of any size does, bit for bit.
//
// One instance is not safe for concurrent calls; callers keep one per
// image (PotCaps::perImageInstances).

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "rgpot/MetatomicPot/vesin_compat.hpp"
#include "vesin.h"

namespace rgpot {
namespace nlist {

/// Canonical pair list: parallel arrays of length size().
struct VerletPairList {
  std::vector<std::array<int64_t, 2>> pairs;
  std::vector<std::array<int32_t, 3>> shifts;
  std::vector<std::array<double, 3>> vectors;
  [[nodiscard]] size_t size() const noexcept { return pairs.size(); }
};

class VerletPairCache {
public:
  /// `skin` <= 0 rebuilds on every call.
  explicit VerletPairCache(double skin = 0.0) : skin_(std::max(skin, 0.0)) {}

  [[nodiscard]] double skin() const noexcept { return skin_; }
  /// Candidate builds so far (one vesin call each).
  [[nodiscard]] uint64_t rebuilds() const noexcept { return rebuilds_; }
  /// Calls served, rebuilt or not.
  [[nodiscard]] uint64_t calls() const noexcept { return calls_; }
  void invalidate() noexcept { valid_ = false; }

  /// Pairs within `cutoff` for `n` points. `box` is row-major, rows the
  /// cell vectors (vesin's convention).
  VerletPairList compute(const double *positions, size_t n, const double *box,
                         const bool periodic[3], double cutoff, bool full) {
    ++calls_;
    if (needsRebuild(positions, n, box, periodic, cutoff, full))
      rebuild(positions, n, box, periodic, cutoff, full);
    return filter(positions, box, cutoff);
  }

private:
  double skin_;
  bool valid_ = false;
  uint64_t rebuilds_ = 0;
  uint64_t calls_ = 0;
  double cutoff_ = 0.0;
  bool full_ = false;
  std::array<double, 9> box_{};
  std::array<bool, 3> periodic_{};
  std::vector<double> ref_;
  // Candidates, already oriented and sorted.
  std::vector<std::array<int64_t, 2>> cand_pairs_;
  std::vector<std::array<int32_t, 3>> cand_shifts_;

  bool needsRebuild(const double *pos, size_t n, const double *box,
                    const bool periodic[3], double cutoff, bool full) const {
    if (!valid_ || skin_ <= 0.0 || cutoff != cutoff_ || full != full_ ||
        n * 3 != ref_.size())
      return true;
    for (int d = 0; d < 3; ++d)
      if (periodic[d] != periodic_[d])
        return true;
    if (std::memcmp(box, box_.data(), sizeof(double) * 9) != 0)
      return true;
    const double half_sq = 0.25 * skin_ * skin_;
    for (size_t k = 0; k < n; ++k) {
      const double dx = pos[3 * k] - ref_[3 * k];
      const double dy = pos[3 * k + 1] - ref_[3 * k + 1];
      const double dz = pos[3 * k + 2] - ref_[3 * k + 2];
      if (dx * dx + dy * dy + dz * dz > half_sq)
        return true;
    }
    return false;
  }

  static bool shiftPositive(const int32_t s[3]) {
    for (int d = 0; d < 3; ++d)
      if (s[d] != 0)
        return s[d] > 0;
    return false;
  }

  void rebuild(const double *pos, size_t n, const double *box,
               const bool periodic[3], double cutoff, bool full) {
    VesinOptions options{};
    // A small margin keeps pairs at the cutoff in the candidate set even
    // when vesin rounds their distance differently from filter().
    const double margin = 1e-9 * std::max(cutoff, 1.0);
    vesin_compat::fill_neighbor_options(options, cutoff + skin_ + margin, full);
    // One thread: the candidate order is then a function of the input.
    vesin_compat::set_n_threads(options, 1);
    options.return_vectors = false;
    options.return_distances = false;
    options.return_shifts = true;

    VesinNeighborList nl{};
    const char *err = nullptr;
    bool pbc[3] = {periodic[0], periodic[1], periodic[2]};
    const int status = vesin_neighbors(
        reinterpret_cast<const double(*)[3]>(pos), n,
        reinterpret_cast<const double(*)[3]>(box), pbc,
        vesin_compat::make_cpu_device(), options, &nl, &err);
    if (status != EXIT_SUCCESS) {
      std::string msg = "VerletPairCache: vesin_neighbors failed";
      if (err != nullptr)
        msg += std::string(": ") + err;
      vesin_free(&nl);
      valid_ = false;
      throw std::runtime_error(msg);
    }

    struct Cand {
      int64_t i, j;
      std::array<int32_t, 3> s;
    };
    std::vector<Cand> c;
    c.reserve(nl.length);
    for (size_t k = 0; k < nl.length; ++k) {
      Cand x{static_cast<int64_t>(nl.pairs[k][0]),
             static_cast<int64_t>(nl.pairs[k][1]),
             {nl.shifts[k][0], nl.shifts[k][1], nl.shifts[k][2]}};
      if (!full && (x.i > x.j || (x.i == x.j && !shiftPositive(x.s.data())))) {
        std::swap(x.i, x.j);
        for (auto &v : x.s)
          v = -v;
      }
      c.push_back(x);
    }
    vesin_free(&nl);
    std::sort(c.begin(), c.end(), [](const Cand &a, const Cand &b) {
      if (a.i != b.i)
        return a.i < b.i;
      if (a.j != b.j)
        return a.j < b.j;
      return a.s < b.s;
    });
    c.erase(std::unique(c.begin(), c.end(),
                        [](const Cand &a, const Cand &b) {
                          return a.i == b.i && a.j == b.j && a.s == b.s;
                        }),
            c.end());
    cand_pairs_.resize(c.size());
    cand_shifts_.resize(c.size());
    for (size_t k = 0; k < c.size(); ++k) {
      cand_pairs_[k] = {c[k].i, c[k].j};
      cand_shifts_[k] = c[k].s;
    }
    ref_.assign(pos, pos + 3 * n);
    std::memcpy(box_.data(), box, sizeof(double) * 9);
    for (int d = 0; d < 3; ++d)
      periodic_[d] = periodic[d];
    cutoff_ = cutoff;
    full_ = full;
    valid_ = true;
    ++rebuilds_;
  }

  VerletPairList filter(const double *pos, const double *box,
                        double cutoff) const {
    VerletPairList out;
    const double cut_sq = cutoff * cutoff;
    for (size_t k = 0; k < cand_pairs_.size(); ++k) {
      const int64_t i = cand_pairs_[k][0];
      const int64_t j = cand_pairs_[k][1];
      const auto &s = cand_shifts_[k];
      std::array<double, 3> v{};
      for (int d = 0; d < 3; ++d) {
        const double sh = static_cast<double>(s[0]) * box[d] +
                          static_cast<double>(s[1]) * box[3 + d] +
                          static_cast<double>(s[2]) * box[6 + d];
        v[d] = (pos[3 * j + d] - pos[3 * i + d]) + sh;
      }
      const double d2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
      if (d2 < cut_sq) {
        out.pairs.push_back(cand_pairs_[k]);
        out.shifts.push_back(s);
        out.vectors.push_back(v);
      }
    }
    return out;
  }
};

} // namespace nlist
} // namespace rgpot
