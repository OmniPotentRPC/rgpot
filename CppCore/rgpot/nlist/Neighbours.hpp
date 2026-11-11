#pragma once
// MIT License
// Copyright 2023--present rgpot developers
//
// One neighbour layer for the classical potentials: cell binning, the
// minimum image, an optional Verlet list, a streaming walk that does not
// store pairs, and a deterministic OpenMP split.

#include "rgpot/nlist/MinimumImage.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace rgpot::nlist {

/// How the minimum image is formed. Round is the orthorhombic truncation
/// used by the pair kernels. Half is one correction about the half-box,
/// used after coordinates have been wrapped into the cell. General is
/// MinimumImage, for a non-orthogonal cell or a periodic side shorter
/// than the search.
enum class Image : std::uint8_t { Round, Half, General };

/// Stored keeps a Verlet list of candidate pairs. Stream walks the cells
/// and stores nothing beyond the atoms and the bins. Auto stores when the
/// candidate index array fits in the memory the OS reports as available,
/// and streams otherwise.
enum class Mode : std::uint8_t { Auto, Stored, Stream };

struct Spec {
  double cutoff = 0;
  double skin = 0;
  std::array<bool, 3> periodic{{true, true, true}};
  Image image = Image::Round;
  Mode mode = Mode::Auto;
};

struct Range {
  std::int64_t begin = 0;
  std::int64_t end = 0;
};

/// Thread count from OMP_NUM_THREADS. Unset, empty, or a non-positive
/// value is one thread, so a call that does not ask for threads matches
/// the serial pair order.
[[nodiscard]] int threadsRequested();

/// Contiguous slice of `n` items for `thread` in `[0, nthreads)`. The
/// slices cover `n` without overlap, in thread order.
[[nodiscard]] Range partition(std::int64_t n, int thread, int nthreads);

/// Bytes the process may still allocate, from MemAvailable. Zero when
/// that figure is not reported.
[[nodiscard]] std::uint64_t availableBytes();

/// Candidate pairs around one configuration.
///
/// A stored list is reused while every atom has moved by at most half
/// the skin, and the cell, the count and the spec are unchanged. Skin 0
/// reuses the list only when the coordinates and the cell match.
class Neighbours {
public:
  void update(const double *xyz, std::size_t n, const double *box9,
              const Spec &spec);

  [[nodiscard]] Mode mode() const noexcept { return mode_; }
  [[nodiscard]] bool streaming() const noexcept { return streaming_; }
  [[nodiscard]] std::uint64_t builds() const noexcept { return builds_; }
  [[nodiscard]] std::size_t slots() const noexcept { return nbr_.size(); }
  [[nodiscard]] const std::vector<std::int32_t> &offsets() const noexcept {
    return off_;
  }
  [[nodiscard]] const std::vector<std::int32_t> &neighbours() const noexcept {
    return nbr_;
  }

  /// Directed pairs for atoms in `[begin, end)`. `dx` is `r_i - r_j`
  /// after the selected image. `r2` is the squared length of that vector.
  /// Candidates lie inside the cutoff plus the skin; the caller drops the
  /// rest. In stored mode the partners of each atom are in increasing
  /// index order.
  template <typename Fn>
  void forDirected(std::int64_t begin, std::int64_t end, Fn &&fn) const {
    if (streaming_) {
      walkCells(begin, end, fn);
    } else {
      walkStored(begin, end, fn);
    }
  }

private:
  struct SpecKey {
    double cutoff = 0;
    double skin = 0;
    Image image = Image::Round;
    Mode mode = Mode::Auto;
    std::array<bool, 3> periodic{{true, true, true}};
    bool operator==(const SpecKey &o) const {
      return cutoff == o.cutoff && skin == o.skin && image == o.image &&
             mode == o.mode && periodic == o.periodic;
    }
  };

  void buildCells(const double *xyz, std::size_t n);
  void buildStored(const double *xyz, std::size_t n);
  [[nodiscard]] std::uint64_t candidateBound() const;
  [[nodiscard]] bool reusable(const double *xyz, std::size_t n,
                              const double *box9, const Spec &spec) const;
  void fold(double &dx, double &dy, double &dz) const;

  template <typename Fn>
  void walkStored(std::int64_t begin, std::int64_t end, Fn &fn) const {
    for (std::int64_t i = begin; i < end; ++i) {
      const auto a = static_cast<std::size_t>(i);
      const auto lo = off_[a];
      const auto hi = off_[a + 1];
      for (std::int32_t p = lo; p < hi; ++p) {
        const auto j = nbr_[static_cast<std::size_t>(p)];
        double dx = 0;
        double dy = 0;
        double dz = 0;
        foldPair(static_cast<std::int32_t>(i), j, dx, dy, dz);
        const double r2 = dx * dx + dy * dy + dz * dz;
        fn(static_cast<std::int32_t>(i), j, dx, dy, dz, r2);
      }
    }
  }

  template <typename Fn>
  void walkCells(std::int64_t begin, std::int64_t end, Fn &fn) const {
    const double search2 = search_ * search_;
    for (std::int64_t i = begin; i < end; ++i) {
      const auto a = static_cast<std::size_t>(i);
      const int cell = cell_of_[a];
      int iz = cell % nz_;
      int iy = (cell / nz_) % ny_;
      int ix = cell / (nz_ * ny_);
      int seen[27];
      int nseen = 0;
      for (int ox = -1; ox <= 1; ++ox) {
        for (int oy = -1; oy <= 1; ++oy) {
          for (int oz = -1; oz <= 1; ++oz) {
            const int id = cellId(ix + ox, iy + oy, iz + oz);
            if (id < 0) {
              continue;
            }
            bool dup = false;
            for (int s = 0; s < nseen; ++s) {
              if (seen[s] == id) {
                dup = true;
                break;
              }
            }
            if (dup) {
              continue;
            }
            seen[nseen++] = id;
            const auto c = static_cast<std::size_t>(id);
            const auto a0 = cell_start_[c];
            const auto a1 = cell_start_[c + 1];
            for (std::int32_t q = a0; q < a1; ++q) {
              const auto j = cell_atoms_[static_cast<std::size_t>(q)];
              if (j == static_cast<std::int32_t>(i)) {
                continue;
              }
              double dx = 0;
              double dy = 0;
              double dz = 0;
              foldPair(static_cast<std::int32_t>(i), j, dx, dy, dz);
              const double r2 = dx * dx + dy * dy + dz * dz;
              if (r2 <= search2) {
                fn(static_cast<std::int32_t>(i), j, dx, dy, dz, r2);
              }
            }
          }
        }
      }
    }
  }

  void foldPair(std::int32_t i, std::int32_t j, double &dx, double &dy,
                double &dz) const;
  [[nodiscard]] int cellId(int ix, int iy, int iz) const;

  std::vector<std::int32_t> off_;
  std::vector<std::int32_t> nbr_;
  std::vector<std::int32_t> cell_of_;
  std::vector<std::int32_t> cell_start_;
  std::vector<std::int32_t> cell_atoms_;
  std::vector<double> ref_;
  double box_[9]{};
  double len_[3]{};
  double inv_[3]{};
  bool periodic_[3]{};
  int nx_ = 1;
  int ny_ = 1;
  int nz_ = 1;
  double search_ = 0;
  Image image_ = Image::Round;
  Mode mode_ = Mode::Auto;
  SpecKey key_{};
  MinimumImage general_;
  bool use_general_ = false;
  bool streaming_ = true;
  bool built_ = false;
  std::size_t n_ = 0;
  std::uint64_t builds_ = 0;
  const double *xyz_ = nullptr;
};

} // namespace rgpot::nlist
