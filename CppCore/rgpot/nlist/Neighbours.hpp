#pragma once
// MIT License
// Copyright 2023--present rgpot developers
//
// One neighbour layer for the classical potentials: cell binning, the
// minimum image, an optional Verlet list, a streaming walk that does not
// store pairs, and a deterministic OpenMP split.

#include "rgpot/nlist/MinimumImage.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif

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
      walkCells(begin, end, fn, false);
    } else {
      walkStored(begin, end, fn);
    }
  }

  /// The pairs of `forDirected` with `j > i` only. Geometry for `j <= i`
  /// is not computed. A serial two-pass potential uses this so each
  /// unordered pair is visited once per pass.
  template <typename Fn>
  void forHalf(std::int64_t begin, std::int64_t end, Fn &&fn) const {
    if (streaming_) {
      walkCells(begin, end, fn, true);
    } else {
      auto half = [&](std::int32_t i, std::int32_t j, double dx, double dy, double dz,
                      double r2) {
        if (j > i) {
          fn(i, j, dx, dy, dz, r2);
        }
      };
      walkStored(begin, end, half);
    }
  }

  /// One walk of the directed pairs, split by cell across `nthreads`.
  /// The bins are built once and shared. `fn(thread, i, j, dx, dy, dz, r2)`.
  /// A thread owns each cell, so a pair is visited once.
  template <typename Fn>
  void forCellTeam(bool half, int nthreads, Fn &&fn) const {
    if (nthreads < 1) {
      nthreads = 1;
    }
    if (!fastTiles([&](const auto &tile, const auto &neigh, const auto &nneigh, std::size_t ncells,
                       bool cell_shift, double search2, double L0, double L1, double L2, double i0,
                       double i1, double i2, bool p0, bool p1, bool p2) {
#if defined(_OPENMP)
#pragma omp parallel num_threads(nthreads)
#endif
          {
#if defined(_OPENMP)
            const int thread = omp_get_thread_num();
#else
            const int thread = 0;
#endif
            const auto range = partition(static_cast<std::int64_t>(ncells), thread, nthreads);
            visitTiles(tile, neigh, nneigh, static_cast<std::size_t>(range.begin),
                       static_cast<std::size_t>(range.end), 0, n_, half, cell_shift, search2, L0,
                       L1, L2, i0, i1, i2, p0, p1, p2,
                       [&](std::int32_t i, std::int32_t j, double dx, double dy, double dz,
                           double r2) { fn(thread, i, j, dx, dy, dz, r2); });
          }
        })) {
#if defined(_OPENMP)
#pragma omp parallel num_threads(nthreads)
#endif
      {
#if defined(_OPENMP)
        const int thread = omp_get_thread_num();
#else
        const int thread = 0;
#endif
        const auto range = partition(static_cast<std::int64_t>(n_), thread, nthreads);
        auto adapt = [&](std::int32_t i, std::int32_t j, double dx, double dy, double dz,
                         double r2) { fn(thread, i, j, dx, dy, dz, r2); };
        if (half) {
          forHalf(range.begin, range.end, adapt);
        } else {
          forDirected(range.begin, range.end, adapt);
        }
      }
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

  template <typename Loop>
  bool fastTiles(Loop &&loop) const {
    if (use_general_ || image_ != Image::Round || n_ < 2) {
      return false;
    }
    const auto ncells = static_cast<std::size_t>(nx_) * static_cast<std::size_t>(ny_) *
                        static_cast<std::size_t>(nz_);
    const bool cell_shift = (!periodic_[0] || nx_ >= 5) && (!periodic_[1] || ny_ >= 5) &&
                            (!periodic_[2] || nz_ >= 5);
    struct Nbor {
      int id;
      double sx;
      double sy;
      double sz;
    };
    std::vector<Nbor> neigh(ncells * 27);
    std::vector<int> nneigh(ncells, 0);
    for (std::size_t c = 0; c < ncells; ++c) {
      const int cell = static_cast<int>(c);
      int iz = cell % nz_;
      int iy = (cell / nz_) % ny_;
      int ix = cell / (nz_ * ny_);
      int seen[27];
      int nseen = 0;
      for (int ox = -1; ox <= 1; ++ox) {
        for (int oy = -1; oy <= 1; ++oy) {
          for (int oz = -1; oz <= 1; ++oz) {
            int jx = ix + ox;
            int jy = iy + oy;
            int jz = iz + oz;
            int sa = 0;
            int sb = 0;
            int sc = 0;
            if (periodic_[0]) {
              if (jx < 0) {
                jx += nx_;
                sa = -1;
              } else if (jx >= nx_) {
                jx -= nx_;
                sa = 1;
              }
            } else if (jx < 0 || jx >= nx_) {
              continue;
            }
            if (periodic_[1]) {
              if (jy < 0) {
                jy += ny_;
                sb = -1;
              } else if (jy >= ny_) {
                jy -= ny_;
                sb = 1;
              }
            } else if (jy < 0 || jy >= ny_) {
              continue;
            }
            if (periodic_[2]) {
              if (jz < 0) {
                jz += nz_;
                sc = -1;
              } else if (jz >= nz_) {
                jz -= nz_;
                sc = 1;
              }
            } else if (jz < 0 || jz >= nz_) {
              continue;
            }
            const int id = (jx * ny_ + jy) * nz_ + jz;
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
            seen[nseen] = id;
            neigh[c * 27 + static_cast<std::size_t>(nseen)] =
                Nbor{id, sa * len_[0], sb * len_[1], sc * len_[2]};
            ++nseen;
          }
        }
      }
      nneigh[c] = nseen;
    }
    struct Tile {
      double x;
      double y;
      double z;
      std::int32_t id;
    };
    std::vector<Tile> tile(n_);
    for (std::size_t q = 0; q < n_; ++q) {
      const auto j = cell_atoms_[q];
      const auto u = static_cast<std::size_t>(j) * 3;
      tile[q] = Tile{xyz_[u], xyz_[u + 1], xyz_[u + 2], j};
    }
    loop(tile, neigh, nneigh, ncells, cell_shift, search_ * search_, len_[0], len_[1], len_[2],
         inv_[0], inv_[1], inv_[2], periodic_[0], periodic_[1], periodic_[2]);
    return true;
  }

  template <typename Tile, typename Nbor, typename Fn>
  void visitTiles(const std::vector<Tile> &tile, const std::vector<Nbor> &neigh,
                  const std::vector<int> &nneigh, std::size_t c0, std::size_t c1,
                  std::int64_t atom0, std::size_t atom1, bool half, bool cell_shift,
                  double search2, double L0, double L1, double L2, double i0, double i1,
                  double i2, bool p0, bool p1, bool p2, Fn &&fn) const {
    auto round_axis = [](double delta, double inv, double length, bool periodic, int &shift) {
      if (!periodic) {
        shift = 0;
        return delta;
      }
      const double t = delta * inv;
      if (!(t > -1.0e15 && t < 1.0e15)) {
        shift = 0;
        return delta;
      }
      const double shifted = t + std::copysign(0.5, t);
      const auto image = static_cast<long long>(shifted);
      shift = static_cast<int>(image);
      return delta - length * static_cast<double>(image);
    };
    for (std::size_t c = c0; c < c1; ++c) {
      const auto a0 = cell_start_[c];
      const auto a1 = cell_start_[c + 1];
      if (a0 == a1) {
        continue;
      }
      const int nn = nneigh[c];
      const Nbor *nb = neigh.data() + c * 27;
      for (std::int32_t q = a0; q < a1; ++q) {
        const Tile &a = tile[static_cast<std::size_t>(q)];
        if (a.id < atom0 || static_cast<std::size_t>(a.id) >= atom1) {
          continue;
        }
        for (int k = 0; k < nn; ++k) {
          const Nbor &src = nb[k];
          const auto b0 = static_cast<std::size_t>(cell_start_[static_cast<std::size_t>(src.id)]);
          const auto b1 =
              static_cast<std::size_t>(cell_start_[static_cast<std::size_t>(src.id) + 1]);
          for (std::size_t r = b0; r < b1; ++r) {
            const Tile &b = tile[r];
            if (b.id == a.id || (half && b.id <= a.id)) {
              continue;
            }
            double dx;
            double dy;
            double dz;
            if (cell_shift) {
              dx = (a.x - b.x) - src.sx;
              dy = (a.y - b.y) - src.sy;
              dz = (a.z - b.z) - src.sz;
            } else {
              int sa = 0;
              int sb = 0;
              int sc = 0;
              dx = round_axis(a.x - b.x, i0, L0, p0, sa);
              dy = round_axis(a.y - b.y, i1, L1, p1, sb);
              dz = round_axis(a.z - b.z, i2, L2, p2, sc);
            }
            const double r2 = dx * dx + dy * dy + dz * dz;
            if (r2 <= search2) {
              fn(a.id, b.id, dx, dy, dz, r2);
            }
          }
        }
      }
    }
  }

  template <typename Fn>
  void walkCells(std::int64_t begin, std::int64_t end, Fn &fn, bool half) const {
    const double search2 = search_ * search_;
    if (!fastTiles([&](const auto &tile, const auto &neigh, const auto &nneigh,
                       std::size_t ncells, bool cell_shift, double search2_fast, double L0,
                       double L1, double L2, double i0, double i1, double i2, bool p0, bool p1,
                       bool p2) {
          visitTiles(tile, neigh, nneigh, 0, ncells, begin, static_cast<std::size_t>(end), half,
                     cell_shift, search2_fast, L0, L1, L2, i0, i1, i2, p0, p1, p2, fn);
        })) {
      walkCellsScattered(begin, end, fn, half, search2);
    }
  }

  template <typename Fn>
  void walkCellsScattered(std::int64_t begin, std::int64_t end, Fn &fn, bool half,
                          double search2) const {
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
              if (half && !(j > static_cast<std::int32_t>(i))) {
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

inline void Neighbours::fold(double &dx, double &dy, double &dz) const {
  if (use_general_) {
    general_.fold(dx, dy, dz);
    return;
  }
  auto round_visit = [](double t) {
    if (!std::isfinite(t)) {
      return 0.0;
    }
    const double shifted = t + std::copysign(0.5, t);
    if (!std::isfinite(shifted) || shifted >= 9223372036854775808.0 ||
        shifted <= -9223372036854775808.0) {
      return 0.0;
    }
    return static_cast<double>(static_cast<long long>(shifted));
  };
  if (image_ == Image::Half) {
    const double h0 = 0.5 * len_[0];
    const double h1 = 0.5 * len_[1];
    const double h2 = 0.5 * len_[2];
    if (periodic_[0]) {
      if (dx > h0) {
        dx -= len_[0];
      } else if (dx < -h0) {
        dx += len_[0];
      }
    }
    if (periodic_[1]) {
      if (dy > h1) {
        dy -= len_[1];
      } else if (dy < -h1) {
        dy += len_[1];
      }
    }
    if (periodic_[2]) {
      if (dz > h2) {
        dz -= len_[2];
      } else if (dz < -h2) {
        dz += len_[2];
      }
    }
    return;
  }
  if (periodic_[0]) {
    dx -= len_[0] * round_visit(dx * inv_[0]);
  }
  if (periodic_[1]) {
    dy -= len_[1] * round_visit(dy * inv_[1]);
  }
  if (periodic_[2]) {
    dz -= len_[2] * round_visit(dz * inv_[2]);
  }
}

inline void Neighbours::foldPair(std::int32_t i, std::int32_t j, double &dx, double &dy,
                                 double &dz) const {
  const auto ui = static_cast<std::size_t>(i) * 3;
  const auto uj = static_cast<std::size_t>(j) * 3;
  dx = xyz_[ui] - xyz_[uj];
  dy = xyz_[ui + 1] - xyz_[uj + 1];
  dz = xyz_[ui + 2] - xyz_[uj + 2];
  fold(dx, dy, dz);
}

inline int Neighbours::cellId(int ix, int iy, int iz) const {
  if (periodic_[0]) {
    if (ix < 0) {
      ix += nx_;
    } else if (ix >= nx_) {
      ix -= nx_;
    }
  } else if (ix < 0 || ix >= nx_) {
    return -1;
  }
  if (periodic_[1]) {
    if (iy < 0) {
      iy += ny_;
    } else if (iy >= ny_) {
      iy -= ny_;
    }
  } else if (iy < 0 || iy >= ny_) {
    return -1;
  }
  if (periodic_[2]) {
    if (iz < 0) {
      iz += nz_;
    } else if (iz >= nz_) {
      iz -= nz_;
    }
  } else if (iz < 0 || iz >= nz_) {
    return -1;
  }
  return (ix * ny_ + iy) * nz_ + iz;
}

} // namespace rgpot::nlist
