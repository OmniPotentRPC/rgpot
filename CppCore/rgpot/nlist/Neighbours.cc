// MIT License
// Copyright 2023--present rgpot developers

#include "rgpot/nlist/Neighbours.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace rgpot::nlist {
namespace {

int axisCount(double length, double edge) {
  if (!(length > 0.0) || !(edge > 0.0)) {
    return 1;
  }
  auto n = static_cast<long>(std::floor(length / edge));
  if (n < 1) {
    n = 1;
  }
  while (n > 1 && length / static_cast<double>(n) < edge) {
    --n;
  }
  return static_cast<int>(n);
}

} // namespace

int threadsRequested() {
#if defined(_OPENMP)
  const char *env = std::getenv("OMP_NUM_THREADS");
  if (env == nullptr || env[0] == '\0') {
    return 1;
  }
  char *end = nullptr;
  const long parsed = std::strtol(env, &end, 10);
  if (end == env || parsed < 1) {
    return 1;
  }
  if (parsed > 0x7fffffffL) {
    return 0x7fffffff;
  }
  return static_cast<int>(parsed);
#else
  return 1;
#endif
}

Range partition(std::int64_t n, int thread, int nthreads) {
  if (nthreads < 1) {
    nthreads = 1;
  }
  if (thread < 0 || thread >= nthreads || n <= 0) {
    return {};
  }
  const auto nt = static_cast<std::int64_t>(nthreads);
  const auto t = static_cast<std::int64_t>(thread);
  Range out;
  out.begin = (n * t) / nt;
  out.end = (n * (t + 1)) / nt;
  return out;
}

std::uint64_t availableBytes() {
  std::ifstream in("/proc/meminfo");
  std::string key;
  std::uint64_t kb = 0;
  std::string unit;
  while (in >> key >> kb >> unit) {
    if (key == "MemAvailable:") {
      return kb * 1024ULL;
    }
  }
  return 0;
}

bool Neighbours::reusable(const double *xyz, std::size_t n, const double *box9,
                          const Spec &spec) const {
  if (!built_ || n != n_) {
    return false;
  }
  SpecKey key;
  key.cutoff = spec.cutoff;
  key.skin = spec.skin;
  key.image = spec.image;
  key.mode = spec.mode;
  key.periodic = spec.periodic;
  if (!(key == key_) || std::memcmp(box_, box9, sizeof(box_)) != 0) {
    return false;
  }
  // A streaming walk trusts the bins. Atoms that have moved may sit in
  // another cell, so the bins are reused only for an identical geometry.
  // A stored list keeps its candidates out to the skin.
  if (streaming_) {
    return std::memcmp(ref_.data(), xyz, n * 3 * sizeof(double)) == 0;
  }
  return withinSkin(ref_.data(), xyz, n, spec.skin);
}

void Neighbours::buildCells(const double *xyz, std::size_t n) {
  const double edge = search_ > 0.0 ? search_ : 1.0;
  nx_ = axisCount(len_[0], edge);
  ny_ = axisCount(len_[1], edge);
  nz_ = axisCount(len_[2], edge);
  const double bytes =
      static_cast<double>(nx_) * static_cast<double>(ny_) *
      static_cast<double>(nz_) * static_cast<double>(sizeof(std::int32_t));
  const auto avail = availableBytes();
  if (avail > 0 && bytes > static_cast<double>(avail)) {
    nx_ = ny_ = nz_ = 1;
  }
  const auto ncells = static_cast<std::size_t>(nx_) * static_cast<std::size_t>(ny_) *
                      static_cast<std::size_t>(nz_);
  cell_start_.assign(ncells + 1, 0);
  cell_of_.resize(n);
  for (std::size_t a = 0; a < n; ++a) {
    int ix = 0;
    int iy = 0;
    int iz = 0;
    auto bin = [](double x, double length, int naxis) {
      if (!(length > 0.0) || naxis <= 1) {
        return 0;
      }
      double t = x / length * static_cast<double>(naxis);
      if (!(t >= 0.0)) {
        t = 0.0;
      }
      auto i = static_cast<int>(t);
      if (i >= naxis) {
        i = naxis - 1;
      }
      if (i < 0) {
        i = 0;
      }
      return i;
    };
    ix = bin(xyz[3 * a], len_[0], nx_);
    iy = bin(xyz[3 * a + 1], len_[1], ny_);
    iz = bin(xyz[3 * a + 2], len_[2], nz_);
    const int id = (ix * ny_ + iy) * nz_ + iz;
    cell_of_[a] = id;
    ++cell_start_[static_cast<std::size_t>(id) + 1];
  }
  for (std::size_t c = 0; c < ncells; ++c) {
    cell_start_[c + 1] += cell_start_[c];
  }
  cell_atoms_.assign(n, 0);
  cell_x_.resize(n);
  cell_y_.resize(n);
  cell_z_.resize(n);
  std::vector<std::int32_t> fill(cell_start_.begin(), cell_start_.end() - 1);
  for (std::size_t a = 0; a < n; ++a) {
    const auto c = static_cast<std::size_t>(cell_of_[a]);
    const auto slot = static_cast<std::size_t>(fill[c]++);
    cell_atoms_[slot] = static_cast<std::int32_t>(a);
    cell_x_[slot] = xyz[3 * a];
    cell_y_[slot] = xyz[3 * a + 1];
    cell_z_[slot] = xyz[3 * a + 2];
  }
}

void Neighbours::buildStencil() const {
  const auto ncells = static_cast<std::size_t>(nx_) * static_cast<std::size_t>(ny_) *
                      static_cast<std::size_t>(nz_);
  // A periodic side shorter than three cells maps two of {-1, 0, +1} onto
  // the same bin. The first offset in that loop wins, matching the walk.
  const bool nodup = (!periodic_[0] || nx_ >= 3) && (!periodic_[1] || ny_ >= 3) &&
                     (!periodic_[2] || nz_ >= 3);
  sten_off_.assign(ncells + 1, 0);
  sten_.clear();
  sten_.reserve(ncells * (nodup ? 27 : 14));
  for (std::size_t c = 0; c < ncells; ++c) {
    sten_off_[c] = static_cast<std::int32_t>(sten_.size());
    const int cell = static_cast<int>(c);
    const int iz = cell % nz_;
    const int iy = (cell / nz_) % ny_;
    const int ix = cell / (nz_ * ny_);
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
          if (!nodup) {
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
          }
          sten_.push_back(Sten{id, static_cast<std::int8_t>(sa), static_cast<std::int8_t>(sb),
                               static_cast<std::int8_t>(sc)});
        }
      }
    }
  }
  sten_off_[ncells] = static_cast<std::int32_t>(sten_.size());
}

std::uint64_t Neighbours::candidateBound() const {
  // Atoms in one cell share the neighbour cells. The old per-atom walk
  // added (occupancy of those cells, minus the atom itself) once per atom.
  std::uint64_t bound = 0;
  const auto ncells = static_cast<std::size_t>(nx_) * static_cast<std::size_t>(ny_) *
                      static_cast<std::size_t>(nz_);
  for (std::size_t c = 0; c < ncells; ++c) {
    const auto occ = static_cast<std::uint64_t>(cell_start_[c + 1] - cell_start_[c]);
    if (occ == 0) {
      continue;
    }
    std::uint64_t here = 0;
    const auto s0 = sten_off_[c];
    const auto s1 = sten_off_[c + 1];
    for (std::int32_t k = s0; k < s1; ++k) {
      const auto id = static_cast<std::size_t>(sten_[static_cast<std::size_t>(k)].id);
      here += static_cast<std::uint64_t>(cell_start_[id + 1] - cell_start_[id]);
    }
    if (here == 0) {
      continue;
    }
    bound += occ * (here - 1);
  }
  return bound;
}

void Neighbours::buildStored(const double *xyz, std::size_t n, std::uint64_t bound) {
  (void)xyz;
  off_.assign(n + 1, 0);
  nbr_.clear();
  nbr_sa_.clear();
  nbr_sb_.clear();
  nbr_sc_.clear();
  nbr_shift_ = shift_dist_;
  if (bound > 0 && bound <= static_cast<std::uint64_t>(0x7fffffff)) {
    nbr_.reserve(static_cast<std::size_t>(bound));
    if (nbr_shift_) {
      nbr_sa_.reserve(static_cast<std::size_t>(bound));
      nbr_sb_.reserve(static_cast<std::size_t>(bound));
      nbr_sc_.reserve(static_cast<std::size_t>(bound));
    }
  }
  struct Item {
    std::int32_t j;
    std::int8_t sa;
    std::int8_t sb;
    std::int8_t sc;
  };
  std::vector<Item> gather;
  const double search2 = search_ * search_;
  const double L0 = len_[0];
  const double L1 = len_[1];
  const double L2 = len_[2];
  for (std::size_t a = 0; a < n; ++a) {
    off_[a] = static_cast<std::int32_t>(nbr_.size());
    const auto cell = static_cast<std::size_t>(cell_of_[a]);
    const double xi = xyz_[3 * a];
    const double yi = xyz_[3 * a + 1];
    const double zi = xyz_[3 * a + 2];
    gather.clear();
    const auto s0 = sten_off_[cell];
    const auto s1 = sten_off_[cell + 1];
    for (std::int32_t k = s0; k < s1; ++k) {
      const Sten &src = sten_[static_cast<std::size_t>(k)];
      const auto nb = static_cast<std::size_t>(src.id);
      const double shx = static_cast<double>(src.sa) * L0;
      const double shy = static_cast<double>(src.sb) * L1;
      const double shz = static_cast<double>(src.sc) * L2;
      for (std::int32_t q = cell_start_[nb]; q < cell_start_[nb + 1]; ++q) {
        const auto slot = static_cast<std::size_t>(q);
        const auto j = cell_atoms_[slot];
        if (j == static_cast<std::int32_t>(a)) {
          continue;
        }
        double dx;
        double dy;
        double dz;
        if (shift_dist_) {
          dx = xi - cell_x_[slot];
          dy = yi - cell_y_[slot];
          dz = zi - cell_z_[slot];
          if (periodic_[0]) {
            dx -= shx;
          }
          if (periodic_[1]) {
            dy -= shy;
          }
          if (periodic_[2]) {
            dz -= shz;
          }
        } else {
          dx = 0;
          dy = 0;
          dz = 0;
          foldPair(static_cast<std::int32_t>(a), j, dx, dy, dz);
        }
        const double r2 = dx * dx + dy * dy + dz * dz;
        if (r2 <= search2) {
          gather.push_back(Item{j, src.sa, src.sb, src.sc});
        }
      }
    }
    std::sort(gather.begin(), gather.end(),
              [](const Item &p, const Item &q) { return p.j < q.j; });
    for (const Item &item : gather) {
      nbr_.push_back(item.j);
      if (nbr_shift_) {
        nbr_sa_.push_back(item.sa);
        nbr_sb_.push_back(item.sb);
        nbr_sc_.push_back(item.sc);
      }
    }
  }
  off_[n] = static_cast<std::int32_t>(nbr_.size());
}

void Neighbours::update(const double *xyz, std::size_t n, const double *box9,
                        const Spec &spec) {
  xyz_ = xyz;
  if (reusable(xyz, n, box9, spec)) {
    reused_ = true;
    return;
  }
  reused_ = false;
  n_ = n;
  std::memcpy(box_, box9, sizeof(box_));
  len_[0] = box9[0];
  len_[1] = box9[4];
  len_[2] = box9[8];
  for (int k = 0; k < 3; ++k) {
    periodic_[k] = spec.periodic[static_cast<std::size_t>(k)] && len_[k] > 0.0;
    inv_[k] = periodic_[k] ? 1.0 / len_[k] : 0.0;
  }
  const bool sheared = box9[1] != 0.0 || box9[2] != 0.0 || box9[3] != 0.0 ||
                       box9[5] != 0.0 || box9[6] != 0.0 || box9[7] != 0.0;
  image_ = spec.image;
  use_general_ = image_ == Image::General || sheared;
  if (use_general_) {
    general_ = MinimumImage(box9, spec.periodic);
  }
  search_ = spec.cutoff + spec.skin;
  if (!(search_ > 0.0)) {
    search_ = 0.0;
  }
  key_.cutoff = spec.cutoff;
  key_.skin = spec.skin;
  key_.image = spec.image;
  key_.mode = spec.mode;
  key_.periodic = spec.periodic;
  off_.clear();
  nbr_.clear();
  nbr_sa_.clear();
  nbr_sb_.clear();
  nbr_sc_.clear();
  nbr_shift_ = false;
  streaming_ = true;
  if (n < 2 || !(search_ > 0.0)) {
    streaming_ = false;
    mode_ = Mode::Stored;
    off_.assign(n + 1, 0);
    nbr_.clear();
    shift_dist_ = false;
    sten_.clear();
    sten_off_.clear();
    built_ = true;
    ref_.assign(xyz, xyz + 3 * n);
    return;
  }
  buildCells(xyz, n);
  // Five cells on a periodic side put every in-range pair strictly inside
  // half the box, so the cell shift is the round minimum image.
  shift_dist_ = !use_general_ && image_ == Image::Round &&
                (!periodic_[0] || nx_ >= 5) && (!periodic_[1] || ny_ >= 5) &&
                (!periodic_[2] || nz_ >= 5);
  sten_.clear();
  sten_off_.clear();
  // A stream builds the stencil on the walk. Doing it here made the binning
  // pass pay for a structure the caller may not use.
  if (spec.mode == Mode::Stream) {
    streaming_ = true;
    mode_ = Mode::Stream;
    built_ = true;
    ref_.assign(xyz, xyz + 3 * n);
    return;
  }
  buildStencil();
  const std::uint64_t bound = candidateBound();
  const auto avail = availableBytes();
  const auto index_bytes = static_cast<std::uint64_t>(sizeof(std::int32_t));
  const bool fits = avail > 0 && bound <= avail / index_bytes &&
                    bound <= static_cast<std::uint64_t>(0x7fffffff);
  // The stream keeps the bins. The list adds one index per candidate.
  // Auto stores only when that index fits. A short cutoff that does not
  // fit the 27-slot stencil keeps the smaller mask in the same cache.
  const bool want_store = spec.mode == Mode::Stored ? fits
                          : spec.mode == Mode::Auto && fits;
  if (want_store) {
    buildStored(xyz, n, bound);
    streaming_ = false;
    mode_ = Mode::Stored;
    ++builds_;
  } else {
    streaming_ = true;
    mode_ = Mode::Stream;
  }
  built_ = true;
  ref_.assign(xyz, xyz + 3 * n);
}

} // namespace rgpot::nlist
