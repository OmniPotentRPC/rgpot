// MIT License
// Copyright 2023--present rgpot developers

#include "rgpot/nlist/Neighbours.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#elif defined(__APPLE__)
#include <mach/mach.h>
#else
#include <fstream>
#endif

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

// Row-major inverse. Binning only; the pair geometry still goes through
// MinimumImage. A singular cell yields one bin, which searches every pair.
bool invertRows(const double a[9], double inv[9]) {
  const double m0 = a[4] * a[8] - a[5] * a[7];
  const double m1 = a[3] * a[8] - a[5] * a[6];
  const double m2 = a[3] * a[7] - a[4] * a[6];
  const double det = a[0] * m0 - a[1] * m1 + a[2] * m2;
  if (!std::isfinite(det) || det == 0.0) {
    return false;
  }
  inv[0] = m0 / det;
  inv[1] = (a[2] * a[7] - a[1] * a[8]) / det;
  inv[2] = (a[1] * a[5] - a[2] * a[4]) / det;
  inv[3] = -m1 / det;
  inv[4] = (a[0] * a[8] - a[2] * a[6]) / det;
  inv[5] = (a[2] * a[3] - a[0] * a[5]) / det;
  inv[6] = m2 / det;
  inv[7] = (a[1] * a[6] - a[0] * a[7]) / det;
  inv[8] = (a[0] * a[4] - a[1] * a[3]) / det;
  for (int k = 0; k < 9; ++k) {
    if (!std::isfinite(inv[k])) {
      return false;
    }
  }
  return true;
}

// Bins along one fractional axis. The cutoff ball reaches at most
// `search * ||column||` in that fractional coordinate, so a bin at least
// that wide makes every neighbour sit in the same bin or a neighbour of it.
int fractionalBins(const double inv[9], int axis, double search, bool periodic) {
  if (!periodic || !(search > 0.0)) {
    return 1;
  }
  const double dual = std::hypot(inv[axis], inv[3 + axis], inv[6 + axis]);
  const double reach = search * dual * (1.0 + 1e-9);
  if (!(reach > 0.0) || !(reach < 1.0)) {
    return 1;
  }
  const double nf = std::floor(1.0 / reach);
  if (!(nf >= 1.0) || !(nf < 2.0e9)) {
    return 1;
  }
  int n = static_cast<int>(nf);
  while (n > 1 && 1.0 / static_cast<double>(n) < reach) {
    --n;
  }
  return n;
}

// An open axis is not wrapped. Bins cover the atoms' fractional extent,
// each at least as wide as the cutoff ball in that coordinate. One bin
// when every atom already lies inside that ball.
int openExtentBins(const double inv[9], int axis, double search, double extent) {
  if (!(search > 0.0) || !(extent > 0.0)) {
    return 1;
  }
  const double dual = std::hypot(inv[axis], inv[3 + axis], inv[6 + axis]);
  const double reach = search * dual * (1.0 + 1e-9);
  if (!(reach > 0.0) || !(extent > reach)) {
    return 1;
  }
  const double nf = std::floor(extent / reach);
  if (!(nf >= 1.0) || !(nf < 2.0e9)) {
    return 1;
  }
  int n = static_cast<int>(nf);
  while (n > 1 && extent / static_cast<double>(n) < reach) {
    --n;
  }
  return n;
}

// Image indices appended to cell_of_ after the per-atom cell ids.
// magic, pair count, fresh, then a bit field: bit 0 tiny (-1, 0, 1),
// bit 1 every index stays valid for any move inside the skin,
// bit 2 general-cell coefficients of MinimumImage.
// The shifts follow, three int32 values per pair. If bit 1 is clear,
// one byte per pair follows those, and a nonzero byte is folded again.
constexpr std::int32_t kShiftMagic = 0x52534846;
constexpr std::int32_t kShiftTiny = 1;
constexpr std::int32_t kShiftStable = 2;
constexpr std::int32_t kShiftGeneral = 4;

// Each atom may move by at most skin/2. The squared test in withinSkin is
// a few ulps loose, so the pair's coordinate difference can change by a
// little more than skin. skin * (1 + 2^-40) covers that. A later reduced
// coordinate t = fl(dx * inv) also rounds, which the slack below covers.
// 2^-40 is far above (1+u)/(1-u)^(3/2) - 1 for u = 2^-52.
constexpr double kSkinSlop = 0x1p-40;

// True when every reduced coordinate within `slack` of `t` has image `idx`.
// The intervals are those of trunc(t + copysign(0.5, t)).
bool indexHolds(double t, double idx, double slack) noexcept {
  if (!(slack >= 0.0) || !std::isfinite(t) || !std::isfinite(idx) || !std::isfinite(slack)) {
    return false;
  }
  if (idx == 0.0) {
    return std::abs(t) + slack < 0.5;
  }
  if (idx > 0.0) {
    return (t - slack) >= (idx - 0.5) && (t + slack) < (idx + 0.5);
  }
  return (t - slack) > (idx - 0.5) && (t + slack) <= (idx + 0.5);
}

// Upper bound on |fl(dx_new * inv) - fl(dx_ref * inv)| while both atoms stay
// inside the skin of the reference. dx is the reference difference.
double shiftSlack(double dx, double inv, double skin) noexcept {
  if (!(skin >= 0.0) || !std::isfinite(dx) || !std::isfinite(inv) || !std::isfinite(skin)) {
    return 1.0;
  }
  const double reach = skin * (1.0 + kSkinSlop);
  const double abs_inv = std::abs(inv);
  const double t = std::abs(dx * inv);
  return abs_inv * reach * (1.0 + kSkinSlop) + t * kSkinSlop + kSkinSlop;
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
#if defined(_WIN32)
  MEMORYSTATUSEX status{};
  status.dwLength = sizeof(status);
  if (!GlobalMemoryStatusEx(&status)) {
    return 0;
  }
  return static_cast<std::uint64_t>(status.ullAvailPhys);
#elif defined(__APPLE__)
  vm_statistics64_data_t vm{};
  mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
  if (host_statistics64(mach_host_self(), HOST_VM_INFO64,
                        reinterpret_cast<host_info64_t>(&vm),
                        &count) != KERN_SUCCESS) {
    return 0;
  }
  const auto page = static_cast<std::uint64_t>(vm_page_size);
  return (static_cast<std::uint64_t>(vm.free_count) +
          static_cast<std::uint64_t>(vm.inactive_count)) *
         page;
#else
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
#endif
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

void Neighbours::installRoundShifts(const std::int32_t *shift, std::size_t npairs, bool tiny,
                                    bool all_stable, bool general, const std::uint8_t *unstable) {
  const auto base = n_;
  if (npairs > (static_cast<std::size_t>(-1) - 4 - base) / 4) {
    return;
  }
  const std::size_t shift_bytes = npairs * 3 * sizeof(std::int32_t);
  const std::size_t flag_bytes = all_stable ? 0 : npairs;
  const std::size_t nints = (shift_bytes + flag_bytes + sizeof(std::int32_t) - 1) / sizeof(std::int32_t);
  cell_of_.resize(base + 4 + nints);
  cell_of_[base] = kShiftMagic;
  cell_of_[base + 1] = static_cast<std::int32_t>(npairs);
  cell_of_[base + 2] = 1;
  cell_of_[base + 3] = (tiny ? kShiftTiny : 0) | (all_stable ? kShiftStable : 0) |
                       (general ? kShiftGeneral : 0);
  auto *raw = reinterpret_cast<unsigned char *>(cell_of_.data() + base + 4);
  if (npairs > 0) {
    std::memcpy(raw, shift, shift_bytes);
  }
  if (!all_stable && npairs > 0 && unstable != nullptr) {
    std::memcpy(raw + shift_bytes, unstable, flag_bytes);
  }
}

void Neighbours::noteShiftFresh(bool fresh) noexcept {
  if (n_ > cell_of_.size() || cell_of_.size() - n_ < 4) {
    return;
  }
  if (cell_of_[n_] != kShiftMagic) {
    return;
  }
  cell_of_[n_ + 2] = fresh ? 1 : 0;
}

const std::int32_t *Neighbours::roundShifts(bool &tiny, const unsigned char *&unstable,
                                           bool &general) const noexcept {
  tiny = false;
  unstable = nullptr;
  general = false;
  if (n_ > cell_of_.size() || cell_of_.size() - n_ < 4 || cell_of_[n_] != kShiftMagic) {
    return nullptr;
  }
  const bool fresh = cell_of_[n_ + 2] != 0;
  const auto bits = cell_of_[n_ + 3];
  const bool all_stable = (bits & kShiftStable) != 0;
  general = (bits & kShiftGeneral) != 0;
  const auto count = static_cast<std::size_t>(cell_of_[n_ + 1]);
  if (count != nbr_.size()) {
    return nullptr;
  }
  const std::size_t shift_bytes = count * 3 * sizeof(std::int32_t);
  const std::size_t extra = all_stable ? 0 : count;
  const std::size_t nints =
      (shift_bytes + extra + sizeof(std::int32_t) - 1) / sizeof(std::int32_t);
  if (cell_of_.size() - n_ - 4 < nints) {
    return nullptr;
  }
  // Exact coordinates make every stored orthorhombic index the one
  // rounding would return. A general-cell pair that the search corrected
  // has no coefficient, so that byte is read even when the coordinates
  // still match. A skin move may keep an index only when the build proved
  // the image cannot change.
  if (!fresh && !all_stable && extra == 0) {
    return nullptr;
  }
  tiny = (bits & kShiftTiny) != 0;
  if (!all_stable && (!fresh || general)) {
    const auto *raw = reinterpret_cast<const unsigned char *>(cell_of_.data() + n_ + 4);
    unstable = raw + shift_bytes;
  }
  return cell_of_.data() + n_ + 4;
}

void Neighbours::buildCells(const double *xyz, std::size_t n) {
  const double edge = search_ > 0.0 ? search_ : 1.0;
  double frac_inv[9] = {};
  const bool frac_bins = use_general_ && invertRows(box_, frac_inv);
  double open_lo[3] = {};
  double open_hi[3] = {};
  if (frac_bins) {
    for (int axis = 0; axis < 3; ++axis) {
      if (periodic_[axis]) {
        continue;
      }
      double lo = std::numeric_limits<double>::infinity();
      double hi = -lo;
      for (std::size_t a = 0; a < n; ++a) {
        const double *r = xyz + 3 * a;
        const double f = r[0] * frac_inv[axis] + r[1] * frac_inv[3 + axis] +
                         r[2] * frac_inv[6 + axis];
        lo = std::min(lo, f);
        hi = std::max(hi, f);
      }
      if (!(hi >= lo)) {
        lo = 0.0;
        hi = 0.0;
      }
      open_lo[axis] = lo;
      open_hi[axis] = hi;
    }
  }
  auto axis_bins = [&](int axis, double length) {
    if (!frac_bins) {
      return axisCount(length, edge);
    }
    if (periodic_[axis]) {
      return fractionalBins(frac_inv, axis, search_, true);
    }
    return openExtentBins(frac_inv, axis, search_, open_hi[axis] - open_lo[axis]);
  };
  int nx = axis_bins(0, len_[0]);
  int ny = axis_bins(1, len_[1]);
  int nz = axis_bins(2, len_[2]);
  const double bytes =
      static_cast<double>(nx) * static_cast<double>(ny) *
      static_cast<double>(nz) * static_cast<double>(sizeof(std::int32_t));
  const auto avail = availableBytes();
  if (avail > 0 && bytes > static_cast<double>(avail)) {
    nx = ny = nz = 1;
  }
  const auto ncells = static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny) *
                      static_cast<std::size_t>(nz);
  // A sheared cell is binned in fractional coordinates. Cartesian bins
  // along the diagonal miss a pair whose connecting lattice vector is
  // not that diagonal, even when the minimum image is inside the cutoff.
  auto bin = [&](const double *r, int axis, double length, int naxis) {
    if (naxis <= 1) {
      return 0;
    }
    double t = 0.0;
    if (frac_bins) {
      const double f = r[0] * frac_inv[axis] + r[1] * frac_inv[3 + axis] +
                       r[2] * frac_inv[6 + axis];
      if (!periodic_[axis]) {
        const double extent = open_hi[axis] - open_lo[axis];
        t = extent > 0.0 ? (f - open_lo[axis]) / extent * static_cast<double>(naxis)
                         : 0.0;
      } else {
        double w = f - std::floor(f);
        if (w < 0.0) {
          w += 1.0;
        }
        if (!(w < 1.0)) {
          w = 0.0;
        }
        t = w * static_cast<double>(naxis);
      }
    } else {
      if (!(length > 0.0)) {
        return 0;
      }
      t = r[axis] / length * static_cast<double>(naxis);
      if (!(t >= 0.0)) {
        t = 0.0;
      }
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
  // A rebuild whose atoms are still in the cells they occupied keeps
  // those cells. The pairs are searched again; only the binning is kept.
  // Atoms were inserted in index order, so an unchanged assignment is
  // the same cell_atoms_ sequence.
  if (nx == nx_ && ny == ny_ && nz == nz_ && cell_atoms_.size() == n &&
      cell_of_.size() >= n && cell_start_.size() == ncells + 1) {
    bool same = true;
    for (std::size_t a = 0; a < n; ++a) {
      const double *r = xyz + 3 * a;
      const int ix = bin(r, 0, len_[0], nx);
      const int iy = bin(r, 1, len_[1], ny);
      const int iz = bin(r, 2, len_[2], nz);
      const int id = (ix * ny + iy) * nz + iz;
      if (id != cell_of_[a]) {
        same = false;
        break;
      }
    }
    if (same) {
      return;
    }
  }
  nx_ = nx;
  ny_ = ny;
  nz_ = nz;
  cell_start_.assign(ncells + 1, 0);
  cell_of_.resize(n);
  for (std::size_t a = 0; a < n; ++a) {
    const double *r = xyz + 3 * a;
    const int ix = bin(r, 0, len_[0], nx_);
    const int iy = bin(r, 1, len_[1], ny_);
    const int iz = bin(r, 2, len_[2], nz_);
    const int id = (ix * ny_ + iy) * nz_ + iz;
    cell_of_[a] = id;
    ++cell_start_[static_cast<std::size_t>(id) + 1];
  }
  for (std::size_t c = 0; c < ncells; ++c) {
    cell_start_[c + 1] += cell_start_[c];
  }
  cell_atoms_.assign(n, 0);
  std::vector<std::int32_t> fill(cell_start_.begin(), cell_start_.end() - 1);
  for (std::size_t a = 0; a < n; ++a) {
    const auto c = static_cast<std::size_t>(cell_of_[a]);
    cell_atoms_[static_cast<std::size_t>(fill[c]++)] = static_cast<std::int32_t>(a);
  }
}

std::uint64_t Neighbours::candidateBound() const {
  std::uint64_t bound = 0;
  const auto n = n_;
  for (std::size_t a = 0; a < n; ++a) {
    const int cell = cell_of_[a];
    int iz = cell % nz_;
    int iy = (cell / nz_) % ny_;
    int ix = cell / (nz_ * ny_);
    int seen[27];
    int nseen = 0;
    std::uint64_t here = 0;
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
          here += static_cast<std::uint64_t>(cell_start_[c + 1] - cell_start_[c]);
        }
      }
    }
    if (here > 0) {
      bound += here - 1;
    }
  }
  return bound;
}


void Neighbours::buildStored(const double *xyz, std::size_t n) {
  (void)xyz;
  off_.assign(n + 1, 0);
  nbr_.clear();
  struct Cand {
    std::int32_t j;
    std::int32_t sx;
    std::int32_t sy;
    std::int32_t sz;
    std::uint8_t unstable;
  };
  std::vector<Cand> gather;
  std::vector<Cand> scratch;
  std::vector<std::int32_t> shifts;
  std::vector<std::uint8_t> flags;
  // A pair is the same geometry from either end, with the image index
  // and the skin flag negated together. Evaluating j > i and writing
  // both rows avoids a second fold. For a sheared cell the index is the
  // rounded MinimumImage coefficient; subtractRounded is an odd map, so
  // the negated coefficient is the opposite end.
  struct Half {
    std::int32_t i;
    std::int32_t j;
    std::int32_t sx;
    std::int32_t sy;
    std::int32_t sz;
    std::uint8_t flag;
  };
  // The mirrored rows are assembled again on the next rebuild. Keeping
  // the buffers avoids asking the allocator for them on every call.
  struct HalfBuf {
    std::vector<Half> halves;
    std::vector<Cand> slots;
    std::vector<Cand> scratch;
    std::vector<std::int32_t> degree;
    std::vector<std::int32_t> cursor;
    std::vector<std::int32_t> shifts;
    std::vector<std::uint8_t> flags;
  };
  thread_local HalfBuf half_buf;
  half_buf.halves.clear();
  std::vector<Half> &halves = half_buf.halves;
  // Orthorhombic and sheared builds both fold each unordered pair once.
  // The opposite row stores the negated image. For a sheared cell that
  // image is the rounded MinimumImage coefficient, and subtractRounded
  // is an odd map, so the stored geometry matches the search.
  const bool half_pairs = true;
#if defined(__GNUC__) || defined(__clang__)
#define RGPOT_ACCEPT_INLINE __attribute__((always_inline))
#else
#define RGPOT_ACCEPT_INLINE
#endif
  auto accept = [&](std::size_t a, std::int32_t j, std::int32_t sx, std::int32_t sy,
                    std::int32_t sz, bool holds) RGPOT_ACCEPT_INLINE {
    const auto flag = static_cast<std::uint8_t>(holds ? 0 : 1);
    if (!half_pairs) {
      gather.push_back(Cand{j, sx, sy, sz, flag});
      return;
    }
    halves.push_back(Half{static_cast<std::int32_t>(a), j, sx, sy, sz, flag});
  };
#undef RGPOT_ACCEPT_INLINE
  auto order = [&]() {
    if (gather.size() <= 1) {
      return;
    }
    int nbytes = 1;
    std::int32_t maxj = 0;
    for (const Cand &c : gather) {
      if (c.j > maxj) {
        maxj = c.j;
      }
    }
    if (maxj > 0x0000ffff) {
      nbytes = maxj > 0x00ffffff ? 4 : 3;
    } else if (maxj > 0x000000ff) {
      nbytes = 2;
    }
    scratch.resize(gather.size());
    int counts[256];
    for (int shift = 0; shift < 8 * nbytes; shift += 8) {
      for (int b = 0; b < 256; ++b) {
        counts[b] = 0;
      }
      for (const Cand &c : gather) {
        ++counts[(static_cast<unsigned>(c.j) >> shift) & 255u];
      }
      int sum = 0;
      for (int b = 0; b < 256; ++b) {
        const int nbin = counts[b];
        counts[b] = sum;
        sum += nbin;
      }
      for (const Cand &c : gather) {
        scratch[static_cast<std::size_t>(
            counts[(static_cast<unsigned>(c.j) >> shift) & 255u]++)] = c;
      }
      gather.swap(scratch);
    }
  };
  const bool record_round = !use_general_ && image_ == Image::Round;
  const bool record_general = use_general_;
  const SearchBound bound = searchBound();
  const bool record = record_round || record_general;
  bool fit = record;
  bool tiny = record_round;
  bool all_stable = record;
  const double lambda = record_general ? general_.shortestPeriodic() : 0.0;
  const double search2 = search_ * search_;
  // The image index and the skin test. The skin test is the heavy part
  // and is needed only for a pair that is kept, so a pair past the search
  // folds and returns without it. Flags still move on every folded axis:
  // a shift that does not fit, or that is outside {-1, 0, 1}, is recorded
  // before the distance test.
  auto fold_axis = [&](double &delta, int axis, std::int32_t &slot) {
    if (!periodic_[axis]) {
      slot = 0;
      return true;
    }
    const double inv = inv_[axis];
    const double t = delta * inv;
    const double idx = orthorhombicImageIndex(t);
    delta -= len_[axis] * idx;
    if (!(idx >= -2147483648.0 && idx <= 2147483647.0)) {
      fit = false;
      tiny = false;
      return false;
    }
    const auto as_int = static_cast<std::int32_t>(idx);
    if (static_cast<double>(as_int) != idx) {
      fit = false;
      tiny = false;
      return false;
    }
    slot = as_int;
    if (as_int < -1 || as_int > 1) {
      tiny = false;
    }
    return true;
  };
  // The original combined fold, used when the box is too small for the
  // early reject to remove any pair.
  auto legacy_axis = [&](double &delta, int axis, std::int32_t &slot) {
    if (!periodic_[axis]) {
      slot = 0;
      return true;
    }
    const double inv = inv_[axis];
    const double t = delta * inv;
    const double idx = orthorhombicImageIndex(t);
    const bool holds = indexHolds(t, idx, shiftSlack(delta, inv, key_.skin));
    delta -= len_[axis] * idx;
    if (!(idx >= -2147483648.0 && idx <= 2147483647.0)) {
      fit = false;
      tiny = false;
      return false;
    }
    const auto as_int = static_cast<std::int32_t>(idx);
    if (static_cast<double>(as_int) != idx) {
      fit = false;
      tiny = false;
      return false;
    }
    slot = as_int;
    if (as_int < -1 || as_int > 1) {
      tiny = false;
    }
    return holds;
  };
  auto axis_holds = [&](double raw, int axis, std::int32_t slot, bool folded) {
    if (!periodic_[axis]) {
      return true;
    }
    if (!folded) {
      return false;
    }
    const double inv = inv_[axis];
    const double t = raw * inv;
    return indexHolds(t, static_cast<double>(slot), shiftSlack(raw, inv, key_.skin));
  };
  // Lower bound on the folded separation. 1e-6 A keeps a pair the exact
  // test would accept. A raw separation past 1.5 box lengths is not
  // classified here, so its image flags still come from fold_axis.
  const double far = search_ + 1e-6;
  const double lim15_0 = 1.5 * len_[0];
  const double lim15_1 = 1.5 * len_[1];
  const double lim15_2 = 1.5 * len_[2];
  // A periodic side no larger than twice the search cannot put a pair
  // outside the search on that axis. The early reject is then pure
  // overhead, and the original skin test stays on that build.
  bool wide = record_round;
  if (wide && (periodic_[0] || periodic_[1] || periodic_[2])) {
    for (int axis = 0; axis < 3; ++axis) {
      if (periodic_[axis] && !(len_[axis] > 2.0 * far)) {
        wide = false;
      }
    }
  }
  auto clearly_out = [&](double delta, int axis) {
    if (!periodic_[axis]) {
      return std::abs(delta) > far;
    }
    const double L = len_[axis];
    if (!(L > 0.0)) {
      return false;
    }
    const double lim15 = axis == 0 ? lim15_0 : (axis == 1 ? lim15_1 : lim15_2);
    const double a = std::abs(delta);
    if (a > lim15) {
      return false;
    }
    const double other = std::abs(a - L);
    const double folded = other < a ? other : a;
    return folded > far;
  };
  // Four or more cells make the 27 stencil slots distinct. With five
  // or more, a neighbour is at most two cell widths away and only an
  // edge cell can hold an atom past the box. Nine tenths of the gap
  // between that span and half a box still rounds to the stencil wrap.
  // Four cells leave no such gap. The folded axis is the unique nearest
  // image only while it lies strictly inside half a box; any other pair
  // keeps the original fold. That test is what admits an atom outside
  // the box. The separation then stays under 1.5 box lengths, so the
  // early reject matches clearly_out. {-L, 0, L} is the multiple
  // fold_axis would subtract. A sheared cell is not handled here. Its
  // hash wraps the Cartesian bins, and that integer is not the lattice
  // coefficient.
  const double sub0[3] = {-len_[0], 0.0, len_[0]};
  const double sub1[3] = {-len_[1], 0.0, len_[1]};
  const double sub2[3] = {-len_[2], 0.0, len_[2]};
  const double half0 = 0.5 * len_[0];
  const double half1 = 0.5 * len_[1];
  const double half2 = 0.5 * len_[2];
  bool use_cell_shift = wide;
  if (use_cell_shift && ((periodic_[0] && nx_ < 4) || (periodic_[1] && ny_ < 4) ||
                         (periodic_[2] && nz_ < 4))) {
    use_cell_shift = false;
  }
  const bool proof_cell4 =
      use_cell_shift && ((periodic_[0] && nx_ == 4) || (periodic_[1] && ny_ == 4) ||
                         (periodic_[2] && nz_ == 4));
  auto axis_near = [](double x, int cell, int naxis, double len) {
    const double w = len / static_cast<double>(naxis);
    const double allowed = 0.9 * (0.5 * len - 2.0 * w);
    if (!(allowed > 1e-6)) {
      return false;
    }
    if (cell <= 0) {
      return x >= -allowed && x < w;
    }
    if (cell >= naxis - 1) {
      return x >= static_cast<double>(naxis - 1) * w && x < len + allowed;
    }
    const double lo = static_cast<double>(cell) * w;
    return x >= lo && x < lo + w;
  };
  auto axis_in = [&](int axis, double x, int cell, int naxis, double len) {
    if (!periodic_[axis]) {
      return true;
    }
    if (naxis >= 5) {
      return axis_near(x, cell, naxis, len);
    }
    // Four cells have no slack that is exact for every pair. The folded
    // half-box test below keeps the original fold for a pair whose
    // stencil image is not the unique nearest one, including atoms that
    // sit outside the box.
    return true;
  };
  for (std::size_t a = 0; a < n && use_cell_shift; ++a) {
    const int cell = cell_of_[a];
    const int iz = cell % nz_;
    const int iy = (cell / nz_) % ny_;
    const int ix = cell / (nz_ * ny_);
    const auto u = a * 3;
    if (!axis_in(0, xyz_[u], ix, nx_, len_[0]) ||
        !axis_in(1, xyz_[u + 1], iy, ny_, len_[1]) ||
        !axis_in(2, xyz_[u + 2], iz, nz_, len_[2])) {
      use_cell_shift = false;
    }
  }
  auto fold_wide = [&](std::size_t atom, std::int32_t partner, double dx, double dy, double dz) {
    if (clearly_out(dx, 0) || clearly_out(dy, 1) || clearly_out(dz, 2)) {
      return;
    }
    const double rawx = dx;
    const double rawy = dy;
    const double rawz = dz;
    std::int32_t sx = 0;
    std::int32_t sy = 0;
    std::int32_t sz = 0;
    const bool fx = fold_axis(dx, 0, sx);
    const bool fy = fold_axis(dy, 1, sy);
    const bool fz = fold_axis(dz, 2, sz);
    const double r2_round = dx * dx + dy * dy + dz * dz;
    if (r2_round <= search2) {
      const bool holds = axis_holds(rawx, 0, sx, fx) && axis_holds(rawy, 1, sy, fy) &&
                         axis_holds(rawz, 2, sz, fz);
      accept(atom, partner, sx, sy, sz, holds);
    }
  };
  for (std::size_t a = 0; a < n; ++a) {
    off_[a] = static_cast<std::int32_t>(nbr_.size());
    const int cell = cell_of_[a];
    int iz = cell % nz_;
    int iy = (cell / nz_) % ny_;
    int ix = cell / (nz_ * ny_);
    int seen[27];
    int nseen = 0;
    gather.clear();
    for (int ox = -1; ox <= 1; ++ox) {
      for (int oy = -1; oy <= 1; ++oy) {
        for (int oz = -1; oz <= 1; ++oz) {
          int sax = 0;
          int say = 0;
          int saz = 0;
          if (use_cell_shift) {
            const int jx = ix + ox;
            const int jy = iy + oy;
            const int jz = iz + oz;
            if (periodic_[0]) {
              if (jx < 0) {
                sax = -1;
              } else if (jx >= nx_) {
                sax = 1;
              }
            }
            if (periodic_[1]) {
              if (jy < 0) {
                say = -1;
              } else if (jy >= ny_) {
                say = 1;
              }
            }
            if (periodic_[2]) {
              if (jz < 0) {
                saz = -1;
              } else if (jz >= nz_) {
                saz = 1;
              }
            }
          }
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
          for (std::int32_t q = cell_start_[c]; q < cell_start_[c + 1]; ++q) {
            const auto j = cell_atoms_[static_cast<std::size_t>(q)];
            if (j == static_cast<std::int32_t>(a)) {
              continue;
            }
            // The partner with the larger index owns the fold. Its row
            // receives the negated index when the rows are assembled.
            if (half_pairs && j < static_cast<std::int32_t>(a)) {
              continue;
            }
            std::int32_t sx = 0;
            std::int32_t sy = 0;
            std::int32_t sz = 0;
            bool holds = true;
            double dx = 0;
            double dy = 0;
            double dz = 0;
            if (!record) {
              foldPair(static_cast<std::int32_t>(a), j, dx, dy, dz);
            } else if (record_general) {
              const auto ui = a * 3;
              const auto uj = static_cast<std::size_t>(j) * 3;
              dx = xyz_[ui] - xyz_[uj];
              dy = xyz_[ui + 1] - xyz_[uj + 1];
              dz = xyz_[ui + 2] - xyz_[uj + 2];
              if (outsideSearch(bound, dx, dy, dz)) {
                continue;
              }
              std::int32_t nimg[3] = {};
              const bool coded = general_.tryRounded(dx, dy, dz, nimg);
              holds = coded && general_.roundedHolds(dx, dy, dz, nimg, key_.skin, lambda);
              if (coded) {
                sx = nimg[0];
                sy = nimg[1];
                sz = nimg[2];
                general_.subtractRounded(sx, sy, sz, dx, dy, dz);
              } else {
                fold(dx, dy, dz);
              }
            } else {
              const auto ui = a * 3;
              const auto uj = static_cast<std::size_t>(j) * 3;
              dx = xyz_[ui] - xyz_[uj];
              dy = xyz_[ui + 1] - xyz_[uj + 1];
              dz = xyz_[ui + 2] - xyz_[uj + 2];
              // Half a box no larger than the search means every axis
              // separation is inside the search, so the early test cannot
              // reject and the skin test still runs on almost every pair.
              if (!wide) {
                const bool hx = legacy_axis(dx, 0, sx);
                const bool hy = legacy_axis(dy, 1, sy);
                const bool hz = legacy_axis(dz, 2, sz);
                holds = hx && hy && hz;
              } else if (use_cell_shift) {
                const double rawx = dx;
                const double rawy = dy;
                const double rawz = dz;
                dx -= sub0[sax + 1];
                dy -= sub1[say + 1];
                dz -= sub2[saz + 1];
                // Four cells sit on the rounding boundary. A folded axis
                // that is not strictly inside half a box is not the unique
                // nearest image, so that pair keeps the original fold.
                const bool proven = !proof_cell4 ||
                                    ((!periodic_[0] || std::abs(dx) < half0) &&
                                     (!periodic_[1] || std::abs(dy) < half1) &&
                                     (!periodic_[2] || std::abs(dz) < half2));
                if (!proven) {
                  fold_wide(a, j, rawx, rawy, rawz);
                  continue;
                }
                if (std::abs(dx) > far || std::abs(dy) > far || std::abs(dz) > far) {
                  continue;
                }
                const double r2_round = dx * dx + dy * dy + dz * dz;
                if (r2_round <= search2) {
                  holds = axis_holds(rawx, 0, sax, true) && axis_holds(rawy, 1, say, true) &&
                          axis_holds(rawz, 2, saz, true);
                  accept(a, j, sax, say, saz, holds);
                }
                continue;
              } else {
                fold_wide(a, j, dx, dy, dz);
                continue;
              }
            }
            const double r2 = dx * dx + dy * dy + dz * dz;
            if (r2 <= search2) {
              accept(a, j, sx, sy, sz, holds);
            }
          }
        }
      }
    }
    // Each neighbour cell is appended as a run already sorted by partner
    // index, because the atoms of a cell are binned in index order. A
    // radix pass on those bytes puts the row in the same order as a
    // comparison sort, which is what the half-list walk assumes. The
    // mirrored orthorhombic rows are ordered once, after both ends exist.
    order();
    for (const Cand &c : gather) {
      nbr_.push_back(c.j);
      if (record) {
        shifts.push_back(c.sx);
        shifts.push_back(c.sy);
        shifts.push_back(c.sz);
        flags.push_back(c.unstable);
        if (c.unstable != 0) {
          all_stable = false;
        }
      }
    }
  }
  if (half_pairs) {
    auto &degree = half_buf.degree;
    degree.assign(n, 0);
    for (const Half &h : halves) {
      ++degree[static_cast<std::size_t>(h.i)];
      ++degree[static_cast<std::size_t>(h.j)];
    }
    off_[0] = 0;
    for (std::size_t a = 0; a < n; ++a) {
      off_[a + 1] = off_[a] + degree[a];
    }
    auto &cursor = half_buf.cursor;
    cursor.assign(off_.begin(), off_.end() - 1);
    auto &slots = half_buf.slots;
    slots.resize(halves.size() * 2);
    for (const Half &h : halves) {
      const auto si = static_cast<std::size_t>(h.i);
      const auto sj = static_cast<std::size_t>(h.j);
      slots[static_cast<std::size_t>(cursor[si]++)] = Cand{h.j, h.sx, h.sy, h.sz, h.flag};
      const auto nsx = static_cast<std::int32_t>(-static_cast<std::int64_t>(h.sx));
      const auto nsy = static_cast<std::int32_t>(-static_cast<std::int64_t>(h.sy));
      const auto nsz = static_cast<std::int32_t>(-static_cast<std::int64_t>(h.sz));
      slots[static_cast<std::size_t>(cursor[sj]++)] = Cand{h.i, nsx, nsy, nsz, h.flag};
    }
    auto &row_scratch = half_buf.scratch;
    auto order_row = [&](Cand *row, std::size_t len) {
      if (len <= 1) {
        return;
      }
      int nbytes = 1;
      std::int32_t maxj = 0;
      for (std::size_t k = 0; k < len; ++k) {
        if (row[k].j > maxj) {
          maxj = row[k].j;
        }
      }
      if (maxj > 0x0000ffff) {
        nbytes = maxj > 0x00ffffff ? 4 : 3;
      } else if (maxj > 0x000000ff) {
        nbytes = 2;
      }
      row_scratch.resize(len);
      int counts[256];
      for (int shift = 0; shift < 8 * nbytes; shift += 8) {
        for (int b = 0; b < 256; ++b) {
          counts[b] = 0;
        }
        for (std::size_t k = 0; k < len; ++k) {
          ++counts[(static_cast<unsigned>(row[k].j) >> shift) & 255u];
        }
        int sum = 0;
        for (int b = 0; b < 256; ++b) {
          const int nbin = counts[b];
          counts[b] = sum;
          sum += nbin;
        }
        for (std::size_t k = 0; k < len; ++k) {
          row_scratch[static_cast<std::size_t>(
              counts[(static_cast<unsigned>(row[k].j) >> shift) & 255u]++)] = row[k];
        }
        for (std::size_t k = 0; k < len; ++k) {
          row[k] = row_scratch[k];
        }
      }
    };
    const auto npairs = slots.size();
    nbr_.resize(npairs);
    auto &out_shifts = half_buf.shifts;
    auto &out_flags = half_buf.flags;
    if (record) {
      out_shifts.resize(npairs * 3);
      out_flags.resize(npairs);
    }
    all_stable = record;
    std::size_t w = 0;
    for (std::size_t a = 0; a < n; ++a) {
      const auto begin = static_cast<std::size_t>(off_[a]);
      const auto end = static_cast<std::size_t>(off_[a + 1]);
      Cand *row = slots.data() + begin;
      const auto len = end - begin;
      order_row(row, len);
      for (std::size_t k = 0; k < len; ++k, ++w) {
        const Cand &c = row[k];
        nbr_[w] = c.j;
        if (record) {
          out_shifts[3 * w] = c.sx;
          out_shifts[3 * w + 1] = c.sy;
          out_shifts[3 * w + 2] = c.sz;
          out_flags[w] = c.unstable;
          if (c.unstable != 0) {
            all_stable = false;
          }
        }
      }
    }
  }
  off_[n] = static_cast<std::int32_t>(nbr_.size());
  if (fit && (record_round || record_general)) {
    const std::int32_t *shift_data = half_pairs ? half_buf.shifts.data() : shifts.data();
    const std::uint8_t *flag_data = half_pairs ? half_buf.flags.data() : flags.data();
    installRoundShifts(shift_data, nbr_.size(), tiny, all_stable, record_general,
                       all_stable ? nullptr : flag_data);
  }
}

Neighbours::SearchBound Neighbours::searchBound() const {
  SearchBound bound;
  if (!use_general_ || !invertRows(box_, bound.inv)) {
    return bound;
  }
  for (int k = 0; k < 3; ++k) {
    bound.dual[k] = std::hypot(bound.inv[k], bound.inv[3 + k], bound.inv[6 + k]);
  }
  bound.on = true;
  return bound;
}

void Neighbours::update(const double *xyz, std::size_t n, const double *box9,
                        const Spec &spec) {
  xyz_ = xyz;
  if (reusable(xyz, n, box9, spec)) {
    reused_ = true;
    const bool same = std::memcmp(ref_.data(), xyz, n * 3 * sizeof(double)) == 0;
    noteShiftFresh(same);
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
  streaming_ = true;
  if (n < 2 || !(search_ > 0.0)) {
    streaming_ = false;
    mode_ = Mode::Stored;
    off_.assign(n + 1, 0);
    nbr_.clear();
    built_ = true;
    ref_.assign(xyz, xyz + 3 * n);
    return;
  }
  buildCells(xyz, n);
  // A stream never stores the pairs, so the candidate count is unused.
  // Counting it walks every stencil the search is about to walk.
  if (spec.mode == Mode::Stream) {
    streaming_ = true;
    mode_ = Mode::Stream;
    built_ = true;
    ref_.assign(xyz, xyz + 3 * n);
    return;
  }
  const std::uint64_t bound = candidateBound();
  const auto avail = availableBytes();
  const auto index_bytes = static_cast<std::uint64_t>(sizeof(std::int32_t));
  // Zero means the probe failed, not that the machine has no free memory.
  // An explicit stored list, and Auto, still keep the index in that case.
  const bool bounded = bound <= static_cast<std::uint64_t>(0x7fffffff);
  const bool over_budget = avail > 0 && bound > avail / index_bytes;
  const bool fits = bounded && !over_budget;
  // The stream keeps the bins. The list adds one index per candidate.
  // Auto stores only when that index fits. A short cutoff that does not
  // fit the 27-slot stencil keeps the smaller mask in the same cache.
  const bool want_store = spec.mode == Mode::Stored ? fits
                          : spec.mode == Mode::Auto && fits;
  if (want_store) {
    buildStored(xyz, n);
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
