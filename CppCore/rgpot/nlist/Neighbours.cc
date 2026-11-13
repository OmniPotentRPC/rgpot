// MIT License
// Copyright 2023--present rgpot developers

#include "rgpot/nlist/Neighbours.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
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
  std::vector<std::int32_t> shifts;
  std::vector<std::uint8_t> flags;
  const bool record_round = !use_general_ && image_ == Image::Round;
  const bool record_general = use_general_;
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
    const double a = std::abs(delta);
    if (a > 1.5 * L) {
      return false;
    }
    const double other = std::abs(a - L);
    const double folded = other < a ? other : a;
    return folded > far;
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
              } else {
                if (clearly_out(dx, 0) || clearly_out(dy, 1) || clearly_out(dz, 2)) {
                  continue;
                }
                const double rawx = dx;
                const double rawy = dy;
                const double rawz = dz;
                const bool fx = fold_axis(dx, 0, sx);
                const bool fy = fold_axis(dy, 1, sy);
                const bool fz = fold_axis(dz, 2, sz);
                const double r2_round = dx * dx + dy * dy + dz * dz;
                if (r2_round <= search2) {
                  holds = axis_holds(rawx, 0, sx, fx) && axis_holds(rawy, 1, sy, fy) &&
                          axis_holds(rawz, 2, sz, fz);
                  gather.push_back(
                      Cand{j, sx, sy, sz, static_cast<std::uint8_t>(holds ? 0 : 1)});
                }
                continue;
              }
            }
            const double r2 = dx * dx + dy * dy + dz * dz;
            if (r2 <= search2) {
              gather.push_back(Cand{j, sx, sy, sz, static_cast<std::uint8_t>(holds ? 0 : 1)});
            }
          }
        }
      }
    }
    std::sort(gather.begin(), gather.end(),
              [](const Cand &a, const Cand &b) { return a.j < b.j; });
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
  off_[n] = static_cast<std::int32_t>(nbr_.size());
  if (fit && (record_round || record_general)) {
    installRoundShifts(shifts.data(), nbr_.size(), tiny, all_stable, record_general,
                       all_stable ? nullptr : flags.data());
  }
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
