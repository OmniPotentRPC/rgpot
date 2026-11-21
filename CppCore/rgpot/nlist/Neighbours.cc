// MIT License
// Copyright 2023--present rgpot developers

#define RGPOT_NLIST_BUILDING
#include "rgpot/nlist/Neighbours.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
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

namespace rgpot::nlist {
namespace {

/// Largest grid whose cell ids, and the offsets past them, fit an int.
constexpr std::uint64_t kMaxCells = 0x7ffffffe;

/// Cells of edge at least `edge` along `length`. A count past kMaxCells
/// is returned as kMaxCells + 1, which no grid accepts.
std::uint64_t axisCount(double length, double edge) {
  if (!(length > 0.0) || !(edge > 0.0)) {
    return 1;
  }
  const double cells = std::floor(length / edge);
  if (!(cells >= 1.0)) {
    return 1;
  }
  if (!(cells <= static_cast<double>(kMaxCells))) {
    return kMaxCells + 1;
  }
  auto n = static_cast<std::uint64_t>(cells);
  while (n > 1 && length / static_cast<double>(n) < edge) {
    --n;
  }
  return n;
}

/// Cells of the grid, or zero when there are more than kMaxCells.
std::uint64_t gridCells(const std::uint64_t (&axes)[3]) {
  std::uint64_t cells = 1;
  for (const std::uint64_t axis : axes) {
    if (axis > kMaxCells / cells) {
      return 0;
    }
    cells *= axis;
  }
  return cells;
}

/// Reciprocal basis of a row-major cell. `ok` is false when the
/// determinant is not finite and nonzero, an inverse element is not
/// finite, or a dual-lattice column norm is not finite and positive.
/// That is the condition under which a fully periodic general cell is
/// not binned in Cartesian coordinates.
struct ReciprocalBasis {
  bool ok = false;
  double inv[9]{};
  double dual[3]{};
};

ReciprocalBasis reciprocalBasis(const double box[9]) {
  ReciprocalBasis out;
  const double m0 = box[4] * box[8] - box[5] * box[7];
  const double m1 = box[3] * box[8] - box[5] * box[6];
  const double m2 = box[3] * box[7] - box[4] * box[6];
  const double det = box[0] * m0 - box[1] * m1 + box[2] * m2;
  if (!std::isfinite(det) || det == 0.0) {
    return out;
  }
  out.inv[0] = m0 / det;
  out.inv[1] = (box[2] * box[7] - box[1] * box[8]) / det;
  out.inv[2] = (box[1] * box[5] - box[2] * box[4]) / det;
  out.inv[3] = -m1 / det;
  out.inv[4] = (box[0] * box[8] - box[2] * box[6]) / det;
  out.inv[5] = (box[2] * box[3] - box[0] * box[5]) / det;
  out.inv[6] = m2 / det;
  out.inv[7] = (box[1] * box[6] - box[0] * box[7]) / det;
  out.inv[8] = (box[0] * box[4] - box[1] * box[3]) / det;
  for (double value : out.inv) {
    if (!std::isfinite(value)) {
      return out;
    }
  }
  out.dual[0] = std::hypot(out.inv[0], out.inv[3], out.inv[6]);
  out.dual[1] = std::hypot(out.inv[1], out.inv[4], out.inv[7]);
  out.dual[2] = std::hypot(out.inv[2], out.inv[5], out.inv[8]);
  out.ok = std::isfinite(out.dual[0]) && std::isfinite(out.dual[1]) &&
           std::isfinite(out.dual[2]) && out.dual[0] > 0.0 &&
           out.dual[1] > 0.0 && out.dual[2] > 0.0;
  return out;
}

[[noreturn]] void refuseCartesianBinning() {
  throw std::invalid_argument(
      "Neighbours: fully periodic general cell has no finite positive "
      "reciprocal basis");
}

} // namespace

std::uint64_t Neighbours::nextGeneration() noexcept {
  static std::atomic<std::uint64_t> counter{0};
  return counter.fetch_add(1, std::memory_order_relaxed) + 1;
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
                          const Spec &spec, bool &wrapped) const {
  wrapped = false;
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
  if (allPairs()) {
    return true;
  }
  // A streaming walk trusts the bins. Atoms that have moved may sit in
  // another cell, so the bins are reused only for an identical geometry.
  // A stored list keeps its candidates out to the skin.
  if (streaming_) {
    return std::memcmp(ref_.data(), xyz, n * 3 * sizeof(double)) == 0;
  }
  const double limit2 = 0.25 * spec.skin * spec.skin;
  for (std::size_t a = 0; a < n; ++a) {
    double dx = xyz[3 * a] - ref_[3 * a];
    double dy = xyz[3 * a + 1] - ref_[3 * a + 1];
    double dz = xyz[3 * a + 2] - ref_[3 * a + 2];
    if (dx * dx + dy * dy + dz * dz <= limit2) {
      continue;
    }
    fold(dx, dy, dz);
    if (!(dx * dx + dy * dy + dz * dz <= limit2)) {
      return false;
    }
    wrapped = true;
  }
  return true;
}

void Neighbours::buildCells(const double *xyz, std::size_t n) {
  const double edge = search_ > 0.0 ? search_ : 1.0;
  // A grid that does not fit, because an int cannot name its cells or
  // the OS reports too little memory for its offsets, is one cell. An
  // open axis can instead be binned on the atom span, which is what a
  // cluster in a huge box needs. A periodic axis keeps the cell length.
  // One cell remains when the tight grid does not fit either. A grid
  // that fits is unchanged, so subtracting a zero origin leaves those
  // bins as they were.
  double origin[3] = {0.0, 0.0, 0.0};
  double span[3] = {len_[0], len_[1], len_[2]};
  std::uint64_t axes[3] = {1, 1, 1};
  auto set_axes = [&]() {
    for (int k = 0; k < 3; ++k) {
      axes[k] = axisCount(span[k], edge);
    }
  };
  // A grid fits when an int names every cell and, when the OS reports
  // it, the offsets fit in the memory available.
  auto fits = [](const std::uint64_t (&grid)[3], std::uint64_t avail) {
    const std::uint64_t cells = gridCells(grid);
    return cells != 0 &&
           (avail == 0 || cells <= avail / sizeof(std::int32_t));
  };
  set_axes();
  double frac_inv[9]{};
  bool use_frac = false;
  // Fractional bins are required for a fully periodic general cell,
  // including every sheared cell whose three edges are positive.
  // Cartesian bins are the path when the cell is not general, or when
  // an axis is open or its diagonal entry is not positive. A fully
  // periodic
  // general cell whose reciprocal basis is not finite and positive is
  // an error: it is not binned in Cartesian coordinates.
  const bool fractional_required =
      use_general_ && periodic_[0] && periodic_[1] && periodic_[2];
  if (fractional_required) {
    const ReciprocalBasis basis = reciprocalBasis(box_);
    if (!basis.ok) {
      refuseCartesianBinning();
    }
    for (int k = 0; k < 9; ++k) {
      frac_inv[k] = basis.inv[k];
    }
    for (int k = 0; k < 3; ++k) {
      axes[k] = axisCount(1.0, edge * basis.dual[k]);
    }
    use_frac = true;
  }
  const auto avail = availableBytes();
  if (!fits(axes, avail)) {
    bool any_open = false;
    for (int k = 0; k < 3; ++k) {
      if (periodic_[k] || n == 0) {
        continue;
      }
      any_open = true;
      double lo = xyz[static_cast<std::size_t>(k)];
      double hi = lo;
      for (std::size_t a = 1; a < n; ++a) {
        const double v = xyz[3 * a + static_cast<std::size_t>(k)];
        if (v < lo) {
          lo = v;
        }
        if (v > hi) {
          hi = v;
        }
      }
      const double extent = hi - lo;
      span[k] = extent > edge ? extent : edge;
      origin[k] = lo;
    }
    if (any_open) {
      set_axes();
    }
    if (!fits(axes, avail)) {
      axes[0] = axes[1] = axes[2] = 1;
    }
  }
  nx_ = static_cast<int>(axes[0]);
  ny_ = static_cast<int>(axes[1]);
  nz_ = static_cast<int>(axes[2]);
  const auto ncells = static_cast<std::size_t>(nx_) *
                      static_cast<std::size_t>(ny_) *
                      static_cast<std::size_t>(nz_);
  cell_start_.assign(ncells + 1, 0);
  cell_of_.resize(n);
  auto bin = [](double x, double length, int naxis) {
    if (!(length > 0.0) || naxis <= 1) {
      return 0;
    }
    const double t = x / length * static_cast<double>(naxis);
    if (!(t >= 0.0)) {
      return 0;
    }
    if (t >= static_cast<double>(naxis)) {
      return naxis - 1;
    }
    return static_cast<int>(t);
  };
  for (std::size_t a = 0; a < n; ++a) {
    int ix = 0;
    int iy = 0;
    int iz = 0;
    if (use_frac) {
      auto bin_frac = [](double s, int naxis) {
        if (!std::isfinite(s)) {
          throw std::invalid_argument(
              "Neighbours: non-finite fractional coordinate");
        }
        if (naxis <= 1) {
          return 0;
        }
        double wrapped = s - std::floor(s);
        if (!(wrapped >= 0.0) || wrapped >= 1.0) {
          wrapped = 0.0;
        }
        auto i = static_cast<int>(wrapped * static_cast<double>(naxis));
        if (i >= naxis) {
          i = naxis - 1;
        }
        if (i < 0) {
          i = 0;
        }
        return i;
      };
      const double x = xyz[3 * a];
      const double y = xyz[3 * a + 1];
      const double z = xyz[3 * a + 2];
      ix = bin_frac(x * frac_inv[0] + y * frac_inv[3] + z * frac_inv[6], nx_);
      iy = bin_frac(x * frac_inv[1] + y * frac_inv[4] + z * frac_inv[7], ny_);
      iz = bin_frac(x * frac_inv[2] + y * frac_inv[5] + z * frac_inv[8], nz_);
    } else {
      ix = bin(xyz[3 * a] - origin[0], span[0], nx_);
      iy = bin(xyz[3 * a + 1] - origin[1], span[1], ny_);
      iz = bin(xyz[3 * a + 2] - origin[2], span[2], nz_);
    }
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
    cell_atoms_[static_cast<std::size_t>(fill[c]++)] =
        static_cast<std::int32_t>(a);
  }
}

std::uint64_t Neighbours::candidateBound() const {
  // Integer adds over the stencil. A dilute system above kTeamMinAtoms
  // has almost none, and a region would cost more than the count, so
  // this stays on the calling thread. The gather below is the build.
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
          here +=
              static_cast<std::uint64_t>(cell_start_[c + 1] - cell_start_[c]);
        }
      }
    }
    if (here > 0) {
      bound += here - 1;
    }
  }
  return bound;
}

void Neighbours::buildStored(const double *xyz, std::size_t n,
                             std::uint64_t bound) {
  (void)xyz;
  const double search2 = search_ * search_;
  // Partners of one atom, unsorted. The caller sorts and appends them.
  auto gather = [&](std::size_t a, std::vector<std::int32_t> &out) {
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
          for (std::int32_t q = cell_start_[c]; q < cell_start_[c + 1]; ++q) {
            const auto j = cell_atoms_[static_cast<std::size_t>(q)];
            if (j == static_cast<std::int32_t>(a)) {
              continue;
            }
            double dx = 0;
            double dy = 0;
            double dz = 0;
            if (!foldPairWithin(static_cast<std::int32_t>(a), j, search2, dx,
                                dy, dz)) {
              continue;
            }
            const double r2 = dx * dx + dy * dy + dz * dz;
            if (r2 <= search2) {
              out.push_back(j);
            }
          }
        }
      }
    }
  };
  // A region costs more than a short gather. The pair kernels use 4096
  // before they thread a half list. A dilute system can still report
  // about one stencil candidate per atom, which is the same cheap pass,
  // so the gather also stays serial unless the stencil holds more
  // candidates than atoms.
  constexpr std::uint64_t kMinParallelCandidates = 4096;
  const bool worth = bound >= kMinParallelCandidates && bound > n;
  const int nt = worth ? teamSize(n) : 1;
  if (nt <= 1) {
    off_.assign(n + 1, 0);
    nbr_.clear();
    std::vector<std::int32_t> row;
    for (std::size_t a = 0; a < n; ++a) {
      off_[a] = static_cast<std::int32_t>(nbr_.size());
      row.clear();
      gather(a, row);
      std::sort(row.begin(), row.end());
      nbr_.insert(nbr_.end(), row.begin(), row.end());
    }
    off_[n] = static_cast<std::int32_t>(nbr_.size());
    return;
  }
  // One buffer per slice, reserved once, written only by the slice that
  // owns it. Joining in slice order is the atom order of one thread.
  struct Part {
    std::vector<std::int32_t> nbr;
    std::vector<std::int32_t> count;
  };
  std::vector<Part> parts(static_cast<std::size_t>(nt));
  forEachSlice(nt, [&](int t) {
    const auto range = partition(static_cast<std::int64_t>(n), t, nt);
    auto &part = parts[static_cast<std::size_t>(t)];
    const auto len = static_cast<std::size_t>(range.end - range.begin);
    part.count.assign(len, 0);
    part.nbr.reserve(len * 64);
    std::vector<std::int32_t> row;
    row.reserve(64);
    std::size_t k = 0;
    for (std::int64_t i = range.begin; i < range.end; ++i, ++k) {
      row.clear();
      gather(static_cast<std::size_t>(i), row);
      std::sort(row.begin(), row.end());
      part.count[k] = static_cast<std::int32_t>(row.size());
      part.nbr.insert(part.nbr.end(), row.begin(), row.end());
    }
  });
  std::size_t total = 0;
  for (const Part &part : parts) {
    total += part.nbr.size();
  }
  nbr_.clear();
  nbr_.reserve(total);
  off_.assign(n + 1, 0);
  std::size_t cursor = 0;
  std::size_t atom = 0;
  for (const Part &part : parts) {
    nbr_.insert(nbr_.end(), part.nbr.begin(), part.nbr.end());
    for (const std::int32_t c : part.count) {
      off_[atom] = static_cast<std::int32_t>(cursor);
      cursor += static_cast<std::size_t>(c);
      ++atom;
    }
  }
  off_[n] = static_cast<std::int32_t>(nbr_.size());
}

void Neighbours::update(const double *xyz, std::size_t n, const double *box9,
                        const Spec &spec) {
  // A partial refresh can keep the reference coordinates of the last
  // build beside bins or a list it has already cleared. reusable() would
  // then accept those coordinates and walk the cleared state.
  try {
    refresh(xyz, n, box9, spec);
  } catch (...) {
    reset();
    throw;
  }
}

void Neighbours::refresh(const double *xyz, std::size_t n, const double *box9,
                         const Spec &spec) {
  xyz_ = xyz;
  bool wrapped = false;
  if (reusable(xyz, n, box9, spec, wrapped)) {
    reused_ = true;
    // The tile walk takes the image of a pair from the bins of its atoms,
    // and a wrapped atom left its bin on the far side of the cell.
    if (wrapped) {
      buildCells(xyz, n);
    }
    return;
  }
  reused_ = false;
  generation_ = nextGeneration();
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
  // Bin before MinimumImage when this update builds cells, so a fully
  // periodic general cell with no reciprocal basis reports that error
  // instead of a Cartesian grid. Paths that store no bins still need
  // the image for later walks.
  const bool bins = n >= 2 && search_ > 0.0 && !std::isinf(search_);
  if (!bins && use_general_) {
    general_ = MinimumImage(box9, spec.periodic);
  }
  if (n < 2 || !(search_ > 0.0)) {
    streaming_ = false;
    mode_ = Mode::Stored;
    off_.assign(n + 1, 0);
    nbr_.clear();
    built_ = true;
    ref_.assign(xyz, xyz + 3 * n);
    return;
  }
  // Every pair is a candidate, so the walks need neither bins nor a list,
  // and the coordinates never invalidate the update. allPairs() reads
  // this state back.
  if (std::isinf(search_)) {
    mode_ = Mode::Stream;
    built_ = true;
    return;
  }
  buildCells(xyz, n);
  if (use_general_) {
    general_ = MinimumImage(box9, spec.periodic);
  }
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
  const bool want_store =
      spec.mode == Mode::Stored ? fits : spec.mode == Mode::Auto && fits;
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
