// MIT License
// Copyright 2023--present rgpot developers

#include "rgpot/nlist/Neighbours.hpp"
#include "rgpot/nlist/NeighbourList.hpp"

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

} // namespace

bool needsImages(const double *box9, const std::array<bool, 3> &periodic, double cutoff) {
  if (!(cutoff > 0.0)) {
    return false;
  }
  for (int k = 0; k < 3; ++k) {
    if (!periodic[static_cast<std::size_t>(k)]) {
      continue;
    }
    const double *a = box9 + 3 * k;
    const double *u = box9 + 3 * ((k + 1) % 3);
    const double *v = box9 + 3 * ((k + 2) % 3);
    const double cx = u[1] * v[2] - u[2] * v[1];
    const double cy = u[2] * v[0] - u[0] * v[2];
    const double cz = u[0] * v[1] - u[1] * v[0];
    const double area = std::sqrt(cx * cx + cy * cy + cz * cz);
    const double volume = std::fabs(a[0] * cx + a[1] * cy + a[2] * cz);
    if (!(area > 0.0) || !(volume > 0.0)) {
      return false;
    }
    if (volume / area < 2.0 * cutoff) {
      return true;
    }
  }
  return false;
}

namespace {

void cross(const double *u, const double *v, double *out) {
  out[0] = u[1] * v[2] - u[2] * v[1];
  out[1] = u[2] * v[0] - u[0] * v[2];
  out[2] = u[0] * v[1] - u[1] * v[0];
}

double norm3(const double *v) { return std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]); }

double triple(const double *rows) {
  double c[3];
  cross(rows + 3, rows + 6, c);
  return rows[0] * c[0] + rows[1] * c[1] + rows[2] * c[2];
}

bool isBasis(const double *rows) {
  const double det = triple(rows);
  const double scale = norm3(rows) * norm3(rows + 3) * norm3(rows + 6);
  return std::isfinite(det) && std::fabs(det) > 1.0e-12 * scale;
}

} // namespace

CellFrame cellFrame(const double *box9, const std::array<bool, 3> &periodic) {
  CellFrame frame;
  double rows[9];
  std::memcpy(rows, box9, sizeof(rows));
  int nper = 0;
  for (int k = 0; k < 3; ++k) {
    frame.periodic[k] = periodic[static_cast<std::size_t>(k)] && norm3(rows + 3 * k) > 0.0;
    nper += frame.periodic[k] ? 1 : 0;
  }
  if (!isBasis(rows)) {
    if (nper == 3) {
      throw Error(Code::Box, "the periodic cell rows are not independent");
    }
    if (nper == 2) {
      const int k = !frame.periodic[0] ? 0 : (!frame.periodic[1] ? 1 : 2);
      double *out = rows + 3 * k;
      cross(rows + 3 * ((k + 1) % 3), rows + 3 * ((k + 2) % 3), out);
      const double len = norm3(out);
      if (!(len > 0.0)) {
        throw Error(Code::Box, "the periodic cell rows are parallel");
      }
      for (int c = 0; c < 3; ++c) {
        out[c] /= len;
      }
    } else if (nper == 1) {
      const int k = frame.periodic[0] ? 0 : (frame.periodic[1] ? 1 : 2);
      const double len = norm3(rows + 3 * k);
      const double u[3] = {rows[3 * k] / len, rows[3 * k + 1] / len, rows[3 * k + 2] / len};
      int least = 0;
      for (int c = 1; c < 3; ++c) {
        if (std::fabs(u[c]) < std::fabs(u[least])) {
          least = c;
        }
      }
      double e[3] = {0.0, 0.0, 0.0};
      e[least] = 1.0;
      double v[3];
      cross(u, e, v);
      const double vl = norm3(v);
      for (double &x : v) {
        x /= vl;
      }
      double w[3];
      cross(u, v, w);
      std::memcpy(rows + 3 * ((k + 1) % 3), v, sizeof(v));
      std::memcpy(rows + 3 * ((k + 2) % 3), w, sizeof(w));
    } else {
      static constexpr double eye[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
      std::memcpy(rows, eye, sizeof(rows));
    }
  }
  // f = r H^-1 for row vectors r = f H. Column k of H^-1 is the cross
  // product of the other two rows over the determinant.
  const double det = triple(rows);
  for (int k = 0; k < 3; ++k) {
    double *d = frame.dual + 3 * k;
    cross(rows + 3 * ((k + 1) % 3), rows + 3 * ((k + 2) % 3), d);
    frame.width[k] = std::fabs(det) / norm3(d);
    for (int c = 0; c < 3; ++c) {
      d[c] /= det;
    }
  }
  return frame;
}

void cellFraction(const CellFrame &frame, const double *r, double f[3], std::int32_t o[3]) {
  for (int k = 0; k < 3; ++k) {
    const double *d = frame.dual + 3 * k;
    double x = r[0] * d[0] + r[1] * d[1] + r[2] * d[2];
    if (!std::isfinite(x)) {
      x = 0.0;
    }
    o[k] = 0;
    if (frame.periodic[k]) {
      const double whole = std::floor(x);
      if (!(std::fabs(whole) < 1.0e9)) {
        throw Error(Code::Box, "an atom lies more than 1e9 cells from the box");
      }
      o[k] = static_cast<std::int32_t>(whole);
      x -= whole;
      if (x >= 1.0) {
        x -= 1.0;
        ++o[k];
      }
    }
    f[k] = x;
  }
}

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
  key.images = spec.images;
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
  if (use_general_) {
    buildFrameCells(xyz, n, edge);
    return;
  }
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
  fillCells(n);
}

void Neighbours::buildFrameCells(const double *xyz, std::size_t n, double edge) {
  // Bins along the cell rows, a search thick across, with every atom
  // wrapped into the cell first. Diagonal lengths of a sheared cell are
  // not plane spacings, and an atom outside them is not in its bin.
  const CellFrame frame = cellFrame(box_, key_.periodic);
  std::vector<double> frac(3 * n);
  double lo[3] = {0.0, 0.0, 0.0};
  double hi[3] = {0.0, 0.0, 0.0};
  for (std::size_t a = 0; a < n; ++a) {
    std::int32_t o[3];
    cellFraction(frame, xyz + 3 * a, frac.data() + 3 * a, o);
    for (int k = 0; k < 3; ++k) {
      const double f = frac[3 * a + static_cast<std::size_t>(k)];
      lo[k] = a == 0 ? f : std::min(lo[k], f);
      hi[k] = a == 0 ? f : std::max(hi[k], f);
    }
  }
  int nb[3];
  for (int k = 0; k < 3; ++k) {
    periodic_[k] = frame.periodic[k];
    const double span = frame.periodic[k] ? frame.width[k] : (hi[k] - lo[k]) * frame.width[k];
    nb[k] = axisCount(span, edge);
  }
  const double bytes = static_cast<double>(nb[0]) * static_cast<double>(nb[1]) *
                       static_cast<double>(nb[2]) * static_cast<double>(sizeof(std::int32_t));
  const auto avail = availableBytes();
  if (avail > 0 && bytes > static_cast<double>(avail)) {
    nb[0] = nb[1] = nb[2] = 1;
  }
  nx_ = nb[0];
  ny_ = nb[1];
  nz_ = nb[2];
  const auto ncells = static_cast<std::size_t>(nx_) * static_cast<std::size_t>(ny_) *
                      static_cast<std::size_t>(nz_);
  cell_start_.assign(ncells + 1, 0);
  cell_of_.resize(n);
  for (std::size_t a = 0; a < n; ++a) {
    int bin[3];
    for (int k = 0; k < 3; ++k) {
      const double f = frac[3 * a + static_cast<std::size_t>(k)];
      double t = 0.0;
      if (frame.periodic[k]) {
        t = f * nb[k];
      } else if (hi[k] > lo[k]) {
        t = (f - lo[k]) / (hi[k] - lo[k]) * nb[k];
      }
      const int b = t >= 0.0 ? static_cast<int>(t) : 0;
      bin[k] = std::min(b, nb[k] - 1);
    }
    const int id = (bin[0] * ny_ + bin[1]) * nz_ + bin[2];
    cell_of_[a] = id;
    ++cell_start_[static_cast<std::size_t>(id) + 1];
  }
  fillCells(n);
}

void Neighbours::fillCells(std::size_t n) {
  const auto ncells = cell_start_.size() - 1;
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
  std::vector<std::int32_t> gather;
  const double search2 = search_ * search_;
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
            double dx = 0;
            double dy = 0;
            double dz = 0;
            foldPair(static_cast<std::int32_t>(a), j, dx, dy, dz);
            const double r2 = dx * dx + dy * dy + dz * dz;
            if (r2 <= search2) {
              gather.push_back(j);
            }
          }
        }
      }
    }
    std::sort(gather.begin(), gather.end());
    nbr_.insert(nbr_.end(), gather.begin(), gather.end());
  }
  off_[n] = static_cast<std::int32_t>(nbr_.size());
}

void Neighbours::buildImages(const double *xyz, std::size_t n,
                             const std::array<bool, 3> &periodic) {
  Query query;
  query.cutoff = search_;
  query.cut = Cut::Loose;
  query.shape = Pairs{Shell::Full};
  query.periodic = periodic;
  query.fields.vectors = false;
  query.fields.distances = false;
  Table table;
  build(table, xyz, n, box_, query);
  // build() drops a zero distance. Coincident atoms keep their pair so the
  // kernel meets r = 0, as it does on the minimum-image walk.
  std::vector<std::array<std::int32_t, 2>> same;
  {
    std::vector<std::int32_t> by(n);
    for (std::size_t a = 0; a < n; ++a) {
      by[a] = static_cast<std::int32_t>(a);
    }
    std::sort(by.begin(), by.end(), [&](std::int32_t a, std::int32_t b) {
      return std::lexicographical_compare(xyz + 3 * a, xyz + 3 * a + 3, xyz + 3 * b,
                                          xyz + 3 * b + 3);
    });
    std::size_t lo = 0;
    while (lo < n) {
      std::size_t hi = lo + 1;
      while (hi < n && std::equal(xyz + 3 * by[lo], xyz + 3 * by[lo] + 3, xyz + 3 * by[hi])) {
        ++hi;
      }
      for (std::size_t p = lo; p < hi; ++p) {
        for (std::size_t q = lo; q < hi; ++q) {
          if (p != q) {
            same.push_back({by[p], by[q]});
          }
        }
      }
      lo = hi;
    }
  }
  const std::size_t built = table.size();
  const std::size_t rows = built + same.size();
  if (rows > static_cast<std::size_t>(0x7fffffff)) {
    throw Error(Code::Nomem, "image table exceeds the 32-bit pair index");
  }
  static constexpr std::int32_t kZero[3] = {0, 0, 0};
  const auto first = [&](std::size_t r) { return r < built ? table.first[r] : same[r - built][0]; };
  const auto second = [&](std::size_t r) {
    return r < built ? table.second[r] : same[r - built][1];
  };
  const auto shift = [&](std::size_t r) {
    return r < built ? table.shift.data() + 3 * r : kZero;
  };
  // Group by atom, then by partner and shift, so the walk order does not
  // depend on the build's thread split.
  std::vector<std::size_t> order(rows);
  for (std::size_t p = 0; p < rows; ++p) {
    order[p] = p;
  }
  std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
    if (first(a) != first(b)) {
      return first(a) < first(b);
    }
    if (second(a) != second(b)) {
      return second(a) < second(b);
    }
    return std::lexicographical_compare(shift(a), shift(a) + 3, shift(b), shift(b) + 3);
  });
  img_off_.assign(n + 1, 0);
  img_j_.resize(rows);
  img_vec_.resize(3 * rows);
  img_half_.resize(rows);
  for (std::size_t r = 0; r < rows; ++r) {
    const std::size_t p = order[r];
    const auto i = first(p);
    const auto j = second(p);
    const std::int32_t *sp = shift(p);
    const int sa = sp[0];
    const int sb = sp[1];
    const int sc = sp[2];
    ++img_off_[static_cast<std::size_t>(i) + 1];
    img_j_[r] = j;
    for (int k = 0; k < 3; ++k) {
      img_vec_[3 * r + static_cast<std::size_t>(k)] =
          sa * box_[k] + sb * box_[3 + k] + sc * box_[6 + k];
    }
    img_half_[r] = keepHalf(i, j, sa, sb, sc) ? 1 : 0;
  }
  for (std::size_t a = 0; a < n; ++a) {
    img_off_[a + 1] += img_off_[a];
  }
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
  key_.images = spec.images;
  off_.clear();
  nbr_.clear();
  images_ = false;
  img_off_.clear();
  img_j_.clear();
  img_vec_.clear();
  img_half_.clear();
  streaming_ = true;
  // A lattice vector may have a zero diagonal entry (an fcc primitive
  // cell), so the requested flags decide here, not periodic_. The minimum
  // image is exact while every height is twice the cutoff: the walk folds
  // from the current coordinates, and a pair left out at build is still
  // past the cutoff after moves inside the skin. A sheared fold rounds the
  // fractional coordinates, which finds the minimum image only within half
  // a face spacing, so a sheared cell needs twice the search.
  const double thick = sheared ? search_ : spec.cutoff;
  if (spec.images && n > 0 && needsImages(box9, spec.periodic, thick)) {
    buildImages(xyz, n, spec.periodic);
    images_ = true;
    streaming_ = false;
    mode_ = Mode::Stored;
    ++builds_;
    built_ = true;
    ref_.assign(xyz, xyz + 3 * n);
    return;
  }
  if (n < 2 || !(search_ > 0.0)) {
    streaming_ = false;
    mode_ = Mode::Stored;
    off_.assign(n + 1, 0);
    nbr_.clear();
    // One bin, so forShifts still finds a lone atom's own images.
    nx_ = ny_ = nz_ = 1;
    cell_of_.assign(n, 0);
    cell_start_ = {0, static_cast<std::int32_t>(n)};
    cell_atoms_.resize(n);
    for (std::size_t a = 0; a < n; ++a) {
      cell_atoms_[a] = static_cast<std::int32_t>(a);
    }
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
