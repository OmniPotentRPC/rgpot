// MIT License
// Copyright 2023--present rgpot developers

#include "rgpot/nlist/NeighbourList.hpp"
#include "rgpot/nlist/rgpot_nlist.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <new>
#include <string>
#include <utility>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace rgpot::nlist {
namespace {

struct Rec {
  std::int32_t first = 0;
  std::int32_t second = 0;
  std::int32_t shift[3]{};
  double vector[3]{};
  double distance = 0;
  std::uint8_t mask = 0xff;
};

void check_geometry(const double *positions, std::size_t n, const double *box9,
                    const Query &query) {
  if (n > 0 && positions == nullptr) {
    throw Error(Code::Null, "positions is null");
  }
  if (box9 == nullptr) {
    throw Error(Code::Null, "box is null");
  }
  if (!std::isfinite(query.cutoff) || query.cutoff < 0.0) {
    throw Error(Code::Cutoff, "cutoff must be finite and >= 0");
  }
  if (!std::isfinite(query.skin) || query.skin < 0.0) {
    throw Error(Code::Cutoff, "skin must be finite and >= 0");
  }
  if (query.skin > 0.0 && query.cut == Cut::Strict) {
    throw Error(Code::Skin,
                "a positive skin needs a loose cut; a strict cut stores no buffer");
  }
  if (const auto *pairs = std::get_if<Pairs>(&query.shape)) {
    if (pairs->shell != Shell::Half && pairs->shell != Shell::Full) {
      throw Error(Code::Shape, "a pair list shell must be half or full");
    }
  }
  if (query.cut != Cut::Strict && query.cut != Cut::Loose) {
    throw Error(Code::Shape, "cut must be strict or loose");
  }
  for (int k = 0; k < 9; ++k) {
    if (!std::isfinite(box9[k])) {
      throw Error(Code::Box, "box has a non-finite element");
    }
  }
  const bool any_periodic =
      query.periodic[0] || query.periodic[1] || query.periodic[2];
  if (any_periodic) {
    const double a[3] = {box9[0], box9[1], box9[2]};
    const double b[3] = {box9[3], box9[4], box9[5]};
    const double c[3] = {box9[6], box9[7], box9[8]};
    const double bx = b[1] * c[2] - b[2] * c[1];
    const double by = b[2] * c[0] - b[0] * c[2];
    const double bz = b[0] * c[1] - b[1] * c[0];
    const double vol = std::fabs(a[0] * bx + a[1] * by + a[2] * bz);
    if (!(vol > 0.0)) {
      throw Error(Code::Box, "a periodic box has zero volume");
    }
  }
}

double search_radius(const Query &query) {
  if (query.cut == Cut::Loose) {
    return query.cutoff + query.skin;
  }
  return query.cutoff;
}

bool clustered(const Query &query) { return std::holds_alternative<Cluster4>(query.shape); }

bool full_shell(const Query &query) {
  if (const auto *pairs = std::get_if<Pairs>(&query.shape)) {
    return pairs->shell == Shell::Full;
  }
  return true;
}

bool accept_distance(double r2, const Query &query) {
  if (!(r2 > 0.0)) {
    return false;
  }
  const double cut2 = query.cutoff * query.cutoff;
  if (query.cut == Cut::Strict) {
    return r2 < cut2;
  }
  const double reach = search_radius(query);
  if (query.skin > 0.0) {
    return r2 < reach * reach;
  }
  return r2 <= cut2;
}

/// Borrowed view of a prepared pair-type matrix. `types` has one id per
/// atom. `limit2` is the squared reach, row-major, or null when `typed`
/// is false.
struct CutView {
  bool typed = false;
  bool inclusive = false;
  std::int32_t ntypes = 0;
  const std::int32_t *types = nullptr;
  const double *limit2 = nullptr;
};

struct OwnedCut {
  bool typed = false;
  bool inclusive = false;
  std::int32_t ntypes = 0;
  double search = 0;
  std::vector<std::int32_t> types;
  std::vector<double> limit2;

  [[nodiscard]] CutView view() const {
    CutView cut;
    cut.typed = typed;
    cut.inclusive = inclusive;
    cut.ntypes = ntypes;
    cut.types = types.data();
    cut.limit2 = limit2.data();
    return cut;
  }
};

OwnedCut prepare_cut(std::size_t n, const Query &query, PairCutoffs cuts) {
  OwnedCut out;
  out.search = search_radius(query);
  if (cuts.ntypes == 0) {
    return out;
  }
  if (cuts.ntypes < 0 || cuts.cutoff == nullptr || (n > 0 && cuts.types == nullptr)) {
    throw Error(Code::Cutoff, "a pair-type cutoff needs one type per atom and a square matrix");
  }
  const auto nt = cuts.ntypes;
  const auto n2 = static_cast<std::size_t>(nt) * static_cast<std::size_t>(nt);
  out.typed = true;
  out.inclusive = query.cut == Cut::Loose && !(query.skin > 0.0);
  out.ntypes = nt;
  if (n > 0) {
    out.types.assign(cuts.types, cuts.types + n);
  }
  out.limit2.resize(n2);
  double widest = 0.0;
  for (std::size_t k = 0; k < n2; ++k) {
    const double c = cuts.cutoff[k];
    if (!std::isfinite(c) || c < 0.0) {
      throw Error(Code::Cutoff, "a pair-type cutoff must be finite and >= 0");
    }
    if (!(c > 0.0)) {
      out.limit2[k] = 0.0;
      continue;
    }
    const double reach = query.cut == Cut::Loose ? c + query.skin : c;
    out.limit2[k] = reach * reach;
    if (reach > widest) {
      widest = reach;
    }
  }
  for (std::size_t a = 0; a < n; ++a) {
    const auto t = out.types[a];
    if (t < 0 || t >= nt) {
      throw Error(Code::Shape, "a pair type is outside the cutoff matrix");
    }
  }
  if (widest > out.search) {
    out.search = widest;
  }
  return out;
}

bool type_ok(const CutView &cut, std::int32_t i, std::int32_t j, double r2) {
  const auto ti = cut.types[static_cast<std::size_t>(i)];
  const auto tj = cut.types[static_cast<std::size_t>(j)];
  const double lim = cut.limit2[static_cast<std::size_t>(ti) * static_cast<std::size_t>(cut.ntypes) +
                                static_cast<std::size_t>(tj)];
  if (!(lim > 0.0)) {
    return false;
  }
  return cut.inclusive ? r2 <= lim : r2 < lim;
}

bool keep_r2(const CutView &cut, const Query &query, std::int32_t i, std::int32_t j, double r2) {
  if (cut.typed) {
    return type_ok(cut, i, j, r2);
  }
  return accept_distance(r2, query);
}

// vesin half-list rule for a pair and its periodic self image.
bool keep_half(std::int32_t first, std::int32_t second, int sa, int sb, int sc) {
  if (first > second) {
    return false;
  }
  if (first < second) {
    return true;
  }
  const int sum = sa + sb + sc;
  if (sum < 0) {
    return false;
  }
  if (sum == 0 && (sc < 0 || (sc == 0 && sb < 0))) {
    return false;
  }
  return sa != 0 || sb != 0 || sc != 0;
}

void shift_bounds(const double *box, const std::array<bool, 3> &periodic,
                  double search, int nmax[3]) {
  const double a[3] = {box[0], box[1], box[2]};
  const double b[3] = {box[3], box[4], box[5]};
  const double c[3] = {box[6], box[7], box[8]};
  const double bx[3] = {b[1] * c[2] - b[2] * c[1], b[2] * c[0] - b[0] * c[2],
                        b[0] * c[1] - b[1] * c[0]};
  const double cx[3] = {c[1] * a[2] - c[2] * a[1], c[2] * a[0] - c[0] * a[2],
                        c[0] * a[1] - c[1] * a[0]};
  const double ax[3] = {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
                        a[0] * b[1] - a[1] * b[0]};
  const double vol = std::fabs(a[0] * bx[0] + a[1] * bx[1] + a[2] * bx[2]);
  const double nb = std::sqrt(bx[0] * bx[0] + bx[1] * bx[1] + bx[2] * bx[2]);
  const double nc = std::sqrt(cx[0] * cx[0] + cx[1] * cx[1] + cx[2] * cx[2]);
  const double na = std::sqrt(ax[0] * ax[0] + ax[1] * ax[1] + ax[2] * ax[2]);
  const double height[3] = {nb > 0.0 ? vol / nb : 0.0, nc > 0.0 ? vol / nc : 0.0,
                            na > 0.0 ? vol / na : 0.0};
  for (int k = 0; k < 3; ++k) {
    if (!periodic[static_cast<std::size_t>(k)] || !(height[k] > 0.0) ||
        !(search > 0.0)) {
      nmax[k] = 0;
    } else {
      nmax[k] = static_cast<int>(std::ceil(search / height[k]));
    }
  }
}

void image_vector(const double *pos, std::size_t i, std::size_t j, int sa, int sb,
                  int sc, const double *box, double &vx, double &vy, double &vz) {
  vx = pos[3 * j] - pos[3 * i];
  vy = pos[3 * j + 1] - pos[3 * i + 1];
  vz = pos[3 * j + 2] - pos[3 * i + 2];
  vx += static_cast<double>(sa) * box[0] + static_cast<double>(sb) * box[3] +
        static_cast<double>(sc) * box[6];
  vy += static_cast<double>(sa) * box[1] + static_cast<double>(sb) * box[4] +
        static_cast<double>(sc) * box[7];
  vz += static_cast<double>(sa) * box[2] + static_cast<double>(sb) * box[5] +
        static_cast<double>(sc) * box[8];
}

int axis_shift(double delta, double inv, bool periodic) {
  if (!periodic || !(inv > 0.0) || !std::isfinite(delta)) {
    return 0;
  }
  const double t = delta * inv;
  const double shifted = t + std::copysign(0.5, t);
  if (!std::isfinite(shifted) || shifted >= 9223372036854775808.0 ||
      shifted <= -9223372036854775808.0) {
    return 0;
  }
  return static_cast<int>(static_cast<long long>(shifted));
}

bool one_minimum_image(const double *box, const Query &query, double search) {
  const bool sheared = box[1] != 0.0 || box[2] != 0.0 || box[3] != 0.0 || box[5] != 0.0 ||
                       box[6] != 0.0 || box[7] != 0.0;
  if (sheared) {
    return false;
  }
  const double len[3] = {box[0], box[4], box[8]};
  for (int k = 0; k < 3; ++k) {
    if (!query.periodic[static_cast<std::size_t>(k)]) {
      continue;
    }
    if (!(len[k] >= 2.0 * search)) {
      return false;
    }
  }
  return true;
}

Neighbours prepare_cells(const double *pos, std::size_t n, const double *box,
                         const Query &query, double search) {
  Spec spec;
  spec.cutoff = search;
  spec.skin = 0;
  spec.periodic = query.periodic;
  const bool sheared = box[1] != 0.0 || box[2] != 0.0 || box[3] != 0.0 || box[5] != 0.0 ||
                       box[6] != 0.0 || box[7] != 0.0;
  spec.image = sheared ? Image::General : Image::Round;
  spec.mode = Mode::Stream;
  Neighbours cells;
  cells.update(pos, n, box, spec);
  return cells;
}

// Cell-list candidates from Neighbours, then the same image test the brute
// force used. A box at least two cutoffs wide has one image, so the folded
// vector is that image. A narrower or sheared cell retries the shift stencil
// on those candidates only.
template <typename Fn>
void each_cell(const Neighbours &cells, const double *pos, const double *box,
               const Query &query, const CutView &cut, const int nmax[3], bool one_image,
               std::size_t begin, std::size_t end, Fn &&fn) {
  auto accept = [&](std::int32_t i, std::int32_t j, int sa, int sb, int sc, double vx,
                    double vy, double vz) {
    const double r2 = vx * vx + vy * vy + vz * vz;
    if (!keep_r2(cut, query, i, j, r2)) {
      return;
    }
    if (!full_shell(query) && !keep_half(i, j, sa, sb, sc)) {
      return;
    }
    fn(i, j, sa, sb, sc, vx, vy, vz, std::sqrt(r2));
  };
  if (one_image) {
    const double inv[3] = {box[0] > 0.0 ? 1.0 / box[0] : 0.0, box[4] > 0.0 ? 1.0 / box[4] : 0.0,
                           box[8] > 0.0 ? 1.0 / box[8] : 0.0};
    // forHalf is keep_half when a single image exists: j < i is dropped
    // and a self pair is not emitted.
    auto take = [&](std::int32_t i, std::int32_t j, double dx, double dy, double dz, double r2) {
      if (!keep_r2(cut, query, i, j, r2)) {
        return;
      }
      const auto ui = static_cast<std::size_t>(i);
      const auto uj = static_cast<std::size_t>(j);
      const int sa = axis_shift(pos[3 * ui] - pos[3 * uj], inv[0], query.periodic[0]);
      const int sb = axis_shift(pos[3 * ui + 1] - pos[3 * uj + 1], inv[1], query.periodic[1]);
      const int sc = axis_shift(pos[3 * ui + 2] - pos[3 * uj + 2], inv[2], query.periodic[2]);
      fn(i, j, sa, sb, sc, -dx, -dy, -dz, std::sqrt(r2));
    };
    if (full_shell(query)) {
      cells.forDirected(static_cast<std::int64_t>(begin), static_cast<std::int64_t>(end), take);
    } else {
      cells.forHalf(static_cast<std::int64_t>(begin), static_cast<std::int64_t>(end), take);
    }
    return;
  }
  cells.forDirected(static_cast<std::int64_t>(begin), static_cast<std::int64_t>(end),
                    [&](std::int32_t i, std::int32_t j, double, double, double, double) {
                      for (int sa = -nmax[0]; sa <= nmax[0]; ++sa) {
                        for (int sb = -nmax[1]; sb <= nmax[1]; ++sb) {
                          for (int sc = -nmax[2]; sc <= nmax[2]; ++sc) {
                            double vx = 0;
                            double vy = 0;
                            double vz = 0;
                            image_vector(pos, static_cast<std::size_t>(i),
                                         static_cast<std::size_t>(j), sa, sb, sc, box, vx, vy,
                                         vz);
                            accept(i, j, sa, sb, sc, vx, vy, vz);
                          }
                        }
                      }
                    });
  for (std::size_t i = begin; i < end; ++i) {
    for (int sa = -nmax[0]; sa <= nmax[0]; ++sa) {
      for (int sb = -nmax[1]; sb <= nmax[1]; ++sb) {
        for (int sc = -nmax[2]; sc <= nmax[2]; ++sc) {
          if (sa == 0 && sb == 0 && sc == 0) {
            continue;
          }
          double vx = 0;
          double vy = 0;
          double vz = 0;
          image_vector(pos, i, i, sa, sb, sc, box, vx, vy, vz);
          accept(static_cast<std::int32_t>(i), static_cast<std::int32_t>(i), sa, sb, sc, vx,
                 vy, vz);
        }
      }
    }
  }
}

void push_rec(std::vector<Rec> &out, const Query &query, std::int32_t first,
              std::int32_t second, int sa, int sb, int sc, double vx, double vy,
              double vz, double dist, std::uint8_t mask) {
  Rec rec;
  rec.first = first;
  rec.second = second;
  if (query.fields.shifts) {
    rec.shift[0] = sa;
    rec.shift[1] = sb;
    rec.shift[2] = sc;
  }
  if (query.fields.vectors) {
    rec.vector[0] = vx;
    rec.vector[1] = vy;
    rec.vector[2] = vz;
  }
  if (query.fields.distances) {
    rec.distance = dist;
  }
  rec.mask = mask;
  out.push_back(rec);
}

void merge_clusters(std::vector<Rec> &rows) {
  if (rows.empty()) {
    return;
  }
  std::sort(rows.begin(), rows.end(), [](const Rec &a, const Rec &b) {
    if (a.first != b.first) {
      return a.first < b.first;
    }
    if (a.second != b.second) {
      return a.second < b.second;
    }
    if (a.shift[0] != b.shift[0]) {
      return a.shift[0] < b.shift[0];
    }
    if (a.shift[1] != b.shift[1]) {
      return a.shift[1] < b.shift[1];
    }
    return a.shift[2] < b.shift[2];
  });
  std::vector<Rec> merged;
  merged.reserve(rows.size());
  for (const Rec &rec : rows) {
    if (!merged.empty()) {
      Rec &last = merged.back();
      if (last.first == rec.first && last.second == rec.second &&
          last.shift[0] == rec.shift[0] && last.shift[1] == rec.shift[1] &&
          last.shift[2] == rec.shift[2]) {
        const auto lowest = [](std::uint8_t bits) {
          return static_cast<unsigned>(bits & -static_cast<int>(bits));
        };
        if (lowest(rec.mask) < lowest(last.mask)) {
          last.vector[0] = rec.vector[0];
          last.vector[1] = rec.vector[1];
          last.vector[2] = rec.vector[2];
          last.distance = rec.distance;
        }
        last.mask = static_cast<std::uint8_t>(last.mask | rec.mask);
        continue;
      }
    }
    merged.push_back(rec);
  }
  rows.swap(merged);
}

void collect(const Neighbours &cells, const double *pos, std::size_t n, const double *box,
             const Query &query, const CutView &cut, const int nmax[3], bool one_image,
             std::size_t begin, std::size_t end, std::vector<Rec> &out) {
  if (clustered(query)) {
    const std::size_t c0 = begin & ~std::size_t{3};
    const std::size_t c1 = std::min(n, (end + 3) & ~std::size_t{3});
    Query full = query;
    full.shape = Pairs{Shell::Full};
    std::vector<Rec> raw;
    each_cell(cells, pos, box, full, cut, nmax, one_image, c0, c1,
              [&](std::int32_t i, std::int32_t j, int sa, int sb, int sc, double vx, double vy,
                  double vz, double dist) {
                if (static_cast<std::size_t>(i) < begin || static_cast<std::size_t>(i) >= end) {
                  return;
                }
                const auto lane = static_cast<unsigned>(i & 3);
                push_rec(raw, query, i & ~std::int32_t{3}, j, sa, sb, sc, vx, vy, vz, dist,
                         static_cast<std::uint8_t>(1u << lane));
              });
    merge_clusters(raw);
    out.insert(out.end(), raw.begin(), raw.end());
    return;
  }
  each_cell(cells, pos, box, query, cut, nmax, one_image, begin, end,
            [&](std::int32_t i, std::int32_t j, int sa, int sb, int sc, double vx, double vy,
                double vz, double dist) {
              push_rec(out, query, i, j, sa, sb, sc, vx, vy, vz, dist, 0xff);
            });
}

void store_records(Table &table, const Query &query, const std::vector<Rec> &rows) {
  table.first.resize(rows.size());
  table.second.resize(rows.size());
  table.mask.resize(rows.size());
  table.shift.clear();
  table.vector.clear();
  table.distance.clear();
  if (query.fields.shifts) {
    table.shift.resize(rows.size() * 3);
  }
  if (query.fields.vectors) {
    table.vector.resize(rows.size() * 3);
  }
  if (query.fields.distances) {
    table.distance.resize(rows.size());
  }
  for (std::size_t p = 0; p < rows.size(); ++p) {
    const Rec &rec = rows[p];
    table.first[p] = rec.first;
    table.second[p] = rec.second;
    table.mask[p] = rec.mask;
    if (query.fields.shifts) {
      table.shift[3 * p] = rec.shift[0];
      table.shift[3 * p + 1] = rec.shift[1];
      table.shift[3 * p + 2] = rec.shift[2];
    }
    if (query.fields.vectors) {
      table.vector[3 * p] = rec.vector[0];
      table.vector[3 * p + 1] = rec.vector[1];
      table.vector[3 * p + 2] = rec.vector[2];
    }
    if (query.fields.distances) {
      table.distance[p] = rec.distance;
    }
  }
  table.stored = true;
  table.device = 0;
}

struct Columns {
  std::vector<std::int32_t> first;
  std::vector<std::int32_t> second;
  std::vector<std::int32_t> shift;
  std::vector<double> vector;
  std::vector<double> distance;
  std::vector<std::uint8_t> mask;
};

// The slot count is a guess. Value-initialising the spare tail writes
// gigabytes that the next loop overwrites. libstdc++ lets the size
// advance without that write; other libraries fall back to resize.
template <typename T>
void grow_uninitialized(std::vector<T> &values, std::size_t n) {
#if defined(__GLIBCXX__)
  struct Tail : std::vector<T> {
    void extend(std::size_t count) {
      if (count > this->capacity()) {
        this->reserve(count);
      }
      this->_M_impl._M_finish = this->_M_impl._M_start + count;
    }
  };
  static_assert(sizeof(Tail) == sizeof(std::vector<T>));
  reinterpret_cast<Tail &>(values).extend(n);
#else
  values.resize(n);
#endif
}

std::size_t pair_guess(std::size_t natoms, std::size_t nall, const double *box, double search,
                       bool full, int stencil) {
  const double vol =
      std::abs(box[0] * (box[4] * box[8] - box[5] * box[7]) -
               box[1] * (box[3] * box[8] - box[5] * box[6]) +
               box[2] * (box[3] * box[7] - box[4] * box[6]));
  double per = 48;
  if (vol > 0.0 && search > 0.0 && nall > 0) {
    const double sphere = 4.1887902047863905 * search * search * search;
    per = static_cast<double>(nall) / vol * sphere;
    if (!full) {
      per *= 0.5;
    }
    if (stencil > 1) {
      per *= static_cast<double>(stencil);
    }
    per = per * 1.5 + 4.0;
  }
  if (per < 4.0) {
    per = 4.0;
  }
  if (per > 256.0) {
    per = 256.0;
  }
  return static_cast<std::size_t>(per * static_cast<double>(natoms) + 1.0);
}

// A parallel region costs more than the search itself below this size.
// 64 atoms on 8 threads was several times the serial search.
int teamSize(std::size_t n) {
  int nt = threadsRequested();
  if (nt < 1) {
    nt = 1;
  }
  if (n < 8192 || nt == 1) {
    return 1;
  }
  if (static_cast<std::size_t>(nt) > n) {
    return static_cast<int>(n);
  }
  return nt;
}

struct alignas(64) TeamCursor {
  std::size_t count = 0;
};

void compact_i32(std::vector<std::int32_t> &col, int width, const TeamCursor *cursors,
                 const std::size_t *base, int nt) {
  std::size_t dst = 0;
  for (int t = 0; t < nt; ++t) {
    const std::size_t n = cursors[t].count * static_cast<std::size_t>(width);
    const std::size_t src = base[t] * static_cast<std::size_t>(width);
    if (n != 0 && dst != src) {
      std::memmove(col.data() + dst, col.data() + src, n * sizeof(std::int32_t));
    }
    dst += n;
  }
  col.resize(dst);
}

void compact_f64(std::vector<double> &col, int width, const TeamCursor *cursors,
                 const std::size_t *base, int nt) {
  std::size_t dst = 0;
  for (int t = 0; t < nt; ++t) {
    const std::size_t n = cursors[t].count * static_cast<std::size_t>(width);
    const std::size_t src = base[t] * static_cast<std::size_t>(width);
    if (n != 0 && dst != src) {
      std::memmove(col.data() + dst, col.data() + src, n * sizeof(double));
    }
    dst += n;
  }
  col.resize(dst);
}

void compact_u8(std::vector<std::uint8_t> &col, const TeamCursor *cursors, const std::size_t *base,
                int nt) {
  std::size_t dst = 0;
  for (int t = 0; t < nt; ++t) {
    const std::size_t n = cursors[t].count;
    const std::size_t src = base[t];
    if (n != 0 && dst != src) {
      std::memmove(col.data() + dst, col.data() + src, n);
    }
    dst += n;
  }
  col.resize(dst);
}

void move_columns(Table &table, Columns &cols) {
  table.first = std::move(cols.first);
  table.second = std::move(cols.second);
  table.shift = std::move(cols.shift);
  table.vector = std::move(cols.vector);
  table.distance = std::move(cols.distance);
  table.mask = std::move(cols.mask);
  table.stored = true;
  table.device = 0;
}

} // namespace

void Builder::update(const double *positions, std::size_t n, const double *box9,
                     const Query &query, PairCutoffs cuts) {
  check_geometry(positions, n, box9, query);
  std::vector<std::int32_t> types;
  std::vector<double> matrix;
  if (cuts.ntypes != 0) {
    if (cuts.ntypes < 0 || cuts.cutoff == nullptr || (n > 0 && cuts.types == nullptr)) {
      throw Error(Code::Cutoff, "a pair-type cutoff needs one type per atom and a square matrix");
    }
    if (n > 0) {
      types.assign(cuts.types, cuts.types + n);
    }
    const auto n2 = static_cast<std::size_t>(cuts.ntypes) * static_cast<std::size_t>(cuts.ntypes);
    matrix.assign(cuts.cutoff, cuts.cutoff + n2);
  }
  PairCutoffs local;
  local.types = types.data();
  local.cutoff = matrix.data();
  local.ntypes = cuts.ntypes;
  reused_ = false;
  if (have_ && n_ == n && ntypes_ == cuts.ntypes && types_ == types && matrix_ == matrix &&
      query_.cutoff == query.cutoff && query_.skin == query.skin &&
      query_.shape == query.shape && query_.cut == query.cut &&
      query_.periodic == query.periodic && query_.fields.shifts == query.fields.shifts &&
      query_.fields.vectors == query.fields.vectors &&
      query_.fields.distances == query.fields.distances &&
      std::memcmp(box_, box9, sizeof(box_)) == 0) {
    if (withinSkin(ref_.data(), positions, n, query.skin)) {
      reused_ = true;
      return;
    }
  }
  build(table_, positions, n, box9, query, local);
  ref_.assign(positions, positions + 3 * n);
  std::memcpy(box_, box9, sizeof(box_));
  query_ = query;
  n_ = n;
  ntypes_ = cuts.ntypes;
  types_ = std::move(types);
  matrix_ = std::move(matrix);
  have_ = true;
  ++builds_;
}

Walk::Walk(const double *positions, std::size_t n, const double *box9, const Query &query,
           PairCutoffs cuts)
    : pos_(positions), n_(n), query_(query) {
  check_geometry(positions, n, box9, query);
  OwnedCut owned = prepare_cut(n, query, cuts);
  cut_typed_ = owned.typed;
  cut_inclusive_ = owned.inclusive;
  cut_ntypes_ = owned.ntypes;
  cut_search_ = owned.search;
  cut_types_ = std::move(owned.types);
  cut_limit_ = std::move(owned.limit2);
  std::memcpy(box_, box9, sizeof(box_));
  shift_bounds(box_, query_.periodic, cut_search_, nmax_);
  one_image_ = one_minimum_image(box_, query_, cut_search_);
  cells_ = prepare_cells(positions, n, box9, query_, cut_search_);
}

std::size_t Walk::pull(Buffers dest, bool &done) {
  const std::size_t cap = dest.first.size();
  if (cap > 0 && dest.second.size() != cap) {
    throw Error(Code::Short, "first and second buffers have different lengths");
  }
  if (cap > 0 && query_.fields.shifts && dest.shift.size() < cap * 3) {
    throw Error(Code::Short, "shift buffer needs 3 entries per pair");
  }
  if (cap > 0 && query_.fields.vectors && dest.vector.size() < cap * 3) {
    throw Error(Code::Short, "vector buffer needs 3 entries per pair");
  }
  if (cap > 0 && query_.fields.distances && dest.distance.size() < cap) {
    throw Error(Code::Short, "distance buffer is shorter than the pair buffer");
  }
  if (cap > 0 && clustered(query_) && dest.mask.size() < cap) {
    throw Error(Code::Short, "a cluster list needs one mask byte per pair");
  }
  if (finished_ || cap == 0) {
    done = finished_;
    return 0;
  }
  auto stash = [&](const std::vector<Rec> &rows) {
    for (const Rec &rec : rows) {
      hold_first_.push_back(rec.first);
      hold_second_.push_back(rec.second);
      hold_shift_.push_back(rec.shift[0]);
      hold_shift_.push_back(rec.shift[1]);
      hold_shift_.push_back(rec.shift[2]);
      hold_vector_.push_back(rec.vector[0]);
      hold_vector_.push_back(rec.vector[1]);
      hold_vector_.push_back(rec.vector[2]);
      hold_distance_.push_back(rec.distance);
      hold_mask_.push_back(rec.mask);
    }
  };
  CutView cut;
  cut.typed = cut_typed_;
  cut.inclusive = cut_inclusive_;
  cut.ntypes = cut_ntypes_;
  cut.types = cut_types_.data();
  cut.limit2 = cut_limit_.data();
  std::size_t written = 0;
  while (written < cap) {
    if (hold_at_ >= hold_first_.size()) {
      hold_first_.clear();
      hold_second_.clear();
      hold_shift_.clear();
      hold_vector_.clear();
      hold_distance_.clear();
      hold_mask_.clear();
      hold_at_ = 0;
      if (atom_ >= n_) {
        finished_ = true;
        break;
      }
      const std::size_t end = std::min(
          n_, atom_ + (clustered(query_) ? std::size_t{4} : std::size_t{1}));
      std::vector<Rec> rows;
      collect(cells_, pos_, n_, box_, query_, cut, nmax_, one_image_, atom_, end, rows);
      atom_ = end;
      stash(rows);
      if (hold_first_.empty()) {
        continue;
      }
    }
    dest.first[written] = hold_first_[hold_at_];
    dest.second[written] = hold_second_[hold_at_];
    if (dest.shift.size() >= (written + 1) * 3) {
      dest.shift[3 * written] = hold_shift_[3 * hold_at_];
      dest.shift[3 * written + 1] = hold_shift_[3 * hold_at_ + 1];
      dest.shift[3 * written + 2] = hold_shift_[3 * hold_at_ + 2];
    }
    if (dest.vector.size() >= (written + 1) * 3) {
      dest.vector[3 * written] = hold_vector_[3 * hold_at_];
      dest.vector[3 * written + 1] = hold_vector_[3 * hold_at_ + 1];
      dest.vector[3 * written + 2] = hold_vector_[3 * hold_at_ + 2];
    }
    if (written < dest.distance.size()) {
      dest.distance[written] = hold_distance_[hold_at_];
    }
    if (written < dest.mask.size()) {
      dest.mask[written] = hold_mask_[hold_at_];
    }
    ++hold_at_;
    ++written;
  }
  done = finished_ && hold_at_ >= hold_first_.size();
  return written;
}

void build(Table &table, const double *positions, std::size_t n, const double *box9,
           const Query &query, PairCutoffs cuts) {
  check_geometry(positions, n, box9, query);
  const OwnedCut owned = prepare_cut(n, query, cuts);
  const CutView cut = owned.view();
  table = Table{};
  if (n == 0 || !(owned.search > 0.0)) {
    table.stored = true;
    return;
  }
  int nmax[3];
  const double search = owned.search;
  shift_bounds(box9, query.periodic, search, nmax);
  const bool one_image = one_minimum_image(box9, query, search);
  // A handful of atoms is cheaper as a direct pair loop. The cell list
  // spends its time building bins, and a sheared cell of this size has
  // only a few bins, so the stencil visits every pair anyway.
  if (!clustered(query) && n < 512) {
    Columns cols;
    const bool full = full_shell(query);
    const int stencil =
        one_image ? 1 : (2 * nmax[0] + 1) * (2 * nmax[1] + 1) * (2 * nmax[2] + 1);
    const std::size_t slots = pair_guess(n, n, box9, search, full, stencil);
    cols.first.reserve(slots);
    cols.second.reserve(slots);
    cols.mask.reserve(slots);
    if (query.fields.shifts) {
      cols.shift.reserve(slots * 3);
    }
    if (query.fields.vectors) {
      cols.vector.reserve(slots * 3);
    }
    if (query.fields.distances) {
      cols.distance.reserve(slots);
    }
    for (std::size_t i = 0; i < n; ++i) {
      for (std::size_t j = 0; j < n; ++j) {
        for (int sa = -nmax[0]; sa <= nmax[0]; ++sa) {
          for (int sb = -nmax[1]; sb <= nmax[1]; ++sb) {
            for (int sc = -nmax[2]; sc <= nmax[2]; ++sc) {
              const auto ii = static_cast<std::int32_t>(i);
              const auto jj = static_cast<std::int32_t>(j);
              if (!full && !keep_half(ii, jj, sa, sb, sc)) {
                continue;
              }
              double vx = 0;
              double vy = 0;
              double vz = 0;
              image_vector(positions, i, j, sa, sb, sc, box9, vx, vy, vz);
              const double r2 = vx * vx + vy * vy + vz * vz;
              if (!keep_r2(cut, query, ii, jj, r2)) {
                continue;
              }
              cols.first.push_back(ii);
              cols.second.push_back(jj);
              cols.mask.push_back(0xff);
              if (query.fields.shifts) {
                cols.shift.push_back(sa);
                cols.shift.push_back(sb);
                cols.shift.push_back(sc);
              }
              if (query.fields.vectors) {
                cols.vector.push_back(vx);
                cols.vector.push_back(vy);
                cols.vector.push_back(vz);
              }
              if (query.fields.distances) {
                cols.distance.push_back(std::sqrt(r2));
              }
            }
          }
        }
      }
    }
    move_columns(table, cols);
    return;
  }
  const Neighbours cells = prepare_cells(positions, n, box9, query, search);
  const int nt = teamSize(n);
  if (!clustered(query)) {
    std::vector<Columns> parts(static_cast<std::size_t>(std::max(nt, 1)));
    const bool full = full_shell(query);
    auto fill = [&](Columns &cols, std::size_t begin, std::size_t end) {
      const int stencil = one_image ? 1
                                   : (2 * nmax[0] + 1) * (2 * nmax[1] + 1) * (2 * nmax[2] + 1);
      const std::size_t slots = pair_guess(end - begin, n, box9, search, full, stencil);
      grow_uninitialized(cols.first, slots);
      grow_uninitialized(cols.second, slots);
      grow_uninitialized(cols.mask, slots);
      if (query.fields.shifts) {
        grow_uninitialized(cols.shift, slots * 3);
      }
      if (query.fields.vectors) {
        grow_uninitialized(cols.vector, slots * 3);
      }
      if (query.fields.distances) {
        grow_uninitialized(cols.distance, slots);
      }
      auto *first = cols.first.data();
      auto *second = cols.second.data();
      auto *mask = cols.mask.data();
      auto *shift = cols.shift.data();
      auto *vector = cols.vector.data();
      auto *distance = cols.distance.data();
      std::size_t count = 0;
      Columns extra;
      each_cell(cells, positions, box9, query, cut, nmax, one_image, begin, end,
                [&](std::int32_t i, std::int32_t j, int sa, int sb, int sc, double vx,
                    double vy, double vz, double dist) {
                  if (count == slots) {
                    extra.first.push_back(i);
                    extra.second.push_back(j);
                    extra.mask.push_back(0xff);
                    if (query.fields.shifts) {
                      extra.shift.push_back(sa);
                      extra.shift.push_back(sb);
                      extra.shift.push_back(sc);
                    }
                    if (query.fields.vectors) {
                      extra.vector.push_back(vx);
                      extra.vector.push_back(vy);
                      extra.vector.push_back(vz);
                    }
                    if (query.fields.distances) {
                      extra.distance.push_back(dist);
                    }
                    return;
                  }
                  first[count] = i;
                  second[count] = j;
                  mask[count] = 0xff;
                  if (query.fields.shifts) {
                    shift[3 * count] = sa;
                    shift[3 * count + 1] = sb;
                    shift[3 * count + 2] = sc;
                  }
                  if (query.fields.vectors) {
                    vector[3 * count] = vx;
                    vector[3 * count + 1] = vy;
                    vector[3 * count + 2] = vz;
                  }
                  if (query.fields.distances) {
                    distance[count] = dist;
                  }
                  ++count;
                });
      if (!extra.first.empty()) {
        cols.first.resize(count);
        cols.second.resize(count);
        cols.mask.resize(count);
        if (query.fields.shifts) {
          cols.shift.resize(count * 3);
        }
        if (query.fields.vectors) {
          cols.vector.resize(count * 3);
        }
        if (query.fields.distances) {
          cols.distance.resize(count);
        }
        cols.first.insert(cols.first.end(), extra.first.begin(), extra.first.end());
        cols.second.insert(cols.second.end(), extra.second.begin(), extra.second.end());
        cols.mask.insert(cols.mask.end(), extra.mask.begin(), extra.mask.end());
        if (query.fields.shifts) {
          cols.shift.insert(cols.shift.end(), extra.shift.begin(), extra.shift.end());
        }
        if (query.fields.vectors) {
          cols.vector.insert(cols.vector.end(), extra.vector.begin(), extra.vector.end());
        }
        if (query.fields.distances) {
          cols.distance.insert(cols.distance.end(), extra.distance.begin(), extra.distance.end());
        }
        return;
      }
      cols.first.resize(count);
      cols.second.resize(count);
      cols.mask.resize(count);
      if (query.fields.shifts) {
        cols.shift.resize(count * 3);
      }
      if (query.fields.vectors) {
        cols.vector.resize(count * 3);
      }
      if (query.fields.distances) {
        cols.distance.resize(count);
      }
    };
    if (nt <= 1) {
      fill(parts[0], 0, n);
      move_columns(table, parts[0]);
      return;
    }
    const int stencil = one_image ? 1 : (2 * nmax[0] + 1) * (2 * nmax[1] + 1) * (2 * nmax[2] + 1);
    const std::size_t slice = pair_guess(n, n, box9, search, full, stencil) /
                                  static_cast<std::size_t>(nt) +
                              8;
    const std::size_t slots = slice * static_cast<std::size_t>(nt);
    Columns cols;
    grow_uninitialized(cols.first, slots);
    grow_uninitialized(cols.second, slots);
    grow_uninitialized(cols.mask, slots);
    if (query.fields.shifts) {
      grow_uninitialized(cols.shift, slots * 3);
    }
    if (query.fields.vectors) {
      grow_uninitialized(cols.vector, slots * 3);
    }
    if (query.fields.distances) {
      grow_uninitialized(cols.distance, slots);
    }
    std::vector<std::size_t> base(static_cast<std::size_t>(nt));
    for (int t = 0; t < nt; ++t) {
      base[static_cast<std::size_t>(t)] = slice * static_cast<std::size_t>(t);
    }
    std::vector<TeamCursor> cursors(static_cast<std::size_t>(nt));
    std::vector<Columns> extra(static_cast<std::size_t>(nt));
    const double inv0 = box9[0] > 0.0 ? 1.0 / box9[0] : 0.0;
    const double inv1 = box9[4] > 0.0 ? 1.0 / box9[4] : 0.0;
    const double inv2 = box9[8] > 0.0 ? 1.0 / box9[8] : 0.0;
    cells.forCellTeam(
        !full, nt,
        [&](int thread, std::int32_t i, std::int32_t j, double dx, double dy, double dz,
            double r2) {
          if (!keep_r2(cut, query, i, j, r2)) {
            return;
          }
          const auto ui = static_cast<std::size_t>(i);
          const auto uj = static_cast<std::size_t>(j);
          const int sa = axis_shift(positions[3 * ui] - positions[3 * uj], inv0, query.periodic[0]);
          const int sb =
              axis_shift(positions[3 * ui + 1] - positions[3 * uj + 1], inv1, query.periodic[1]);
          const int sc =
              axis_shift(positions[3 * ui + 2] - positions[3 * uj + 2], inv2, query.periodic[2]);
          const double dist = std::sqrt(r2);
          const auto t = static_cast<std::size_t>(thread);
          auto &cursor = cursors[t];
          if (cursor.count == slice) {
            extra[t].first.push_back(i);
            extra[t].second.push_back(j);
            extra[t].mask.push_back(0xff);
            if (query.fields.shifts) {
              extra[t].shift.push_back(sa);
              extra[t].shift.push_back(sb);
              extra[t].shift.push_back(sc);
            }
            if (query.fields.vectors) {
              extra[t].vector.push_back(-dx);
              extra[t].vector.push_back(-dy);
              extra[t].vector.push_back(-dz);
            }
            if (query.fields.distances) {
              extra[t].distance.push_back(dist);
            }
            return;
          }
          const std::size_t p = base[t] + cursor.count;
          cols.first[p] = i;
          cols.second[p] = j;
          cols.mask[p] = 0xff;
          if (query.fields.shifts) {
            cols.shift[3 * p] = sa;
            cols.shift[3 * p + 1] = sb;
            cols.shift[3 * p + 2] = sc;
          }
          if (query.fields.vectors) {
            cols.vector[3 * p] = -dx;
            cols.vector[3 * p + 1] = -dy;
            cols.vector[3 * p + 2] = -dz;
          }
          if (query.fields.distances) {
            cols.distance[p] = dist;
          }
          ++cursor.count;
        });
    compact_i32(cols.first, 1, cursors.data(), base.data(), nt);
    compact_i32(cols.second, 1, cursors.data(), base.data(), nt);
    compact_u8(cols.mask, cursors.data(), base.data(), nt);
    if (query.fields.shifts) {
      compact_i32(cols.shift, 3, cursors.data(), base.data(), nt);
    }
    if (query.fields.vectors) {
      compact_f64(cols.vector, 3, cursors.data(), base.data(), nt);
    }
    if (query.fields.distances) {
      compact_f64(cols.distance, 1, cursors.data(), base.data(), nt);
    }
    for (int t = 0; t < nt; ++t) {
      auto &over = extra[static_cast<std::size_t>(t)];
      if (over.first.empty()) {
        continue;
      }
      cols.first.insert(cols.first.end(), over.first.begin(), over.first.end());
      cols.second.insert(cols.second.end(), over.second.begin(), over.second.end());
      cols.mask.insert(cols.mask.end(), over.mask.begin(), over.mask.end());
      if (query.fields.shifts) {
        cols.shift.insert(cols.shift.end(), over.shift.begin(), over.shift.end());
      }
      if (query.fields.vectors) {
        cols.vector.insert(cols.vector.end(), over.vector.begin(), over.vector.end());
      }
      if (query.fields.distances) {
        cols.distance.insert(cols.distance.end(), over.distance.begin(), over.distance.end());
      }
    }
    move_columns(table, cols);
    return;
  }
  std::vector<std::vector<Rec>> parts(static_cast<std::size_t>(std::max(nt, 1)));
  if (nt <= 1) {
    collect(cells, positions, n, box9, query, cut, nmax, one_image, 0, n, parts[0]);
  } else {
#if defined(_OPENMP)
#pragma omp parallel num_threads(nt)
#endif
    {
#if defined(_OPENMP)
      const int t = omp_get_thread_num();
#else
      const int t = 0;
#endif
      const auto range = partition(static_cast<std::int64_t>(n), t, nt);
      collect(cells, positions, n, box9, query, cut, nmax, one_image,
              static_cast<std::size_t>(range.begin), static_cast<std::size_t>(range.end),
              parts[static_cast<std::size_t>(t)]);
    }
  }
  std::vector<Rec> rows;
  for (const auto &part : parts) {
    rows.insert(rows.end(), part.begin(), part.end());
  }
  if (nt > 1) {
    merge_clusters(rows);
  }
  store_records(table, query, rows);
}

void buildByAtom(Table &table, const double *positions, std::size_t n, const double *box9,
                 const Query &query, PairCutoffs cuts) {
  check_geometry(positions, n, box9, query);
  const OwnedCut owned = prepare_cut(n, query, cuts);
  const CutView cut = owned.view();
  table = Table{};
  if (n == 0 || !(owned.search > 0.0) || n < 512 || clustered(query)) {
    build(table, positions, n, box9, query, cuts);
    return;
  }
  int nmax[3];
  const double search = owned.search;
  shift_bounds(box9, query.periodic, search, nmax);
  const bool one_image = one_minimum_image(box9, query, search);
  const Neighbours cells = prepare_cells(positions, n, box9, query, search);
  std::vector<Rec> rows;
  rows.reserve(n * 8);
  for (std::size_t i = 0; i < n; ++i) {
    collect(cells, positions, n, box9, query, cut, nmax, one_image, i, i + 1, rows);
  }
  store_records(table, query, rows);
}

void build(Batch &batch, const Systems &systems, const Query &query, PairCutoffs cuts) {
  if (systems.systems > 0 &&
      (systems.positions == nullptr || systems.count == nullptr || systems.box9 == nullptr)) {
    throw Error(Code::Null, "a batch needs positions, a count per system and a box per system");
  }
  Batch out;
  std::size_t origin = 0;
  for (std::size_t s = 0; s < systems.systems; ++s) {
    const auto count = systems.count[s];
    if (count < 0) {
      throw Error(Code::Shape, "a system count must be >= 0");
    }
    const auto n = static_cast<std::size_t>(count);
    Table table;
    PairCutoffs slice = cuts;
    if (cuts.ntypes != 0 && cuts.types != nullptr) {
      slice.types = cuts.types + origin;
    }
    build(table, systems.positions + 3 * origin, n, systems.box9 + 9 * s, query, slice);
    const auto base = static_cast<std::int32_t>(origin);
    for (std::size_t p = 0; p < table.size(); ++p) {
      out.pairs.first.push_back(table.first[p] + base);
      out.pairs.second.push_back(table.second[p] + base);
      out.pairs.mask.push_back(table.mask[p]);
      out.system.push_back(static_cast<std::int32_t>(s));
    }
    if (query.fields.shifts) {
      out.pairs.shift.insert(out.pairs.shift.end(), table.shift.begin(), table.shift.end());
    }
    if (query.fields.vectors) {
      out.pairs.vector.insert(out.pairs.vector.end(), table.vector.begin(), table.vector.end());
    }
    if (query.fields.distances) {
      out.pairs.distance.insert(out.pairs.distance.end(), table.distance.begin(),
                                table.distance.end());
    }
    origin += n;
  }
  out.pairs.stored = true;
  out.pairs.device = 0;
  batch = std::move(out);
}

void each_raw(const double *positions, std::size_t n, const double *box9, const Query &query,
              void (*fn)(void *, std::int32_t, std::int32_t, int, int, int, double, double, double,
                         double),
              void *user, PairCutoffs cuts) {
  check_geometry(positions, n, box9, query);
  const OwnedCut owned = prepare_cut(n, query, cuts);
  const CutView cut = owned.view();
  if (fn == nullptr) {
    throw Error(Code::Null, "neighbour callback is null");
  }
  if (n == 0 || !(owned.search > 0.0)) {
    return;
  }
  int nmax[3];
  shift_bounds(box9, query.periodic, owned.search, nmax);
  const bool one_image = one_minimum_image(box9, query, owned.search);
  const Neighbours cells = prepare_cells(positions, n, box9, query, owned.search);
  auto emit = [&](std::int32_t i, std::int32_t j, int sa, int sb, int sc, double vx, double vy,
                  double vz, double dist) { fn(user, i, j, sa, sb, sc, vx, vy, vz, dist); };
  if (!clustered(query)) {
    each_cell(cells, positions, box9, query, cut, nmax, one_image, 0, n, emit);
    return;
  }
  Query geometry = query;
  geometry.fields.shifts = true;
  geometry.fields.vectors = true;
  geometry.fields.distances = true;
  for (std::size_t atom = 0; atom < n;) {
    const std::size_t end = std::min(n, atom + 4);
    std::vector<Rec> rows;
    collect(cells, positions, n, box9, geometry, cut, nmax, one_image, atom, end, rows);
    for (const Rec &rec : rows) {
      fn(user, rec.first, rec.second, rec.shift[0], rec.shift[1], rec.shift[2], rec.vector[0],
         rec.vector[1], rec.vector[2], rec.distance);
    }
    atom = end;
  }
}

void Sweep::update(const double *positions, std::size_t n, const double *box9, const Query &query,
                   PairCutoffs cuts) {
  if (std::holds_alternative<Cluster4>(query.shape)) {
    throw Error(Code::Shape, "a sweep visits a pair shell; a cluster uses build");
  }
  check_geometry(positions, n, box9, query);
  OwnedCut owned = prepare_cut(n, query, cuts);
  const double search = owned.search;
  const bool sheared = box9[1] != 0.0 || box9[2] != 0.0 || box9[3] != 0.0 || box9[5] != 0.0 ||
                       box9[6] != 0.0 || box9[7] != 0.0;
  one_image_ = one_minimum_image(box9, query, search);
  Spec spec;
  spec.cutoff = query.cutoff;
  spec.skin = query.skin;
  spec.image = sheared ? Image::General : Image::Round;
  spec.periodic = query.periodic;
  spec.mode = (query.skin > 0.0 && one_image_) ? Mode::Stored : Mode::Stream;
  cells_.update(positions, n, box9, spec);
  reused_ = cells_.reused();
  pos_ = positions;
  n_ = n;
  query_ = query;
  query_.shape = Pairs{};
  std::memcpy(box_, box9, sizeof(box_));
  search_ = search;
  shift_bounds(box_, query_.periodic, search, nmax_);
  cut_typed_ = owned.typed;
  cut_inclusive_ = owned.inclusive;
  cut_ntypes_ = owned.ntypes;
  cut_types_ = std::move(owned.types);
  cut_limit_ = std::move(owned.limit2);
  have_ = true;
}

void Sweep::each_raw(Shell shell, int threads, GeomFn fn, void *user) const {
  if (!have_) {
    throw Error(Code::Shape, "a sweep has not been updated");
  }
  if (fn == nullptr) {
    throw Error(Code::Null, "neighbour callback is null");
  }
  if (shell != Shell::Half && shell != Shell::Full) {
    throw Error(Code::Shape, "a pair list shell must be half or full");
  }
  if (threads < 1) {
    throw Error(Code::Shape, "thread count must be >= 1");
  }
  if (n_ == 0 || !(search_ > 0.0)) {
    return;
  }
  Query walked = query_;
  walked.shape = Pairs{shell};
  CutView cut;
  cut.typed = cut_typed_;
  cut.inclusive = cut_inclusive_;
  cut.ntypes = cut_ntypes_;
  cut.types = cut_types_.data();
  cut.limit2 = cut_limit_.data();
  const int nt = std::min(threads, static_cast<int>(std::max<std::size_t>(n_, 1)));
  // A wide orthorhombic cell has one image. The stored index is the list
  // that stays valid across the skin, so a reused sweep walks that index.
  // A fresh stream uses forCellTeam. A narrow or sheared cell keeps the
  // shift stencil on the calling thread.
  if (one_image_) {
    const double inv0 = box_[0] > 0.0 ? 1.0 / box_[0] : 0.0;
    const double inv1 = box_[4] > 0.0 ? 1.0 / box_[4] : 0.0;
    const double inv2 = box_[8] > 0.0 ? 1.0 / box_[8] : 0.0;
    auto publish = [&](int thread, std::int32_t i, std::int32_t j, double dx, double dy, double dz,
                       double r2) {
      if (!keep_r2(cut, walked, i, j, r2)) {
        return;
      }
      const auto ui = static_cast<std::size_t>(i);
      const auto uj = static_cast<std::size_t>(j);
      const int sa = axis_shift(pos_[3 * ui] - pos_[3 * uj], inv0, query_.periodic[0]);
      const int sb = axis_shift(pos_[3 * ui + 1] - pos_[3 * uj + 1], inv1, query_.periodic[1]);
      const int sc = axis_shift(pos_[3 * ui + 2] - pos_[3 * uj + 2], inv2, query_.periodic[2]);
      fn(user, thread, i, j, sa, sb, sc, -dx, -dy, -dz, std::sqrt(r2));
    };
    if (!cells_.streaming()) {
#if defined(_OPENMP)
#pragma omp parallel num_threads(nt)
#endif
      {
#if defined(_OPENMP)
        const int thread = omp_get_thread_num();
#else
        const int thread = 0;
#endif
        const auto range = partition(static_cast<std::int64_t>(n_), thread, nt);
        if (shell == Shell::Half) {
          cells_.forHalf(range.begin, range.end,
                         [&](std::int32_t i, std::int32_t j, double dx, double dy, double dz,
                             double r2) { publish(thread, i, j, dx, dy, dz, r2); });
        } else {
          cells_.forDirected(range.begin, range.end,
                             [&](std::int32_t i, std::int32_t j, double dx, double dy, double dz,
                                 double r2) { publish(thread, i, j, dx, dy, dz, r2); });
        }
      }
      return;
    }
    cells_.forCellTeam(shell == Shell::Half, nt,
                       [&](int thread, std::int32_t i, std::int32_t j, double dx, double dy,
                           double dz, double r2) { publish(thread, i, j, dx, dy, dz, r2); });
    return;
  }
  each_cell(cells_, pos_, box_, walked, cut, nmax_, one_image_, 0, n_,
            [&](std::int32_t i, std::int32_t j, int sa, int sb, int sc, double vx, double vy,
                double vz, double dist) { fn(user, 0, i, j, sa, sb, sc, vx, vy, vz, dist); });
}

void Cells::update(const double *positions, std::size_t n, const double *box9, double cutoff,
                   double skin, Image image, Mode mode, const std::array<bool, 3> &periodic,
                   PairCutoffs cuts) {
  if ((n > 0 && positions == nullptr) || box9 == nullptr) {
    throw Error(Code::Null, "positions or box is null");
  }
  if (!std::isfinite(cutoff) || cutoff < 0.0 || !std::isfinite(skin) || skin < 0.0) {
    throw Error(Code::Cutoff, "cutoff and skin must be finite and >= 0");
  }
  typed_ = false;
  ntypes_ = 0;
  types_.clear();
  limit_.clear();
  double widest = cutoff;
  if (cuts.ntypes != 0) {
    if (cuts.ntypes < 0 || cuts.cutoff == nullptr || (n > 0 && cuts.types == nullptr)) {
      throw Error(Code::Cutoff, "a pair-type cutoff needs one type per atom and a square matrix");
    }
    ntypes_ = cuts.ntypes;
    const auto n2 = static_cast<std::size_t>(ntypes_) * static_cast<std::size_t>(ntypes_);
    if (n > 0) {
      types_.assign(cuts.types, cuts.types + n);
    }
    limit_.assign(n2, 0.0);
    for (std::size_t k = 0; k < n2; ++k) {
      const double c = cuts.cutoff[k];
      if (!std::isfinite(c) || c < 0.0) {
        throw Error(Code::Cutoff, "a pair-type cutoff must be finite and >= 0");
      }
      if (!(c > 0.0)) {
        continue;
      }
      const double reach = c + skin;
      limit_[k] = reach * reach;
      if (c > widest) {
        widest = c;
      }
    }
    for (std::size_t a = 0; a < n; ++a) {
      const auto t = types_[a];
      if (t < 0 || t >= ntypes_) {
        throw Error(Code::Shape, "a pair type is outside the cutoff matrix");
      }
    }
    typed_ = true;
  }
  Spec spec;
  spec.cutoff = widest;
  spec.skin = skin;
  spec.image = image;
  spec.mode = mode;
  spec.periodic = periodic;
  cells_.update(positions, n, box9, spec);
}

void Cells::filter_directed(std::int64_t begin, std::int64_t end, PairFn fn, void *user) const {
  cells_.forDirected(begin, end, [&](std::int32_t i, std::int32_t j, double dx, double dy, double dz,
                                    double r2) {
    if (allow(i, j, r2)) {
      fn(user, i, j, dx, dy, dz, r2);
    }
  });
}

void Cells::filter_half(std::int64_t begin, std::int64_t end, PairFn fn, void *user) const {
  cells_.forHalf(begin, end, [&](std::int32_t i, std::int32_t j, double dx, double dy, double dz,
                                double r2) {
    if (allow(i, j, r2)) {
      fn(user, i, j, dx, dy, dz, r2);
    }
  });
}

void Cells::filter_atom(std::int32_t i, PairFn fn, void *user) const {
  cells_.forAtom(i, [&](std::int32_t a, std::int32_t j, double dx, double dy, double dz, double r2) {
    if (allow(a, j, r2)) {
      fn(user, a, j, dx, dy, dz, r2);
    }
  });
}

void Cells::filter_team(bool half, int nthreads, TeamFn fn, void *user) const {
  cells_.forCellTeam(half, nthreads,
                     [&](int thread, std::int32_t i, std::int32_t j, double dx, double dy, double dz,
                         double r2) {
                       if (allow(i, j, r2)) {
                         fn(user, thread, i, j, dx, dy, dz, r2);
                       }
                     });
}

} // namespace rgpot::nlist

namespace {

struct ParsedRequest {
  rgpot::nlist::Query query;
  rgpot::nlist::PairCutoffs cuts;
};

ParsedRequest query_from_c(const rgpot_nlist_request *request) {
  if (request == nullptr) {
    throw rgpot::nlist::Error(rgpot::nlist::Code::Null, "request is null");
  }
  const auto minimum =
      offsetof(rgpot_nlist_request, want_distances) + sizeof(std::int32_t);
  if (request->bytes < minimum || request->abi != RGPOT_NLIST_ABI) {
    throw rgpot::nlist::Error(rgpot::nlist::Code::Abi,
                              "request abi or bytes does not match this library");
  }
  rgpot_nlist_request taken{};
  std::memcpy(&taken, request,
              std::min(static_cast<std::size_t>(request->bytes), sizeof(taken)));
  if (taken.shape != 0 && taken.shape != 1) {
    throw rgpot::nlist::Error(rgpot::nlist::Code::Shape,
                              "shape must be 0 (pairs) or 1 (cluster of four)");
  }
  if (taken.shell != 0 && taken.shell != 1) {
    throw rgpot::nlist::Error(rgpot::nlist::Code::Shape, "shell must be 0 (half) or 1 (full)");
  }
  if (taken.shape == 1 && taken.shell != 1) {
    throw rgpot::nlist::Error(
        rgpot::nlist::Code::Shape,
        "a cluster list stores every neighbour; set shell to 1 (full)");
  }
  if (taken.cut != 0 && taken.cut != 1) {
    throw rgpot::nlist::Error(rgpot::nlist::Code::Shape,
                              "cut must be 0 (strict) or 1 (loose)");
  }
  rgpot::nlist::Query query;
  query.cutoff = taken.cutoff;
  query.skin = taken.skin;
  if (taken.shape == 1) {
    query.shape = rgpot::nlist::Cluster4{};
  } else {
    query.shape = rgpot::nlist::Pairs{taken.shell == 1 ? rgpot::nlist::Shell::Full
                                                       : rgpot::nlist::Shell::Half};
  }
  query.cut = taken.cut == 0 ? rgpot::nlist::Cut::Strict : rgpot::nlist::Cut::Loose;
  query.periodic = {taken.periodic[0] != 0, taken.periodic[1] != 0, taken.periodic[2] != 0};
  query.fields.shifts = taken.want_shifts != 0;
  query.fields.vectors = taken.want_vectors != 0;
  query.fields.distances = taken.want_distances != 0;
  ParsedRequest parsed;
  parsed.query = query;
  parsed.cuts.types = taken.types;
  parsed.cuts.cutoff = taken.pair_cutoff;
  parsed.cuts.ntypes = taken.ntypes;
  return parsed;
}

int32_t code_of(const rgpot::nlist::Error &err, size_t *count) {
  if (count != nullptr) {
    *count = 0;
  }
  return static_cast<int32_t>(err.code());
}

rgpot_nlist_soa load_soa(const rgpot_nlist_soa *out) {
  if (out == nullptr) {
    throw rgpot::nlist::Error(rgpot::nlist::Code::Null, "neighbour buffers are null");
  }
  const auto minimum = offsetof(rgpot_nlist_soa, cap) + sizeof(size_t);
  if (out->bytes < minimum) {
    throw rgpot::nlist::Error(rgpot::nlist::Code::Abi,
                              "neighbour buffer bytes do not cover this library's struct");
  }
  rgpot_nlist_soa taken{};
  std::memcpy(&taken, out, std::min(static_cast<std::size_t>(out->bytes), sizeof(taken)));
  return taken;
}

void require_columns(const rgpot::nlist::Query &query, const rgpot_nlist_soa &out) {
  if (out.cap == 0) {
    return;
  }
  if (out.first == nullptr || out.second == nullptr) {
    throw rgpot::nlist::Error(rgpot::nlist::Code::Null, "first and second buffers are null");
  }
  if (query.fields.shifts && out.shift == nullptr) {
    throw rgpot::nlist::Error(rgpot::nlist::Code::Null, "shift buffer is null");
  }
  if (query.fields.vectors && out.vector == nullptr) {
    throw rgpot::nlist::Error(rgpot::nlist::Code::Null, "vector buffer is null");
  }
  if (query.fields.distances && out.distance == nullptr) {
    throw rgpot::nlist::Error(rgpot::nlist::Code::Null, "distance buffer is null");
  }
  if (std::holds_alternative<rgpot::nlist::Cluster4>(query.shape) && out.mask == nullptr) {
    throw rgpot::nlist::Error(rgpot::nlist::Code::Null, "a cluster list needs a mask buffer");
  }
}

rgpot::nlist::Buffers buffers_from(const rgpot_nlist_soa &out) {
  rgpot::nlist::Buffers dest;
  if (out.cap == 0) {
    return dest;
  }
  dest.first = std::span<std::int32_t>(out.first, out.cap);
  dest.second = std::span<std::int32_t>(out.second, out.cap);
  if (out.shift != nullptr) {
    dest.shift = std::span<std::int32_t>(out.shift, out.cap * 3);
  }
  if (out.vector != nullptr) {
    dest.vector = std::span<double>(out.vector, out.cap * 3);
  }
  if (out.distance != nullptr) {
    dest.distance = std::span<double>(out.distance, out.cap);
  }
  if (out.mask != nullptr) {
    dest.mask = std::span<std::uint8_t>(out.mask, out.cap);
  }
  return dest;
}

} // namespace

extern "C" {

int32_t rgpot_nlist_abi_version(void) { return RGPOT_NLIST_ABI; }

const char *rgpot_nlist_strerror(int32_t code) {
  switch (code) {
  case RGPOT_NLIST_OK:
    return "ok";
  case RGPOT_NLIST_ERR_ABI:
    return "neighbour request abi or size does not match this library";
  case RGPOT_NLIST_NULL:
    return "a required neighbour pointer is null";
  case RGPOT_NLIST_CUTOFF:
    return "cutoff and skin must be finite and >= 0";
  case RGPOT_NLIST_BOX:
    return "box must be finite, and a periodic box must have positive volume";
  case RGPOT_NLIST_SHAPE:
    return "neighbour shape, shell or cut is not a supported value";
  case RGPOT_NLIST_SHORT:
    return "neighbour buffer is shorter than the list; needed holds the full count";
  case RGPOT_NLIST_NOMEM:
    return "neighbour list allocation failed";
  case RGPOT_NLIST_SKIN:
    return "a positive skin needs a loose cut; a strict cut stores no buffer";
  default:
    return "unknown neighbour error";
  }
}

void rgpot_nlist_request_init(rgpot_nlist_request *request) {
  if (request == nullptr) {
    return;
  }
  std::memset(request, 0, sizeof(*request));
  request->bytes = static_cast<uint32_t>(sizeof(*request));
  request->abi = RGPOT_NLIST_ABI;
  request->shape = 0;
  request->shell = 0;
  request->cut = 0;
  request->periodic[0] = 1;
  request->periodic[1] = 1;
  request->periodic[2] = 1;
  request->want_shifts = 1;
  request->want_vectors = 1;
  request->want_distances = 1;
}

int32_t rgpot_nlist_build(const double *positions, size_t n, const double *box9,
                          const rgpot_nlist_request *request, rgpot_nlist_soa *out,
                          size_t *count, size_t *needed) {
  try {
    const auto parsed = query_from_c(request);
    const auto &query = parsed.query;
    rgpot::nlist::Table table;
    rgpot::nlist::build(table, positions, n, box9, query, parsed.cuts);
    if (needed != nullptr) {
      *needed = table.size();
    }
    if (out == nullptr) {
      if (count != nullptr) {
        *count = 0;
      }
      return RGPOT_NLIST_OK;
    }
    const rgpot_nlist_soa taken = load_soa(out);
    if (taken.cap == 0) {
      if (count != nullptr) {
        *count = 0;
      }
      return RGPOT_NLIST_OK;
    }
    require_columns(query, taken);
    const size_t nwrite = std::min(taken.cap, table.size());
    const auto view = table.view();
    for (size_t p = 0; p < nwrite; ++p) {
      taken.first[p] = view.first[p];
      taken.second[p] = view.second[p];
      if (taken.shift != nullptr && view.shift.size() >= (p + 1) * 3) {
        taken.shift[3 * p] = view.shift[3 * p];
        taken.shift[3 * p + 1] = view.shift[3 * p + 1];
        taken.shift[3 * p + 2] = view.shift[3 * p + 2];
      }
      if (taken.vector != nullptr && view.vector.size() >= (p + 1) * 3) {
        taken.vector[3 * p] = view.vector[3 * p];
        taken.vector[3 * p + 1] = view.vector[3 * p + 1];
        taken.vector[3 * p + 2] = view.vector[3 * p + 2];
      }
      if (taken.distance != nullptr && p < view.distance.size()) {
        taken.distance[p] = view.distance[p];
      }
      if (taken.mask != nullptr && p < view.mask.size()) {
        taken.mask[p] = view.mask[p];
      }
    }
    if (count != nullptr) {
      *count = nwrite;
    }
    if (nwrite < table.size()) {
      return RGPOT_NLIST_SHORT;
    }
    return RGPOT_NLIST_OK;
  } catch (const rgpot::nlist::Error &err) {
    return code_of(err, count);
  } catch (const std::bad_alloc &) {
    if (count != nullptr) {
      *count = 0;
    }
    return RGPOT_NLIST_NOMEM;
  }
}

struct rgpot_nlist_walk {
  rgpot::nlist::Walk walk;
  rgpot::nlist::Query query;
  rgpot_nlist_walk(const double *positions, size_t n, const double *box9,
                   const rgpot::nlist::Query &q, rgpot::nlist::PairCutoffs cuts)
      : walk(positions, n, box9, q, cuts), query(q) {}
};

int32_t rgpot_nlist_walk_open(const double *positions, size_t n, const double *box9,
                              const rgpot_nlist_request *request,
                              rgpot_nlist_walk **walk) {
  if (walk == nullptr) {
    return RGPOT_NLIST_NULL;
  }
  *walk = nullptr;
  try {
    const auto parsed = query_from_c(request);
    *walk = new (std::nothrow)
        rgpot_nlist_walk(positions, n, box9, parsed.query, parsed.cuts);
    if (*walk == nullptr) {
      return RGPOT_NLIST_NOMEM;
    }
    return RGPOT_NLIST_OK;
  } catch (const rgpot::nlist::Error &err) {
    return static_cast<int32_t>(err.code());
  } catch (const std::bad_alloc &) {
    return RGPOT_NLIST_NOMEM;
  }
}

int32_t rgpot_nlist_walk_pull(rgpot_nlist_walk *walk, rgpot_nlist_soa *out, size_t *count,
                              int32_t *done) {
  if (walk == nullptr || out == nullptr) {
    if (count != nullptr) {
      *count = 0;
    }
    return RGPOT_NLIST_NULL;
  }
  try {
    const rgpot_nlist_soa taken = load_soa(out);
    require_columns(walk->query, taken);
    bool finished = false;
    const std::size_t nwrite = walk->walk.pull(buffers_from(taken), finished);
    if (count != nullptr) {
      *count = nwrite;
    }
    if (done != nullptr) {
      *done = finished ? 1 : 0;
    }
    return RGPOT_NLIST_OK;
  } catch (const rgpot::nlist::Error &err) {
    return code_of(err, count);
  } catch (const std::bad_alloc &) {
    if (count != nullptr) {
      *count = 0;
    }
    return RGPOT_NLIST_NOMEM;
  }
}

void rgpot_nlist_walk_close(rgpot_nlist_walk *walk) { delete walk; }

int32_t rgpot_nlist_each(const double *positions, size_t n, const double *box9,
                         const rgpot_nlist_request *request, rgpot_nlist_each_fn fn, void *user) {
  try {
    const auto parsed = query_from_c(request);
    rgpot::nlist::each_raw(positions, n, box9, parsed.query, fn, user, parsed.cuts);
    return RGPOT_NLIST_OK;
  } catch (const rgpot::nlist::Error &err) {
    return static_cast<int32_t>(err.code());
  } catch (const std::bad_alloc &) {
    return RGPOT_NLIST_NOMEM;
  }
}

struct rgpot_nlist_sweep {
  rgpot::nlist::Sweep sweep;
};

int32_t rgpot_nlist_sweep_open(rgpot_nlist_sweep **sweep) {
  if (sweep == nullptr) {
    return RGPOT_NLIST_NULL;
  }
  *sweep = new (std::nothrow) rgpot_nlist_sweep;
  if (*sweep == nullptr) {
    return RGPOT_NLIST_NOMEM;
  }
  return RGPOT_NLIST_OK;
}

int32_t rgpot_nlist_sweep_each(rgpot_nlist_sweep *sweep, const double *positions, size_t n,
                              const double *box9, const rgpot_nlist_request *request,
                              rgpot_nlist_each_team_fn fn, void *user, int32_t threads,
                              int32_t shell, int32_t *reused) {
  if (sweep == nullptr || fn == nullptr) {
    return RGPOT_NLIST_NULL;
  }
  try {
    auto parsed = query_from_c(request);
    if (shell != 0 && shell != 1) {
      throw rgpot::nlist::Error(rgpot::nlist::Code::Shape, "shell must be 0 (half) or 1 (full)");
    }
    parsed.query.shape = shell == 1 ? rgpot::nlist::Shape{rgpot::nlist::Pairs{rgpot::nlist::Shell::Full}}
                                    : rgpot::nlist::Shape{rgpot::nlist::Pairs{rgpot::nlist::Shell::Half}};
    sweep->sweep.update(positions, n, box9, parsed.query, parsed.cuts);
    if (reused != nullptr) {
      *reused = sweep->sweep.reused() ? 1 : 0;
    }
    const auto which = shell == 1 ? rgpot::nlist::Shell::Full : rgpot::nlist::Shell::Half;
    sweep->sweep.eachTeam(which, threads,
                          [&](int thread, std::int32_t i, std::int32_t j, int sa, int sb, int sc,
                              double x, double y, double z, double r) {
                            fn(user, static_cast<int32_t>(thread), i, j, sa, sb, sc, x, y, z, r);
                          });
    return RGPOT_NLIST_OK;
  } catch (const rgpot::nlist::Error &err) {
    return static_cast<int32_t>(err.code());
  } catch (const std::bad_alloc &) {
    return RGPOT_NLIST_NOMEM;
  }
}

void rgpot_nlist_sweep_close(rgpot_nlist_sweep *sweep) { delete sweep; }

int32_t rgpot_nlist_build_systems(const double *positions, const int32_t *counts,
                                  size_t nsystems, const double *box9,
                                  const rgpot_nlist_request *request, rgpot_nlist_soa *out,
                                  int32_t *system, size_t *count, size_t *needed) {
  try {
    const auto parsed = query_from_c(request);
    const auto &query = parsed.query;
    rgpot::nlist::Systems systems;
    systems.positions = positions;
    systems.count = counts;
    systems.systems = nsystems;
    systems.box9 = box9;
    rgpot::nlist::Batch batch;
    rgpot::nlist::build(batch, systems, query, parsed.cuts);
    if (needed != nullptr) {
      *needed = batch.pairs.size();
    }
    if (out == nullptr) {
      if (count != nullptr) {
        *count = 0;
      }
      return RGPOT_NLIST_OK;
    }
    const rgpot_nlist_soa taken = load_soa(out);
    if (taken.cap == 0) {
      if (count != nullptr) {
        *count = 0;
      }
      return RGPOT_NLIST_OK;
    }
    require_columns(query, taken);
    const size_t nwrite = std::min(taken.cap, batch.pairs.size());
    const auto view = batch.pairs.view();
    for (size_t p = 0; p < nwrite; ++p) {
      taken.first[p] = view.first[p];
      taken.second[p] = view.second[p];
      if (system != nullptr) {
        system[p] = batch.system[p];
      }
      if (taken.shift != nullptr && view.shift.size() >= (p + 1) * 3) {
        taken.shift[3 * p] = view.shift[3 * p];
        taken.shift[3 * p + 1] = view.shift[3 * p + 1];
        taken.shift[3 * p + 2] = view.shift[3 * p + 2];
      }
      if (taken.vector != nullptr && view.vector.size() >= (p + 1) * 3) {
        taken.vector[3 * p] = view.vector[3 * p];
        taken.vector[3 * p + 1] = view.vector[3 * p + 1];
        taken.vector[3 * p + 2] = view.vector[3 * p + 2];
      }
      if (taken.distance != nullptr && p < view.distance.size()) {
        taken.distance[p] = view.distance[p];
      }
      if (taken.mask != nullptr && p < view.mask.size()) {
        taken.mask[p] = view.mask[p];
      }
    }
    if (count != nullptr) {
      *count = nwrite;
    }
    if (nwrite < batch.pairs.size()) {
      return RGPOT_NLIST_SHORT;
    }
    return RGPOT_NLIST_OK;
  } catch (const rgpot::nlist::Error &err) {
    return code_of(err, count);
  } catch (const std::bad_alloc &) {
    if (count != nullptr) {
      *count = 0;
    }
    return RGPOT_NLIST_NOMEM;
  }
}

struct rgpot_nlist_builder {
  rgpot::nlist::Builder builder;
};

int32_t rgpot_nlist_builder_open(rgpot_nlist_builder **builder) {
  if (builder == nullptr) {
    return RGPOT_NLIST_NULL;
  }
  *builder = new (std::nothrow) rgpot_nlist_builder;
  if (*builder == nullptr) {
    return RGPOT_NLIST_NOMEM;
  }
  return RGPOT_NLIST_OK;
}

int32_t rgpot_nlist_builder_update(rgpot_nlist_builder *builder, const double *positions, size_t n,
                                   const double *box9, const rgpot_nlist_request *request,
                                   rgpot_nlist_soa *out, size_t *count, size_t *needed,
                                   int32_t *reused) {
  if (builder == nullptr) {
    return RGPOT_NLIST_NULL;
  }
  try {
    const auto parsed = query_from_c(request);
    const auto &query = parsed.query;
    builder->builder.update(positions, n, box9, query, parsed.cuts);
    if (reused != nullptr) {
      *reused = builder->builder.reused() ? 1 : 0;
    }
    const auto &table = builder->builder.table();
    if (needed != nullptr) {
      *needed = table.size();
    }
    if (out == nullptr) {
      if (count != nullptr) {
        *count = 0;
      }
      return RGPOT_NLIST_OK;
    }
    const rgpot_nlist_soa taken = load_soa(out);
    if (taken.cap == 0) {
      if (count != nullptr) {
        *count = 0;
      }
      return RGPOT_NLIST_OK;
    }
    require_columns(query, taken);
    const size_t nwrite = std::min(taken.cap, table.size());
    const auto view = table.view();
    for (size_t p = 0; p < nwrite; ++p) {
      taken.first[p] = view.first[p];
      taken.second[p] = view.second[p];
      if (taken.shift != nullptr && view.shift.size() >= (p + 1) * 3) {
        taken.shift[3 * p] = view.shift[3 * p];
        taken.shift[3 * p + 1] = view.shift[3 * p + 1];
        taken.shift[3 * p + 2] = view.shift[3 * p + 2];
      }
      if (taken.vector != nullptr && view.vector.size() >= (p + 1) * 3) {
        taken.vector[3 * p] = view.vector[3 * p];
        taken.vector[3 * p + 1] = view.vector[3 * p + 1];
        taken.vector[3 * p + 2] = view.vector[3 * p + 2];
      }
      if (taken.distance != nullptr && p < view.distance.size()) {
        taken.distance[p] = view.distance[p];
      }
      if (taken.mask != nullptr && p < view.mask.size()) {
        taken.mask[p] = view.mask[p];
      }
    }
    if (count != nullptr) {
      *count = nwrite;
    }
    if (nwrite < table.size()) {
      return RGPOT_NLIST_SHORT;
    }
    return RGPOT_NLIST_OK;
  } catch (const rgpot::nlist::Error &err) {
    return code_of(err, count);
  } catch (const std::bad_alloc &) {
    if (count != nullptr) {
      *count = 0;
    }
    return RGPOT_NLIST_NOMEM;
  }
}

void rgpot_nlist_builder_close(rgpot_nlist_builder *builder) { delete builder; }

struct rgpot_nlist_fortran_list {
  rgpot::nlist::Builder builder;
};

rgpot_nlist_fortran_list *rgpot_nlist_fortran_open(void) {
  return new (std::nothrow) rgpot_nlist_fortran_list;
}

void rgpot_nlist_fortran_close(rgpot_nlist_fortran_list *list) { delete list; }

int32_t rgpot_nlist_fortran_build(rgpot_nlist_fortran_list *list, const double *positions,
                                  size_t natoms, const double *box9, double cutoff, double skin,
                                  const int32_t periodic[3], int32_t *first, int32_t *second,
                                  double *vector, double *distance, size_t cap, size_t *count,
                                  int32_t *reused) {
  if (list == nullptr || positions == nullptr || box9 == nullptr || periodic == nullptr) {
    if (count != nullptr) {
      *count = 0;
    }
    return RGPOT_NLIST_NULL;
  }
  try {
    rgpot::nlist::Query query;
    query.cutoff = cutoff;
    query.skin = skin;
    query.cut = skin > 0.0 ? rgpot::nlist::Cut::Loose : rgpot::nlist::Cut::Strict;
    query.shape = rgpot::nlist::Pairs{rgpot::nlist::Shell::Full};
    query.fields = {true, true, true};
    for (int k = 0; k < 3; ++k) {
      query.periodic[static_cast<std::size_t>(k)] = periodic[k] != 0;
    }
    list->builder.update(positions, natoms, box9, query);
    if (reused != nullptr) {
      *reused = list->builder.reused() ? 1 : 0;
    }
    const auto &table = list->builder.table();
    const double cut2 = cutoff * cutoff;
    std::size_t kept = 0;
    for (std::size_t p = 0; p < table.size(); ++p) {
      const auto i = static_cast<std::size_t>(table.first[p]);
      const auto j = static_cast<std::size_t>(table.second[p]);
      const int sa = table.shift[3 * p];
      const int sb = table.shift[3 * p + 1];
      const int sc = table.shift[3 * p + 2];
      const double vx = positions[3 * j] - positions[3 * i] + sa * box9[0] + sb * box9[3] +
                        sc * box9[6];
      const double vy = positions[3 * j + 1] - positions[3 * i + 1] + sa * box9[1] +
                        sb * box9[4] + sc * box9[7];
      const double vz = positions[3 * j + 2] - positions[3 * i + 2] + sa * box9[2] +
                        sb * box9[5] + sc * box9[8];
      const double r2 = vx * vx + vy * vy + vz * vz;
      if (!(r2 > 0.0) || !(r2 < cut2)) {
        continue;
      }
      if (first != nullptr && kept < cap) {
        first[kept] = table.first[p];
        second[kept] = table.second[p];
        vector[3 * kept] = vx;
        vector[3 * kept + 1] = vy;
        vector[3 * kept + 2] = vz;
        distance[kept] = std::sqrt(r2);
      }
      ++kept;
    }
    if (count != nullptr) {
      *count = kept;
    }
    if (first == nullptr) {
      return RGPOT_NLIST_OK;
    }
    if (kept > cap) {
      return RGPOT_NLIST_SHORT;
    }
    return RGPOT_NLIST_OK;
  } catch (const rgpot::nlist::Error &err) {
    if (count != nullptr) {
      *count = 0;
    }
    return static_cast<int32_t>(err.code());
  } catch (const std::bad_alloc &) {
    if (count != nullptr) {
      *count = 0;
    }
    return RGPOT_NLIST_NOMEM;
  }
}

} // extern "C"
