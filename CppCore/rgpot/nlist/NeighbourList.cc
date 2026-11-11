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

template <typename Fn>
void each_image(const double *pos, std::size_t n, const double *box, const Query &query,
                const int nmax[3], std::size_t begin, std::size_t end, Fn &&fn) {
  for (std::size_t i = begin; i < end; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      for (int sa = -nmax[0]; sa <= nmax[0]; ++sa) {
        for (int sb = -nmax[1]; sb <= nmax[1]; ++sb) {
          for (int sc = -nmax[2]; sc <= nmax[2]; ++sc) {
            if (!full_shell(query) &&
                !keep_half(static_cast<std::int32_t>(i), static_cast<std::int32_t>(j),
                           sa, sb, sc)) {
              continue;
            }
            double vx = 0;
            double vy = 0;
            double vz = 0;
            image_vector(pos, i, j, sa, sb, sc, box, vx, vy, vz);
            const double r2 = vx * vx + vy * vy + vz * vz;
            if (!accept_distance(r2, query)) {
              continue;
            }
            fn(static_cast<std::int32_t>(i), static_cast<std::int32_t>(j), sa, sb, sc, vx,
               vy, vz, std::sqrt(r2));
          }
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

void collect(const double *pos, std::size_t n, const double *box, const Query &query,
             const int nmax[3], std::size_t begin, std::size_t end,
             std::vector<Rec> &out) {
  if (clustered(query)) {
    const std::size_t c0 = begin & ~std::size_t{3};
    const std::size_t c1 = std::min(n, (end + 3) & ~std::size_t{3});
    Query full = query;
    full.shape = Pairs{Shell::Full};
    std::vector<Rec> raw;
    each_image(pos, n, box, full, nmax, c0, c1,
               [&](std::int32_t i, std::int32_t j, int sa, int sb, int sc, double vx,
                   double vy, double vz, double dist) {
                 if (static_cast<std::size_t>(i) < begin ||
                     static_cast<std::size_t>(i) >= end) {
                   return;
                 }
                 const auto lane = static_cast<unsigned>(i & 3);
                 push_rec(raw, query, i & ~std::int32_t{3}, j, sa, sb, sc, vx, vy, vz,
                          dist, static_cast<std::uint8_t>(1u << lane));
               });
    merge_clusters(raw);
    out.insert(out.end(), raw.begin(), raw.end());
    return;
  }
  each_image(pos, n, box, query, nmax, begin, end,
             [&](std::int32_t i, std::int32_t j, int sa, int sb, int sc, double vx,
                 double vy, double vz, double dist) {
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

} // namespace

void Builder::update(const double *positions, std::size_t n, const double *box9,
                     const Query &query) {
  check_geometry(positions, n, box9, query);
  reused_ = false;
  if (have_ && n_ == n && query_.cutoff == query.cutoff && query_.skin == query.skin &&
      query_.shape == query.shape && query_.cut == query.cut &&
      query_.periodic == query.periodic && query_.fields.shifts == query.fields.shifts &&
      query_.fields.vectors == query.fields.vectors &&
      query_.fields.distances == query.fields.distances &&
      std::memcmp(box_, box9, sizeof(box_)) == 0) {
    const double limit2 = 0.25 * query.skin * query.skin;
    bool same = true;
    for (std::size_t a = 0; a < n; ++a) {
      const double dx = positions[3 * a] - ref_[3 * a];
      const double dy = positions[3 * a + 1] - ref_[3 * a + 1];
      const double dz = positions[3 * a + 2] - ref_[3 * a + 2];
      if (dx * dx + dy * dy + dz * dz > limit2) {
        same = false;
        break;
      }
    }
    if (same) {
      reused_ = true;
      return;
    }
  }
  build(table_, positions, n, box9, query);
  ref_.assign(positions, positions + 3 * n);
  std::memcpy(box_, box9, sizeof(box_));
  query_ = query;
  n_ = n;
  have_ = true;
  ++builds_;
}

Walk::Walk(const double *positions, std::size_t n, const double *box9, const Query &query)
    : pos_(positions), n_(n), query_(query) {
  check_geometry(positions, n, box9, query);
  std::memcpy(box_, box9, sizeof(box_));
  shift_bounds(box_, query_.periodic, search_radius(query_), nmax_);
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
      collect(pos_, n_, box_, query_, nmax_, atom_, end, rows);
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
           const Query &query) {
  check_geometry(positions, n, box9, query);
  table = Table{};
  if (n == 0 || !(query.cutoff > 0.0)) {
    table.stored = true;
    return;
  }
  int nmax[3];
  shift_bounds(box9, query.periodic, search_radius(query), nmax);
  const int nt = threadsRequested();
  std::vector<std::vector<Rec>> parts(static_cast<std::size_t>(std::max(nt, 1)));
  if (nt <= 1) {
    collect(positions, n, box9, query, nmax, 0, n, parts[0]);
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
      collect(positions, n, box9, query, nmax, static_cast<std::size_t>(range.begin),
              static_cast<std::size_t>(range.end), parts[static_cast<std::size_t>(t)]);
    }
  }
  std::vector<Rec> rows;
  for (const auto &part : parts) {
    rows.insert(rows.end(), part.begin(), part.end());
  }
  if (clustered(query) && nt > 1) {
    merge_clusters(rows);
  }
  store_records(table, query, rows);
}

} // namespace rgpot::nlist

namespace {

rgpot::nlist::Query query_from_c(const rgpot_nlist_request *request) {
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
  return query;
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
    const auto query = query_from_c(request);
    rgpot::nlist::Table table;
    rgpot::nlist::build(table, positions, n, box9, query);
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
                   const rgpot::nlist::Query &q)
      : walk(positions, n, box9, q), query(q) {}
};

int32_t rgpot_nlist_walk_open(const double *positions, size_t n, const double *box9,
                              const rgpot_nlist_request *request,
                              rgpot_nlist_walk **walk) {
  if (walk == nullptr) {
    return RGPOT_NLIST_NULL;
  }
  *walk = nullptr;
  try {
    const auto query = query_from_c(request);
    *walk = new (std::nothrow) rgpot_nlist_walk(positions, n, box9, query);
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

} // extern "C"
