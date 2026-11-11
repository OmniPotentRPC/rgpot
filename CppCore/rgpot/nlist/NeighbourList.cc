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
  if (query.format != Format::Half && query.format != Format::Full &&
      query.format != Format::Cluster4) {
    throw Error(Code::Shape, "format must be half, full, or a cluster of four");
  }
  if (query.cut != Cut::Strict && query.cut != Cut::Loose) {
    throw Error(Code::Shape, "cut must be strict or loose");
  }
  if (query.mode != Mode::Auto && query.mode != Mode::Stored &&
      query.mode != Mode::Stream) {
    throw Error(Code::Shape, "mode must be auto, stored, or stream");
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
            if (query.format != Format::Full &&
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
  if (query.format == Format::Cluster4) {
    const std::size_t c0 = begin & ~std::size_t{3};
    const std::size_t c1 = std::min(n, (end + 3) & ~std::size_t{3});
    Query full = query;
    full.format = Format::Full;
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

std::uint64_t pair_bound(std::size_t n, const int nmax[3]) {
  const auto nx = static_cast<std::uint64_t>(2 * nmax[0] + 1);
  const auto ny = static_cast<std::uint64_t>(2 * nmax[1] + 1);
  const auto nz = static_cast<std::uint64_t>(2 * nmax[2] + 1);
  return static_cast<std::uint64_t>(n) * static_cast<std::uint64_t>(n) * nx * ny * nz;
}

} // namespace

void Builder::update(const double *positions, std::size_t n, const double *box9,
                     const Query &query) {
  check_geometry(positions, n, box9, query);
  reused_ = false;
  if (have_ && n_ == n && query_.cutoff == query.cutoff && query_.skin == query.skin &&
      query_.format == query.format && query_.cut == query.cut &&
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

std::size_t Walk::pull(OutSoA dest, std::size_t cap, bool &done) {
  if (cap > 0 && (dest.first == nullptr || dest.second == nullptr)) {
    throw Error(Code::Null, "walk buffers for first and second are null");
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
          n_, atom_ + (query_.format == Format::Cluster4 ? std::size_t{4} : std::size_t{1}));
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
    if (dest.shift != nullptr) {
      dest.shift[3 * written] = hold_shift_[3 * hold_at_];
      dest.shift[3 * written + 1] = hold_shift_[3 * hold_at_ + 1];
      dest.shift[3 * written + 2] = hold_shift_[3 * hold_at_ + 2];
    }
    if (dest.vector != nullptr) {
      dest.vector[3 * written] = hold_vector_[3 * hold_at_];
      dest.vector[3 * written + 1] = hold_vector_[3 * hold_at_ + 1];
      dest.vector[3 * written + 2] = hold_vector_[3 * hold_at_ + 2];
    }
    if (dest.distance != nullptr) {
      dest.distance[written] = hold_distance_[hold_at_];
    }
    if (dest.mask != nullptr) {
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
    table.stored = query.mode != Mode::Stream;
    return;
  }
  int nmax[3];
  shift_bounds(box9, query.periodic, search_radius(query), nmax);
  const bool stream = query.mode == Mode::Stream ||
                      (query.mode == Mode::Auto &&
                       pair_bound(n, nmax) > availableBytes() / 32);
  if (stream) {
    table.stored = false;
    return;
  }
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
  if (query.format == Format::Cluster4 && nt > 1) {
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
  const auto minimum = offsetof(rgpot_nlist_request, mode) + sizeof(std::int32_t);
  if (request->bytes < minimum || request->abi != RGPOT_NLIST_ABI) {
    throw rgpot::nlist::Error(rgpot::nlist::Code::Abi,
                              "request abi or bytes does not match this library");
  }
  rgpot_nlist_request taken{};
  std::memcpy(&taken, request,
              std::min(static_cast<std::size_t>(request->bytes), sizeof(taken)));
  if (taken.format != 0 && taken.format != 1 && taken.format != 2) {
    throw rgpot::nlist::Error(rgpot::nlist::Code::Shape,
                              "format must be 0 (half), 1 (full) or 2 (cluster of four)");
  }
  if (taken.cut != 0 && taken.cut != 1) {
    throw rgpot::nlist::Error(rgpot::nlist::Code::Shape,
                              "cut must be 0 (strict) or 1 (loose)");
  }
  if (taken.mode != 0 && taken.mode != 1 && taken.mode != 2) {
    throw rgpot::nlist::Error(rgpot::nlist::Code::Shape,
                              "mode must be 0 (auto), 1 (stored) or 2 (stream)");
  }
  rgpot::nlist::Query query;
  query.cutoff = taken.cutoff;
  query.skin = taken.skin;
  query.format = taken.format == 1   ? rgpot::nlist::Format::Full
                 : taken.format == 2 ? rgpot::nlist::Format::Cluster4
                                      : rgpot::nlist::Format::Half;
  query.cut = taken.cut == 0 ? rgpot::nlist::Cut::Strict : rgpot::nlist::Cut::Loose;
  query.periodic = {taken.periodic[0] != 0, taken.periodic[1] != 0, taken.periodic[2] != 0};
  query.fields.shifts = taken.want_shifts != 0;
  query.fields.vectors = taken.want_vectors != 0;
  query.fields.distances = taken.want_distances != 0;
  query.mode = taken.mode == 2   ? rgpot::nlist::Mode::Stream
               : taken.mode == 0 ? rgpot::nlist::Mode::Auto
                                 : rgpot::nlist::Mode::Stored;
  return query;
}

int32_t code_of(const rgpot::nlist::Error &err, size_t *count) {
  if (count != nullptr) {
    *count = 0;
  }
  return static_cast<int32_t>(err.code());
}

void write_pair(rgpot_nlist_pair &dst, const rgpot::nlist::SoA &src, std::size_t p) {
  dst.first = src.first[p];
  dst.second = src.second[p];
  dst.shift[0] = src.shift != nullptr ? src.shift[3 * p] : 0;
  dst.shift[1] = src.shift != nullptr ? src.shift[3 * p + 1] : 0;
  dst.shift[2] = src.shift != nullptr ? src.shift[3 * p + 2] : 0;
  dst.vector[0] = src.vector != nullptr ? src.vector[3 * p] : 0;
  dst.vector[1] = src.vector != nullptr ? src.vector[3 * p + 1] : 0;
  dst.vector[2] = src.vector != nullptr ? src.vector[3 * p + 2] : 0;
  dst.distance = src.distance != nullptr ? src.distance[p] : 0;
  dst.mask = src.mask != nullptr ? src.mask[p] : 0xff;
}

struct WalkBox {
  rgpot::nlist::Walk walk;
  rgpot::nlist::Query query;
};

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
    return "neighbour format, cut or mode is not a supported value";
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
  request->format = 0;
  request->cut = 0;
  request->periodic[0] = 1;
  request->periodic[1] = 1;
  request->periodic[2] = 1;
  request->want_shifts = 1;
  request->want_vectors = 1;
  request->want_distances = 1;
  request->mode = 1;
}

int32_t rgpot_nlist_build(const double *positions, size_t n, const double *box9,
                          const rgpot_nlist_request *request, rgpot_nlist_pair *out,
                          size_t cap, size_t *count, size_t *needed) {
  try {
    if (out == nullptr && cap > 0) {
      if (count != nullptr) {
        *count = 0;
      }
      return RGPOT_NLIST_NULL;
    }
    rgpot::nlist::Table table;
    rgpot::nlist::build(table, positions, n, box9, query_from_c(request));
    if (needed != nullptr) {
      *needed = table.size();
    }
    const size_t nwrite = out == nullptr ? 0 : std::min(cap, table.size());
    const auto view = table.view();
    for (size_t p = 0; p < nwrite; ++p) {
      write_pair(out[p], view, p);
    }
    if (count != nullptr) {
      *count = nwrite;
    }
    if (out != nullptr && nwrite < table.size()) {
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
  explicit rgpot_nlist_walk(const double *positions, size_t n, const double *box9,
                            const rgpot::nlist::Query &query)
      : walk(positions, n, box9, query) {}
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

int32_t rgpot_nlist_walk_pull(rgpot_nlist_walk *walk, rgpot_nlist_pair *out, size_t cap,
                              size_t *count, int32_t *done) {
  if (walk == nullptr || (out == nullptr && cap > 0)) {
    if (count != nullptr) {
      *count = 0;
    }
    return RGPOT_NLIST_NULL;
  }
  try {
    if (cap == 0) {
      bool finished = false;
      rgpot::nlist::OutSoA dest{};
      walk->walk.pull(dest, 0, finished);
      if (count != nullptr) {
        *count = 0;
      }
      if (done != nullptr) {
        *done = finished ? 1 : 0;
      }
      return RGPOT_NLIST_OK;
    }
    std::vector<std::int32_t> first(cap), second(cap), shift(cap * 3);
    std::vector<double> vector(cap * 3), distance(cap);
    std::vector<std::uint8_t> mask(cap);
    rgpot::nlist::OutSoA dest{first.data(),   second.data(), shift.data(),
                              vector.data(), distance.data(), mask.data()};
    bool finished = false;
    const std::size_t nwrite = walk->walk.pull(dest, cap, finished);
    for (std::size_t p = 0; p < nwrite; ++p) {
      out[p].first = first[p];
      out[p].second = second[p];
      out[p].shift[0] = shift[3 * p];
      out[p].shift[1] = shift[3 * p + 1];
      out[p].shift[2] = shift[3 * p + 2];
      out[p].vector[0] = vector[3 * p];
      out[p].vector[1] = vector[3 * p + 1];
      out[p].vector[2] = vector[3 * p + 2];
      out[p].distance = distance[p];
      out[p].mask = mask[p];
    }
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
