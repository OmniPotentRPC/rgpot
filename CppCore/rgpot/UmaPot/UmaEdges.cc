// Copyright 2023--present rgpot developers

#include "rgpot/UmaPot/UmaEdges.hpp"

#include "rgpot/MetatomicPot/vesin_compat.hpp"
#include "rgpot/UmaPot/UmaLinkedCell.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <type_traits>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace rgpot {

struct UmaEdgeSession::CellCache {
  linked_cell::Scratch scratch;
};

UmaEdgeList take_edges();
std::unique_ptr<UmaEdgeSession::CellCache> take_cell();

namespace {

struct EdgeError : std::exception {
  const char *text;
  explicit EdgeError(const char *t) noexcept : text(t) {}
  const char *what() const noexcept override { return text; }
};

int call_vesin(const double (*points)[3], std::size_t n_points, const double (*box)[3],
               bool pbc[3], VesinDevice device, const VesinOptions &options,
               VesinNeighborList *neighbors, const char **error) noexcept;
void free_vesin(VesinNeighborList *neighbors) noexcept;

// Vesin rebuilds the cached list after an atom moves more than half of this.
constexpr double kVerletSkin = 1.0;

// libstdc++ and libc++ both store a vector as three pointers, and the
// second one is the end of the constructed range. A large resize
// value-initializes that range on the calling thread, which faults
// every page before the workers run. The edge write fills every slot,
// so the size is advanced and the workers touch the pages themselves.
template <typename T>
void resize_no_init(std::vector<T> &vec, std::size_t n) {
  static_assert(std::is_trivially_copyable<T>::value, "trivial elements only");
  static_assert(sizeof(std::vector<T>) == 3 * sizeof(void *), "vector is three pointers");
  if (n <= vec.size()) {
    grow_resize(vec, n);
    return;
  }
  if (vec.capacity() < n)
    grow_reserve(vec, n);
  const std::size_t bytes = n * sizeof(T);
  auto *ptr = reinterpret_cast<unsigned char **>(&vec);
  ptr[1] = ptr[0] + bytes;
}

double axis_length(const double *box, int axis) noexcept {
  const double x = box[3 * axis];
  const double y = box[3 * axis + 1];
  const double z = box[3 * axis + 2];
  return std::sqrt(x * x + y * y + z * z);
}

// A cell list with fewer than three cells along a side walks the same
// pairs as a direct search. Brute force is the minimum image, so it
// matches the cell list only when a second image sits past the cutoff.
bool brute_force_box(const double *box, double cutoff) noexcept {
  double lo = 1e300;
  double hi = 0.0;
  for (int axis = 0; axis < 3; ++axis) {
    const double len = axis_length(box, axis);
    lo = std::min(lo, len);
    hi = std::max(hi, len);
  }
  return lo >= 2.0 * cutoff && hi < 3.0 * cutoff;
}

int worker_count(int requested, std::size_t n) noexcept {
  int threads = requested > 0 ? requested : 1;
  if (n > 0 && threads > static_cast<int>(n))
    threads = static_cast<int>(n);
  // The line above already yields at least 1.
  return threads;
}

template <typename Work>
void run_ranges(int threads, std::size_t n, Work work) noexcept {
  if (threads <= 1 || n <= 1) {
    work(0, n);
    return;
  }
  const bool started = spawn_workers(threads, [&](int t) {
    const std::size_t begin =
        (static_cast<std::size_t>(t) * n) / static_cast<std::size_t>(threads);
    const std::size_t end =
        (static_cast<std::size_t>(t + 1) * n) / static_cast<std::size_t>(threads);
    return [&, begin, end] { work(begin, end); };
  });
  if (!started)
    work(0, n);
}

void exercise_empty_ranges_impl(int threads) {
  run_ranges(worker_count(threads, 0), 0, [](std::size_t, std::size_t) {});
}

UmaEdgeList molecular_edges(const double *pos, std::size_t n, UmaEdgeSession &session) {
  grow_resize(session.rows, n);
  const int threads = worker_count(session.threads, n);
  run_ranges(threads, n, [&](std::size_t begin, std::size_t end) {
    for (std::size_t c = begin; c < end; ++c) {
      auto &row = session.rows[c];
      row.neighbor.clear();
      row.dist.clear();
      grow_reserve(row.neighbor, n - 1);
      grow_reserve(row.dist, n - 1);
      for (std::size_t j = 0; j < n; ++j) {
        if (j == c)
          continue;
        const double dx = pos[3 * j] - pos[3 * c];
        const double dy = pos[3 * j + 1] - pos[3 * c + 1];
        const double dz = pos[3 * j + 2] - pos[3 * c + 2];
        grow_push(row.neighbor, static_cast<int64_t>(j));
        grow_push(row.dist, std::sqrt(dx * dx + dy * dy + dz * dz));
      }
      grow_resize(row.order, row.neighbor.size());
      std::iota(row.order.begin(), row.order.end(), 0);
      grow_sort(row.order.begin(), row.order.end(), [&](std::size_t a, std::size_t b) {
        if (row.dist[a] != row.dist[b])
          return row.dist[a] < row.dist[b];
        return row.neighbor[a] < row.neighbor[b];
      });
    }
  });

  UmaEdgeList out;
  const std::size_t total = n * (n - 1);
  grow_reserve(out.neighbor, total);
  grow_reserve(out.center, total);
  grow_reserve(out.offsets, total * 3);
  for (std::size_t c = 0; c < n; ++c) {
    const auto &row = session.rows[c];
    for (std::size_t id : row.order) {
      grow_push(out.neighbor, row.neighbor[id]);
      grow_push(out.center, static_cast<int64_t>(c));
      grow_push(out.offsets, 0.f);
      grow_push(out.offsets, 0.f);
      grow_push(out.offsets, 0.f);
    }
  }
  return out;
}

void emit_sorted(const VesinNeighborList &nl, const ForceInput &in, int max_neighbors,
                 int threads, UmaEdgeSession &session) noexcept {
  const std::size_t n = in.nAtoms;
  (void)threads;
  const double *pos = in.pos;
  const double *box = in.box;
  grow_assign_n(session.bucket, n + 1, std::size_t{0});
  for (std::size_t k = 0; k < nl.length; ++k) {
    const auto center = static_cast<std::size_t>(nl.pairs[k][0]);
    ++session.bucket[center + 1];
  }
  for (std::size_t i = 1; i <= n; ++i)
    session.bucket[i] += session.bucket[i - 1];
  if (session.keys.size() < session.bucket[n])
    grow_resize(session.keys, session.bucket[n]);
  std::vector<std::size_t> write_idx;
  grow_assign_range(write_idx, session.bucket.data(), session.bucket.data() + session.bucket.size());
  for (std::size_t k = 0; k < nl.length; ++k) {
    const auto center = static_cast<std::size_t>(nl.pairs[k][0]);
    const std::size_t j = nl.pairs[k][1];
    const double s0 = static_cast<double>(nl.shifts[k][0]);
    const double s1 = static_cast<double>(nl.shifts[k][1]);
    const double s2 = static_cast<double>(nl.shifts[k][2]);
    const double dx = pos[3 * j] - pos[3 * center] + s0 * box[0] + s1 * box[3] + s2 * box[6];
    const double dy = pos[3 * j + 1] - pos[3 * center + 1] + s0 * box[1] + s1 * box[4] + s2 * box[7];
    const double dz = pos[3 * j + 2] - pos[3 * center + 2] + s0 * box[2] + s1 * box[5] + s2 * box[8];
    auto &key = session.keys[write_idx[center]++];
    key.dist2 = dx * dx + dy * dy + dz * dz;
    key.neighbor = static_cast<std::uint32_t>(j);
    key.s0 = nl.shifts[k][0];
    key.s1 = nl.shifts[k][1];
    key.s2 = nl.shifts[k][2];
  }
  auto &out = session.edges;
  out.neighbor.clear();
  out.center.clear();
  out.offsets.clear();
  if (out.neighbor.capacity() < session.bucket[n])
    grow_reserve(out.neighbor, session.bucket[n]);
  if (out.center.capacity() < session.bucket[n])
    grow_reserve(out.center, session.bucket[n]);
  if (out.offsets.capacity() < session.bucket[n] * 3)
    grow_reserve(out.offsets, session.bucket[n] * 3);
  const int workers = worker_count(threads, n);
  run_ranges(workers, n, [&](std::size_t begin, std::size_t end) {
    for (std::size_t c = begin; c < end; ++c) {
      auto first = session.keys.begin() + static_cast<std::ptrdiff_t>(session.bucket[c]);
      auto last = session.keys.begin() + static_cast<std::ptrdiff_t>(session.bucket[c + 1]);
      grow_sort(first, last, [](const UmaEdgeSession::EdgeKey &a, const UmaEdgeSession::EdgeKey &b) {
        if (a.dist2 != b.dist2)
          return a.dist2 < b.dist2;
        if (a.neighbor != b.neighbor)
          return a.neighbor < b.neighbor;
        if (a.s0 != b.s0)
          return a.s0 < b.s0;
        if (a.s1 != b.s1)
          return a.s1 < b.s1;
        return a.s2 < b.s2;
      });
    }
  });
  for (std::size_t c = 0; c < n; ++c) {
    std::size_t written = 0;
    for (std::size_t k = session.bucket[c]; k < session.bucket[c + 1]; ++k) {
      const auto &key = session.keys[k];
      if (key.dist2 < 1e-16)
        continue;
      if (max_neighbors > 0 && written >= static_cast<std::size_t>(max_neighbors))
        break;
      grow_push(out.neighbor, static_cast<int64_t>(key.neighbor));
      grow_push(out.center, static_cast<int64_t>(c));
      grow_push(out.offsets, static_cast<float>(key.s0));
      grow_push(out.offsets, static_cast<float>(key.s1));
      grow_push(out.offsets, static_cast<float>(key.s2));
      ++written;
    }
  }
}

void emit_slices(linked_cell::Scratch &scratch, const double *pos, const double *box,
                 std::size_t n, double cut2, int max_neighbors, int threads, bool recompute,
                 UmaEdgeSession &session) noexcept {
  const double side[3] = {box[0], box[4], box[8]};
  const int workers = worker_count(threads, n);
  if (recompute) {
    run_ranges(workers, n, [&](std::size_t begin, std::size_t end) {
      for (std::size_t c = begin; c < end; ++c) {
        const auto t = scratch.slice_thread[c];
        auto &part = scratch.parts[t];
        const auto at = scratch.slice_at[c];
        const auto count = scratch.slice_n[c];
        for (std::uint32_t i = 0; i < count; ++i) {
          auto &row = part[at + i];
          const double dx = pos[3 * row.neighbor] - pos[3 * c] + row.s0 * side[0];
          const double dy = pos[3 * row.neighbor + 1] - pos[3 * c + 1] + row.s1 * side[1];
          const double dz = pos[3 * row.neighbor + 2] - pos[3 * c + 2] + row.s2 * side[2];
          row.dist2 = dx * dx + dy * dy + dz * dz;
        }
        if (count > 1) {
          grow_sort(part.begin() + static_cast<std::ptrdiff_t>(at),
                    part.begin() + static_cast<std::ptrdiff_t>(at + count),
                    linked_cell::before);
        }
      }
    });
  }
  grow_assign_n(session.bucket, n + 1, std::size_t{0});
  auto keep_of = [&](std::size_t c) {
    const auto &part = scratch.parts[scratch.slice_thread[c]];
    const auto at = scratch.slice_at[c];
    const auto count = scratch.slice_n[c];
    std::size_t keep = 0;
    const int cap = max_neighbors > 0 ? max_neighbors : static_cast<int>(count);
    for (std::uint32_t i = 0; i < count; ++i) {
      const double d2 = part[at + i].dist2;
      if (d2 < 1e-16)
        continue;
      if (!(d2 < cut2))
        break;
      ++keep;
      if (keep >= static_cast<std::size_t>(cap))
        break;
    }
    return keep;
  };
  run_ranges(workers, n, [&](std::size_t begin, std::size_t end) {
    for (std::size_t c = begin; c < end; ++c)
      session.bucket[c + 1] = keep_of(c);
  });
  for (std::size_t i = 1; i <= n; ++i)
    session.bucket[i] += session.bucket[i - 1];
  const std::size_t total = session.bucket[n];
  auto &out = session.edges;
  out.neighbor.clear();
  out.center.clear();
  out.offsets.clear();
  resize_no_init(out.neighbor, total);
  resize_no_init(out.center, total);
  resize_no_init(out.offsets, total * 3);
  run_ranges(workers, n, [&](std::size_t begin, std::size_t end) {
    for (std::size_t c = begin; c < end; ++c) {
      const auto &part = scratch.parts[scratch.slice_thread[c]];
      const auto at = scratch.slice_at[c];
      const auto count = scratch.slice_n[c];
      std::size_t slot = session.bucket[c];
      const std::size_t stop = session.bucket[c + 1];
      for (std::uint32_t i = 0; i < count && slot < stop; ++i) {
        const auto &row = part[at + i];
        if (row.dist2 < 1e-16)
          continue;
        out.neighbor[slot] = static_cast<int64_t>(row.neighbor);
        out.center[slot] = static_cast<int64_t>(c);
        out.offsets[3 * slot] = static_cast<float>(row.s0);
        out.offsets[3 * slot + 1] = static_cast<float>(row.s1);
        out.offsets[3 * slot + 2] = static_cast<float>(row.s2);
        ++slot;
      }
    }
  });
}

void periodic_edges(const ForceInput &in, double cutoff, int max_neighbors,
                     UmaEdgeSession &session) {
  VesinOptions options{};
  vesin_compat::fill_neighbor_options(options, cutoff, /*full_list=*/true);
  vesin_compat::set_n_threads(options, session.threads > 0 ? session.threads : 1);
  const bool brute = brute_force_box(in.box, cutoff);
  if (brute)
    options.algorithm = VesinBruteForce;
  else
    options.algorithm = VesinCellList;
  // The direct pair loop matches the cell list only when every side
  // is at least two cutoffs and under three. A fresh session builds
  // at the bare cutoff. The skin turns on after that call finishes.
  double skin = 0.0;
  if (!brute && session.skin_policy != 1) {
    if (session.skin_policy == 2 || session.calls > 0)
      skin = kVerletSkin;
  }
  vesin_compat::set_skin(options, skin);
  session.last_skin = skin;
  options.return_distances = false;
  options.return_vectors = false;

  const double search = cutoff + skin;
  if (!brute) {
    if (!session.cell_cache)
      session.cell_cache = take_cell();
    auto &scratch = session.cell_cache->scratch;
    const double half = 0.5 * skin;
    bool box_same = scratch.ref_n == static_cast<int>(in.nAtoms);
    for (int a = 0; box_same && a < 9; ++a)
      box_same = scratch.ref_box[a] == in.box[a];
    const bool keep = skin > 0.0 && box_same && scratch.ref_search == search &&
                      linked_cell::max_move(in.pos, scratch.ref_pos) <= half * half;
    bool ready = keep;
    if (!keep &&
        linked_cell::build(in.pos, in.nAtoms, in.box, search, session.threads, scratch)) {
      grow_assign_range(scratch.ref_pos, in.pos, in.pos + 3 * in.nAtoms);
      for (int a = 0; a < 9; ++a)
        scratch.ref_box[a] = in.box[a];
      scratch.ref_n = static_cast<int>(in.nAtoms);
      scratch.ref_search = search;
      ready = true;
    }
    if (ready) {
      session.list_ptr = scratch.stable;
      session.list_len = scratch.stable_n;
      emit_slices(scratch, in.pos, in.box, in.nAtoms, cutoff * cutoff, max_neighbors,
                  session.threads, /*recompute=*/keep, session);
      return;
    }
  }

  if (!session.held_live) {
    session.held = VesinNeighborList{};
    session.held_live = true;
  }
  const char *err = nullptr;
  bool pbc[3] = {true, true, true};
  const int status = call_vesin(
      reinterpret_cast<const double(*)[3]>(in.pos), in.nAtoms,
      reinterpret_cast<const double(*)[3]>(in.box), pbc,
      vesin_compat::make_cpu_device(), options, &session.held, &err);
  if (status != EXIT_SUCCESS) {
    free_vesin(&session.held);
    session.held = VesinNeighborList{};
    session.held_live = false;
    throw EdgeError(err ? "UmaPot: vesin_neighbors failed: vesin"
                        : "UmaPot: vesin_neighbors failed");
  }

  emit_sorted(session.held, in, max_neighbors, session.threads, session);
}

double edge_span(const ForceInput &in, int64_t neighbor, int64_t center, double s0,
                 double s1, double s2) noexcept {
  const double *box = in.box;
  const double dx = in.pos[3 * neighbor] - in.pos[3 * center] + s0 * box[0] +
                    s1 * box[3] + s2 * box[6];
  const double dy = in.pos[3 * neighbor + 1] - in.pos[3 * center + 1] + s0 * box[1] +
                    s1 * box[4] + s2 * box[7];
  const double dz = in.pos[3 * neighbor + 2] - in.pos[3 * center + 2] + s0 * box[2] +
                    s1 * box[5] + s2 * box[8];
  return std::sqrt(dx * dx + dy * dy + dz * dz);
}

UmaEdgeList fit_edge_count(const ForceInput &in, UmaEdgeList edges, int64_t want,
                           double range) {
  const int64_t have = static_cast<int64_t>(edges.neighbor.size());
  if (want <= 0 || have == want)
    return edges;
  if (have > want) {
    std::vector<double> dist;
    std::vector<std::size_t> order;
    grow_resize(dist, static_cast<std::size_t>(have));
    grow_resize(order, static_cast<std::size_t>(have));
    for (int64_t e = 0; e < have; ++e) {
      dist[static_cast<std::size_t>(e)] =
          edge_span(in, edges.neighbor[static_cast<std::size_t>(e)],
                    edges.center[static_cast<std::size_t>(e)],
                    edges.offsets[static_cast<std::size_t>(3 * e)],
                    edges.offsets[static_cast<std::size_t>(3 * e + 1)],
                    edges.offsets[static_cast<std::size_t>(3 * e + 2)]);
      order[static_cast<std::size_t>(e)] = static_cast<std::size_t>(e);
    }
    grow_sort(order.begin(), order.end(),
              [&](std::size_t a, std::size_t b) { return dist[a] > dist[b]; });
    std::vector<char> drop;
    grow_assign_n(drop, static_cast<std::size_t>(have), char{0});
    int64_t kept = have;
    for (std::size_t idx : order) {
      if (kept <= want)
        break;
      if (dist[idx] >= range) {
        drop[idx] = 1;
        --kept;
      }
    }
    UmaEdgeList trimmed;
    grow_reserve(trimmed.neighbor, static_cast<std::size_t>(kept));
    grow_reserve(trimmed.center, static_cast<std::size_t>(kept));
    grow_reserve(trimmed.offsets, static_cast<std::size_t>(kept) * 3);
    for (int64_t e = 0; e < have; ++e) {
      if (drop[static_cast<std::size_t>(e)])
        continue;
      grow_push(trimmed.neighbor, edges.neighbor[static_cast<std::size_t>(e)]);
      grow_push(trimmed.center, edges.center[static_cast<std::size_t>(e)]);
      grow_push(trimmed.offsets, edges.offsets[static_cast<std::size_t>(3 * e)]);
      grow_push(trimmed.offsets, edges.offsets[static_cast<std::size_t>(3 * e + 1)]);
      grow_push(trimmed.offsets, edges.offsets[static_cast<std::size_t>(3 * e + 2)]);
    }
    return trimmed;
  }
  if (in.nAtoms < 2)
    throw EdgeError("UmaPot: static edge count needs at least two atoms to pad");
  const double pair = edge_span(in, 1, 0, 0.0, 0.0, 0.0);
  const int column[3] = {0, 3, 6};
  int axis = -1;
  double longest = 0.0;
  for (int a = 0; a < 3; ++a) {
    const double x = in.box[column[a]];
    const double y = in.box[column[a] + 1];
    const double z = in.box[column[a] + 2];
    const double len = std::sqrt(x * x + y * y + z * z);
    if (len > longest) {
      longest = len;
      axis = a;
    }
  }
  if (!(longest > 0.0))
    throw EdgeError("UmaPot: cannot pad an edge in a zero cell");
  int shift = static_cast<int>(std::ceil((range + pair) / longest));
  // The ceiling of a positive ratio is at least 1.
  auto span_at = [&](int s) {
    const double v = static_cast<double>(s);
    if (axis == 0)
      return edge_span(in, 1, 0, v, 0.0, 0.0);
    if (axis == 1)
      return edge_span(in, 1, 0, 0.0, v, 0.0);
    return edge_span(in, 1, 0, 0.0, 0.0, v);
  };
  if (!(span_at(shift) > range))
    ++shift;
  // One more cell along a positive axis leaves the range.
  constexpr int kMaxExactFloatShift = 1 << 24;
  if (shift > kMaxExactFloatShift)
    throw EdgeError("UmaPot: padded image shift is not an exact float32 integer");
  const float s0 = axis == 0 ? static_cast<float>(shift) : 0.f;
  const float s1 = axis == 1 ? static_cast<float>(shift) : 0.f;
  const float s2 = axis == 2 ? static_cast<float>(shift) : 0.f;
  grow_reserve(edges.neighbor, static_cast<size_t>(want));
  grow_reserve(edges.center, static_cast<size_t>(want));
  grow_reserve(edges.offsets, static_cast<size_t>(want) * 3);
  while (static_cast<int64_t>(edges.neighbor.size()) < want) {
    grow_push(edges.neighbor, static_cast<int64_t>(1));
    grow_push(edges.center, static_cast<int64_t>(0));
    grow_push(edges.offsets, s0);
    grow_push(edges.offsets, s1);
    grow_push(edges.offsets, s2);
  }
  return edges;
}

int64_t static_edges_per_system(const UmaContract &contract, int64_t n_systems) {
  if (!contract.nedges_min || !contract.nedges_max ||
      *contract.nedges_min != *contract.nedges_max || n_systems < 1)
    return 0;
  const int64_t total = *contract.nedges_max;
  if (n_systems == 1)
    return total;
  if (total % n_systems != 0)
    return total;
  return total / n_systems;
}

int call_vesin(const double (*points)[3], std::size_t n_points, const double (*box)[3],
               bool pbc[3], VesinDevice device, const VesinOptions &options,
               VesinNeighborList *neighbors, const char **error) noexcept {
  return vesin_neighbors(points, n_points, box, pbc, device, options, neighbors, error);
}

void free_vesin(VesinNeighborList *neighbors) noexcept { vesin_free(neighbors); }

} // namespace

void exercise_empty_ranges(int threads) { exercise_empty_ranges_impl(threads); }

// glibc hands multi-megabyte buffers back to the kernel. The next
// one-shot call would fault them again. A short recycle list keeps the
// pages, and drops the Verlet keys so a new session still rebuilds.
struct EdgePool {
  std::mutex mu;
  std::vector<UmaEdgeList> lists;
  std::vector<std::unique_ptr<UmaEdgeSession::CellCache>> cells;
};

static EdgePool g_edge_pool;

EdgePool &edge_pool() { return g_edge_pool; }

UmaEdgeList take_edges() {
  auto &pool = edge_pool();
  QuietLock lock(pool.mu);
  if (pool.lists.empty())
    return {};
  UmaEdgeList out = std::move(pool.lists.back());
  pool.lists.pop_back();
  return out;
}

void give_edges(UmaEdgeList edges) {
  if (edges.neighbor.capacity() < 4096)
    return;
  auto &pool = edge_pool();
  QuietLock lock(pool.mu);
  if (pool.lists.size() >= 4)
    return;
  grow_push(pool.lists, std::move(edges));
}

std::unique_ptr<UmaEdgeSession::CellCache> take_cell() {
  auto &pool = edge_pool();
  QuietLock lock(pool.mu);
  if (pool.cells.empty()) {
    return std::make_unique<UmaEdgeSession::CellCache>();
  }
  auto cell = std::move(pool.cells.back());
  pool.cells.pop_back();
  auto &scratch = cell->scratch;
  scratch.ref_n = -1;
  scratch.ref_search = -1;
  scratch.stable = nullptr;
  scratch.stable_n = 0;
  scratch.ref_pos.clear();
  return cell;
}

void give_cell(std::unique_ptr<UmaEdgeSession::CellCache> cell) {
  if (!cell)
    return;
  auto &pool = edge_pool();
  QuietLock lock(pool.mu);
  if (pool.cells.size() >= 4)
    return;
  grow_push(pool.cells, std::move(cell));
}

UmaEdgeList fit_recorded_edges(const ForceInput &in, UmaEdgeList edges, int64_t want,
                               double range) {
  return fit_edge_count(in, std::move(edges), want, range);
}

UmaEdgeSession::UmaEdgeSession() : edges(take_edges()) {}

UmaEdgeSession::~UmaEdgeSession() {
  if (held_live)
    vesin_free(&held);
  give_cell(std::move(cell_cache));
  give_edges(std::move(edges));
}

const UmaEdgeList &build_system_edges(const ForceInput &local, double cutoff,
                                      double model_cutoff, double molecular_box,
                                      int max_neighbors, const UmaContract &contract,
                                      int64_t n_systems, UmaEdgeSession &session) {
  QuietLock lock(session.mutex);
  const int64_t n = static_cast<int64_t>(local.nAtoms);
  const int64_t complete = n > 1 ? n * (n - 1) : 0;
  const int64_t want = static_edges_per_system(contract, n_systems);
  const bool molecular = molecular_box > 0.0;
  if (complete > 0 && (want == complete || (molecular && want == 0))) {
    session.edges = molecular_edges(local.pos, local.nAtoms, session);
  } else {
    const double range = (model_cutoff > 0.0 && std::isfinite(model_cutoff))
                             ? model_cutoff
                             : cutoff;
    periodic_edges(local, range, max_neighbors, session);
    if (want > 0 && static_cast<int64_t>(session.edges.neighbor.size()) != want)
      session.edges = fit_edge_count(local, std::move(session.edges), want, range);
  }
  session.calls += 1;
  return session.edges;
}

} // namespace rgpot