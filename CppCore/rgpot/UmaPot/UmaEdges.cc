// Copyright (c) rgpot developers

#include "rgpot/UmaPot/UmaEdges.hpp"

#include "rgpot/MetatomicPot/vesin_compat.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace rgpot {
namespace {

// Vesin rebuilds the cached list after an atom moves more than half of this.
constexpr double kVerletSkin = 1.0;

struct Cand {
  int64_t n{0};
  int64_t c{0};
  float s0{0};
  float s1{0};
  float s2{0};
  double dist{0};
};

bool closer(const Cand &a, const Cand &b) {
  if (a.dist != b.dist)
    return a.dist < b.dist;
  if (a.n != b.n)
    return a.n < b.n;
  if (a.s0 != b.s0)
    return a.s0 < b.s0;
  if (a.s1 != b.s1)
    return a.s1 < b.s1;
  return a.s2 < b.s2;
}

double axis_length(const double *box, int axis) {
  const double x = box[3 * axis];
  const double y = box[3 * axis + 1];
  const double z = box[3 * axis + 2];
  return std::sqrt(x * x + y * y + z * z);
}

// A cell list with fewer than three cells along a side walks the same
// pairs as a direct search. Brute force is the minimum image, so it
// matches the cell list only when a second image sits past the cutoff.
bool brute_force_box(const double *box, double cutoff) {
  double lo = 1e300;
  double hi = 0.0;
  for (int axis = 0; axis < 3; ++axis) {
    const double len = axis_length(box, axis);
    lo = std::min(lo, len);
    hi = std::max(hi, len);
  }
  return lo >= 2.0 * cutoff && hi < 3.0 * cutoff;
}

int worker_count(int requested, std::size_t n) {
  int threads = requested > 0 ? requested : 1;
  if (n > 0 && threads > static_cast<int>(n))
    threads = static_cast<int>(n);
  if (threads < 1)
    threads = 1;
  return threads;
}

template <typename Work>
void run_ranges(int threads, std::size_t n, Work work) {
  if (threads <= 1 || n <= 1) {
    work(0, n);
    return;
  }
  std::vector<std::thread> pool;
  pool.reserve(static_cast<std::size_t>(threads));
  for (int t = 0; t < threads; ++t) {
    const std::size_t begin =
        (static_cast<std::size_t>(t) * n) / static_cast<std::size_t>(threads);
    const std::size_t end =
        (static_cast<std::size_t>(t + 1) * n) / static_cast<std::size_t>(threads);
    pool.emplace_back(work, begin, end);
  }
  for (auto &worker : pool)
    worker.join();
}

UmaEdgeList molecular_edges(const double *pos, std::size_t n, UmaEdgeSession &session) {
  session.rows.resize(n);
  const int threads = worker_count(session.threads, n);
  run_ranges(threads, n, [&](std::size_t begin, std::size_t end) {
    for (std::size_t c = begin; c < end; ++c) {
      auto &row = session.rows[c];
      row.neighbor.clear();
      row.dist.clear();
      if (n > 1) {
        row.neighbor.reserve(n - 1);
        row.dist.reserve(n - 1);
      }
      for (std::size_t j = 0; j < n; ++j) {
        if (j == c)
          continue;
        const double dx = pos[3 * j] - pos[3 * c];
        const double dy = pos[3 * j + 1] - pos[3 * c + 1];
        const double dz = pos[3 * j + 2] - pos[3 * c + 2];
        row.neighbor.push_back(static_cast<int64_t>(j));
        row.dist.push_back(std::sqrt(dx * dx + dy * dy + dz * dz));
      }
      row.order.resize(row.neighbor.size());
      std::iota(row.order.begin(), row.order.end(), 0);
      std::sort(row.order.begin(), row.order.end(), [&](std::size_t a, std::size_t b) {
        if (row.dist[a] != row.dist[b])
          return row.dist[a] < row.dist[b];
        return row.neighbor[a] < row.neighbor[b];
      });
    }
  });

  UmaEdgeList out;
  const std::size_t total = n > 1 ? n * (n - 1) : 0;
  out.neighbor.reserve(total);
  out.center.reserve(total);
  out.offsets.reserve(total * 3);
  for (std::size_t c = 0; c < n; ++c) {
    const auto &row = session.rows[c];
    for (std::size_t id : row.order) {
      out.neighbor.push_back(row.neighbor[id]);
      out.center.push_back(static_cast<int64_t>(c));
      out.offsets.push_back(0.f);
      out.offsets.push_back(0.f);
      out.offsets.push_back(0.f);
    }
  }
  return out;
}

UmaEdgeList emit_sorted(std::vector<Cand> keep, std::size_t n_atoms, int max_neighbors, int threads) {
  const std::size_t n = n_atoms;
  std::vector<std::size_t> counts(n + 1, 0);
  for (const auto &cand : keep) {
    const auto center = static_cast<std::size_t>(cand.c);
    if (center < n)
      ++counts[center + 1];
  }
  for (std::size_t i = 1; i <= n; ++i)
    counts[i] += counts[i - 1];
  std::vector<Cand> grouped(counts[n]);
  auto write_idx = counts;
  for (const auto &cand : keep) {
    const auto center = static_cast<std::size_t>(cand.c);
    if (center < n)
      grouped[write_idx[center]++] = cand;
  }

  const int workers = worker_count(threads, n);
  run_ranges(workers, n, [&](std::size_t begin, std::size_t end) {
    for (std::size_t c = begin; c < end; ++c) {
      auto first = grouped.begin() + static_cast<std::ptrdiff_t>(counts[c]);
      auto last = grouped.begin() + static_cast<std::ptrdiff_t>(counts[c + 1]);
      std::sort(first, last, closer);
    }
  });

  UmaEdgeList out;
  out.neighbor.reserve(grouped.size());
  out.center.reserve(grouped.size());
  out.offsets.reserve(grouped.size() * 3);
  for (std::size_t c = 0; c < n; ++c) {
    const std::size_t begin = counts[c];
    std::size_t end = counts[c + 1];
    if (max_neighbors > 0) {
      const std::size_t cap = begin + static_cast<std::size_t>(max_neighbors);
      if (end > cap)
        end = cap;
    }
    for (std::size_t k = begin; k < end; ++k) {
      out.neighbor.push_back(grouped[k].n);
      out.center.push_back(grouped[k].c);
      out.offsets.push_back(grouped[k].s0);
      out.offsets.push_back(grouped[k].s1);
      out.offsets.push_back(grouped[k].s2);
    }
  }
  return out;
}

UmaEdgeList periodic_edges(const ForceInput &in, double cutoff, int max_neighbors,
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
  options.return_distances = true;

  if (!session.held_live) {
    session.held = VesinNeighborList{};
    session.held_live = true;
  }
  const char *err = nullptr;
  bool pbc[3] = {true, true, true};
  const int status = vesin_neighbors(
      reinterpret_cast<const double(*)[3]>(in.pos), in.nAtoms,
      reinterpret_cast<const double(*)[3]>(in.box), pbc,
      vesin_compat::make_cpu_device(), options, &session.held, &err);
  if (status != EXIT_SUCCESS) {
    std::string msg = "UmaPot: vesin_neighbors failed";
    if (err)
      msg += ": " + std::string(err);
    vesin_free(&session.held);
    session.held = VesinNeighborList{};
    session.held_live = false;
    throw std::runtime_error(msg);
  }

  std::vector<Cand> keep;
  keep.reserve(session.held.length);
  const double *box = in.box;
  for (std::size_t k = 0; k < session.held.length; ++k) {
    const int64_t i = static_cast<int64_t>(session.held.pairs[k][0]);
    const int64_t j = static_cast<int64_t>(session.held.pairs[k][1]);
    double dist = 0.0;
    if (session.held.distances != nullptr) {
      dist = session.held.distances[k];
    } else {
      const double sx = static_cast<double>(session.held.shifts[k][0]);
      const double sy = static_cast<double>(session.held.shifts[k][1]);
      const double sz = static_cast<double>(session.held.shifts[k][2]);
      const double dx = in.pos[3 * j] - in.pos[3 * i] + sx * box[0] + sy * box[3] +
                        sz * box[6];
      const double dy = in.pos[3 * j + 1] - in.pos[3 * i + 1] + sx * box[1] +
                        sy * box[4] + sz * box[7];
      const double dz = in.pos[3 * j + 2] - in.pos[3 * i + 2] + sx * box[2] +
                        sy * box[5] + sz * box[8];
      dist = std::sqrt(dx * dx + dy * dy + dz * dz);
    }
    if (dist < 1e-8)
      continue;
    Cand cand;
    cand.n = j;
    cand.c = i;
    cand.s0 = static_cast<float>(session.held.shifts[k][0]);
    cand.s1 = static_cast<float>(session.held.shifts[k][1]);
    cand.s2 = static_cast<float>(session.held.shifts[k][2]);
    cand.dist = dist;
    keep.push_back(cand);
  }
  return emit_sorted(std::move(keep), in.nAtoms, max_neighbors, session.threads);
}

double edge_span(const ForceInput &in, int64_t neighbor, int64_t center, double s0,
                 double s1, double s2) {
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
    std::vector<double> dist(static_cast<std::size_t>(have));
    std::vector<std::size_t> order(static_cast<std::size_t>(have));
    for (int64_t e = 0; e < have; ++e) {
      dist[static_cast<std::size_t>(e)] =
          edge_span(in, edges.neighbor[static_cast<std::size_t>(e)],
                    edges.center[static_cast<std::size_t>(e)],
                    edges.offsets[static_cast<std::size_t>(3 * e)],
                    edges.offsets[static_cast<std::size_t>(3 * e + 1)],
                    edges.offsets[static_cast<std::size_t>(3 * e + 2)]);
      order[static_cast<std::size_t>(e)] = static_cast<std::size_t>(e);
    }
    std::sort(order.begin(), order.end(),
              [&](std::size_t a, std::size_t b) { return dist[a] > dist[b]; });
    std::vector<char> drop(static_cast<std::size_t>(have), 0);
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
    trimmed.neighbor.reserve(static_cast<std::size_t>(kept));
    trimmed.center.reserve(static_cast<std::size_t>(kept));
    trimmed.offsets.reserve(static_cast<std::size_t>(kept) * 3);
    for (int64_t e = 0; e < have; ++e) {
      if (drop[static_cast<std::size_t>(e)])
        continue;
      trimmed.neighbor.push_back(edges.neighbor[static_cast<std::size_t>(e)]);
      trimmed.center.push_back(edges.center[static_cast<std::size_t>(e)]);
      trimmed.offsets.push_back(edges.offsets[static_cast<std::size_t>(3 * e)]);
      trimmed.offsets.push_back(edges.offsets[static_cast<std::size_t>(3 * e + 1)]);
      trimmed.offsets.push_back(edges.offsets[static_cast<std::size_t>(3 * e + 2)]);
    }
    return trimmed;
  }
  if (in.nAtoms < 2)
    throw std::runtime_error("UmaPot: static edge count needs at least two atoms to pad");
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
    throw std::runtime_error("UmaPot: cannot pad an edge in a zero cell");
  int shift = static_cast<int>(std::ceil((range + pair) / longest));
  if (shift < 1)
    shift = 1;
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
  if (!(span_at(shift) > range))
    throw std::runtime_error("UmaPot: cannot pad an edge past the model cutoff");
  constexpr int kMaxExactFloatShift = 1 << 24;
  if (shift > kMaxExactFloatShift)
    throw std::runtime_error("UmaPot: padded image shift is not an exact float32 integer");
  const float s0 = axis == 0 ? static_cast<float>(shift) : 0.f;
  const float s1 = axis == 1 ? static_cast<float>(shift) : 0.f;
  const float s2 = axis == 2 ? static_cast<float>(shift) : 0.f;
  edges.neighbor.reserve(static_cast<size_t>(want));
  edges.center.reserve(static_cast<size_t>(want));
  edges.offsets.reserve(static_cast<size_t>(want) * 3);
  while (static_cast<int64_t>(edges.neighbor.size()) < want) {
    edges.neighbor.push_back(1);
    edges.center.push_back(0);
    edges.offsets.push_back(s0);
    edges.offsets.push_back(s1);
    edges.offsets.push_back(s2);
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

} // namespace

UmaEdgeSession::UmaEdgeSession() = default;

UmaEdgeSession::~UmaEdgeSession() {
  if (held_live)
    vesin_free(&held);
}

UmaEdgeList build_system_edges(const ForceInput &local, double cutoff,
                               double model_cutoff, double molecular_box,
                               int max_neighbors, const UmaContract &contract,
                               int64_t n_systems, UmaEdgeSession &session) {
  std::lock_guard<std::mutex> lock(session.mutex);
  const int64_t n = static_cast<int64_t>(local.nAtoms);
  const int64_t complete = n > 1 ? n * (n - 1) : 0;
  const int64_t want = static_edges_per_system(contract, n_systems);
  const bool molecular = molecular_box > 0.0;
  UmaEdgeList edges;
  if (complete > 0 && (want == complete || (molecular && want == 0))) {
    edges = molecular_edges(local.pos, local.nAtoms, session);
  } else {
    const double range = (model_cutoff > 0.0 && std::isfinite(model_cutoff))
                             ? model_cutoff
                             : cutoff;
    edges = periodic_edges(local, range, max_neighbors, session);
    if (want > 0 && static_cast<int64_t>(edges.neighbor.size()) != want)
      edges = fit_edge_count(local, std::move(edges), want, range);
  }
  session.calls += 1;
  return edges;
}

} // namespace rgpot