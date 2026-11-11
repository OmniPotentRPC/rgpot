// Copyright (c) rgpot developers
//
// The optimized edge builders must match a single-thread reference.
// A Verlet step must match a full rebuild of the same coordinates.

#include "rgpot/MetatomicPot/vesin_compat.hpp"
#include "rgpot/UmaPot/UmaEdges.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

int g_fails = 0;

void expect(bool ok, const char *what) {
  if (ok)
    return;
  std::fprintf(stderr, "FAIL %s\n", what);
  ++g_fails;
}

bool same_edges(const rgpot::UmaEdgeList &a, const rgpot::UmaEdgeList &b) {
  return a.neighbor == b.neighbor && a.center == b.center &&
         a.offsets == b.offsets;
}

std::vector<double> lattice(int n_side, double spacing) {
  const int n = n_side * n_side * n_side;
  std::vector<double> pos(static_cast<std::size_t>(n) * 3);
  int k = 0;
  for (int z = 0; z < n_side; ++z) {
    for (int y = 0; y < n_side; ++y) {
      for (int x = 0; x < n_side; ++x, ++k) {
        pos[static_cast<std::size_t>(3 * k)] = x * spacing;
        pos[static_cast<std::size_t>(3 * k + 1)] = y * spacing;
        pos[static_cast<std::size_t>(3 * k + 2)] = z * spacing;
      }
    }
  }
  return pos;
}

rgpot::UmaEdgeList reference_molecular(const double *pos, std::size_t n) {
  struct Row {
    int64_t neighbor;
    int64_t center;
    double dist;
  };
  std::vector<Row> keep;
  keep.reserve(n * (n - 1));
  for (std::size_t c = 0; c < n; ++c) {
    for (std::size_t j = 0; j < n; ++j) {
      if (c == j)
        continue;
      const double dx = pos[3 * j] - pos[3 * c];
      const double dy = pos[3 * j + 1] - pos[3 * c + 1];
      const double dz = pos[3 * j + 2] - pos[3 * c + 2];
      keep.push_back({static_cast<int64_t>(j), static_cast<int64_t>(c),
                      std::sqrt(dx * dx + dy * dy + dz * dz)});
    }
  }
  std::sort(keep.begin(), keep.end(), [](const Row &a, const Row &b) {
    if (a.center != b.center)
      return a.center < b.center;
    if (a.dist != b.dist)
      return a.dist < b.dist;
    return a.neighbor < b.neighbor;
  });
  rgpot::UmaEdgeList out;
  for (const auto &row : keep) {
    out.neighbor.push_back(row.neighbor);
    out.center.push_back(row.center);
    out.offsets.push_back(0.f);
    out.offsets.push_back(0.f);
    out.offsets.push_back(0.f);
  }
  return out;
}

rgpot::UmaEdgeList reference_periodic(const rgpot::ForceInput &in, double cutoff,
                                      int max_neighbors) {
  VesinOptions options{};
  rgpot::vesin_compat::fill_neighbor_options(options, cutoff, true);
  rgpot::vesin_compat::set_n_threads(options, 1);
  options.return_distances = true;
  VesinNeighborList nl{};
  const char *err = nullptr;
  bool pbc[3] = {true, true, true};
  const int status = vesin_neighbors(
      reinterpret_cast<const double(*)[3]>(in.pos), in.nAtoms,
      reinterpret_cast<const double(*)[3]>(in.box), pbc,
      rgpot::vesin_compat::make_cpu_device(), options, &nl, &err);
  if (status != 0) {
    std::fprintf(stderr, "FAIL reference vesin\n");
    vesin_free(&nl);
    ++g_fails;
    return {};
  }
  struct Cand {
    int64_t n;
    int64_t c;
    float s0, s1, s2;
    double dist;
  };
  std::vector<Cand> keep;
  keep.reserve(nl.length);
  for (std::size_t k = 0; k < nl.length; ++k) {
    const double dist = nl.distances[k];
    if (dist < 1e-8)
      continue;
    keep.push_back({static_cast<int64_t>(nl.pairs[k][1]),
                    static_cast<int64_t>(nl.pairs[k][0]),
                    static_cast<float>(nl.shifts[k][0]),
                    static_cast<float>(nl.shifts[k][1]),
                    static_cast<float>(nl.shifts[k][2]), dist});
  }
  vesin_free(&nl);
  std::sort(keep.begin(), keep.end(), [](const Cand &a, const Cand &b) {
    if (a.c != b.c)
      return a.c < b.c;
    if (a.dist != b.dist)
      return a.dist < b.dist;
    if (a.n != b.n)
      return a.n < b.n;
    if (a.s0 != b.s0)
      return a.s0 < b.s0;
    if (a.s1 != b.s1)
      return a.s1 < b.s1;
    return a.s2 < b.s2;
  });
  if (max_neighbors > 0) {
    std::vector<Cand> trimmed;
    int64_t center = -1;
    int count = 0;
    for (const auto &c : keep) {
      if (c.c != center) {
        center = c.c;
        count = 0;
      }
      if (count < max_neighbors) {
        trimmed.push_back(c);
        ++count;
      }
    }
    keep.swap(trimmed);
  }
  rgpot::UmaEdgeList out;
  for (const auto &c : keep) {
    out.neighbor.push_back(c.n);
    out.center.push_back(c.c);
    out.offsets.push_back(c.s0);
    out.offsets.push_back(c.s1);
    out.offsets.push_back(c.s2);
  }
  return out;
}

std::vector<double> forces_of(const rgpot::ForceInput &in,
                              const rgpot::UmaEdgeList &edges) {
  std::vector<double> f(in.nAtoms * 3, 0.0);
  const double *box = in.box;
  for (std::size_t k = 0; k < edges.neighbor.size(); ++k) {
    const int64_t j = edges.neighbor[k];
    const int64_t i = edges.center[k];
    const double s0 = edges.offsets[3 * k];
    const double s1 = edges.offsets[3 * k + 1];
    const double s2 = edges.offsets[3 * k + 2];
    const double dx = in.pos[3 * j] - in.pos[3 * i] + s0 * box[0] +
                      s1 * box[3] + s2 * box[6];
    const double dy = in.pos[3 * j + 1] - in.pos[3 * i + 1] + s0 * box[1] +
                      s1 * box[4] + s2 * box[7];
    const double dz = in.pos[3 * j + 2] - in.pos[3 * i + 2] + s0 * box[2] +
                      s1 * box[5] + s2 * box[8];
    const double r2 = dx * dx + dy * dy + dz * dz;
    if (r2 < 1e-12)
      continue;
    const double inv = 1.0 / (r2 * std::sqrt(r2));
    f[static_cast<std::size_t>(3 * i)] += dx * inv;
    f[static_cast<std::size_t>(3 * i + 1)] += dy * inv;
    f[static_cast<std::size_t>(3 * i + 2)] += dz * inv;
  }
  return f;
}

bool forces_match(const std::vector<double> &a, const std::vector<double> &b) {
  if (a.size() != b.size())
    return false;
  for (std::size_t i = 0; i < a.size(); ++i) {
    if (std::abs(a[i] - b[i]) > 1e-9)
      return false;
  }
  return true;
}

rgpot::UmaEdgeList call_edges(const rgpot::ForceInput &in, double cutoff,
                              double molecular_box, int max_neighbors,
                              int threads, rgpot::UmaEdgeSession &session) {
  session.threads = threads;
  rgpot::UmaContract contract;
  return rgpot::build_system_edges(in, cutoff, cutoff, molecular_box,
                                   max_neighbors, contract, 1, session);
}

void check_molecular() {
  const auto pos = lattice(4, 1.7);
  const std::size_t n = pos.size() / 3;
  double box[9] = {8, 0, 0, 0, 8, 0, 0, 0, 8};
  std::vector<int> z(n, 1);
  rgpot::ForceInput in{n, pos.data(), z.data(), box};
  rgpot::UmaEdgeSession one;
  rgpot::UmaEdgeSession many;
  const auto a = call_edges(in, 6.0, 8.0, 300, 1, one);
  const auto b = call_edges(in, 6.0, 8.0, 300, 4, many);
  const auto ref = reference_molecular(pos.data(), n);
  expect(a.neighbor.size() == n * (n - 1), "molecular edge count");
  expect(same_edges(a, ref), "molecular matches the row sort");
  expect(same_edges(a, b), "molecular threads share one edge list");
}

void check_periodic_threads() {
  const auto pos = lattice(6, 2.2);
  const std::size_t n = pos.size() / 3;
  const double side = 6 * 2.2;
  double box[9] = {side, 0, 0, 0, side, 0, 0, 0, side};
  std::vector<int> z(n, 6);
  rgpot::ForceInput in{n, pos.data(), z.data(), box};
  rgpot::UmaEdgeSession one;
  rgpot::UmaEdgeSession many;
  const auto a = call_edges(in, 6.0, 0.0, 300, 1, one);
  const auto b = call_edges(in, 6.0, 0.0, 300, 8, many);
  const auto ref = reference_periodic(in, 6.0, 300);
  expect(same_edges(a, ref), "periodic matches the single thread reference");
  expect(same_edges(a, b), "periodic threads share one edge list");
  rgpot::UmaEdgeSession again;
  const auto c = call_edges(in, 6.0, 0.0, 300, 8, again);
  expect(same_edges(b, c), "periodic output is deterministic");
}

void check_trim() {
  const auto pos = lattice(5, 1.5);
  const std::size_t n = pos.size() / 3;
  const double side = 5 * 1.5;
  double box[9] = {side, 0, 0, 0, side, 0, 0, 0, side};
  std::vector<int> z(n, 1);
  rgpot::ForceInput in{n, pos.data(), z.data(), box};
  rgpot::UmaEdgeSession session;
  const auto got = call_edges(in, 6.0, 0.0, 4, 1, session);
  const auto ref = reference_periodic(in, 6.0, 4);
  expect(same_edges(got, ref), "trimmed list matches the linear pass");
  int run = 0;
  int64_t center = -1;
  for (int64_t c : got.center) {
    if (c != center) {
      center = c;
      run = 0;
    }
    ++run;
    if (run > 4) {
      expect(false, "max neighbors holds");
      return;
    }
  }
  expect(true, "max neighbors holds");
}

void shift_all(std::vector<double> &pos, double delta) {
  for (std::size_t i = 0; i < pos.size(); i += 3)
    pos[i] += delta;
}

void check_verlet_forces() {
  // Wider than three cutoffs, so the periodic path keeps a Verlet list.
  const auto base = lattice(8, 3.0);
  const std::size_t n = base.size() / 3;
  const double side = 8 * 3.0;
  double box[9] = {side, 0, 0, 0, side, 0, 0, 0, side};
  std::vector<int> z(n, 1);
  auto pos = base;
  rgpot::UmaEdgeSession kept;
  auto step = [&](std::vector<double> &p, rgpot::UmaEdgeSession &session) {
    rgpot::ForceInput in{n, p.data(), z.data(), box};
    return call_edges(in, 6.0, 0.0, 300, 1, session);
  };
  auto fresh_forces = [&](std::vector<double> &p) {
    rgpot::UmaEdgeSession session;
    rgpot::ForceInput in{n, p.data(), z.data(), box};
    return forces_of(in, step(p, session));
  };
  {
    rgpot::ForceInput in{n, pos.data(), z.data(), box};
    const auto kept_edges = step(pos, kept);
    expect(forces_match(forces_of(in, kept_edges), fresh_forces(pos)),
           "first step matches a full rebuild");
  }
  shift_all(pos, 0.05);
  {
    rgpot::ForceInput in{n, pos.data(), z.data(), box};
    const auto kept_edges = step(pos, kept);
    expect(forces_match(forces_of(in, kept_edges), fresh_forces(pos)),
           "small step matches a full rebuild");
  }
  shift_all(pos, 2.0);
  {
    rgpot::ForceInput in{n, pos.data(), z.data(), box};
    const auto kept_edges = step(pos, kept);
    expect(forces_match(forces_of(in, kept_edges), fresh_forces(pos)),
           "large step matches a full rebuild");
  }
}

} // namespace

int main() {
  check_molecular();
  check_periodic_threads();
  check_trim();
  check_verlet_forces();
  if (g_fails != 0) {
    std::fprintf(stderr, "FAIL checks=%d\n", g_fails);
    return 1;
  }
  std::printf("PASS uma edge identity\n");
  return 0;
}
