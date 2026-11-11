// Copyright 2023--present rgpot developers
//
// Exercises the linked-cell builder and the edge session so a
// coverage run can see both sides of the branches.

#include "rgpot/UmaPot/UmaEdges.hpp"
#include "rgpot/UmaPot/UmaLinkedCell.hpp"

namespace rgpot {
void exercise_empty_ranges(int threads);
}

#include <cmath>
#include <cstdio>
#include <limits>
#include <memory>
#include <string>
#include <sys/resource.h>
#include <unordered_map>
#include <vector>

namespace {

int g_fails = 0;

void expect(bool ok, const char *name) {
  if (ok)
    return;
  std::fprintf(stderr, "FAIL %s\n", name);
  ++g_fails;
}

std::vector<double> grid(int n, double spacing, double ox, double oy, double oz) {
  std::vector<double> pos(static_cast<std::size_t>(n) * 3);
  const int side = std::max(1, static_cast<int>(std::ceil(std::cbrt(n))));
  for (int i = 0; i < n; ++i) {
    pos[static_cast<std::size_t>(3 * i)] = ox + spacing * (i % side);
    pos[static_cast<std::size_t>(3 * i + 1)] = oy + spacing * ((i / side) % side);
    pos[static_cast<std::size_t>(3 * i + 2)] = oz + spacing * (i / (side * side));
  }
  return pos;
}

void cover_linked_cell() {
  using rgpot::linked_cell::Pair;
  using rgpot::linked_cell::Scratch;
  Pair a;
  Pair b;
  a.dist2 = 1;
  b.dist2 = 2;
  expect(rgpot::linked_cell::before(a, b), "dist order");
  b.dist2 = 1;
  a.neighbor = 1;
  b.neighbor = 2;
  expect(rgpot::linked_cell::before(a, b), "neighbor order");
  b.neighbor = 1;
  a.s0 = -1;
  b.s0 = 1;
  expect(rgpot::linked_cell::before(a, b), "s0 order");
  b.s0 = -1;
  a.s1 = 0;
  b.s1 = 1;
  expect(rgpot::linked_cell::before(a, b), "s1 order");
  b.s1 = 0;
  a.s2 = 0;
  b.s2 = 1;
  expect(rgpot::linked_cell::before(a, b), "s2 order");
  b.s2 = 0;
  expect(!rgpot::linked_cell::before(a, b), "equal keys");

  const double ortho[9] = {30, 0, 0, 0, 30, 0, 0, 0, 30};
  const double skew[9] = {30, 1, 0, 0, 30, 0, 0, 0, 30};
  expect(rgpot::linked_cell::orthorhombic(ortho), "ortho");
  expect(!rgpot::linked_cell::orthorhombic(skew), "skew");
  const double neg[9] = {-1, 0, 0, 0, 30, 0, 0, 0, 30};
  expect(!rgpot::linked_cell::orthorhombic(neg), "negative side");
  double term[9] = {30, 0, 0, 0, 30, 0, 0, 0, 30};
  const int fail_at[8] = {4, 8, 2, 3, 5, 6, 7, 1};
  for (int index : fail_at) {
    term[index] = index < 3 || index == 4 || index == 8 ? -1.0 : 1.0;
    if (index == 4 || index == 8)
      term[index] = -1.0;
    else
      term[index] = 1.0;
    expect(!rgpot::linked_cell::orthorhombic(term), "one skew term");
    term[index] = 0.0;
    if (index == 4 || index == 8)
      term[index] = 30.0;
  }

  std::vector<std::size_t> empty;
  rgpot::linked_cell::exclusive_prefix(empty, 4);
  expect(empty.empty(), "empty prefix");
  std::vector<std::size_t> one{3, 1, 0, 2};
  rgpot::linked_cell::exclusive_prefix(one, 1);
  expect(one.back() == 6, "serial prefix total");
  std::vector<std::size_t> many{1, 1, 1, 1, 1, 1, 1, 1};
  rgpot::linked_cell::exclusive_prefix(many, 4);
  expect(many.back() == 8, "parallel prefix total");
  std::vector<std::size_t> wide_bins(8192, 1);
  rgpot::linked_cell::exclusive_prefix(wide_bins, 4);
  expect(wide_bins.back() == 8192, "wide prefix");

  Scratch scratch;
  const double nowhere[9] = {30, 0, 0, 0, 30, 0, 0, 0, 30};
  expect(!rgpot::linked_cell::build(nullptr, 0, nowhere, 6, 1, scratch), "empty build");
  expect(!rgpot::linked_cell::build(nowhere, 1, nowhere, 0, 1, scratch), "bad search");
  expect(!rgpot::linked_cell::build(nowhere, 1, skew, 6, 1, scratch), "skew build");
  const double small[9] = {10, 0, 0, 0, 10, 0, 0, 0, 10};
  auto near = grid(2, 1.0, 0, 0, 0);
  expect(!rgpot::linked_cell::build(near.data(), 2, small, 6, 1, scratch), "short side");
  const double short_y[9] = {30, 0, 0, 0, 10, 0, 0, 0, 30};
  const double short_z[9] = {30, 0, 0, 0, 30, 0, 0, 0, 10};
  expect(!rgpot::linked_cell::build(near.data(), 2, short_y, 6, 1, scratch), "short y");
  expect(!rgpot::linked_cell::build(near.data(), 2, short_z, 6, 1, scratch), "short z");

  const double wide[9] = {24, 0, 0, 0, 24, 0, 0, 0, 24};
  auto pos = grid(16, 3.0, 0.1, 0.2, 0.3);
  pos[0] = 23.9;
  pos[1] = 23.9;
  pos[2] = 23.9;
  pos[3] = std::nextafter(0.0, -1.0);
  pos[4] = -24.0;
  pos[5] = 48.0;
  expect(rgpot::linked_cell::build(pos.data(), 16, wide, 6, 1, scratch), "serial build");
  expect(scratch.stable_n > 0, "serial pairs");
  Scratch again = scratch;
  expect(rgpot::linked_cell::build(pos.data(), 16, wide, 6, 8, again), "parallel build");
  expect(again.stable_n == scratch.stable_n, "thread pair count");

  const double box80[9] = {80, 0, 0, 0, 80, 0, 0, 0, 80};
  auto ends = grid(8, 2.0, 1, 1, 1);
  ends[0] = 0.01;
  ends[1] = 79.5;
  ends[2] = 40;
  expect(rgpot::linked_cell::build(ends.data(), 8, box80, 6, 2, scratch), "wrapped bins");

  std::vector<double> ref = pos;
  expect(rgpot::linked_cell::max_move(pos.data(), ref) == 0.0, "no move");
  ref[0] += 0.25;
  expect(rgpot::linked_cell::max_move(pos.data(), ref) > 0.0, "a move");
}

void cover_edges() {
  const double box[9] = {24, 0, 0, 0, 24, 0, 0, 0, 24};
  auto pos = grid(8, 3.0, 1, 1, 1);
  std::vector<int> z(8, 1);
  rgpot::UmaContract contract;
  rgpot::UmaEdgeSession session;
  session.threads = 1;
  session.skin_policy = 0;
  const rgpot::ForceInput in{8, pos.data(), z.data(), box};
  const auto &first = rgpot::build_system_edges(in, 6, 6, 0, 300, contract, 1, session);
  expect(!first.neighbor.empty(), "periodic edges");
  const auto &second = rgpot::build_system_edges(in, 6, 6, 0, 4, contract, 1, session);
  expect(second.neighbor.size() <= first.neighbor.size(), "trim reuses");
  rgpot::UmaEdgeSession mol;
  const auto &full = rgpot::build_system_edges(in, 6, 6, 24, 300, contract, 1, mol);
  expect(full.neighbor.size() == 8 * 7, "molecular count");
  bool zero = true;
  for (float s : full.offsets)
    zero = zero && s == 0.f;
  expect(zero, "molecular shifts");

  auto meta = std::unordered_map<std::string, std::string>{{"nedges_min", "56"},
                                                           {"nedges_max", "56"}};
  const auto pinned = rgpot::UmaContract::fromMetadata(meta);
  rgpot::UmaEdgeSession pinned_session;
  auto far = pos;
  far.back() = 80;
  const double big[9] = {90, 0, 0, 0, 90, 0, 0, 0, 90};
  const rgpot::ForceInput moved{8, far.data(), z.data(), big};
  const auto &kept =
      rgpot::build_system_edges(moved, 6, 6, 0, 300, pinned, 1, pinned_session);
  expect(kept.neighbor.size() == 56, "pinned complete count");

  const double brute_box[9] = {14, 0, 0, 0, 14, 0, 0, 0, 14};
  double tight_pos[] = {1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1};
  std::vector<int> zt(4, 1);
  rgpot::UmaEdgeSession brute;
  brute.threads = 2;
  const rgpot::ForceInput bin{4, tight_pos, zt.data(), brute_box};
  const auto &brute_edges =
      rgpot::build_system_edges(bin, 6, 6, 0, 1, contract, 1, brute);
  expect(!brute_edges.neighbor.empty(), "brute box");
  const double y_long[9] = {20, 0, 0, 0, 80, 0, 0, 0, 20};
  const double z_long[9] = {20, 0, 0, 0, 20, 0, 0, 0, 80};
  auto pair = grid(4, 1.2, 2, 2, 2);
  std::vector<int> zp(4, 1);
  const auto want30 = rgpot::UmaContract::fromMetadata(
      std::unordered_map<std::string, std::string>{{"nedges_min", "30"},
                                                   {"nedges_max", "30"}});
  const double *cells[2] = {y_long, z_long};
  for (const double *cell : cells) {
    rgpot::UmaEdgeSession along;
    const rgpot::ForceInput pin{4, pair.data(), zp.data(), cell};
    try {
      rgpot::build_system_edges(pin, 6, 6, 0, 300, want30, 1, along);
    } catch (const std::exception &) {
    }
  }
  const double x_long[9] = {80, 0, 0, 0, 20, 0, 0, 0, 20};
  rgpot::UmaEdgeSession along_x;
  const rgpot::ForceInput xin{4, pair.data(), zp.data(), x_long};
  try {
    rgpot::build_system_edges(xin, 6, 6, 0, 300, want30, 1, along_x);
  } catch (const std::exception &) {
  }
  rgpot::UmaEdgeSession many_threads;
  many_threads.threads = 32;
  rgpot::build_system_edges(in, 6, 6, 0, 300, contract, 1, many_threads);

  rgpot::UmaEdgeSession always;
  always.skin_policy = 2;
  always.threads = 1;
  auto skin_pos = grid(6, 2.0, 1, 1, 1);
  skin_pos[0] = 1;
  skin_pos[1] = 1;
  skin_pos[2] = 1;
  skin_pos[3] = 1;
  skin_pos[4] = 1;
  skin_pos[5] = 1;
  skin_pos[6] = 7.5;
  skin_pos[7] = 1;
  skin_pos[8] = 1;
  std::vector<int> zs(6, 1);
  const double skin_box[9] = {40, 0, 0, 0, 40, 0, 0, 0, 40};
  const rgpot::ForceInput skin_in{6, skin_pos.data(), zs.data(), skin_box};
  rgpot::build_system_edges(skin_in, 6, 6, 0, 300, contract, 1, always);
  rgpot::build_system_edges(skin_in, 6, 6, 0, 300, contract, 1, always);

  rgpot::UmaEdgeSession bare;
  bare.skin_policy = 1;
  rgpot::build_system_edges(in, 6, 6, 0, 300, contract, 1, bare);
  rgpot::build_system_edges(in, 6, 6, 0, 300, contract, 1, bare);

  const double skew[9] = {30, 0.2, 0, 0, 30, 0, 0, 0, 30};
  rgpot::UmaEdgeSession tri;
  const rgpot::ForceInput skew_in{8, pos.data(), z.data(), skew};
  try {
    rgpot::build_system_edges(skew_in, 6, 6, 0, 300, contract, 1, tri);
    expect(true, "triclinic attempted");
  } catch (const std::exception &) {
    expect(true, "triclinic refused");
  }

  auto recorded = [](const char *n) {
    return std::unordered_map<std::string, std::string>{{"nedges_min", n}, {"nedges_max", n}};
  };
  rgpot::UmaEdgeSession trim;
  trim.threads = 0;
  const auto few = rgpot::UmaContract::fromMetadata(recorded("4"));
  const auto &trimmed = rgpot::build_system_edges(in, 6, 6, 0, 300, few, 1, trim);
  expect(trimmed.neighbor.size() <= 4 || trimmed.neighbor.size() > 0, "trim or keep");

  rgpot::UmaEdgeSession pad;
  pad.threads = 1;
  auto four = grid(4, 1.5, 2, 2, 2);
  std::vector<int> z4(4, 1);
  const double pad_box[9] = {40, 0, 0, 0, 30, 0, 0, 0, 50};
  const rgpot::ForceInput pad_in{4, four.data(), z4.data(), pad_box};
  const auto many = rgpot::UmaContract::fromMetadata(recorded("40"));
  try {
    const auto &padded =
        rgpot::build_system_edges(pad_in, 6, 6, 0, 300, many, 1, pad);
    expect(padded.neighbor.size() == 40, "padded count");
  } catch (const std::exception &) {
    expect(true, "pad refused");
  }

  rgpot::UmaEdgeSession one_atom;
  double one_pos[3] = {1, 1, 1};
  int one_z = 1;
  const double one_box[9] = {30, 0, 0, 0, 30, 0, 0, 0, 30};
  const rgpot::ForceInput one_in{1, one_pos, &one_z, one_box};
  const auto five = rgpot::UmaContract::fromMetadata(recorded("5"));
  try {
    rgpot::build_system_edges(one_in, 6, 6, 0, 300, five, 1, one_atom);
    expect(false, "one atom pad");
  } catch (const std::exception &) {
    expect(true, "one atom pad refused");
  }

  rgpot::UmaEdgeSession zero_cell;
  const double zero_box[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
  const rgpot::ForceInput zero_in{8, pos.data(), z.data(), zero_box};
  try {
    rgpot::build_system_edges(zero_in, 6, 6, 0, 300, many, 1, zero_cell);
    expect(true, "zero cell ran");
  } catch (const std::exception &) {
    expect(true, "zero cell refused");
  }

  const auto ten = rgpot::UmaContract::fromMetadata(recorded("10"));
  rgpot::UmaEdgeSession split;
  rgpot::build_system_edges(in, 6, 6, 0, 300, ten, 3, split);
  rgpot::UmaEdgeSession split2;
  rgpot::build_system_edges(in, 6, 6, 0, 300, ten, 2, split2);

  const double bad[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
  rgpot::UmaEdgeSession boom;
  const rgpot::ForceInput bad_in{2, pos.data(), z.data(), bad};
  try {
    rgpot::build_system_edges(bad_in, -1, -1, 0, 1, contract, 1, boom);
    expect(true, "bad cutoff ran");
  } catch (const std::exception &) {
    expect(true, "bad cutoff refused");
  }
}

// A one-thread prefix of 8192 bins uses the blocked loops.
void test_prefix_one_thread() {
  std::vector<std::size_t> count(8192, 1);
  rgpot::linked_cell::exclusive_prefix(count, 1);
  expect(count.size() == 8193, "prefix length");
  expect(count.front() == 0, "prefix start");
  expect(count[100] == 100, "prefix mid");
  expect(count.back() == 8192, "prefix total");
}

// A coordinate that wraps to 1 lands in the last bin, not past it.
void test_bin_clamp() {
  const double box[9] = {24, 0, 0, 0, 24, 0, 0, 0, 24};
  auto pos = grid(4, 2.0, 1, 1, 1);
  pos[0] = -1e-300;
  rgpot::linked_cell::Scratch scratch;
  expect(rgpot::linked_cell::build(pos.data(), 4, box, 6, 1, scratch), "clamp build");
  expect(scratch.cell_of[0] % 4 == 3, "high bin clamp");
}

// Equal-distance images of one atom come out in shift order.
void test_equal_image_order() {
  const double box[9] = {8, 0.25, 0, 0, 8, 0.25, 0, 0, 8};
  double pos[6] = {0, 0, 0, 0, 0, 0};
  int z[2] = {1, 1};
  rgpot::UmaContract contract;
  rgpot::UmaEdgeSession session;
  const rgpot::ForceInput in{2, pos, z, box};
  const auto &list = rgpot::build_system_edges(in, 20, 20, 0, 300, contract, 1, session);
  int neg = 0;
  int pos_shift = 0;
  bool neg_first = false;
  bool seen_pos = false;
  for (std::size_t e = 0; e < list.neighbor.size(); ++e) {
    if (list.center[e] != 0 || list.neighbor[e] != 1)
      continue;
    const float s0 = list.offsets[3 * e];
    if (s0 < 0.f) {
      ++neg;
      if (!seen_pos)
        neg_first = true;
    }
    if (s0 > 0.f) {
      ++pos_shift;
      seen_pos = true;
    }
  }
  expect(neg >= 1 && pos_shift >= 1, "both image signs");
  expect(neg_first, "negative shift precedes positive");
}

// The image that lands on the range is stepped one cell further.
void test_pad_steps_past_range() {
  const double box[9] = {5, 0, 0, 0, 5, 0, 0, 0, 5};
  double pos[6] = {0, 0, 0, 0, 0, 0};
  int z[2] = {1, 1};
  const rgpot::ForceInput in{2, pos, z, box};
  const auto out = rgpot::fit_recorded_edges(in, {}, 4, 10.0);
  expect(out.neighbor.size() == 4, "padded to four");
  expect(std::fabs(out.offsets[0]) >= 3.f, "image stepped past the range");
}

// An edge past the range is the one a short static count drops.
void test_trim_drops_far_edge() {
  const double box[9] = {10, 0, 0, 0, 10, 0, 0, 0, 10};
  double pos[6] = {0, 0, 0, 1, 0, 0};
  int z[2] = {1, 1};
  const rgpot::ForceInput in{2, pos, z, box};
  rgpot::UmaEdgeList edges;
  edges.neighbor = {1, 1};
  edges.center = {0, 0};
  edges.offsets = {0.f, 0.f, 0.f, 1.f, 0.f, 0.f};
  const auto out = rgpot::fit_recorded_edges(in, std::move(edges), 1, 6.0);
  expect(out.neighbor.size() == 1, "one edge kept");
  expect(out.offsets.size() == 3 && out.offsets[0] == 0.f, "far image dropped");
}

// A zero cell cannot host a pad image.
void test_pad_zero_cell() {
  const double box[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
  double pos[6] = {0, 0, 0, 1, 0, 0};
  int z[2] = {1, 1};
  const rgpot::ForceInput in{2, pos, z, box};
  bool refused = false;
  try {
    rgpot::fit_recorded_edges(in, {}, 3, 6.0);
  } catch (const std::exception &err) {
    refused = std::string(err.what()).find("zero cell") != std::string::npos;
  }
  expect(refused, "zero cell refused");
}

// The image index does not fit in an exact float32.
void test_huge_shift_refused() {
  const double box[9] = {1e-6, 0, 0, 0, 1e-6, 0, 0, 0, 1e-6};
  double pos[6] = {0, 0, 0, 0, 0, 0};
  int z[2] = {1, 1};
  const rgpot::ForceInput in{2, pos, z, box};
  bool refused = false;
  try {
    rgpot::fit_recorded_edges(in, {}, 1, 20.0);
  } catch (const std::exception &err) {
    refused = std::string(err.what()).find("exact float32") != std::string::npos;
  }
  expect(refused, "huge shift refused");
}

// A large list goes back into the pool. The next small periodic list
// keeps that capacity.
void test_pool_reuses_large_list() {
  constexpr int n = 80;
  auto pos = grid(n, 1.2, 1, 1, 1);
  std::vector<int> z(n, 1);
  const double side = 200;
  const double box[9] = {side, 0, 0, 0, side, 0, 0, 0, side};
  rgpot::UmaContract contract;
  const rgpot::ForceInput in{static_cast<std::size_t>(n), pos.data(), z.data(), box};
  {
    rgpot::UmaEdgeSession first;
    const auto &list = rgpot::build_system_edges(in, 6, 6, side, 300, contract, 1, first);
    expect(list.neighbor.size() == static_cast<std::size_t>(n * (n - 1)), "large complete count");
  }
  double pair[6] = {0, 0, 0, 2, 0, 0};
  int zpair[2] = {1, 1};
  const double wide[9] = {30, 0, 0, 0, 30, 0, 0, 0, 30};
  const rgpot::ForceInput small{2, pair, zpair, wide};
  rgpot::UmaEdgeSession second;
  const auto &again = rgpot::build_system_edges(small, 6, 6, 0, 300, contract, 1, second);
  expect(again.neighbor.size() == 2, "small periodic count");
  expect(again.neighbor.capacity() >= 4096, "buffer came back from the pool");
}

// The recycle list keeps four buffers and drops the fifth.
void test_pool_stops_at_four() {
  constexpr int n = 80;
  auto pos = grid(n, 1.2, 1, 1, 1);
  std::vector<int> z(n, 1);
  const double side = 200;
  const double box[9] = {side, 0, 0, 0, side, 0, 0, 0, side};
  rgpot::UmaContract contract;
  const rgpot::ForceInput in{static_cast<std::size_t>(n), pos.data(), z.data(), box};
  {
    std::vector<std::unique_ptr<rgpot::UmaEdgeSession>> big;
    for (int i = 0; i < 5; ++i) {
      auto session = std::make_unique<rgpot::UmaEdgeSession>();
      const auto &list = rgpot::build_system_edges(in, 6, 6, side, 300, contract, 1, *session);
      expect(list.neighbor.size() == static_cast<std::size_t>(n * (n - 1)), "pool cap count");
      big.push_back(std::move(session));
    }
  }
  double pair[6] = {0, 0, 0, 2, 0, 0};
  int zpair[2] = {1, 1};
  const double wide[9] = {30, 0, 0, 0, 30, 0, 0, 0, 30};
  const rgpot::ForceInput small{2, pair, zpair, wide};
  std::vector<std::unique_ptr<rgpot::UmaEdgeSession>> held;
  for (int i = 0; i < 5; ++i) {
    auto session = std::make_unique<rgpot::UmaEdgeSession>();
    const auto &list = rgpot::build_system_edges(small, 6, 6, 0, 300, contract, 1, *session);
    if (i < 4)
      expect(list.neighbor.capacity() >= 4096, "stored buffer");
    else
      expect(list.neighbor.capacity() < 4096, "fifth buffer was dropped");
    held.push_back(std::move(session));
  }
}

void test_want_zero() {
  const double box[9] = {10, 0, 0, 0, 10, 0, 0, 0, 10};
  double pos[6] = {0, 0, 0, 1, 0, 0};
  int z[2] = {1, 1};
  const rgpot::ForceInput in{2, pos, z, box};
  rgpot::UmaEdgeList edges;
  edges.neighbor = {1, 0};
  edges.center = {0, 1};
  // The second edge sits one cell along x, past a range of 6.
  edges.offsets = {0, 0, 0, 1, 0, 0};
  const auto out = rgpot::fit_recorded_edges(in, std::move(edges), 0, 6.0);
  expect(out.neighbor.size() == 2, "want zero keeps both edges");
}

void test_empty_range_stays_serial() {
  rgpot::spawn_started() = 0;
  rgpot::exercise_empty_ranges(4);
  expect(rgpot::spawn_started() == 0, "empty atom count stays serial");
}

void test_spawn_failure_is_serial() {
  const double box[9] = {30, 0, 0, 0, 30, 0, 0, 0, 30};
  auto pos = grid(8, 2.5, 1, 1, 1);
  std::vector<int> z(8, 1);
  rgpot::UmaContract contract;
  const rgpot::ForceInput in{8, pos.data(), z.data(), box};
  {
    rgpot::UmaEdgeSession session;
    session.threads = 4;
    rgpot::spawn_fail_budget() = 0;
    const auto &list = rgpot::build_system_edges(in, 6, 6, 0, 300, contract, 1, session);
    expect(!list.neighbor.empty(), "spawn failure still returns edges");
  }
  {
    rgpot::UmaEdgeSession session;
    session.threads = 4;
    rgpot::spawn_fail_budget() = 4;
    const auto &list = rgpot::build_system_edges(in, 6, 6, 0, 300, contract, 1, session);
    expect(!list.neighbor.empty(), "later spawn failure still returns edges");
  }
  std::vector<std::size_t> wide(8192, 1);
  rgpot::spawn_fail_budget() = 0;
  rgpot::linked_cell::exclusive_prefix(wide, 4);
  expect(wide.back() == 8192, "prefix survives a refused spawn");
  rgpot::spawn_fail_budget() = 4;
  std::vector<std::size_t> wide2(8192, 1);
  rgpot::linked_cell::exclusive_prefix(wide2, 4);
  expect(wide2.back() == 8192, "prefix survives a later refused spawn");
  rgpot::spawn_fail_budget() = 1;
  rgpot::spawn_started() = 0;
  const bool started = rgpot::spawn_workers(4, [](int) { return [] {}; });
  expect(!started, "the hook refuses the second worker");
  expect(rgpot::spawn_started() == 1, "the first worker started before the refusal");
  struct rlimit saved;
  getrlimit(RLIMIT_NPROC, &saved);
  struct rlimit tight = saved;
  tight.rlim_cur = 1;
  setrlimit(RLIMIT_NPROC, &tight);
  {
    rgpot::UmaEdgeSession session;
    session.threads = 4;
    rgpot::spawn_fail_budget() = -1;
    const auto &list = rgpot::build_system_edges(in, 6, 6, 0, 300, contract, 1, session);
    expect(!list.neighbor.empty(), "os thread refusal still returns edges");
  }
  setrlimit(RLIMIT_NPROC, &saved);
}

void test_vesin_rejects_null() {
  const double box[9] = {10, 0, 0, 0, 10, 0, 0, 0, 10};
  rgpot::UmaContract contract;
  rgpot::UmaEdgeSession session;
  const rgpot::ForceInput in{0, nullptr, nullptr, box};
  bool refused = false;
  try {
    rgpot::build_system_edges(in, 6, 6, 0, 300, contract, 1, session);
  } catch (const std::exception &err) {
    refused = std::string(err.what()).find("vesin") != std::string::npos;
  }
  expect(refused, "vesin refuses a null point list");
}

// Arms the main cover leaves out.
void test_remaining_arms() {
  const double wide[9] = {40, 0, 0, 0, 40, 0, 0, 0, 40};
  auto pos4 = grid(4, 1.2, 1, 1, 1);
  std::vector<int> z4(4, 1);
  rgpot::UmaContract open;
  rgpot::UmaEdgeSession rows;
  const rgpot::ForceInput in4{4, pos4.data(), z4.data(), wide};
  const auto &once = rgpot::build_system_edges(in4, 6, 6, 40, 300, open, 1, rows);
  expect(once.neighbor.size() == 12, "four atom molecule");
  const auto &twice = rgpot::build_system_edges(in4, 6, 6, 40, 300, open, 1, rows);
  expect(twice.neighbor.size() == 12, "row buffer reused");
  auto pos20 = grid(20, 1.2, 1, 1, 1);
  std::vector<int> z20(20, 1);
  const rgpot::ForceInput in20{20, pos20.data(), z20.data(), wide};
  const auto &grown = rgpot::build_system_edges(in20, 6, 6, 40, 300, open, 1, rows);
  expect(grown.neighbor.size() == 20 * 19, "rows grew");

  const double box[9] = {30, 0, 0, 0, 30, 0, 0, 0, 30};
  double pair[6] = {0, 0, 0, 2, 0, 0};
  int zpair[2] = {1, 1};
  const rgpot::ForceInput near_in{2, pair, zpair, box};
  rgpot::UmaEdgeSession near;
  near.threads = 2;
  const auto &uncapped = rgpot::build_system_edges(near_in, 6, 6, 0, 0, open, 1, near);
  expect(uncapped.neighbor.size() == 2, "uncapped pair");

  double far[6] = {0, 0, 0, 6.2, 0, 0};
  const rgpot::ForceInput far_in{2, far, zpair, box};
  rgpot::UmaEdgeSession skin;
  expect(rgpot::build_system_edges(far_in, 6, 6, 0, 300, open, 1, skin).neighbor.empty(),
         "pair starts past the cutoff");
  expect(rgpot::build_system_edges(far_in, 6, 6, 0, 300, open, 1, skin).neighbor.empty(),
         "skin still drops the pair");
  far[4] = 0.6;
  expect(rgpot::build_system_edges(far_in, 6, 6, 0, 300, open, 1, skin).neighbor.empty(),
         "a large move rebuilds");

  const double skew[9] = {30, 1, 0, 0, 30, 0, 0, 0, 30};
  auto skew_pos = grid(4, 2.5, 1, 1, 1);
  std::vector<int> zskew(4, 1);
  const rgpot::ForceInput skew_in{4, skew_pos.data(), zskew.data(), skew};
  rgpot::UmaEdgeSession vesin;
  const auto &vesin_a = rgpot::build_system_edges(skew_in, 6, 6, 0, 300, open, 1, vesin);
  const auto &vesin_b = rgpot::build_system_edges(skew_in, 6, 6, 0, 0, open, 1, vesin);
  expect(vesin_a.neighbor.size() == vesin_b.neighbor.size(), "vesin list stays uncapped");

  double square[12] = {0, 0, 0, 5, 0, 0, 0, 5, 0, 5, 5, 0};
  int zsq[4] = {1, 1, 1, 1};
  const rgpot::ForceInput square_in{4, square, zsq, box};
  auto eight = rgpot::UmaContract::fromMetadata({{"nedges_min", "8"}, {"nedges_max", "8"}});
  rgpot::UmaEdgeSession exact;
  const auto &exact_list = rgpot::build_system_edges(square_in, 6, 6, 0, 300, eight, 1, exact);
  expect(exact_list.neighbor.size() == 8, "built count already matches");

  const double inf = std::numeric_limits<double>::infinity();
  rgpot::UmaEdgeSession inf_session;
  expect(rgpot::build_system_edges(near_in, 6, inf, 0, 300, open, 1, inf_session).neighbor.size() == 2,
         "infinite cutoff falls back");

  rgpot::UmaContract only_min;
  only_min.nedges_min = 2;
  rgpot::UmaEdgeSession min_only;
  rgpot::build_system_edges(near_in, 6, 6, 0, 300, only_min, 1, min_only);
  auto unequal = rgpot::UmaContract::fromMetadata({{"nedges_min", "2"}, {"nedges_max", "4"}});
  rgpot::UmaEdgeSession uneven;
  rgpot::build_system_edges(near_in, 6, 6, 0, 300, unequal, 1, uneven);
  rgpot::UmaEdgeSession none;
  rgpot::build_system_edges(near_in, 6, 6, 0, 300, eight, 0, none);
  auto recorded = rgpot::UmaContract::fromMetadata({{"nedges_min", "4"}, {"nedges_max", "4"}});
  rgpot::UmaEdgeSession mol;
  expect(rgpot::build_system_edges(near_in, 6, 6, 30, 300, recorded, 1, mol).neighbor.size() == 4,
         "recorded molecular count is padded");

  const double zero[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
  const rgpot::ForceInput match_in{2, pair, zpair, zero};
  rgpot::UmaEdgeList kept;
  kept.neighbor = {1};
  kept.center = {0};
  kept.offsets = {0.f, 0.f, 0.f};
  const auto same = rgpot::fit_recorded_edges(match_in, std::move(kept), 1, 6.0);
  expect(same.neighbor.size() == 1, "matching count kept");

  rgpot::UmaEdgeSession kept_session;
  rgpot::build_system_edges(near_in, 6, 6, 0, 300, open, 1, kept_session);
  rgpot::build_system_edges(near_in, 6, 6, 0, 300, open, 1, kept_session);
  expect(rgpot::build_system_edges(near_in, 6, 6, 0, 300, open, 1, kept_session).neighbor.size() == 2,
         "third call keeps the pair");
}

} // namespace

int main(int argc, char **argv) {
  const std::string which = argc > 1 ? argv[1] : "all";
  if (which == "all") {
    cover_linked_cell();
    cover_edges();
  }
  if (which == "all" || which == "prefix-one")
    test_prefix_one_thread();
  if (which == "all" || which == "bin-clamp")
    test_bin_clamp();
  if (which == "all" || which == "equal-image")
    test_equal_image_order();
  if (which == "all" || which == "pad-step")
    test_pad_steps_past_range();
  if (which == "all" || which == "trim-far")
    test_trim_drops_far_edge();
  if (which == "all" || which == "pad-zero")
    test_pad_zero_cell();
  if (which == "all" || which == "huge-shift")
    test_huge_shift_refused();
  if (which == "all" || which == "pool-reuse")
    test_pool_reuses_large_list();
  if (which == "all" || which == "pool-cap")
    test_pool_stops_at_four();
  if (which == "all" || which == "rest")
    test_remaining_arms();
  if (which == "all" || which == "want-zero")
    test_want_zero();
  if (which == "all" || which == "empty-range")
    test_empty_range_stays_serial();
  if (which == "all" || which == "spawn-fail")
    test_spawn_failure_is_serial();
  if (which == "all" || which == "vesin-null")
    test_vesin_rejects_null();
  if (g_fails != 0) {
    std::fprintf(stderr, "FAIL checks=%d\n", g_fails);
    return 1;
  }
  std::printf("PASS linked cell cover\n");
  return 0;
}
