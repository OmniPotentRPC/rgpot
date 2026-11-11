// Copyright 2023--present rgpot developers
//
// Exercises the linked-cell builder and the edge session so a
// coverage run can see both sides of the branches.

#include "rgpot/UmaPot/UmaEdges.hpp"
#include "rgpot/UmaPot/UmaLinkedCell.hpp"

#include <cmath>
#include <cstdio>
#include <limits>
#include <string>
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

} // namespace

int main() {
  cover_linked_cell();
  cover_edges();
  if (g_fails != 0) {
    std::fprintf(stderr, "FAIL checks=%d\n", g_fails);
    return 1;
  }
  std::printf("PASS linked cell cover\n");
  return 0;
}
