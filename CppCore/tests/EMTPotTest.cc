// MIT License
// Copyright 2023--present rgpot developers
//
// Copper FCC cluster pin from eOn EMTCuTest / point_emt_cu.dat.

#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <new>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include <catch2/catch_all.hpp>

#include "rgpot/EMT/Asap/Exception.h"
#include "rgpot/EMT/EMTPot.hpp"
#include "rgpot/ForceStructs.hpp"
#include "rgpot/Slices.hpp"

#include "rgpot/EMT/Asap/Atoms.h"
#include "rgpot/EMT/Asap/EMT.h"
#include "rgpot/EMT/Asap/SuperCell.h"
#include "rgpot/EMT/Asap/Vec.h"

using Catch::Matchers::WithinAbs;

namespace {

constexpr std::size_t kTrackedBlockSlots = std::size_t{1} << 17;
constexpr std::size_t kTrackedBlockMask = kTrackedBlockSlots - 1;
// Linear probing stays short only while the table is at most half full.
constexpr std::size_t kTrackedBlockCapacity = kTrackedBlockSlots / 2;

std::mutex g_trackedBlocksMutex;
void *g_trackedBlocks[kTrackedBlockSlots]{};
std::size_t g_trackedBlockCount = 0;
std::atomic<long> g_liveBlocks{0};

std::size_t homeSlot(const void *block) {
  std::uint64_t bits = reinterpret_cast<std::uintptr_t>(block);
  bits ^= bits >> 33;
  bits *= 0xff51afd7ed558ccdULL;
  bits ^= bits >> 33;
  return static_cast<std::size_t>(bits) & kTrackedBlockMask;
}

void trackAllocation(void *block) {
  const std::lock_guard<std::mutex> lock(g_trackedBlocksMutex);
  if (g_trackedBlockCount == kTrackedBlockCapacity) {
    std::abort();
  }
  std::size_t slot = homeSlot(block);
  while (g_trackedBlocks[slot] != nullptr) {
    slot = (slot + 1) & kTrackedBlockMask;
  }
  g_trackedBlocks[slot] = block;
  ++g_trackedBlockCount;
  g_liveBlocks.fetch_add(1, std::memory_order_relaxed);
}

void trackDeallocation(void *block) {
  // Replacement delete can see allocations that replacement new did not,
  // notably across MinGW runtime module boundaries. Only a matched pointer
  // belongs in this test's live-block count.
  const std::lock_guard<std::mutex> lock(g_trackedBlocksMutex);
  std::size_t hole = homeSlot(block);
  while (g_trackedBlocks[hole] != block) {
    if (g_trackedBlocks[hole] == nullptr) {
      return;
    }
    hole = (hole + 1) & kTrackedBlockMask;
  }
  // Backward-shift deletion: no tombstones, so a probe for an unmatched
  // pointer still stops at the first empty slot.
  for (std::size_t next = (hole + 1) & kTrackedBlockMask;
       g_trackedBlocks[next] != nullptr;
       next = (next + 1) & kTrackedBlockMask) {
    const std::size_t home = homeSlot(g_trackedBlocks[next]);
    if (((next - home) & kTrackedBlockMask) >=
        ((next - hole) & kTrackedBlockMask)) {
      g_trackedBlocks[hole] = g_trackedBlocks[next];
      hole = next;
    }
  }
  g_trackedBlocks[hole] = nullptr;
  --g_trackedBlockCount;
  g_liveBlocks.fetch_sub(1, std::memory_order_relaxed);
}

} // namespace

void *operator new(std::size_t size) {
  void *block = std::malloc(size == 0 ? 1 : size);
  if (block == nullptr) {
    throw std::bad_alloc();
  }
  trackAllocation(block);
  return block;
}

void operator delete(void *block) noexcept {
  if (block != nullptr) {
    trackDeallocation(block);
    std::free(block);
  }
}

void operator delete(void *block, std::size_t) noexcept {
  ::operator delete(block);
}

void *operator new[](std::size_t size) { return ::operator new(size); }

void operator delete[](void *block) noexcept { ::operator delete(block); }

void operator delete[](void *block, std::size_t) noexcept {
  ::operator delete(block);
}

TEST_CASE("EMT energy matches the copper cluster pin", "[emt]") {
  const std::vector<std::array<double, 3>> xyz = {{
      {8.1925, 8.1925, 8.1925},
      {10.0, 10.0, 8.1925},
      {10.0, 8.1925, 10.0},
      {8.1925, 10.0, 10.0},
  }};
  rgpot::types::AtomMatrix positions(4, 3);
  std::vector<int> types(4, 29);
  for (int i = 0; i < 4; ++i) {
    for (int k = 0; k < 3; ++k) {
      positions(i, k) = xyz[static_cast<size_t>(i)][static_cast<size_t>(k)];
    }
  }
  const std::array<std::array<double, 3>, 3> box = {{
      {20.0, 0.0, 0.0},
      {0.0, 20.0, 0.0},
      {0.0, 0.0, 20.0},
  }};
  rgpot::EMTPot pot(rgpot::EMTConfig{.rasmussen = false});
  const auto [energy, forces, variance] = pot(positions, types, box);
  REQUIRE_THAT(energy, WithinAbs(5.129167, 5e-4));
  double worst = 0.0;
  for (int i = 0; i < 4; ++i) {
    const double n2 = forces(i, 0) * forces(i, 0) +
                      forces(i, 1) * forces(i, 1) + forces(i, 2) * forces(i, 2);
    worst = std::max(worst, std::sqrt(n2));
  }
  REQUIRE_THAT(worst, WithinAbs(1.914263, 2e-3));
  REQUIRE(pot.config().rasmussen == false);
  REQUIRE(pot.paramsKey() != 0);
}

TEST_CASE("EMT reads the species again when only the species change", "[emt]") {
  rgpot::types::AtomMatrix positions(4, 3);
  const std::array<std::array<double, 3>, 4> xyz = {{
      {8.1925, 8.1925, 8.1925},
      {10.0, 10.0, 8.1925},
      {10.0, 8.1925, 10.0},
      {8.1925, 10.0, 10.0},
  }};
  for (int i = 0; i < 4; ++i) {
    for (int k = 0; k < 3; ++k) {
      positions(i, k) = xyz[static_cast<size_t>(i)][static_cast<size_t>(k)];
    }
  }
  const std::array<std::array<double, 3>, 3> box = {{
      {20.0, 0.0, 0.0},
      {0.0, 20.0, 0.0},
      {0.0, 0.0, 20.0},
  }};
  const std::vector<int> copper(4, 29);
  const std::vector<int> mixed = {29, 79, 29, 79};

  rgpot::EMTPot reused;
  (void)reused(positions, copper, box);
  const auto [e_reused, f_reused, v_reused] = reused(positions, mixed, box);

  rgpot::EMTPot fresh;
  const auto [e_fresh, f_fresh, v_fresh] = fresh(positions, mixed, box);

  REQUIRE_THAT(e_reused, WithinAbs(e_fresh, 1e-12));
  for (int i = 0; i < 4; ++i) {
    for (int k = 0; k < 3; ++k) {
      REQUIRE_THAT(f_reused(i, k), WithinAbs(f_fresh(i, k), 1e-12));
    }
  }
}

namespace {

struct ThreadGuard {
  std::string saved;
  bool had = false;

  ThreadGuard() {
    if (const char *v = std::getenv("OMP_NUM_THREADS")) {
      saved = v;
      had = true;
    }
  }

  ~ThreadGuard() { set(had ? saved.c_str() : nullptr); }

  static void set(const char *v) {
#if defined(_WIN32)
    _putenv_s("OMP_NUM_THREADS", v == nullptr ? "" : v);
#else
    if (v == nullptr || v[0] == '\0') {
      unsetenv("OMP_NUM_THREADS");
    } else {
      setenv("OMP_NUM_THREADS", v, 1);
    }
#endif
  }
};

std::uint64_t f64Bits(double v) {
  std::uint64_t out = 0;
  std::memcpy(&out, &v, sizeof(out));
  return out;
}

struct Eval {
  double energy = 0.0;
  std::vector<double> forces;
  std::array<double, 9> stress{};
  int has_stress = 0;
};

Eval runEval(rgpot::EMTPot &pot, const std::vector<double> &pos,
             const std::vector<int> &z, const std::array<double, 9> &box,
             bool stress = false) {
  Eval outv;
  outv.forces.assign(pos.size(), 0.0);
  rgpot::ForceOut out{};
  out.F = outv.forces.data();
  std::fill(std::begin(out.stress), std::end(out.stress), 0.0);
  rgpot::ForceInput in{z.size(), pos.data(), z.data(), box.data()};
  in.compute_stress = stress;
  pot.forceImpl(in, &out);
  outv.energy = out.energy;
  outv.has_stress = out.has_stress;
  std::copy(std::begin(out.stress), std::end(out.stress), outv.stress.begin());
  return outv;
}

std::size_t countBitDiffs(const Eval &a, const Eval &b) {
  std::size_t diffs = 0;
  diffs += (f64Bits(a.energy) == f64Bits(b.energy)) ? 0 : 1;
  diffs += (a.has_stress == b.has_stress) ? 0 : 1;
  for (std::size_t k = 0; k < 9; ++k) {
    diffs += (f64Bits(a.stress[k]) == f64Bits(b.stress[k])) ? 0 : 1;
  }
  for (std::size_t k = 0; k < a.forces.size(); ++k) {
    diffs += (f64Bits(a.forces[k]) == f64Bits(b.forces[k])) ? 0 : 1;
  }
  return diffs;
}

struct FccBox {
  std::vector<double> pos;
  std::vector<int> z;
  std::array<double, 9> box{};
};

FccBox fccCu(int side, double lattice) {
  static constexpr double kBasis[4][3] = {
      {0.0, 0.0, 0.0}, {0.0, 0.5, 0.5}, {0.5, 0.0, 0.5}, {0.5, 0.5, 0.0}};
  FccBox out;
  const double L = static_cast<double>(side) * lattice;
  out.box = {L, 0.0, 0.0, 0.0, L, 0.0, 0.0, 0.0, L};
  for (int ix = 0; ix < side; ++ix) {
    for (int iy = 0; iy < side; ++iy) {
      for (int iz = 0; iz < side; ++iz) {
        for (const auto &b : kBasis) {
          const double fx =
              (static_cast<double>(ix) + b[0]) / static_cast<double>(side);
          const double fy =
              (static_cast<double>(iy) + b[1]) / static_cast<double>(side);
          const double fz =
              (static_cast<double>(iz) + b[2]) / static_cast<double>(side);
          const std::size_t idx = out.z.size();
          const double jx =
              0.01 * lattice * std::sin(0.23 * static_cast<double>(idx) + 0.1);
          const double jy =
              0.01 * lattice * std::cos(0.31 * static_cast<double>(idx) + 0.2);
          const double jz =
              0.01 * lattice * std::sin(0.29 * static_cast<double>(idx) + 0.3);
          out.pos.push_back(fx * L + jx);
          out.pos.push_back(fy * L + jy);
          out.pos.push_back(fz * L + jz);
          out.z.push_back(29);
        }
      }
    }
  }
  return out;
}

std::vector<double> shifted(const std::vector<double> &pos, double step) {
  std::vector<double> out = pos;
  for (std::size_t i = 0; i < out.size(); ++i) {
    out[i] += step * std::sin(0.71 * static_cast<double>(i) + 0.4);
  }
  return out;
}

constexpr bool kPeriodic[3] = {true, true, true};

// Writes a marker over the force buffer EMT keeps. A later call that
// returns the marker came from the cache, and one that overwrites it
// recomputed.
class MarkableEMT : public EMT {
public:
  MarkableEMT() : EMT(nullptr) {}

  static constexpr double kMarker = 1234.5;

  void markForces() {
    for (Vec &f : force) {
      f[0] = f[1] = f[2] = kMarker;
    }
  }
};

struct DirectEmt {
  Vec basis[3];
  SuperCell cell;
  Atoms atoms;
  MarkableEMT emt;

  explicit DirectEmt(const FccBox &sys)
      : basis{Vec(sys.box[0], sys.box[1], sys.box[2]),
              Vec(sys.box[3], sys.box[4], sys.box[5]),
              Vec(sys.box[6], sys.box[7], sys.box[8])},
        cell(basis, kPeriodic),
        atoms(reinterpret_cast<const Vec *>(sys.pos.data()),
              static_cast<int>(sys.z.size()), &cell) {
    atoms.SetAtomicNumbers(sys.z.data());
    atoms.SetUnitCell(basis, true);
    atoms.SetCalculator(&emt);
  }

  void move(const std::vector<double> &pos) {
    atoms.SetCartesianPositions(reinterpret_cast<const Vec *>(pos.data()));
    atoms.SetUnitCell(basis, true);
  }

  std::vector<double> forces(std::size_t n) {
    const Vec *f = emt.GetCartesianForces();
    std::vector<double> out(3 * n);
    for (std::size_t i = 0; i < n; ++i) {
      for (int k = 0; k < 3; ++k) {
        out[3 * i + static_cast<std::size_t>(k)] = f[i][k];
      }
    }
    return out;
  }
};

std::size_t countBitDiffs(const std::vector<double> &a,
                          const std::vector<double> &b) {
  std::size_t diffs = a.size() == b.size() ? 0 : 1;
  for (std::size_t k = 0; k < std::min(a.size(), b.size()); ++k) {
    diffs += (f64Bits(a[k]) == f64Bits(b[k])) ? 0 : 1;
  }
  return diffs;
}

void rightTransform(std::vector<double> &pos, std::array<double, 9> &box,
                    const std::array<double, 9> &map) {
  auto transform = [&](double *v) {
    const std::array<double, 3> old{v[0], v[1], v[2]};
    for (int b = 0; b < 3; ++b) {
      v[b] = old[0] * map[static_cast<std::size_t>(b)] +
             old[1] * map[static_cast<std::size_t>(3 + b)] +
             old[2] * map[static_cast<std::size_t>(6 + b)];
    }
  };
  for (std::size_t i = 0; i < pos.size(); i += 3) {
    transform(pos.data() + i);
  }
  for (int i = 0; i < 3; ++i) {
    transform(box.data() + 3 * i);
  }
}

} // namespace

TEST_CASE("EMT direct repeated force calls reuse the cached forces",
          "[emt][threads]") {
  ThreadGuard guard;
  const auto large = fccCu(13, 3.615 / std::sqrt(2.0));
  const std::size_t n = large.z.size();
  REQUIRE(n >= rgpot::kTeamMinAtoms);
  const auto moved = shifted(large.pos, 0.02);

  std::vector<double> ref_forces;
  std::vector<double> ref_moved;
  for (const char *th : {"1", "4"}) {
    INFO("OMP_NUM_THREADS=" << th);
    ThreadGuard::set(th);

    DirectEmt direct(large);
    const auto first = direct.forces(n);

    direct.emt.markForces();
    const double marker = MarkableEMT::kMarker;
    const Vec *again = direct.emt.GetCartesianForces();
    bool cached = true;
    for (std::size_t i = 0; i < n; ++i) {
      for (int k = 0; k < 3; ++k) {
        cached = cached && f64Bits(again[i][k]) == f64Bits(marker);
      }
    }
    CHECK(cached);
    const auto before = rgpot::teamsOpened();
    CHECK(direct.emt.GetCartesianForces() == again);
    CHECK(rgpot::teamsOpened() == before);

    // A moved atom set is a new evaluation, not the cached one.
    direct.move(moved);
    const auto after_move = direct.forces(n);
    CHECK(countBitDiffs(after_move, first) != 0);

    // The same forces, bit for bit, as an evaluation through EMTPot.
    rgpot::EMTPot pot;
    const auto through_pot = runEval(pot, large.pos, large.z, large.box);
    CHECK(countBitDiffs(first, through_pot.forces) == 0);
    const auto through_pot_moved = runEval(pot, moved, large.z, large.box);
    CHECK(countBitDiffs(after_move, through_pot_moved.forces) == 0);

    if (ref_forces.empty()) {
      ref_forces = first;
      ref_moved = after_move;
    } else {
      CHECK(countBitDiffs(first, ref_forces) == 0);
      CHECK(countBitDiffs(after_move, ref_moved) == 0);
    }
  }
}

namespace {

using Box = std::array<std::array<double, 3>, 3>;

rgpot::types::AtomMatrix copperCluster() {
  const std::array<std::array<double, 3>, 4> xyz = {{
      {8.1925, 8.1925, 8.1925},
      {10.0, 10.0, 8.1925},
      {10.0, 8.1925, 10.0},
      {8.1925, 10.0, 10.0},
  }};
  rgpot::types::AtomMatrix positions(4, 3);
  for (int i = 0; i < 4; ++i) {
    for (int k = 0; k < 3; ++k) {
      positions(i, k) = xyz[static_cast<size_t>(i)][static_cast<size_t>(k)];
    }
  }
  return positions;
}

constexpr Box kCube = {{
    {20.0, 0.0, 0.0},
    {0.0, 20.0, 0.0},
    {0.0, 0.0, 20.0},
}};

} // namespace

TEST_CASE("EMT turns an Asap exception into std::invalid_argument",
          "[emt][errors]") {
  const auto positions = copperCluster();
  const std::vector<int> carbon(4, 6);
  rgpot::EMTPot pot;
  REQUIRE_THROWS_AS(pot(positions, carbon, kCube), std::invalid_argument);
  REQUIRE_THROWS_WITH(pot(positions, carbon, kCube),
                      Catch::Matchers::ContainsSubstring("EMT"));
}

TEST_CASE("EMT keeps no state from a call that threw", "[emt][errors]") {
  const auto positions = copperCluster();
  const std::vector<int> carbon(4, 6);
  const std::vector<int> copper(4, 29);
  rgpot::EMTPot pot;
  for (int attempt = 0; attempt < 3; ++attempt) {
    REQUIRE_THROWS_AS(pot(positions, carbon, kCube), std::invalid_argument);
  }
  const auto [e_after, f_after, v_after] = pot(positions, copper, kCube);
  rgpot::EMTPot fresh;
  const auto [e_fresh, f_fresh, v_fresh] = fresh(positions, copper, kCube);
  REQUIRE(e_after == e_fresh);
  for (int i = 0; i < 4; ++i) {
    for (int k = 0; k < 3; ++k) {
      REQUIRE(f_after(i, k) == f_fresh(i, k));
    }
  }
}

TEST_CASE("EMT recovers after a cell that Asap refuses", "[emt][errors]") {
  const auto positions = copperCluster();
  const std::vector<int> copper(4, 29);
  const Box thin = {{{6.0, 0.0, 0.0}, {0.0, 20.0, 0.0}, {0.0, 0.0, 20.0}}};
  rgpot::EMTPot pot;
  (void)pot(positions, copper, kCube);
  REQUIRE_THROWS_AS(pot(positions, copper, thin), std::invalid_argument);
  const auto [energy, forces, variance] = pot(positions, copper, kCube);
  REQUIRE_THAT(energy, WithinAbs(5.129167, 5e-4));
}

TEST_CASE("EMT frees what a refused neighbour list construction allocated",
          "[emt][errors]") {
  const auto positions = copperCluster();
  const std::vector<int> copper(4, 29);
  const Box thin = {{{6.0, 0.0, 0.0}, {0.0, 20.0, 0.0}, {0.0, 0.0, 20.0}}};
  rgpot::EMTPot pot;
  REQUIRE_THROWS_AS(pot(positions, copper, thin), std::invalid_argument);

  constexpr int kAttempts = 32;
  int refused = 0;
  const long before = g_liveBlocks.load();
  for (int attempt = 0; attempt < kAttempts; ++attempt) {
    try {
      (void)pot(positions, copper, thin);
    } catch (const std::invalid_argument &) {
      ++refused;
    }
  }
  const long after = g_liveBlocks.load();
  CHECK(refused == kAttempts);
  CHECK(after == before);
}

TEST_CASE("EMT rejects zero atoms", "[emt][errors]") {
  rgpot::types::AtomMatrix none(0, 3);
  const std::vector<int> nothing;
  rgpot::EMTPot pot;
  REQUIRE_THROWS_AS(pot(none, nothing, kCube), std::invalid_argument);
  REQUIRE_THROWS_WITH(
      pot(none, nothing, kCube),
      Catch::Matchers::ContainsSubstring("nAtoms must be positive"));
}

TEST_CASE("EMT refuses a cell it cannot invert", "[emt][errors]") {
  const auto positions = copperCluster();
  const std::vector<int> copper(4, 29);
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const std::array<Box, 4> bad = {{
      {{{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}}},
      {{{20.0, 0.0, 0.0}, {40.0, 0.0, 0.0}, {0.0, 0.0, 20.0}}},
      {{{20.0, 0.0, 0.0}, {0.0, 20.0, 0.0}, {20.0, 20.0, 0.0}}},
      {{{20.0, 0.0, 0.0}, {0.0, nan, 0.0}, {0.0, 0.0, 20.0}}},
  }};
  rgpot::EMTPot pot;
  for (const Box &box : bad) {
    REQUIRE_THROWS_AS(pot(positions, copper, box), std::invalid_argument);
  }
  const auto [energy, forces, variance] = pot(positions, copper, kCube);
  REQUIRE_THAT(energy, WithinAbs(5.129167, 5e-4));
}

TEST_CASE("EMT refuses non-finite positions", "[emt][errors]") {
  auto positions = copperCluster();
  positions(2, 1) = std::numeric_limits<double>::infinity();
  const std::vector<int> copper(4, 29);
  rgpot::EMTPot pot;
  REQUIRE_THROWS_AS(pot(positions, copper, kCube), std::invalid_argument);
}

TEST_CASE("EMT reports coincident atoms as std::domain_error",
          "[emt][errors]") {
  auto positions = copperCluster();
  for (int k = 0; k < 3; ++k) {
    positions(1, k) = positions(0, k);
  }
  const std::vector<int> copper(4, 29);
  rgpot::EMTPot pot;
  REQUIRE_THROWS_AS(pot(positions, copper, kCube), std::domain_error);
}

TEST_CASE("EMT keeps a cached kernel unless the kernel throws",
          "[emt][errors]") {
  // g_liveBlocks counts allocations that outlive the call. A validation
  // failure and a non-finite force leave the cached kernel in place, so the
  // next call with the same cell does not build another one. An Asap
  // exception drops it, and the next accepted call builds it again.
  const auto positions = copperCluster();
  auto shifted = positions;
  shifted(0, 0) += 0.02;
  auto coincident = positions;
  for (int k = 0; k < 3; ++k) {
    coincident(1, k) = coincident(0, k);
  }
  auto infinite = positions;
  infinite(2, 1) = std::numeric_limits<double>::infinity();
  const std::vector<int> copper(4, 29);
  const std::vector<int> carbon(4, 6);
  const Box singular = {{{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}}};
  rgpot::types::AtomMatrix none(0, 3);
  const std::vector<int> nothing;

  const long before = g_liveBlocks.load();
  rgpot::EMTPot pot;
  double energy = 0.0;
  rgpot::types::AtomMatrix forces;
  std::tie(energy, forces, std::ignore) = pot(positions, copper, kCube);
  const long kernel = g_liveBlocks.load() - before;
  REQUIRE(kernel >= 8);
  REQUIRE_THAT(energy, WithinAbs(5.129167, 5e-4));

  rgpot::EMTPot reference;
  double shifted_energy = 0.0;
  rgpot::types::AtomMatrix shifted_forces;
  std::tie(shifted_energy, shifted_forces, std::ignore) =
      reference(shifted, copper, kCube);

  const auto kept = [](long mark, long kernel_blocks) {
    const long after = g_liveBlocks.load();
    REQUIRE(after + kernel_blocks / 2 > mark);
  };
  const auto reused = [](long mark, long kernel_blocks) {
    const long grown = g_liveBlocks.load() - mark;
    REQUIRE(grown < kernel_blocks / 2);
  };

  long mark = g_liveBlocks.load();
  REQUIRE_THROWS_AS(pot(positions, copper, singular), std::invalid_argument);
  kept(mark, kernel);

  mark = g_liveBlocks.load();
  REQUIRE_THROWS_AS(pot(infinite, copper, kCube), std::invalid_argument);
  kept(mark, kernel);

  mark = g_liveBlocks.load();
  REQUIRE_THROWS_AS(pot(none, nothing, kCube), std::invalid_argument);
  kept(mark, kernel);

  mark = g_liveBlocks.load();
  double again = 0.0;
  rgpot::types::AtomMatrix again_forces;
  std::tie(again, again_forces, std::ignore) = pot(shifted, copper, kCube);
  reused(mark, kernel);
  REQUIRE_THAT(again, WithinAbs(shifted_energy, 1e-12));
  for (int i = 0; i < 4; ++i) {
    for (int k = 0; k < 3; ++k) {
      REQUIRE_THAT(again_forces(i, k), WithinAbs(shifted_forces(i, k), 1e-12));
    }
  }

  mark = g_liveBlocks.load();
  REQUIRE_THROWS_AS(pot(coincident, copper, kCube), std::domain_error);
  kept(mark, kernel);

  mark = g_liveBlocks.load();
  double after_domain = 0.0;
  rgpot::types::AtomMatrix after_forces;
  std::tie(after_domain, after_forces, std::ignore) =
      pot(shifted, copper, kCube);
  reused(mark, kernel);
  REQUIRE_THAT(after_domain, WithinAbs(shifted_energy, 1e-12));
  for (int i = 0; i < 4; ++i) {
    for (int k = 0; k < 3; ++k) {
      REQUIRE_THAT(after_forces(i, k), WithinAbs(shifted_forces(i, k), 1e-12));
    }
  }

  mark = g_liveBlocks.load();
  REQUIRE_THROWS_AS(pot(positions, carbon, kCube), std::invalid_argument);
  const long dropped = mark - g_liveBlocks.load();
  REQUIRE(dropped > kernel / 2);

  mark = g_liveBlocks.load();
  double rebuilt = 0.0;
  rgpot::types::AtomMatrix rebuilt_forces;
  std::tie(rebuilt, rebuilt_forces, std::ignore) =
      pot(positions, copper, kCube);
  const long grown = g_liveBlocks.load() - mark;
  REQUIRE(grown > kernel / 2);
  REQUIRE_THAT(rebuilt, WithinAbs(energy, 1e-12));
  for (int i = 0; i < 4; ++i) {
    for (int k = 0; k < 3; ++k) {
      REQUIRE_THAT(rebuilt_forces(i, k), WithinAbs(forces(i, k), 1e-12));
    }
  }
}

TEST_CASE("EMT keeps a cached kernel when a stress request fails",
          "[emt][errors][stress]") {
  // A stress request follows the retention rules of a force-only call. A
  // failure before the kernel runs and a non-finite energy leave the cached
  // kernel in place, an Asap exception drops it, and a call that threw leaves
  // has_stress clear and the tensor zero.
  const auto cluster = copperCluster();
  std::array<double, 12> pos{};
  for (int i = 0; i < 4; ++i) {
    for (int k = 0; k < 3; ++k) {
      pos[static_cast<std::size_t>(3 * i + k)] = cluster(i, k);
    }
  }
  auto shifted = pos;
  shifted[0] += 0.02;
  auto coincident = pos;
  for (std::size_t k = 0; k < 3; ++k) {
    coincident[3 + k] = coincident[k];
  }
  auto infinite = pos;
  infinite[7] = std::numeric_limits<double>::infinity();
  const std::array<int, 4> copper{29, 29, 29, 29};
  const std::array<int, 4> carbon{6, 6, 6, 6};
  const std::array<double, 9> cube{20.0, 0.0, 0.0, 0.0, 20.0,
                                   0.0,  0.0, 0.0, 20.0};
  const std::array<double, 9> singular{};

  struct Call {
    std::array<double, 12> forces{};
    rgpot::ForceOut out{};
  };
  const auto stressCall =
      [](const rgpot::EMTPot &pot, const std::array<double, 12> &positions,
         const std::array<int, 4> &numbers, const std::array<double, 9> &box,
         std::size_t n, Call &call) {
        call.out.F = call.forces.data();
        call.out.has_stress = 1;
        std::fill(std::begin(call.out.stress), std::end(call.out.stress), 7.0);
        rgpot::ForceInput in{n, positions.data(), numbers.data(), box.data()};
        in.compute_stress = true;
        pot.forceImpl(in, &call.out);
      };
  const auto cleared = [](const Call &call) {
    CHECK(call.out.has_stress == 0);
    for (const double component : call.out.stress) {
      CHECK(component == 0.0);
    }
  };

  const long before = g_liveBlocks.load();
  rgpot::EMTPot pot;
  Call first;
  stressCall(pot, pos, copper, cube, 4, first);
  const long kernel = g_liveBlocks.load() - before;
  REQUIRE(kernel >= 8);
  REQUIRE(first.out.has_stress == 1);

  rgpot::EMTPot reference;
  Call shifted_reference;
  stressCall(reference, shifted, copper, cube, 4, shifted_reference);
  REQUIRE(shifted_reference.out.has_stress == 1);

  const auto sameStress = [](const Call &a, const Call &b) {
    CHECK_THAT(a.out.energy, WithinAbs(b.out.energy, 1e-12));
    for (std::size_t k = 0; k < 9; ++k) {
      CHECK_THAT(a.out.stress[k], WithinAbs(b.out.stress[k], 1e-12));
    }
  };

  const auto kept = [&](long mark) {
    REQUIRE(g_liveBlocks.load() + kernel / 2 > mark);
  };

  Call failed;
  long mark = g_liveBlocks.load();
  REQUIRE_THROWS_AS(stressCall(pot, pos, copper, singular, 4, failed),
                    std::invalid_argument);
  cleared(failed);
  kept(mark);

  mark = g_liveBlocks.load();
  REQUIRE_THROWS_AS(stressCall(pot, infinite, copper, cube, 4, failed),
                    std::invalid_argument);
  cleared(failed);
  kept(mark);

  mark = g_liveBlocks.load();
  REQUIRE_THROWS_AS(stressCall(pot, pos, copper, cube, 0, failed),
                    std::invalid_argument);
  cleared(failed);
  kept(mark);

  mark = g_liveBlocks.load();
  Call again;
  stressCall(pot, shifted, copper, cube, 4, again);
  REQUIRE(g_liveBlocks.load() - mark < kernel / 2);
  REQUIRE(again.out.has_stress == 1);
  sameStress(again, shifted_reference);

  mark = g_liveBlocks.load();
  REQUIRE_THROWS_AS(stressCall(pot, coincident, copper, cube, 4, failed),
                    std::domain_error);
  cleared(failed);
  kept(mark);

  mark = g_liveBlocks.load();
  Call after_domain;
  stressCall(pot, shifted, copper, cube, 4, after_domain);
  REQUIRE(g_liveBlocks.load() - mark < kernel / 2);
  REQUIRE(after_domain.out.has_stress == 1);
  sameStress(after_domain, shifted_reference);

  mark = g_liveBlocks.load();
  REQUIRE_THROWS_AS(stressCall(pot, pos, carbon, cube, 4, failed),
                    std::invalid_argument);
  cleared(failed);
  REQUIRE(mark - g_liveBlocks.load() > kernel / 2);

  mark = g_liveBlocks.load();
  Call rebuilt;
  stressCall(pot, pos, copper, cube, 4, rebuilt);
  REQUIRE(g_liveBlocks.load() - mark > kernel / 2);
  REQUIRE(rebuilt.out.has_stress == 1);
  sameStress(rebuilt, first);
}

TEST_CASE("EMT follows a change of the cell between calls", "[emt]") {
  constexpr int kRepeat = 4;
  constexpr double kLattice = 3.61;
  const std::array<std::array<double, 3>, 4> basis = {{
      {0.0, 0.0, 0.0},
      {0.0, 0.5, 0.5},
      {0.5, 0.0, 0.5},
      {0.5, 0.5, 0.0},
  }};
  const int n = 4 * kRepeat * kRepeat * kRepeat;
  rgpot::types::AtomMatrix strained(n, 3);
  rgpot::types::AtomMatrix relaxed(n, 3);
  int at = 0;
  for (int i = 0; i < kRepeat; ++i) {
    for (int j = 0; j < kRepeat; ++j) {
      for (int k = 0; k < kRepeat; ++k) {
        for (const auto &site : basis) {
          const std::array<double, 3> frac = {i + site[0], j + site[1],
                                              k + site[2]};
          for (int d = 0; d < 3; ++d) {
            const double wobble = 0.03 * std::sin(1.7 * at + 2.3 * d);
            relaxed(at, d) = kLattice * frac[static_cast<size_t>(d)] + wobble;
            strained(at, d) = 1.05 * relaxed(at, d);
          }
          ++at;
        }
      }
    }
  }
  const std::vector<int> copper(static_cast<size_t>(n), 29);
  const double side = kLattice * kRepeat;
  const Box cube = {{{side, 0.0, 0.0}, {0.0, side, 0.0}, {0.0, 0.0, side}}};
  const Box stretched = {{{1.05 * side, 0.0, 0.0},
                          {0.0, 1.05 * side, 0.0},
                          {0.0, 0.0, 1.05 * side}}};

  rgpot::EMTPot reused;
  (void)reused(relaxed, copper, cube);
  const auto [e_reused, f_reused, v_reused] =
      reused(strained, copper, stretched);
  rgpot::EMTPot fresh;
  const auto [e_fresh, f_fresh, v_fresh] = fresh(strained, copper, stretched);

  REQUIRE_THAT(e_reused, WithinAbs(e_fresh, 1e-9));
  for (int i = 0; i < n; ++i) {
    for (int d = 0; d < 3; ++d) {
      REQUIRE_THAT(f_reused(i, d), WithinAbs(f_fresh(i, d), 1e-9));
    }
  }
}

TEST_CASE("EMT advertises the stress it computes", "[emt][stress]") {
  rgpot::EMTPot plain;
  rgpot::EMTConfig rasmussen;
  rasmussen.rasmussen = true;
  rgpot::EMTPot other(rasmussen);
  CHECK(plain.caps().stress);
  CHECK(other.caps().stress);
}

TEST_CASE("EMT stress is opt-in and matches triclinic strain",
          "[emt][stress]") {
  auto system = fccCu(4, 3.615);
  const std::array<double, 9> shear = {1.0,  0.12, 0.03,  0.02, 0.96,
                                       0.08, 0.04, -0.03, 1.05};
  rightTransform(system.pos, system.box, shear);

  rgpot::EMTPot plain_pot;
  rgpot::EMTPot stress_pot;
  const Eval plain = runEval(plain_pot, system.pos, system.z, system.box);
  const Eval analytic =
      runEval(stress_pot, system.pos, system.z, system.box, true);
  REQUIRE(plain.has_stress == 0);
  REQUIRE(analytic.has_stress == 1);
  REQUIRE(f64Bits(plain.energy) == f64Bits(analytic.energy));
  REQUIRE(plain.forces.size() == analytic.forces.size());
  for (std::size_t i = 0; i < plain.forces.size(); ++i) {
    REQUIRE(f64Bits(plain.forces[i]) == f64Bits(analytic.forces[i]));
  }
  CHECK(analytic.stress[1] == analytic.stress[3]);
  CHECK(analytic.stress[2] == analytic.stress[6]);
  CHECK(analytic.stress[5] == analytic.stress[7]);

  static constexpr int slots[6] = {0, 4, 8, 5, 2, 1};
  double target_error = 0.0;
  double best_other = std::numeric_limits<double>::infinity();
  for (double step : {1e-3, 1e-4, 1e-5, 1e-6, 1e-7}) {
    double max_error = 0.0;
    double scale = 0.0;
    for (int component = 0; component < 6; ++component) {
      const int a = component < 3 ? component : (component == 3 ? 1 : 0);
      const int b = component < 3 ? component
                                  : (component == 3   ? 2
                                     : component == 4 ? 2
                                                      : 1);
      std::array<double, 9> plus_map{1, 0, 0, 0, 1, 0, 0, 0, 1};
      std::array<double, 9> minus_map = plus_map;
      const double weight = a == b ? step : 0.5 * step;
      plus_map[static_cast<std::size_t>(3 * a + b)] += weight;
      minus_map[static_cast<std::size_t>(3 * a + b)] -= weight;
      if (a != b) {
        plus_map[static_cast<std::size_t>(3 * b + a)] += weight;
        minus_map[static_cast<std::size_t>(3 * b + a)] -= weight;
      }
      auto plus_pos = system.pos;
      auto minus_pos = system.pos;
      auto plus_box = system.box;
      auto minus_box = system.box;
      rightTransform(plus_pos, plus_box, plus_map);
      rightTransform(minus_pos, minus_box, minus_map);
      rgpot::EMTPot plus_pot;
      rgpot::EMTPot minus_pot;
      const double ep = runEval(plus_pot, plus_pos, system.z, plus_box).energy;
      const double em =
          runEval(minus_pot, minus_pos, system.z, minus_box).energy;
      const double volume = std::abs(
          system.box[0] *
              (system.box[4] * system.box[8] - system.box[5] * system.box[7]) -
          system.box[1] *
              (system.box[3] * system.box[8] - system.box[5] * system.box[6]) +
          system.box[2] *
              (system.box[3] * system.box[7] - system.box[4] * system.box[6]));
      const double fd = (ep - em) / (2.0 * step * volume);
      const double value = analytic.stress[slots[component]];
      max_error = std::max(max_error, std::abs(value - fd));
      scale = std::max(scale, std::abs(value));
    }
    const double relative = max_error / std::max(scale, 1e-12);
    if (step == 1e-5) {
      target_error = relative;
    } else {
      best_other = std::min(best_other, relative);
    }
  }
  std::cout << "STRESS_ERROR emt " << std::setprecision(12) << target_error
            << '\n';
  CHECK(target_error <= 2e-7);
  CHECK(target_error <= best_other * 2.0 + 1e-12);
}

TEST_CASE("EMT stress follows a change of the cell on a reused instance",
          "[emt][stress]") {
  auto first = fccCu(4, 3.615);
  auto second = first;
  const std::array<double, 9> strain = {1.03, 0.01, 0.0,  0.0, 0.98,
                                        0.02, 0.0,  0.01, 1.04};
  rightTransform(second.pos, second.box, strain);

  rgpot::EMTPot reused;
  const Eval before = runEval(reused, first.pos, first.z, first.box, true);
  REQUIRE(before.has_stress == 1);
  // The second call changes the cell, so the cached calculator is rebuilt and
  // the stress is read from the rebuilt one.
  const Eval after = runEval(reused, second.pos, second.z, second.box, true);
  rgpot::EMTPot fresh;
  const Eval expected = runEval(fresh, second.pos, second.z, second.box, true);
  REQUIRE(after.has_stress == 1);
  REQUIRE(countBitDiffs(after, expected) == 0);
  CHECK(f64Bits(before.stress[0]) != f64Bits(after.stress[0]));

  // A force-only call on the same instance does not leave a stress behind.
  const Eval plain = runEval(reused, second.pos, second.z, second.box);
  CHECK(plain.has_stress == 0);
  for (const double component : plain.stress) {
    CHECK(component == 0.0);
  }
  CHECK_THAT(plain.energy, WithinAbs(after.energy, 1e-9));
}

TEST_CASE("EMT OpenMP force path is bit-identical across threads",
          "[emt][threads]") {
  ThreadGuard guard;
  rgpot::EMTPot sanity;

  const auto small = fccCu(12, 3.615 / std::sqrt(2.0));
  REQUIRE(small.z.size() < rgpot::kTeamMinAtoms);
  const auto small_reuse = shifted(small.pos, 0.02);
  const auto small_rebuild = shifted(small.pos, 0.35);

  ThreadGuard::set("1");
  auto ref_small = runEval(sanity, small.pos, small.z, small.box);
  auto ref_small_reuse = runEval(sanity, small_reuse, small.z, small.box);
  auto ref_small_rebuild = runEval(sanity, small_rebuild, small.z, small.box);
  REQUIRE(ref_small.has_stress == 0);

  for (const char *th : {"2", "4", "8"}) {
    INFO("small OMP_NUM_THREADS=" << th);
    ThreadGuard::set(th);
    const auto before = rgpot::teamsOpened();
    rgpot::EMTPot pot;
    CHECK(countBitDiffs(runEval(pot, small.pos, small.z, small.box),
                        ref_small) == 0);
    CHECK(countBitDiffs(runEval(pot, small_reuse, small.z, small.box),
                        ref_small_reuse) == 0);
    CHECK(countBitDiffs(runEval(pot, small_rebuild, small.z, small.box),
                        ref_small_rebuild) == 0);
    CHECK(rgpot::teamsOpened() == before);
  }

  const auto large = fccCu(13, 3.615 / std::sqrt(2.0));
  REQUIRE(large.z.size() >= rgpot::kTeamMinAtoms);
  const auto large_reuse = shifted(large.pos, 0.02);
  const auto large_rebuild = shifted(large.pos, 0.35);

  ThreadGuard::set("1");
  rgpot::EMTPot serial;
  auto ref_large = runEval(serial, large.pos, large.z, large.box);
  auto ref_large_reuse = runEval(serial, large_reuse, large.z, large.box);
  auto ref_large_rebuild = runEval(serial, large_rebuild, large.z, large.box);

  for (const char *th : {"2", "4", "8"}) {
    INFO("large OMP_NUM_THREADS=" << th);
    ThreadGuard::set(th);
    rgpot::EMTPot pot;
    CHECK(countBitDiffs(runEval(pot, large.pos, large.z, large.box),
                        ref_large) == 0);
    const auto before = rgpot::teamsOpened();
    CHECK(countBitDiffs(runEval(pot, large_reuse, large.z, large.box),
                        ref_large_reuse) == 0);
#if defined(_OPENMP)
    CHECK(rgpot::teamsOpened() == before + 1);
#endif
    CHECK(countBitDiffs(runEval(pot, large_rebuild, large.z, large.box),
                        ref_large_rebuild) == 0);
  }
}

TEST_CASE("EMT opens its team from a TeamThreshold", "[emt][threads]") {
  ThreadGuard guard;
  const auto small = fccCu(12, 3.615 / std::sqrt(2.0));
  REQUIRE(small.z.size() < rgpot::kTeamMinAtoms);
  const rgpot::TeamThreshold any(small.z.size());
  const auto moved = shifted(small.pos, 0.02);

  ThreadGuard::set("1");
  rgpot::EMTPot serial;
  (void)runEval(serial, small.pos, small.z, small.box);
  const auto ref = runEval(serial, moved, small.z, small.box);

  for (const char *th : {"2", "8"}) {
    INFO("OMP_NUM_THREADS=" << th);
    ThreadGuard::set(th);
    rgpot::EMTPot pot;
    (void)runEval(pot, small.pos, small.z, small.box);
    const auto before = rgpot::teamsOpened();
    CHECK(countBitDiffs(runEval(pot, moved, small.z, small.box), ref) == 0);
#if defined(_OPENMP)
    CHECK(rgpot::teamsOpened() == before + 1);
#else
    CHECK(rgpot::teamsOpened() == before);
#endif
  }
}

TEST_CASE("EMT Exception keeps its message per object and per copy", "[emt]") {
  static_assert(sizeof(Exception) == sizeof(void *),
                "Exception is a single pointer in the installed header");
  static_assert(noexcept(std::declval<const Exception &>().GetMessage()),
                "GetMessage must remain noexcept");

  Exception first("first: ");
  first << 108;
  Exception second("second: ");
  second << 256;

  const char *first_msg = first.GetMessage();
  const char *second_msg = second.GetMessage();
  CHECK(first.GetMessage() == first_msg);
  CHECK(std::string_view(first_msg) == "first: 108");
  CHECK(std::string_view(second_msg) == "second: 256");

  Exception copied(first);
  copied << " copied";
  CHECK(std::string_view(first.GetMessage()) == "first: 108");
  CHECK(std::string_view(copied.GetMessage()) == "first: 108 copied");

  Exception assigned("initial message exceeding small string buffer capacity");
  assigned = second;
  assigned << " assigned";
  CHECK(std::string_view(second.GetMessage()) == "second: 256");
  CHECK(std::string_view(assigned.GetMessage()) == "second: 256 assigned");
  CHECK(std::string_view(first_msg) == "first: 108");

  first << " updated";
  CHECK(std::string_view(first.GetMessage()) == "first: 108 updated");

  Exception &self = assigned;
  assigned = self;
  CHECK(std::string_view(assigned.GetMessage()) == "second: 256 assigned");

  Exception null_message(nullptr);
  CHECK(std::string_view(null_message.GetMessage()).empty());
}
