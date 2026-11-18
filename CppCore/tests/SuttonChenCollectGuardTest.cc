// MIT License
// Copyright 2023--present rgpot developers

// Below kTeamMinAtoms, SuttonChenPot gathers the pairs of a stored list into
// arrays. A stream or an unbounded search has no list to gather, and the
// gather would hold every pair at once, 40 bytes each. The largest single
// allocation of a force call shows whether it took that path.

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <new>
#include <vector>

#include <catch2/catch_all.hpp>

#include "rgpot/SuttonChen/SuttonChenPot.hpp"
#include "rgpot/nlist/Neighbours.hpp"

namespace {

std::atomic<bool> g_tracking{false};
std::atomic<std::size_t> g_largest{0};

void note(std::size_t bytes) noexcept {
  if (!g_tracking.load(std::memory_order_relaxed)) {
    return;
  }
  std::size_t seen = g_largest.load(std::memory_order_relaxed);
  while (bytes > seen && !g_largest.compare_exchange_weak(seen, bytes)) {
  }
}

void *grab(std::size_t bytes) {
  note(bytes);
  if (void *p = std::malloc(bytes != 0 ? bytes : 1)) {
    return p;
  }
  throw std::bad_alloc();
}

/// Largest single allocation made while `call` runs.
template <typename Fn> std::size_t largestAllocation(Fn &&call) {
  g_largest.store(0);
  g_tracking.store(true);
  call();
  g_tracking.store(false);
  return g_largest.load();
}

std::vector<double> cubicGrid(int per_side, double spacing) {
  std::vector<double> pos;
  pos.reserve(static_cast<std::size_t>(per_side) * per_side * per_side * 3);
  for (int ix = 0; ix < per_side; ++ix) {
    for (int iy = 0; iy < per_side; ++iy) {
      for (int iz = 0; iz < per_side; ++iz) {
        pos.push_back(ix * spacing);
        pos.push_back(iy * spacing);
        pos.push_back(iz * spacing);
      }
    }
  }
  return pos;
}

/// The per-atom arrays of a call are a few hundred KiB. A pair array of the
/// systems below is 15 MiB or more.
constexpr std::size_t kLargestAllowed = std::size_t{1} << 20;

constexpr std::size_t kAtoms = 8000;

} // namespace

void *operator new(std::size_t bytes) { return grab(bytes); }
void *operator new[](std::size_t bytes) { return grab(bytes); }
void operator delete(void *p) noexcept { std::free(p); }
void operator delete[](void *p) noexcept { std::free(p); }
void operator delete(void *p, std::size_t) noexcept { std::free(p); }
void operator delete[](void *p, std::size_t) noexcept { std::free(p); }

TEST_CASE("Sutton-Chen below the team size gathers pairs only from a stored list",
          "[SuttonChen][regression]") {
  constexpr int kSide = 20;
  static_assert(kSide * kSide * kSide == kAtoms);
  constexpr double kSpacing = 2.5;
  const std::vector<double> pos = cubicGrid(kSide, kSpacing);
  REQUIRE(pos.size() == 3 * kAtoms);
  double box[9] = {};
  box[0] = box[4] = box[8] = kSide * kSpacing;
  std::vector<double> forces(pos.size(), 0.0);

  auto run = [&](const rgpot::SuttonChenConfig &cfg, rgpot::nlist::Mode mode) {
    rgpot::SuttonChenPot pot(cfg);
    pot.setNeighbourMode(mode);
    pot.setPairTermLimit(0);
    rgpot::ForceOut out{};
    out.F = forces.data();
    const rgpot::ForceInput in{kAtoms, pos.data(), nullptr, cfg.periodic ? box : nullptr};
    return largestAllocation([&] { pot.forceImpl(in, &out); });
  };

  rgpot::SuttonChenConfig bounded;
  bounded.epsilon = 0.01;
  bounded.c = 30.0;
  bounded.a = 2.5;
  bounded.cutoff = 12.0;
  bounded.periodic = true;

  SECTION("a stored list gathers its pairs") {
    const std::size_t stored = run(bounded, rgpot::nlist::Mode::Stored);
    REQUIRE(stored > kLargestAllowed);
  }

  SECTION("a stream gathers nothing") {
    const std::size_t streamed = run(bounded, rgpot::nlist::Mode::Stream);
    REQUIRE(streamed <= kLargestAllowed);
  }

  SECTION("an unbounded search gathers nothing") {
    rgpot::SuttonChenConfig all = bounded;
    all.cutoff = 0.0;
    all.periodic = false;
    const std::size_t unbounded = run(all, rgpot::nlist::Mode::Auto);
    REQUIRE(unbounded <= kLargestAllowed);
  }
}
