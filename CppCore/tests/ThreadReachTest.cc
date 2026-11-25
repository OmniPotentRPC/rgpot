// MIT License
// Copyright 2023--present rgpot developers
//
// A potential's thread count reaches the threads the potential uses. The
// batch fan-out of every shared-instance potential takes the count of the
// object it was called on. Each check watches the team an OpenMP region
// really has, and not only that the call returns.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include <catch2/catch_all.hpp>

#include "lsan_libomp.hpp"
#include "rgpot/ForceStructs.hpp"
#include "rgpot/Potential.hpp"
#include "rgpot/Slices.hpp"

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace {

void setThreads(const char *value) {
#if defined(_WIN32)
  _putenv_s("OMP_NUM_THREADS", value == nullptr ? "" : value);
#else
  if (value == nullptr || value[0] == '\0') {
    unsetenv("OMP_NUM_THREADS");
  } else {
    setenv("OMP_NUM_THREADS", value, 1);
  }
#endif
}

// The largest team a region can start for `wanted`: the runtime caps it at
// OMP_THREAD_LIMIT. Without OpenMP every region is the calling thread.
int teamFor(int wanted) {
#if defined(_OPENMP)
  return std::max(1, std::min(wanted, omp_get_thread_limit()));
#else
  (void)wanted;
  return 1;
#endif
}

bool canSpread() {
#if defined(_OPENMP)
  return omp_get_thread_limit() >= 2;
#else
  return false;
#endif
}

int teamHere() {
#if defined(_OPENMP)
  return omp_get_num_threads();
#else
  return 1;
#endif
}

struct System {
  std::vector<double> pos;
  std::vector<int> z;
  std::array<double, 9> box{};
};

struct Eval {
  double energy = 0.0;
  std::vector<double> f;
  std::array<double, 9> stress{};
  int has_stress = 0;
};

std::vector<Eval> runBatch(rgpot::PotentialBase &pot,
                           const std::vector<System> &systems) {
  std::vector<Eval> evals(systems.size());
  std::vector<rgpot::ForceInput> in;
  in.reserve(systems.size());
  std::vector<rgpot::ForceOut> out(systems.size());
  for (std::size_t i = 0; i < systems.size(); ++i) {
    evals[i].f.assign(systems[i].pos.size(), 0.0);
    in.push_back(rgpot::ForceInput{systems[i].z.size(), systems[i].pos.data(),
                                   systems[i].z.data(),
                                   systems[i].box.data()});
    out[i].F = evals[i].f.data();
  }
  pot.forceBatch(rgpot::ForceBatch{
      .nSystems = systems.size(), .in = in.data(), .out = out.data()});
  for (std::size_t i = 0; i < systems.size(); ++i) {
    evals[i].energy = out[i].energy;
    evals[i].has_stress = out[i].has_stress;
    std::copy(out[i].stress, out[i].stress + 9, evals[i].stress.begin());
  }
  return evals;
}

// What one Probe call saw on the thread that ran it.
struct Seen {
  int explicit_count = -1;
  int requested = -1;
  int outer_team = 0;
  int inner_team = 0;
};

class Log {
public:
  void add(const Seen &s) {
    std::lock_guard<std::mutex> lock(mutex_);
    seen_.push_back(s);
  }
  std::vector<Seen> take() {
    std::lock_guard<std::mutex> lock(mutex_);
    return seen_;
  }
  void clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    seen_.clear();
  }

private:
  std::mutex mutex_;
  std::vector<Seen> seen_;
};

// Binds its own count as every shipped kernel does, reads back the count a
// parent bound for it, and opens one region of the size that count asks for.
class Probe final : public rgpot::Potential<Probe> {
public:
  explicit Probe(std::shared_ptr<Log> log, bool fail = false)
      : rgpot::Potential<Probe>(rgpot::PotType::UNKNOWN),
        log_(std::move(log)), fail_(fail) {}

  void forceImpl(const rgpot::ForceInput &in,
                 rgpot::ForceOut *out) const override {
    const rgpot::ThreadBinding bind_threads{threadCount()};
    Seen s;
    s.explicit_count = rgpot::explicitThreadCount();
    s.requested = rgpot::threadsRequested();
    s.outer_team = teamHere();
    int inner = 1;
    std::mutex gate;
    rgpot::forEachSlice(s.requested, [&](int) {
      const int team = teamHere();
      std::lock_guard<std::mutex> lock(gate);
      inner = std::max(inner, team);
    });
    s.inner_team = inner;
    log_->add(s);
    if (fail_) {
      throw std::runtime_error("probe failed");
    }
    out->energy = 1.0;
    for (std::size_t k = 0; k < 3 * in.nAtoms; ++k) {
      out->F[k] = 0.5;
    }
  }

private:
  std::shared_ptr<Log> log_;
  bool fail_;
};

System pair(int z, double gap) {
  System s;
  s.pos = {0.0, 0.0, 0.0, 0.0, 0.0, gap};
  s.z = {z, z};
  return s;
}

std::vector<System> smallSystems(std::size_t count) {
  std::vector<System> systems;
  for (std::size_t i = 0; i < count; ++i) {
    systems.push_back(pair(1, 2.0 + 0.1 * static_cast<double>(i)));
  }
  REQUIRE(systems.front().z.size() < rgpot::kTeamMinAtoms);
  return systems;
}

int widestOuter(const std::vector<Seen> &seen) {
  int widest = 0;
  for (const Seen &s : seen) {
    widest = std::max(widest, s.outer_team);
  }
  return widest;
}

} // namespace

TEST_CASE("a batch of small systems spreads over the potential's own count",
          "[threads][batch]") {
  setThreads("");
  const auto systems = smallSystems(8);
  auto log = std::make_shared<Log>();

  SECTION("no count and no OMP_NUM_THREADS stays on the calling thread") {
    Probe pot(log);
    const std::uint64_t before = rgpot::teamsOpened();
    runBatch(pot, systems);
    CHECK(rgpot::teamsOpened() == before);
    CHECK(widestOuter(log->take()) == 1);
  }

  SECTION("setThreadCount sets the team of the fan-out") {
    if (!canSpread()) {
      SKIP("this runtime cannot start a team of two");
    }
    for (const int threads : {2, 4}) {
      INFO("setThreadCount(" << threads << ")");
      log->clear();
      Probe pot(log);
      pot.setThreadCount(threads);
      const std::uint64_t before = rgpot::teamsOpened();
      runBatch(pot, systems);
#if defined(_OPENMP)
      CHECK(rgpot::teamsOpened() > before);
      CHECK(widestOuter(log->take()) == teamFor(threads));
#endif
    }
  }

  SECTION("every system sees the count on whichever thread runs it") {
    Probe pot(log);
    pot.setThreadCount(4);
    runBatch(pot, systems);
    const auto seen = log->take();
    REQUIRE(seen.size() == systems.size());
    for (const Seen &s : seen) {
      CHECK(s.explicit_count == 4);
    }
  }

  SECTION("the potential's count beats OMP_NUM_THREADS") {
    if (!canSpread()) {
      SKIP("this runtime cannot start a team of two");
    }
    setThreads("3");
    Probe pot(log);
    pot.setThreadCount(2);
    runBatch(pot, systems);
#if defined(_OPENMP)
    CHECK(widestOuter(log->take()) == teamFor(2));
#endif
    setThreads("");
  }

  SECTION("a potential without a count keeps the OMP_NUM_THREADS team") {
    if (!canSpread()) {
      SKIP("this runtime cannot start a team of two");
    }
    setThreads("3");
    Probe pot(log);
    runBatch(pot, systems);
#if defined(_OPENMP)
    CHECK(widestOuter(log->take()) == teamFor(3));
#endif
    setThreads("");
  }

  SECTION("a ThreadCount on the call wins over the potential") {
    Probe pot(log);
    pot.setThreadCount(4);
    const rgpot::ThreadCount one(1);
    const std::uint64_t before = rgpot::teamsOpened();
    runBatch(pot, systems);
    CHECK(rgpot::teamsOpened() == before);
    for (const Seen &s : log->take()) {
      CHECK(s.explicit_count == 1);
      CHECK(s.outer_team == 1);
    }
  }

  SECTION("the binding ends with the batch, also when a system throws") {
    Probe pot(log, true);
    pot.setThreadCount(4);
    CHECK_THROWS_AS(runBatch(pot, systems), std::runtime_error);
    CHECK(rgpot::explicitThreadCount() == 0);
  }
  setThreads("");
}
