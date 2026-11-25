// MIT License
// Copyright 2023--present rgpot developers
//
// Energy and forces are bit-identical at 1, 2, 4 and 8 threads, whether the
// count comes from OMP_NUM_THREADS or from the opt-in thread API. The
// one-thread loop is the one a call runs today when nothing is set.

#include <algorithm>
#include <atomic>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <catch2/catch_all.hpp>

#include "rgpot/EAM/EAMPot.hpp"
#include "rgpot/EMT/EMTPot.hpp"
#include "rgpot/LennardJones/LJClusterPot.hpp"
#include "rgpot/LennardJones/LJPot.hpp"
#include "rgpot/Morse/MorsePot.hpp"
#include "rgpot/Slices.hpp"
#include "rgpot/StillingerWeber/SWSi.hpp"
#include "rgpot/SuttonChen/SuttonChenPot.hpp"
#include "rgpot/Tersoff/TersoffSiC.hpp"
#include "rgpot/ZBL/ZBLPot.hpp"

#if defined(RGPOT_HAS_LONG_RANGE)
#include "rgpot/LongRange/Dispersion.hpp"
#include "rgpot/LongRange/Electro.hpp"
#include "rgpot/LongRange/Team.hpp"

#include <cstdio>
#include <future>
#endif

#if defined(_OPENMP)
#include <omp.h>
#endif

#if defined(__has_feature)
#if __has_feature(thread_sanitizer)
#define RGPOT_THREAD_SANITIZER 1
#endif
#endif
#if defined(__SANITIZE_THREAD__)
#define RGPOT_THREAD_SANITIZER 1
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

std::uint64_t bits(double v) {
  std::uint64_t out = 0;
  std::memcpy(&out, &v, sizeof(out));
  return out;
}

struct System {
  std::vector<double> pos;
  std::vector<int> z;
  double length = 0;
};

System lattice(const std::vector<std::array<double, 3>> &basis, int n, double a,
               int z) {
  System s;
  for (int ix = 0; ix < n; ++ix) {
    for (int iy = 0; iy < n; ++iy) {
      for (int iz = 0; iz < n; ++iz) {
        for (const auto &p : basis) {
          const double cell[3] = {ix + p[0], iy + p[1], iz + p[2]};
          for (const double c : cell) {
            const auto k = static_cast<double>(s.pos.size());
            s.pos.push_back(c * a + 0.01 * a * std::sin(1.7 * k + 0.3));
          }
          s.z.push_back(z);
        }
      }
    }
  }
  s.length = n * a;
  return s;
}

System fcc(int n, double a, int z) {
  return lattice({{0, 0, 0}, {0, 0.5, 0.5}, {0.5, 0, 0.5}, {0.5, 0.5, 0}}, n, a,
                 z);
}

System diamond(int n, double a, int z) {
  return lattice({{0, 0, 0},
                  {0.5, 0.5, 0},
                  {0.5, 0, 0.5},
                  {0, 0.5, 0.5},
                  {0.25, 0.25, 0.25},
                  {0.75, 0.75, 0.25},
                  {0.75, 0.25, 0.75},
                  {0.25, 0.75, 0.75}},
                 n, a, z);
}

struct Eval {
  double energy = 0;
  std::vector<double> force;
};

template <typename Pot> Eval run(Pot &pot, const System &s) {
  const std::size_t n = s.z.size();
  const double box[9] = {s.length, 0, 0, 0, s.length, 0, 0, 0, s.length};
  Eval e;
  e.force.assign(3 * n, 0.0);
  rgpot::ForceOut out{};
  out.F = e.force.data();
  pot.forceImpl(rgpot::ForceInput{n, s.pos.data(), s.z.data(), box}, &out);
  e.energy = out.energy;
  return e;
}

std::size_t forceApart(const Eval &a, const Eval &b) {
  std::size_t apart = bits(a.energy) == bits(b.energy) ? 0 : 1;
  const std::size_t n = std::min(a.force.size(), b.force.size());
  apart += a.force.size() == b.force.size() ? 0 : 1;
  for (std::size_t i = 0; i < n; ++i) {
    apart += bits(a.force[i]) == bits(b.force[i]) ? 0 : 1;
  }
  return apart;
}

// Counts the suite actually runs. Eight is included: this host has eight
// processors. A build without OpenMP still walks the loop, on one thread.
const int kCounts[] = {1, 2, 4, 8};

template <typename Make>
void checkBits(const char *name, Make make, const System &s,
               std::uint64_t pinned = 0) {
  INFO(name << " on " << s.z.size() << " atoms");
  REQUIRE(s.z.size() >= rgpot::kTeamMinAtoms);
  setThreads("");
  Eval serial = run(*make(), s);
  if (pinned != 0) {
    CHECK(bits(serial.energy) == pinned);
  }
  for (const int threads : kCounts) {
    INFO("OMP_NUM_THREADS=" << threads);
    setThreads(std::to_string(threads).c_str());
    CHECK(forceApart(run(*make(), s), serial) == 0);
    setThreads("");
    INFO("ThreadCount(" << threads << ")");
    rgpot::ThreadCount scope(threads);
#if defined(_OPENMP)
    const int expected_threads = threads;
#else
    const int expected_threads = 1;
#endif
    CHECK(rgpot::threadsRequested() == expected_threads);
    CHECK(forceApart(run(*make(), s), serial) == 0);
  }
  setThreads("");
  auto opted = make();
  opted->setThreadCount(4);
  CHECK(rgpot::threadsRequested() == 1);
  const std::uint64_t before = rgpot::teamsOpened();
  CHECK(forceApart(run(*opted, s), serial) == 0);
#if defined(_OPENMP)
  if (omp_get_thread_limit() >= 2) {
    CHECK(rgpot::teamsOpened() > before);
  }
#endif
  // A call scope wins over the potential. A fresh object: a second
  // evaluation of the same EMT object is not bit-stable on develop.
  {
    auto scoped = make();
    scoped->setThreadCount(4);
    rgpot::ThreadCount one(1);
    CHECK(rgpot::threadsRequested() == 1);
    CHECK(forceApart(run(*scoped, s), serial) == 0);
  }
  setThreads("");
}

#if defined(RGPOT_HAS_LONG_RANGE)
// Rock salt with the same small displacement as `lattice`, +1 on the first
// four basis sites and -1 on the last four.
struct Ionic {
  System s;
  std::vector<double> q;
  double total = 0.0;
};

Ionic rockSalt(int n, double a) {
  Ionic out;
  out.s = lattice({{0, 0, 0},
                   {0, 0.5, 0.5},
                   {0.5, 0, 0.5},
                   {0.5, 0.5, 0},
                   {0.5, 0, 0},
                   {0, 0.5, 0},
                   {0, 0, 0.5},
                   {0.5, 0.5, 0.5}},
                  n, a, 11);
  for (std::size_t i = 0; i < out.s.z.size(); ++i) {
    const bool cation = i % 8 < 4;
    out.s.z[i] = cation ? 11 : 17;
    out.q.push_back(cation ? 1.0 : -1.0);
  }
  return out;
}

struct CoulombEval {
  double energy = 0;
  std::vector<double> force;
  double stress[9]{};
  int has_stress = 0;
};

std::size_t coulombApart(const CoulombEval &a, const CoulombEval &b) {
  std::size_t apart = bits(a.energy) == bits(b.energy) ? 0 : 1;
  apart += a.force.size() == b.force.size() ? 0 : 1;
  for (std::size_t i = 0; i < std::min(a.force.size(), b.force.size()); ++i) {
    apart += bits(a.force[i]) == bits(b.force[i]) ? 0 : 1;
  }
  apart += a.has_stress == b.has_stress ? 0 : 1;
  for (int k = 0; k < 9; ++k) {
    apart += bits(a.stress[k]) == bits(b.stress[k]) ? 0 : 1;
  }
  return apart;
}

// Runs `fn` on its own thread and ends the process when it does not come
// back: a rank stuck at the mesh gate never returns, and a hung suite
// would hide which call it was.
template <typename Fn> auto guarded(const std::string &what, Fn fn) {
  using Result = decltype(fn());
  std::packaged_task<Result()> task(std::move(fn));
  auto done = task.get_future();
  std::thread worker(std::move(task));
  if (done.wait_for(std::chrono::seconds(120)) != std::future_status::ready) {
    std::fprintf(stderr, "%s did not finish in 120 s; the mesh group is deadlocked\n",
                 what.c_str());
    std::fflush(stderr);
    std::_Exit(3);
  }
  worker.join();
  return done.get();
}

CoulombEval runCoulomb(const rgpot::lr::CoulombConfig &cfg, const Ionic &ion,
                       int scope_threads, int pot_threads) {
  return guarded("CoulombPot", [&] {
    rgpot::lr::CoulombPot pot(
        cfg, std::make_shared<rgpot::lr::FixedCharges>(ion.q, ion.total));
    if (pot_threads > 0) {
      pot.setThreadCount(pot_threads);
    }
    std::unique_ptr<rgpot::ThreadCount> scope;
    if (scope_threads > 0) {
      scope = std::make_unique<rgpot::ThreadCount>(scope_threads);
    }
    const std::size_t n = ion.s.z.size();
    const double L = ion.s.length;
    const double box[9] = {L, 0, 0, 0, L, 0, 0, 0, L};
    CoulombEval e;
    e.force.assign(3 * n, 0.0);
    rgpot::ForceOut out{};
    out.F = e.force.data();
    pot.forceImpl(rgpot::ForceInput{n, ion.s.pos.data(), ion.s.z.data(), box}, &out);
    e.energy = out.energy;
    e.has_stress = out.has_stress;
    std::copy(out.stress, out.stress + 9, e.stress);
    return e;
  });
}

// Bit-identical energy, forces and stress for a call scope and for a
// potential binding at every count, against the one-thread call.
void checkCoulombBits(const char *name, const rgpot::lr::CoulombConfig &cfg,
                      const Ionic &ion) {
  INFO(name << " on " << ion.s.z.size() << " atoms");
  const CoulombEval serial = runCoulomb(cfg, ion, 1, 0);
  REQUIRE(std::isfinite(serial.energy));
  CHECK(coulombApart(runCoulomb(cfg, ion, 0, 0), serial) == 0);
  for (const int threads : kCounts) {
    INFO("ThreadCount(" << threads << ")");
    CHECK(coulombApart(runCoulomb(cfg, ion, threads, 0), serial) == 0);
    INFO("setThreadCount(" << threads << ")");
    CHECK(coulombApart(runCoulomb(cfg, ion, 0, threads), serial) == 0);
  }
  // A second call on warm caches takes the same bits.
  CHECK(coulombApart(runCoulomb(cfg, ion, 8, 0), serial) == 0);
}

TEST_CASE("DSF screen caches are safe across concurrent potential instances",
          "[threads][invariant][coulomb]") {
  rgpot::lr::CoulombConfig cfg;
  cfg.method = rgpot::lr::CoulombMethod::DSF;
  cfg.cutoff = 8.0;
  cfg.alpha = 0.25;
  cfg.periodic = false;
  const Ionic small = rockSalt(3, 5.6);
  const Ionic large = rockSalt(4, 5.7);
  const CoulombEval small_ref = runCoulomb(cfg, small, 1, 0);
  const CoulombEval large_ref = runCoulomb(cfg, large, 1, 0);
  std::atomic<int> ready{0};
  std::atomic<bool> go{false};
  auto evaluate = [&](const Ionic &ion, const CoulombEval &reference) {
    ready.fetch_add(1, std::memory_order_release);
    while (!go.load(std::memory_order_acquire)) {
      std::this_thread::yield();
    }
    std::size_t apart = 0;
    for (int repeat = 0; repeat < 3; ++repeat) {
      apart += coulombApart(runCoulomb(cfg, ion, 1, 0), reference);
    }
    return apart;
  };
  auto small_run =
      std::async(std::launch::async, [&] { return evaluate(small, small_ref); });
  auto large_run =
      std::async(std::launch::async, [&] { return evaluate(large, large_ref); });
  while (ready.load(std::memory_order_acquire) != 2) {
    std::this_thread::yield();
  }
  go.store(true, std::memory_order_release);
  CHECK(small_run.get() == 0);
  CHECK(large_run.get() == 0);
}

CoulombEval evalCoulomb(const rgpot::lr::CoulombPot &pot, const Ionic &ion) {
  const std::size_t n = ion.s.z.size();
  const double L = ion.s.length;
  const double box[9] = {L, 0, 0, 0, L, 0, 0, 0, L};
  CoulombEval e;
  e.force.assign(3 * n, 0.0);
  rgpot::ForceOut out{};
  out.F = e.force.data();
  pot.forceImpl(rgpot::ForceInput{n, ion.s.pos.data(), ion.s.z.data(), box}, &out);
  e.energy = out.energy;
  e.has_stress = out.has_stress;
  std::copy(out.stress, out.stress + 9, e.stress);
  return e;
}

// Two instances, each on its own thread with its own team, against the
// one-thread result of a third instance. Every call of both runs while
// the other instance is evaluating. The systems, waves and meshes differ,
// so state one instance shares with the other is resized or overwritten
// under it.
void checkCoulombPair(const char *name, const rgpot::lr::CoulombConfig &a_cfg,
                      const Ionic &a_ion, const rgpot::lr::CoulombConfig &b_cfg,
                      const Ionic &b_ion) {
  INFO(name);
  const CoulombEval a_ref = runCoulomb(a_cfg, a_ion, 1, 0);
  const CoulombEval b_ref = runCoulomb(b_cfg, b_ion, 1, 0);
  REQUIRE(std::isfinite(a_ref.energy));
  REQUIRE(std::isfinite(b_ref.energy));
  for (const int threads : kCounts) {
    INFO("setThreadCount(" << threads << ") on both instances");
    constexpr int kRepeats = 4;
    std::atomic<int> ready{0};
    std::atomic<int> running{0};
    auto evaluate = [&](const rgpot::lr::CoulombConfig &cfg, const Ionic &ion,
                        const CoulombEval &reference) {
      rgpot::lr::CoulombPot pot(
          cfg, std::make_shared<rgpot::lr::FixedCharges>(ion.q, 0.0));
      pot.setThreadCount(threads);
      ready.fetch_add(1, std::memory_order_acq_rel);
      while (ready.load(std::memory_order_acquire) != 2) {
        std::this_thread::yield();
      }
      running.fetch_add(1, std::memory_order_acq_rel);
      std::size_t apart = 0;
      for (int repeat = 0; repeat < kRepeats; ++repeat) {
        apart += coulombApart(evalCoulomb(pot, ion), reference);
      }
      // A finished instance keeps evaluating until the other one is done,
      // so neither runs its last call alone.
      running.fetch_sub(1, std::memory_order_acq_rel);
      while (running.load(std::memory_order_acquire) != 0) {
        apart += coulombApart(evalCoulomb(pot, ion), reference);
      }
      return apart;
    };
    auto a_run = std::async(std::launch::async,
                            [&] { return evaluate(a_cfg, a_ion, a_ref); });
    auto b_run = std::async(std::launch::async,
                            [&] { return evaluate(b_cfg, b_ion, b_ref); });
    for (auto *run : {&a_run, &b_run}) {
      if (run->wait_for(std::chrono::seconds(300)) != std::future_status::ready) {
        std::fprintf(stderr, "%s at %d threads did not finish in 300 s\n", name,
                     threads);
        std::fflush(stderr);
        std::_Exit(3);
      }
    }
    CHECK(a_run.get() == 0);
    CHECK(b_run.get() == 0);
  }
}
#endif

} // namespace

TEST_CASE("the explicit thread count is opt-in and rejects a count below 1",
          "[threads][invariant]") {
  setThreads("");
#if defined(_OPENMP)
  omp_set_num_threads(4);
  CHECK(rgpot::poolThreads() == 4);
#endif
  CHECK(rgpot::threadsRequested() == 1);
  CHECK(rgpot::explicitThreadCount() == 0);
  CHECK_THROWS_AS(rgpot::setThreadCount(0), std::invalid_argument);
  CHECK_THROWS_AS(rgpot::ThreadCount(0), std::invalid_argument);
  rgpot::setThreadCount(3);
  CHECK(rgpot::explicitThreadCount() == 3);
#if defined(_OPENMP)
  CHECK(rgpot::threadsRequested() == 3);
  CHECK(rgpot::activeThreads(8) == 3);
#else
  CHECK(rgpot::threadsRequested() == 1);
  CHECK(rgpot::activeThreads(8) == 1);
#endif
  {
    rgpot::ThreadCount inner(2);
    CHECK(rgpot::explicitThreadCount() == 2);
  }
  CHECK(rgpot::explicitThreadCount() == 3);
  rgpot::clearThreadCount();
  CHECK(rgpot::explicitThreadCount() == 0);
  CHECK(rgpot::threadsRequested() == 1);
  setThreads("");
}

TEST_CASE("potential thread counts are synchronized across callers",
          "[threads][invariant]") {
  rgpot::LJPot pot(rgpot::LJConfig{});
  std::atomic<int> invalid{0};
  std::vector<std::thread> callers;
  for (int worker = 0; worker < 8; ++worker) {
    callers.emplace_back([&, worker] {
      for (int i = 0; i < 2000; ++i) {
        if ((i + worker) % 3 == 0) {
          pot.clearThreadCount();
        } else {
          pot.setThreadCount((i + worker) % 2 == 0 ? 2 : 4);
        }
        const int observed = pot.threadCount();
        if (observed != 0 && observed != 2 && observed != 4) {
          invalid.fetch_add(1, std::memory_order_relaxed);
        }
      }
    });
  }
  for (auto &caller : callers) {
    caller.join();
  }
  CHECK(invalid.load(std::memory_order_relaxed) == 0);
  pot.clearThreadCount();
  CHECK(pot.threadCount() == 0);
}

TEST_CASE("destroying a potential erases its thread count",
          "[threads][invariant]") {
  alignas(rgpot::LJPot) unsigned char storage[sizeof(rgpot::LJPot)];
  auto *first = std::construct_at(reinterpret_cast<rgpot::LJPot *>(storage),
                                  rgpot::LJConfig{});
  first->setThreadCount(4);
  REQUIRE(first->threadCount() == 4);
  std::destroy_at(first);

  auto *second = std::construct_at(reinterpret_cast<rgpot::LJPot *>(storage),
                                   rgpot::LJConfig{});
  CHECK(second->threadCount() == 0);
  std::destroy_at(second);
}

TEST_CASE("classical potentials match at 1, 2, 4 and 8 threads",
          "[threads][invariant]") {
  const double a_lj = std::pow(2.0, 1.0 / 6.0) * std::sqrt(2.0);
  rgpot::LJConfig lj;
  lj.psi = 1.0;
  lj.cutoff = 1.8;
  const System lj_sys = fcc(13, a_lj, 18);
  // Pinned from the unmodified develop build of this lattice.
  checkBits(
      "LJ",
      [&] {
        auto p = std::make_unique<rgpot::LJPot>(lj);
        p->setNeighbourMode(rgpot::nlist::Mode::Stored);
        return p;
      },
      lj_sys, 0xc0e8293f86540e63ull);

  rgpot::LJClusterConfig cluster;
  cluster.cutoff = lj.cutoff;
  checkBits(
      "LJ cluster",
      [&] {
        auto p = std::make_unique<rgpot::LJClusterPot>(cluster);
        p->setNeighbourMode(rgpot::nlist::Mode::Stored);
        return p;
      },
      lj_sys);

  const System zbl_sys = fcc(13, 1.6, 14);
  checkBits(
      "ZBL",
      [] {
        auto p = std::make_unique<rgpot::ZBLPot>(rgpot::ZBLConfig{});
        p->setNeighbourMode(rgpot::nlist::Mode::Stored);
        return p;
      },
      zbl_sys, 0x413b05ac92c8a622ull);

  checkBits(
      "EAM",
      [] {
        auto p = std::make_unique<rgpot::EAMPot>();
        p->setNeighbourMode(rgpot::nlist::Mode::Stored);
        return p;
      },
      fcc(13, 4.05, 13), 0xc0d9eb38227bfc36ull);

  rgpot::MorseConfig morse;
  morse.cutoff = 4.5;
  checkBits(
      "Morse",
      [&] {
        auto p = std::make_unique<rgpot::MorsePot>(morse);
        p->setNeighbourMode(rgpot::nlist::Mode::Stored);
        return p;
      },
      fcc(13, 4.1, 29));

  rgpot::SuttonChenConfig sc;
  sc.periodic = true;
  sc.cutoff = 1.5;
  checkBits(
      "Sutton-Chen",
      [&] {
        auto p = std::make_unique<rgpot::SuttonChenPot>(sc);
        p->setNeighbourMode(rgpot::nlist::Mode::Stored);
        return p;
      },
      fcc(13, std::sqrt(2.0), 78));

  const System si = diamond(11, 5.431, 14);
  checkBits(
      "Stillinger-Weber",
      [] {
        auto p = std::make_unique<rgpot::StillingerWeberSi>();
        p->setNeighbourMode(rgpot::nlist::Mode::Stored);
        return p;
      },
      si);
  checkBits(
      "Tersoff",
      [] {
        auto p = std::make_unique<rgpot::TersoffSiC>();
        p->setNeighbourMode(rgpot::nlist::Mode::Stored);
        return p;
      },
      si);

  checkBits(
      "EMT", [] { return std::make_unique<rgpot::EMTPot>(); },
      fcc(13, 4.05, 13));

#if defined(RGPOT_HAS_LONG_RANGE)
  rgpot::lr::DispersionConfig qdo;
  qdo.kind = rgpot::lr::DispersionKind::QDO;
  qdo.cutoff = 8.0;
  qdo.periodic = true;
  qdo.skin = 0.0;
  checkBits(
      "QDO", [&] { return std::make_unique<rgpot::lr::DispersionPot>(qdo); },
      fcc(13, 3.5, 6));
#endif
}

#if defined(RGPOT_HAS_LONG_RANGE)
TEST_CASE("Coulomb DSF, Ewald and SPME match at 1, 2, 4 and 8 threads",
          "[threads][invariant][coulomb]") {
  setThreads("");
  // 512 ions in a 22.56 A cell. The 25-point mesh runs real space beside
  // the transform at two and at eight threads and on one team at four, so
  // all three SPME paths meet the same reference.
  const Ionic ion = rockSalt(4, 5.64);
  rgpot::lr::CoulombConfig base;
  base.cutoff = 9.0;
  base.alpha = 0.3;
  base.short_cutoff = 3.0;
  base.skin = 0.5;
  base.periodic = true;

  rgpot::lr::CoulombConfig dsf = base;
  dsf.method = rgpot::lr::CoulombMethod::DSF;
  dsf.alpha = 0.2;
  checkCoulombBits("DSF", dsf, ion);

  rgpot::lr::CoulombConfig ewald = base;
  ewald.method = rgpot::lr::CoulombMethod::Ewald;
  ewald.kmax = 6;
  checkCoulombBits("Ewald with stress", ewald, ion);
  ewald.stress = false;
  checkCoulombBits("Ewald without stress", ewald, ion);

  rgpot::lr::CoulombConfig spme = base;
  spme.method = rgpot::lr::CoulombMethod::SPME;
  spme.grid[0] = spme.grid[1] = spme.grid[2] = 25;
  spme.spline_order = 5;
  spme.stress = false;
  checkCoulombBits("SPME float mesh 25", spme, ion);
  spme.stress = true;
  checkCoulombBits("SPME double mesh 25", spme, ion);
  spme.grid[0] = spme.grid[1] = spme.grid[2] = 48;
  spme.spline_order = 6;
  spme.stress = false;
  checkCoulombBits("SPME float mesh 48", spme, ion);
  spme.stress = true;
  checkCoulombBits("SPME double mesh 48", spme, ion);
}

TEST_CASE("a charged cell sums the SPME energy in one order on every path",
          "[threads][invariant][coulomb]") {
  setThreads("");
  // The neutralizing background is not zero here, so the energy has three
  // terms: real space, the mesh and the background. Real space beside the
  // mesh at two and eight threads must add them in the order of the
  // sequential call at one and four. Whether the two orders round apart
  // depends on the values, so several net charges are tried; each of these
  // rounds apart when the background is added before real space.
  rgpot::lr::CoulombConfig cfg;
  cfg.method = rgpot::lr::CoulombMethod::SPME;
  cfg.cutoff = 9.0;
  cfg.alpha = 0.3;
  cfg.short_cutoff = 3.0;
  cfg.skin = 0.5;
  cfg.periodic = true;
  cfg.grid[0] = cfg.grid[1] = cfg.grid[2] = 25;
  cfg.spline_order = 5;
  for (const double extra : {0.2, 0.45, 0.65, 1.1, 1.25}) {
    INFO("net charge " << extra);
    Ionic ion = rockSalt(4, 5.64);
    ion.q[0] += extra;
    ion.total = extra;
    cfg.stress = false;
    checkCoulombBits("charged SPME float mesh 25", cfg, ion);
    cfg.stress = true;
    checkCoulombBits("charged SPME double mesh 25", cfg, ion);
  }
}

namespace {

rgpot::lr::CoulombConfig shortMeshSpme() {
  rgpot::lr::CoulombConfig cfg;
  cfg.method = rgpot::lr::CoulombMethod::SPME;
  cfg.cutoff = 9.0;
  cfg.alpha = 0.3;
  cfg.short_cutoff = 3.0;
  cfg.skin = 0.5;
  cfg.periodic = true;
  cfg.grid[0] = cfg.grid[1] = cfg.grid[2] = 25;
  cfg.spline_order = 5;
  cfg.stress = false;
  return cfg;
}

// One evaluation on the calling thread, unlike runCoulomb.
CoulombEval coulombHere(const rgpot::lr::CoulombConfig &cfg, const Ionic &ion) {
  rgpot::lr::CoulombPot pot(
      cfg, std::make_shared<rgpot::lr::FixedCharges>(ion.q, ion.total));
  const std::size_t n = ion.s.z.size();
  const double L = ion.s.length;
  const double box[9] = {L, 0, 0, 0, L, 0, 0, 0, L};
  CoulombEval e;
  e.force.assign(3 * n, 0.0);
  rgpot::ForceOut out{};
  out.F = e.force.data();
  pot.forceImpl(rgpot::ForceInput{n, ion.s.pos.data(), ion.s.z.data(), box}, &out);
  e.energy = out.energy;
  e.has_stress = out.has_stress;
  std::copy(out.stress, out.stress + 9, e.stress);
  return e;
}

struct CallerTeamReset {
  int previous = rgpot::lr::caller_team();
  ~CallerTeamReset() { rgpot::lr::caller_team() = previous; }
};

} // namespace

TEST_CASE("an explicit count wins over the recorded caller team",
          "[threads][invariant][coulomb]") {
  setThreads("");
  const CallerTeamReset reset;
  const Ionic ion = rockSalt(4, 5.64);
  const rgpot::lr::CoulombConfig cfg = shortMeshSpme();
  const CoulombEval serial = runCoulomb(cfg, ion, 1, 0);
#if defined(_OPENMP)
  omp_set_dynamic(0);
  omp_set_num_threads(8);
#endif
  // A ladder that asked for eight records it. A count the caller sets
  // afterwards is the team the call runs on, not that record: two threads
  // on a 25-point mesh are one mesh rank and one real-space thread, where
  // eight would be four mesh ranks and four real-space threads.
  rgpot::lr::caller_team() = 8;
  CoulombEval scoped;
  {
    const rgpot::ThreadCount two{2};
    scoped = coulombHere(cfg, ion);
  }
  CHECK(rgpot::lr::pme_trace().real_team < 4);
  CHECK(coulombApart(scoped, serial) == 0);
  // The record belongs to the thread that set it.
  CHECK(std::async(std::launch::async, [] { return rgpot::lr::caller_team(); })
            .get() == 1);
  setThreads("");
}

TEST_CASE("an SPME call inside the caller's team opens no team of its own",
          "[threads][invariant][coulomb]") {
  setThreads("");
  const Ionic ion = rockSalt(4, 5.64);
  const rgpot::lr::CoulombConfig cfg = shortMeshSpme();
  const CoulombEval serial = runCoulomb(cfg, ion, 1, 0);
#if defined(_OPENMP)
  // Nested regions would be live here, so a call that opened one would run
  // a second team on every thread of the caller's.
  const int levels = omp_get_max_active_levels();
  omp_set_dynamic(0);
  omp_set_max_active_levels(2);
  omp_set_num_threads(8);
  constexpr int kOuter = 2;
  std::array<std::uint64_t, kOuter> opened{};
  std::array<int, kOuter> pool_moved{};
  std::array<std::size_t, kOuter> apart{};
  std::array<int, kOuter> ran{};
#pragma omp parallel num_threads(kOuter)
  {
    const int t = omp_get_thread_num();
    const int pool = omp_get_max_threads();
    rgpot::lr::caller_team() = 8;
    const std::uint64_t before = rgpot::teamsOpened();
    const CoulombEval e = coulombHere(cfg, ion);
    opened[static_cast<std::size_t>(t)] = rgpot::teamsOpened() - before;
    pool_moved[static_cast<std::size_t>(t)] = omp_get_max_threads() != pool;
    apart[static_cast<std::size_t>(t)] = coulombApart(e, serial);
    ran[static_cast<std::size_t>(t)] = rgpot::inCallerTeam() ? 1 : 0;
    rgpot::lr::caller_team() = 1;
  }
  omp_set_max_active_levels(levels);
  if (ran[0] + ran[1] == 0) {
    setThreads("");
    SKIP("this runtime cannot start a team of two");
  }
  for (int t = 0; t < kOuter; ++t) {
    INFO("outer thread " << t);
    if (ran[static_cast<std::size_t>(t)] == 0) {
      continue;
    }
    CHECK(opened[static_cast<std::size_t>(t)] == 0);
    CHECK(pool_moved[static_cast<std::size_t>(t)] == 0);
    CHECK(apart[static_cast<std::size_t>(t)] == 0);
  }
#else
  CHECK(coulombApart(coulombHere(cfg, ion), serial) == 0);
#endif
  setThreads("");
}

TEST_CASE("SPME beside the mesh finishes under a potential thread count",
          "[threads][invariant][coulomb]") {
  setThreads("");
  // A potential binding is thread-local, so only mesh rank 0 sees it. At
  // eight the mesh group is four ranks and real space takes four more; at
  // two the second thread runs real space beside a one-rank mesh.
  const Ionic ion = rockSalt(4, 5.64);
  rgpot::lr::CoulombConfig cfg;
  cfg.method = rgpot::lr::CoulombMethod::SPME;
  cfg.cutoff = 9.0;
  cfg.alpha = 0.3;
  cfg.short_cutoff = 3.0;
  cfg.skin = 0.5;
  cfg.grid[0] = cfg.grid[1] = cfg.grid[2] = 25;
  cfg.spline_order = 5;
  cfg.stress = false;
  const CoulombEval serial = runCoulomb(cfg, ion, 1, 0);
  for (const int threads : {8, 2}) {
    INFO("setThreadCount(" << threads << ")");
    const CoulombEval bound = runCoulomb(cfg, ion, 0, threads);
#if defined(_OPENMP)
    if (threads == 8 && omp_get_thread_limit() >= 8) {
      CHECK(rgpot::lr::pme_trace().real_team == 4);
    }
#endif
    CHECK(coulombApart(bound, serial) == 0);
  }
}

TEST_CASE("Ewald and SPME instances on two threads keep their serial bits",
          "[threads][invariant][coulomb][concurrent]") {
  setThreads("");
  const Ionic small = rockSalt(3, 5.6);
  const Ionic large = rockSalt(4, 5.64);
  rgpot::lr::CoulombConfig base;
  base.cutoff = 8.0;
  base.alpha = 0.3;
  base.short_cutoff = 3.0;
  base.skin = 0.5;
  base.periodic = true;
  base.stress = false;

  rgpot::lr::CoulombConfig ewald_a = base;
  ewald_a.method = rgpot::lr::CoulombMethod::Ewald;
  ewald_a.kmax = 5;
  rgpot::lr::CoulombConfig ewald_b = ewald_a;
  ewald_b.kmax = 6;
  ewald_b.alpha = 0.32;

  // The 25-point mesh on 512 ions runs real space beside the transform at
  // two and at eight threads; the 20-point mesh stays on one team.
  rgpot::lr::CoulombConfig spme_a = base;
  spme_a.method = rgpot::lr::CoulombMethod::SPME;
  spme_a.grid[0] = spme_a.grid[1] = spme_a.grid[2] = 20;
  spme_a.spline_order = 4;
  rgpot::lr::CoulombConfig spme_b = spme_a;
  spme_b.grid[0] = spme_b.grid[1] = spme_b.grid[2] = 25;
  spme_b.spline_order = 5;
  rgpot::lr::CoulombConfig spme_stress = spme_b;
  spme_stress.stress = true;

  checkCoulombPair("Ewald beside Ewald", ewald_a, small, ewald_b, large);
  checkCoulombPair("SPME beside SPME", spme_a, small, spme_b, large);
  checkCoulombPair("Ewald beside SPME", ewald_a, small, spme_b, large);
  checkCoulombPair("SPME with stress beside Ewald", spme_stress, large, ewald_b, small);
}
#endif
