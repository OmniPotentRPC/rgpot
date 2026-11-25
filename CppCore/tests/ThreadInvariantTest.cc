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

#include <cstdio>
#include <future>
#endif

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
    CHECK(rgpot::threadsRequested() ==
#if defined(_OPENMP)
          threads
#else
          1
#endif
    );
    CHECK(forceApart(run(*make(), s), serial) == 0);
  }
  setThreads("");
  auto opted = make();
  opted->setThreadCount(4);
  CHECK(rgpot::threadsRequested() == 1);
  const std::uint64_t before = rgpot::teamsOpened();
  CHECK(forceApart(run(*opted, s), serial) == 0);
#if defined(_OPENMP)
  CHECK(rgpot::teamsOpened() > before);
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
    rgpot::lr::CoulombPot pot(cfg,
                              std::make_shared<rgpot::lr::FixedCharges>(ion.q, 0.0));
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
    if (threads == 8) {
      CHECK(rgpot::lr::pme_trace().real_team == 4);
    }
#endif
    CHECK(coulombApart(bound, serial) == 0);
  }
}
#endif

TEST_CASE("eight threads still beat one thread on the large Lennard-Jones cell",
          "[threads][invariant][timing]") {
  const double a_lj = std::pow(2.0, 1.0 / 6.0) * std::sqrt(2.0);
  rgpot::LJConfig lj;
  lj.psi = 1.0;
  // The shared pair walk finishes a nearest-neighbour cell faster than a
  // team can start. This cutoff reaches several shells, so the arithmetic
  // is the cost the team is there to share.
  lj.cutoff = 4.5;
  const System s = fcc(13, a_lj, 18);
  const std::size_t n = s.z.size();
  const double box[9] = {s.length, 0, 0, 0, s.length, 0, 0, 0, s.length};
  std::vector<double> force(3 * n);
  rgpot::ForceInput in{n, s.pos.data(), s.z.data(), box};
  auto time_of = [&](int threads) {
    setThreads(std::to_string(threads).c_str());
    rgpot::LJPot pot(lj);
    pot.setNeighbourMode(rgpot::nlist::Mode::Stored);
    rgpot::ForceOut out{};
    out.F = force.data();
    pot.forceImpl(in, &out);
    const auto t0 = std::chrono::steady_clock::now();
    constexpr int kRep = 3;
    for (int i = 0; i < kRep; ++i) {
      std::fill(force.begin(), force.end(), 0.0);
      pot.forceImpl(in, &out);
    }
    const double us = std::chrono::duration<double, std::micro>(
                          std::chrono::steady_clock::now() - t0)
                          .count() /
                      kRep;
    return us;
  };
  const double one = time_of(1);
  const double eight = time_of(8);
  setThreads("");
  WARN("LJ 1-thread " << one << " us, 8-thread " << eight << " us, n=" << n);
#if defined(_OPENMP)
  CHECK(eight < one);
#endif
}
