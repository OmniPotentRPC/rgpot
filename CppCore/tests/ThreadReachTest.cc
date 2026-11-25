// MIT License
// Copyright 2023--present rgpot developers
//
// A potential's thread count reaches the threads the potential uses. The
// batch fan-out of every shared-instance potential, the terms of a SumPot,
// and the kernels of MEAMPot, D2Pot and D2CpmdPot take the count of the
// object they were called on. Each check watches the team an OpenMP region
// really has, or the count a child really reads, and not only that the call
// returns. The dlopen frontends and the Fortran Stillinger-Weber batch are
// checked beside their own suites (XTBPotTest, MetatomicPotTest,
// FortranPotsTest), so this one builds without the Fortran potentials.

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
#include "rgpot/D2Pot/D2CpmdPot.hpp"
#include "rgpot/D2Pot/D2Pot.hpp"
#include "rgpot/ForceStructs.hpp"
#include "rgpot/Potential.hpp"
#include "rgpot/MEAM/MEAMPot.hpp"
#include "rgpot/Slices.hpp"
#include "rgpot/StillingerWeber/SWSi.hpp"
#include "rgpot/SumPot/SumPot.hpp"
#include "rgpot/Tersoff/TersoffSiC.hpp"

#if defined(RGPOT_HAS_LONG_RANGE)
#include "rgpot/LongRange/SumPot.hpp"
#endif

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace {

constexpr std::array<int, 4> kCounts{1, 2, 4, 8};

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

std::uint64_t bits(double v) {
  std::uint64_t out = 0;
  std::memcpy(&out, &v, sizeof(out));
  return out;
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

Eval runOne(rgpot::PotentialBase &pot, const System &s) {
  return runBatch(pot, {s}).front();
}

int bitsApart(const Eval &a, const Eval &b) {
  int apart = bits(a.energy) != bits(b.energy) ? 1 : 0;
  apart += a.has_stress != b.has_stress ? 1 : 0;
  for (std::size_t k = 0; k < 9; ++k) {
    apart += bits(a.stress[k]) != bits(b.stress[k]) ? 1 : 0;
  }
  REQUIRE(a.f.size() == b.f.size());
  for (std::size_t k = 0; k < a.f.size(); ++k) {
    apart += bits(a.f[k]) != bits(b.f[k]) ? 1 : 0;
  }
  return apart;
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

// A potential that takes its thread binding does so at the top of forceImpl.
// The nest holds 32 bindings, so with 32 already on the stack a potential
// that opted in cannot take a 33rd and throws std::logic_error, while one
// that did not opt in binds nothing and evaluates.
template <typename Pot> void checkTakesBinding(Pot &pot, const System &s) {
  std::vector<double> f(s.pos.size(), 0.0);
  const rgpot::ForceInput in{s.z.size(), s.pos.data(), s.z.data(),
                             s.box.data()};
  rgpot::ForceOut out{};
  out.F = f.data();
  std::vector<std::unique_ptr<rgpot::ThreadBinding>> full;
  for (int i = 0; i < 32; ++i) {
    full.push_back(std::make_unique<rgpot::ThreadBinding>(2));
  }
  pot.clearThreadCount();
  CHECK_NOTHROW(pot.forceImpl(in, &out));
  pot.setThreadCount(3);
  CHECK_THROWS_AS(pot.forceImpl(in, &out), std::logic_error);
  pot.clearThreadCount();
  full.clear();
  CHECK(rgpot::explicitThreadCount() == 0);
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

TEST_CASE("direct silicon batch APIs bind their potential thread count",
          "[threads][batch]") {
  if (!canSpread()) {
    SKIP("this runtime cannot start a team of two");
  }
  constexpr std::size_t nconfig = 8;
  constexpr std::size_t natoms = 2;
  const std::array<double, 6> one{0.0, 0.0, 0.0, 2.3, 0.0, 0.0};
  std::vector<double> positions(nconfig * one.size());
  for (std::size_t c = 0; c < nconfig; ++c) {
    std::copy(one.begin(), one.end(), positions.begin() + c * one.size());
  }
  const std::array<int, natoms> z{14, 14};
  const std::array<double, 9> box{10.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0};
  auto check = [&](auto &pot) {
    std::vector<double> energy(nconfig, 0.0);
    std::vector<double> gradient(nconfig * one.size(), 0.0);
    pot.setThreadCount(1);
    const std::uint64_t serial_before = rgpot::teamsOpened();
    pot.energyAndGradientBatch(nconfig, positions.data(), z.data(), box.data(), natoms,
                               energy.data(), gradient.data());
    CHECK(rgpot::teamsOpened() == serial_before);
    pot.setThreadCount(4);
    const std::uint64_t parallel_before = rgpot::teamsOpened();
    pot.energyAndGradientBatch(nconfig, positions.data(), z.data(), box.data(), natoms,
                               energy.data(), gradient.data());
    CHECK(rgpot::teamsOpened() > parallel_before);
    CHECK(std::all_of(energy.begin(), energy.end(),
                      [](double value) { return std::isfinite(value); }));
  };
  rgpot::StillingerWeberSi sw;
  check(sw);
  rgpot::TersoffSiC tersoff;
  check(tersoff);
}

TEST_CASE("a SumPot hands its thread count to the terms", "[threads][SumPot]") {
  setThreads("");
  const System sys = pair(1, 2.0);
  auto log = std::make_shared<Log>();
  auto build = [&](rgpot::SumPot &sum, int terms) {
    for (int i = 0; i < terms; ++i) {
      sum.add(std::make_unique<Probe>(log));
    }
  };
  auto seenByTerms = [&](rgpot::PotentialBase &sum) {
    log->clear();
    runOne(sum, sys);
    return log->take();
  };

  SECTION("the setter rejects a count below 1") {
    rgpot::SumPot sum;
    CHECK_THROWS_AS(sum.setThreadCount(0), std::invalid_argument);
    CHECK_THROWS_AS(sum.setThreadCount(-2), std::invalid_argument);
    CHECK(sum.threadCount() == 0);
  }

  SECTION("a sum without a count leaves every term at its default") {
    rgpot::SumPot sum;
    build(sum, 2);
    const auto seen = seenByTerms(sum);
    REQUIRE(seen.size() == 2);
    for (const Seen &s : seen) {
      CHECK(s.explicit_count == 0);
      CHECK(s.requested == 1);
      CHECK(s.inner_team == 1);
    }
  }

  SECTION("every term reads the count, and its own region has that team") {
    if (!canSpread()) {
      SKIP("this runtime cannot start a team of two");
    }
    for (const int threads : {2, 3, 4}) {
      INFO("setThreadCount(" << threads << ")");
      rgpot::SumPot sum;
      build(sum, 3);
      sum.setThreadCount(threads);
      const auto seen = seenByTerms(sum);
      REQUIRE(seen.size() == 3);
      for (const Seen &s : seen) {
        CHECK(s.explicit_count == threads);
#if defined(_OPENMP)
        CHECK(s.requested == threads);
        CHECK(s.inner_team == teamFor(threads));
#endif
      }
    }
  }

  SECTION("a term with its own count keeps it and the others follow the sum") {
    rgpot::SumPot sum;
    build(sum, 3);
    sum.child(1)->setThreadCount(2);
    sum.setThreadCount(4);
    const auto seen = seenByTerms(sum);
    REQUIRE(seen.size() == 3);
    CHECK(seen[0].explicit_count == 4);
    CHECK(seen[1].explicit_count == 2);
    CHECK(seen[2].explicit_count == 4);
  }

  SECTION("a ThreadCount on the call wins over the sum and its terms") {
    rgpot::SumPot sum;
    build(sum, 2);
    sum.child(0)->setThreadCount(2);
    sum.setThreadCount(4);
    const rgpot::ThreadCount one(1);
    const auto seen = seenByTerms(sum);
    REQUIRE(seen.size() == 2);
    for (const Seen &s : seen) {
      CHECK(s.explicit_count == 1);
      CHECK(s.inner_team == 1);
    }
  }

  SECTION("clearing the count on the sum takes it from every term at once") {
    rgpot::SumPot sum;
    build(sum, 2);
    sum.setThreadCount(4);
    for (const Seen &s : seenByTerms(sum)) {
      CHECK(s.explicit_count == 4);
    }
    sum.clearThreadCount();
    CHECK(sum.threadCount() == 0);
    for (const Seen &s : seenByTerms(sum)) {
      CHECK(s.explicit_count == 0);
    }
  }

  SECTION("a term added after the count is set follows it") {
    rgpot::SumPot sum;
    sum.setThreadCount(3);
    build(sum, 1);
    const auto seen = seenByTerms(sum);
    REQUIRE(seen.size() == 1);
    CHECK(seen[0].explicit_count == 3);
  }

  SECTION("a disabled term is not evaluated and sees nothing") {
    rgpot::SumPot sum;
    build(sum, 2);
    sum.setEnabled(0, false);
    sum.setThreadCount(3);
    const auto seen = seenByTerms(sum);
    REQUIRE(seen.size() == 1);
    CHECK(seen[0].explicit_count == 3);
  }

  SECTION("a sum inside a sum passes the outer count down") {
    rgpot::SumPot outer;
    auto inner = std::make_unique<rgpot::SumPot>();
    build(*inner, 2);
    outer.add(std::move(inner));
    outer.setThreadCount(3);
    const auto seen = seenByTerms(outer);
    REQUIRE(seen.size() == 2);
    for (const Seen &s : seen) {
      CHECK(s.explicit_count == 3);
    }
  }

  SECTION("the inner sum's own count wins for its terms") {
    rgpot::SumPot outer;
    auto inner = std::make_unique<rgpot::SumPot>();
    build(*inner, 2);
    inner->setThreadCount(2);
    outer.add(std::move(inner));
    outer.setThreadCount(5);
    for (const Seen &s : seenByTerms(outer)) {
      CHECK(s.explicit_count == 2);
    }
  }

  SECTION("the binding ends with the call, also when a term throws") {
    rgpot::SumPot sum;
    sum.add(std::make_unique<Probe>(log, true));
    sum.setThreadCount(4);
    CHECK_THROWS_AS(runOne(sum, sys), std::runtime_error);
    CHECK(rgpot::explicitThreadCount() == 0);
  }

  SECTION("the sum takes its binding when it evaluates") {
    rgpot::SumPot sum;
    build(sum, 1);
    checkTakesBinding(sum, sys);
  }

#if defined(RGPOT_HAS_LONG_RANGE)
  SECTION("the long-range sum forwards the count the same way") {
    rgpot::lr::SumPot sum;
    build(sum, 2);
    sum.setThreadCount(3);
    const auto seen = seenByTerms(sum);
    REQUIRE(seen.size() == 2);
    for (const Seen &s : seen) {
      CHECK(s.explicit_count == 3);
    }
  }
#endif
  setThreads("");
}

namespace {

System openMolecule() {
  System s;
  s.pos = {0.0, 0.0, 0.0, 3.4, 0.0, 0.0, 1.7, 2.9, 0.4, -0.5, 3.1, 3.6};
  s.z = {6, 6, 8, 1};
  return s;
}

System nickelCell() {
  System s;
  const double a = 3.52;
  const int cells = 2;
  const double basis[4][3] = {
      {0, 0, 0}, {0, 0.5, 0.5}, {0.5, 0, 0.5}, {0.5, 0.5, 0}};
  for (int ix = 0; ix < cells; ++ix) {
    for (int iy = 0; iy < cells; ++iy) {
      for (int iz = 0; iz < cells; ++iz) {
        for (const auto &b : basis) {
          const double lattice[3] = {ix + b[0], iy + b[1], iz + b[2]};
          const std::size_t i = s.z.size();
          for (int k = 0; k < 3; ++k) {
            // A small fixed displacement so the forces are not all zero.
            const double shake =
                0.02 * std::sin(1.7 * static_cast<double>(i) + 0.9 * k);
            s.pos.push_back(a * lattice[k] + shake);
          }
          s.z.push_back(28);
        }
      }
    }
  }
  s.box = {a * cells, 0, 0, 0, a * cells, 0, 0, 0, a * cells};
  return s;
}

// What a kernel with no region of its own must still do: give one answer on
// any count, take the binding, and let the batch fan-out use the count.
template <typename Make>
void checkSerialKernel(const char *name, Make make, const System &s) {
  INFO(name << " on " << s.z.size() << " atoms");
  setThreads("");
  REQUIRE(s.z.size() < rgpot::kTeamMinAtoms);
  auto reference_pot = make();
  const Eval reference = runOne(*reference_pot, s);

  for (const int threads : kCounts) {
    INFO("setThreadCount(" << threads << ")");
    auto pot = make();
    pot->setThreadCount(threads);
    CHECK(pot->threadCount() == threads);
    CHECK(bitsApart(runOne(*pot, s), reference) == 0);
    CHECK(rgpot::explicitThreadCount() == 0);
  }

  {
    const std::vector<System> batch(8, s);
    {
      auto pot = make();
      const std::uint64_t before = rgpot::teamsOpened();
      for (const Eval &e : runBatch(*pot, batch)) {
        CHECK(bitsApart(e, reference) == 0);
      }
      CHECK(rgpot::teamsOpened() == before);
    }
    if (canSpread()) {
      auto pot = make();
      pot->setThreadCount(4);
      const std::uint64_t before = rgpot::teamsOpened();
      for (const Eval &e : runBatch(*pot, batch)) {
        CHECK(bitsApart(e, reference) == 0);
      }
#if defined(_OPENMP)
      CHECK(rgpot::teamsOpened() > before);
#endif
      auto scoped = make();
      scoped->setThreadCount(4);
      const rgpot::ThreadCount one(1);
      const std::uint64_t held = rgpot::teamsOpened();
      runBatch(*scoped, batch);
      CHECK(rgpot::teamsOpened() == held);
    }
  }

  auto probe = make();
  checkTakesBinding(*probe, s);
}

// An error from the kernel leaves nothing bound on the thread.
template <typename Pot> void checkBindingEndsOnError(Pot &pot, const System &s) {
  std::vector<double> f(s.pos.size(), 0.0);
  const rgpot::ForceInput bad{s.z.size(), s.pos.data(), nullptr,
                              s.box.data()};
  rgpot::ForceOut out{};
  out.F = f.data();
  pot.setThreadCount(4);
  CHECK_THROWS(pot.forceImpl(bad, &out));
  pot.clearThreadCount();
  CHECK(rgpot::explicitThreadCount() == 0);
}

} // namespace

TEST_CASE("D2Pot takes the thread count of its object", "[threads][D2]") {
  rgpot::D2Config cfg;
  cfg.cutoff = 12.0;
  checkSerialKernel(
      "D2Pot", [&] { return std::make_unique<rgpot::D2Pot>(cfg); },
      openMolecule());
  rgpot::D2Pot pot(cfg);
  checkBindingEndsOnError(pot, openMolecule());
}

TEST_CASE("D2CpmdPot takes the thread count of its object",
          "[threads][D2Cpmd]") {
  checkSerialKernel(
      "D2CpmdPot", [] { return std::make_unique<rgpot::D2CpmdPot>(); },
      openMolecule());
  rgpot::D2CpmdPot pot;
  checkBindingEndsOnError(pot, openMolecule());
}

TEST_CASE("MEAMPot takes the thread count of its object", "[threads][MEAM]") {
  const std::string data = "CppCore/tests/data/meam/";
  auto make = [&] {
    rgpot::MEAMConfig c;
    c.library = data + "library.meam";
    c.parameters = data + "Ni.meam";
    c.elements = {"Ni"};
    c.periodic = true;
    c.stress = true;
    return std::make_unique<rgpot::MEAMPot>(c);
  };
  checkSerialKernel("MEAMPot", make, nickelCell());
  auto pot = make();
  checkBindingEndsOnError(*pot, nickelCell());
}
