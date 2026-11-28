// MIT License
// Copyright 2023--present rgpot developers
//
// Small systems stay below kTeamMinAtoms, so a one-thread loop leaves the
// rest of the team idle. A batch of those systems has to spread across
// the team and still match the one-thread bits.

#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif

#include <catch2/catch_all.hpp>

#include "RestoreEnv.hpp"
#include "lsan_libomp.hpp"
#include "rgpot/LennardJones/LJPot.hpp"
#include "rgpot/Slices.hpp"
#include "rgpot/StillingerWeber/SWSi.hpp"
#include "rgpot/Tersoff/TersoffSiC.hpp"

namespace {

constexpr std::size_t kSystems = 16;
constexpr std::size_t kAtoms = 4;

bool sameBits(double a, double b) noexcept {
  return std::bit_cast<std::uint64_t>(a) == std::bit_cast<std::uint64_t>(b);
}

void requireBits(const std::vector<double> &got,
                 const std::vector<double> &ref) {
  REQUIRE(got.size() == ref.size());
  for (std::size_t i = 0; i < got.size(); ++i) {
    INFO("index " << i);
    REQUIRE(sameBits(got[i], ref[i]));
  }
}

std::vector<double> makePositions(std::size_t systems, std::size_t atoms,
                                  double shift) {
  std::vector<double> pos(systems * atoms * 3, 0.0);
  for (std::size_t s = 0; s < systems; ++s) {
    for (std::size_t a = 0; a < atoms; ++a) {
      const std::size_t b = (s * atoms + a) * 3;
      pos[b] = 1.65 * static_cast<double>(a) + shift * static_cast<double>(s);
      pos[b + 1] =
          0.4 * static_cast<double>(a % 2) + 0.01 * static_cast<double>(s);
      pos[b + 2] = 0.25 * static_cast<double>((a * a) % 3);
    }
  }
  return pos;
}

const double kBox[9] = {250.0, 0.0, 0.0, 0.0, 250.0, 0.0, 0.0, 0.0, 250.0};

struct Grad {
  std::vector<double> energy;
  std::vector<double> gradient;
  std::uint64_t teams = 0;
};

template <typename Pot, typename Run>
Grad runGradient(const std::vector<double> &pos, const int *species,
                 const char *threads, Run &&run) {
  rgpot::test_support::RestoreEnv env("OMP_NUM_THREADS");
  env.set(threads);
#if defined(_OPENMP)
  REQUIRE(rgpot::threadsRequested() == std::stoi(threads));
#else
  REQUIRE(rgpot::threadsRequested() == 1);
#endif
  Pot pot;
  Grad out;
  out.energy.assign(kSystems, 0.0);
  out.gradient.assign(kSystems * kAtoms * 3, 0.0);
  const auto before = rgpot::teamsOpened();
  run(pot, pos, species, out);
  out.teams = rgpot::teamsOpened() - before;
  for (double e : out.energy) {
    REQUIRE(std::isfinite(e));
  }
  return out;
}

void requireTeams(const Grad &one, const Grad &more) {
  REQUIRE(one.teams == 0);
#if defined(_OPENMP)
  REQUIRE(more.teams == 1);
#else
  REQUIRE(more.teams == 0);
#endif
  requireBits(more.energy, one.energy);
  requireBits(more.gradient, one.gradient);
}

std::string ratioText(const char *label, std::chrono::nanoseconds serial,
                      std::chrono::nanoseconds threaded) {
  std::ostringstream os;
  os << "smoke wall time ratio for " << label << " (one thread over four) ";
  if (threaded.count() > 0) {
    os << static_cast<double>(serial.count()) /
              static_cast<double>(threaded.count());
  } else {
    os << "undefined";
  }
  os << " serial_ns " << serial.count() << " threaded_ns " << threaded.count();
  return os.str();
}

} // namespace

TEST_CASE("small Stillinger-Weber batches match across thread counts",
          "[batch]") {
  const int species[kAtoms] = {14, 14, 14, 14};
  for (const double shift : {3.0, 0.02}) {
    const std::vector<double> pos = makePositions(kSystems, kAtoms, shift);
    auto run = [&](rgpot::StillingerWeberSi &pot, const std::vector<double> &p,
                   const int *z, Grad &out) {
      pot.energyAndGradientBatch(kSystems, p.data(), z, kBox, kAtoms,
                                 out.energy.data(), out.gradient.data());
    };
    const Grad one =
        runGradient<rgpot::StillingerWeberSi>(pos, species, "1", run);
    const Grad two =
        runGradient<rgpot::StillingerWeberSi>(pos, species, "2", run);
    const Grad four =
        runGradient<rgpot::StillingerWeberSi>(pos, species, "4", run);
    requireTeams(one, two);
    requireTeams(one, four);

    for (std::size_t s = 0; s < kSystems; ++s) {
      rgpot::StillingerWeberSi fresh;
      double energy = 0.0;
      std::vector<double> gradient(kAtoms * 3, 0.0);
      fresh.energyAndGradient(pos.data() + s * kAtoms * 3, species, kBox,
                              kAtoms, &energy, gradient.data());
      INFO("system " << s);
      REQUIRE(sameBits(energy, one.energy[s]));
      for (std::size_t g = 0; g < gradient.size(); ++g) {
        REQUIRE(sameBits(gradient[g], one.gradient[s * kAtoms * 3 + g]));
      }
    }
  }

  const std::vector<double> pos = makePositions(kSystems, kAtoms, 3.0);
  auto timeAt = [&](const char *threads) {
    rgpot::test_support::RestoreEnv env("OMP_NUM_THREADS");
    env.set(threads);
    rgpot::StillingerWeberSi pot;
    std::vector<double> energy(kSystems, 0.0);
    std::vector<double> gradient(kSystems * kAtoms * 3, 0.0);
    const auto start = std::chrono::steady_clock::now();
    for (int rep = 0; rep < 20; ++rep) {
      pot.energyAndGradientBatch(kSystems, pos.data(), species, kBox, kAtoms,
                                 energy.data(), gradient.data());
    }
    return std::chrono::steady_clock::now() - start;
  };
  const auto serialTime = timeAt("1");
  const auto threadedTime = timeAt("4");
  WARN(ratioText("Stillinger-Weber", serialTime, threadedTime));
}

TEST_CASE("small Tersoff batches match across thread counts", "[batch]") {
  const int species[kAtoms] = {14, 14, 6, 14};
  for (const double shift : {3.0, 0.02}) {
    const std::vector<double> pos = makePositions(kSystems, kAtoms, shift);
    auto run = [&](rgpot::TersoffSiC &pot, const std::vector<double> &p,
                   const int *z, Grad &out) {
      pot.energyAndGradientBatch(kSystems, p.data(), z, kBox, kAtoms,
                                 out.energy.data(), out.gradient.data());
    };
    const Grad one = runGradient<rgpot::TersoffSiC>(pos, species, "1", run);
    const Grad two = runGradient<rgpot::TersoffSiC>(pos, species, "2", run);
    const Grad four = runGradient<rgpot::TersoffSiC>(pos, species, "4", run);
    requireTeams(one, two);
    requireTeams(one, four);
    for (std::size_t s = 0; s < kSystems; ++s) {
      rgpot::TersoffSiC fresh;
      double energy = 0.0;
      std::vector<double> gradient(kAtoms * 3, 0.0);
      fresh.energyAndGradient(pos.data() + s * kAtoms * 3, species, kBox,
                              kAtoms, &energy, gradient.data());
      INFO("system " << s);
      REQUIRE(sameBits(energy, one.energy[s]));
      for (std::size_t g = 0; g < gradient.size(); ++g) {
        REQUIRE(sameBits(gradient[g], one.gradient[s * kAtoms * 3 + g]));
      }
    }
  }
}

TEST_CASE("small default-path batches match across thread counts", "[batch]") {
  const std::vector<double> pos = makePositions(kSystems, kAtoms, 3.0);
  const std::vector<double> close = makePositions(kSystems, kAtoms, 0.02);
  auto run = [](const std::vector<double> &coordinates, const char *threads) {
    rgpot::test_support::RestoreEnv env("OMP_NUM_THREADS");
    env.set(threads);
#if defined(_OPENMP)
    REQUIRE(rgpot::threadsRequested() == std::stoi(threads));
#else
    REQUIRE(rgpot::threadsRequested() == 1);
#endif
    std::vector<int> species(kAtoms, 18);
    const std::array<double, 9> box = {250.0, 0.0, 0.0, 0.0,  250.0,
                                       0.0,   0.0, 0.0, 250.0};
    std::vector<double> forces(kSystems * kAtoms * 3, 0.0);
    std::vector<rgpot::ForceInput> inputs;
    std::vector<rgpot::ForceOut> outputs(kSystems);
    inputs.reserve(kSystems);
    for (std::size_t s = 0; s < kSystems; ++s) {
      inputs.push_back(rgpot::ForceInput{
          .nAtoms = kAtoms,
          .pos = coordinates.data() + s * kAtoms * 3,
          .atmnrs = species.data(),
          .box = box.data(),
      });
      outputs[s] = rgpot::ForceOut{
          .F = forces.data() + s * kAtoms * 3,
          .energy = 0.0,
          .variance = 0.0,
          .stress = {},
          .has_stress = 0,
      };
    }
    rgpot::LJPot pot;
    REQUIRE(pot.caps().reentrancy == rgpot::Reentrancy::SharedInstance);
    const auto before = rgpot::teamsOpened();
    pot.forceBatch(rgpot::ForceBatch{
        .nSystems = kSystems,
        .in = inputs.data(),
        .out = outputs.data(),
    });
    struct Got {
      std::vector<double> energy;
      std::vector<double> force;
      std::vector<double> stress;
      std::vector<double> variance;
      std::uint64_t teams = 0;
    } got;
    got.teams = rgpot::teamsOpened() - before;
    got.force = std::move(forces);
    got.energy.resize(kSystems);
    got.variance.resize(kSystems);
    got.stress.resize(kSystems * 9);
    for (std::size_t s = 0; s < kSystems; ++s) {
      got.energy[s] = outputs[s].energy;
      got.variance[s] = outputs[s].variance;
      REQUIRE(std::isfinite(got.energy[s]));
      REQUIRE(outputs[s].has_stress == 1);
      for (int c = 0; c < 9; ++c) {
        got.stress[s * 9 + static_cast<std::size_t>(c)] = outputs[s].stress[c];
      }
    }
    return got;
  };

  for (const std::vector<double> *coordinates : {&pos, &close}) {
    const auto one = run(*coordinates, "1");
    const auto two = run(*coordinates, "2");
    const auto four = run(*coordinates, "4");
    REQUIRE(one.teams == 0);
#if defined(_OPENMP)
    REQUIRE(two.teams == 1);
    REQUIRE(four.teams == 1);
#else
    REQUIRE(two.teams == 0);
    REQUIRE(four.teams == 0);
#endif
    requireBits(two.energy, one.energy);
    requireBits(four.energy, one.energy);
    requireBits(two.force, one.force);
    requireBits(four.force, one.force);
    requireBits(two.stress, one.stress);
    requireBits(four.stress, one.stress);
    requireBits(two.variance, one.variance);
    requireBits(four.variance, one.variance);

    rgpot::LJPot fresh;
    std::vector<int> species(kAtoms, 18);
    const std::array<double, 9> box = {250.0, 0.0, 0.0, 0.0,  250.0,
                                       0.0,   0.0, 0.0, 250.0};
    for (std::size_t s = 0; s < kSystems; ++s) {
      std::vector<double> force(kAtoms * 3, 0.0);
      rgpot::ForceOut out{
          .F = force.data(),
          .energy = 0.0,
          .variance = 0.0,
          .stress = {},
          .has_stress = 0,
      };
      fresh.forceImpl(
          rgpot::ForceInput{.nAtoms = kAtoms,
                            .pos = coordinates->data() + s * kAtoms * 3,
                            .atmnrs = species.data(),
                            .box = box.data()},
          &out);
      INFO("system " << s);
      REQUIRE(sameBits(out.energy, one.energy[s]));
      requireBits(force, std::vector<double>(
                             one.force.begin() +
                                 static_cast<std::ptrdiff_t>(s * kAtoms * 3),
                             one.force.begin() + static_cast<std::ptrdiff_t>(
                                                     (s + 1) * kAtoms * 3)));
    }
  }
}

TEST_CASE("batch spread stops at the in-system crossover", "[batch]") {
  rgpot::test_support::RestoreEnv env("OMP_NUM_THREADS");
  env.set("4");
#if defined(_OPENMP)
  REQUIRE(rgpot::threadsRequested() == 4);
#else
  REQUIRE(rgpot::threadsRequested() == 1);
#endif

  std::size_t calls = 0;
  const auto largeBefore = rgpot::teamsOpened();
  rgpot::forEachBatchSystem(
      8, [](std::size_t) { return rgpot::kTeamMinAtoms; },
      [&](std::size_t) { ++calls; });
  REQUIRE(calls == 8);
  REQUIRE(rgpot::teamsOpened() == largeBefore);

  std::vector<int> seen(kSystems, 0);
  const auto smallBefore = rgpot::teamsOpened();
  rgpot::forEachBatchSystem(
      kSystems, [](std::size_t) { return rgpot::kTeamMinAtoms - 1; },
      [&](std::size_t i) { seen[i] += 1; });
#if defined(_OPENMP)
  REQUIRE(rgpot::teamsOpened() == smallBefore + 1);
#else
  REQUIRE(rgpot::teamsOpened() == smallBefore);
#endif
  for (int hit : seen) {
    REQUIRE(hit == 1);
  }

  const auto oneBefore = rgpot::teamsOpened();
  rgpot::forEachBatchSystem(
      1, [](std::size_t) { return std::size_t{2}; }, [&](std::size_t) {});
  REQUIRE(rgpot::teamsOpened() == oneBefore);

  const auto mixedBefore = rgpot::teamsOpened();
  rgpot::forEachBatchSystem(
      4,
      [](std::size_t i) {
        return i == 2 ? rgpot::kTeamMinAtoms : std::size_t{3};
      },
      [&](std::size_t) {});
  REQUIRE(rgpot::teamsOpened() == mixedBefore);
}

TEST_CASE("two-thread batch spread propagates the team threshold", "[batch]") {
  rgpot::test_support::RestoreEnv env("OMP_NUM_THREADS");
  env.set("2");
  std::vector<int> sizes(2, 0);
  const auto before = rgpot::teamsOpened();
  {
    const rgpot::TeamThreshold no_inner_team(rgpot::kNoTeam);
    rgpot::forEachBatchSystem(
        sizes.size(),
        [](std::size_t) {
          return rgpot::StillingerWeberSi::kTeamMinAtoms;
        },
        [&](std::size_t i) {
          const std::size_t floor = rgpot::teamThreshold(
              rgpot::StillingerWeberSi::kTeamMinAtoms);
          sizes[i] = rgpot::teamSize(
              rgpot::StillingerWeberSi::kTeamMinAtoms, floor,
              rgpot::StillingerWeberSi::kTeamFrom);
        },
        rgpot::StillingerWeberSi::kTeamMinAtoms,
        rgpot::StillingerWeberSi::kTeamFrom);
  }
#if defined(_OPENMP)
  REQUIRE(rgpot::threadsRequested() == 2);
  REQUIRE(rgpot::teamsOpened() == before + 1);
#else
  REQUIRE(rgpot::threadsRequested() == 1);
  REQUIRE(rgpot::teamsOpened() == before);
#endif
  for (int size : sizes) {
    REQUIRE(size == 1);
  }
}

// Systems between a class floor and the shared 8192-atom threshold.
// A spread would put each worker on forRowTeam, which visits every pair
// from both atoms, and the caller's teamsOpened would count one region
// for the spread rather than one region per system.
constexpr std::size_t kMidSystems = 4;
constexpr std::size_t kMidAtoms = 48;
const double kMidBox[9] = {80.0, 0.0, 0.0, 0.0, 80.0, 0.0, 0.0, 0.0, 80.0};

std::vector<double> gridPositions(std::size_t systems, double spacing) {
  std::vector<double> pos(systems * kMidAtoms * 3, 0.0);
  for (std::size_t s = 0; s < systems; ++s) {
    for (std::size_t a = 0; a < kMidAtoms; ++a) {
      const std::size_t b = (s * kMidAtoms + a) * 3;
      pos[b] = spacing * static_cast<double>(a % 4) + 0.02 * static_cast<double>(s);
      pos[b + 1] = spacing * static_cast<double>((a / 4) % 4);
      pos[b + 2] = spacing * static_cast<double>(a / 16);
    }
  }
  return pos;
}

std::size_t halfPairs(const double *pos, double cutoff) {
  rgpot::nlist::Neighbours cells;
  rgpot::nlist::Spec spec;
  spec.cutoff = cutoff;
  spec.skin = 0.0;
  spec.image = rgpot::nlist::Image::Round;
  spec.mode = rgpot::nlist::Mode::Stream;
  cells.update(pos, kMidAtoms, kMidBox, spec);
  std::size_t pairs = 0;
  const double cut2 = cutoff * cutoff;
  cells.forHalf(0, static_cast<std::int64_t>(kMidAtoms),
                [&](std::int32_t i, std::int32_t j, double, double, double, double r2) {
                  if (i < j && r2 > 0.0 && r2 <= cut2) {
                    ++pairs;
                  }
                });
  return pairs;
}

class ActiveLevels {
public:
  explicit ActiveLevels(int levels) {
#if defined(_OPENMP)
    previous_ = omp_get_max_active_levels();
    omp_set_max_active_levels(levels);
#else
    (void)levels;
#endif
  }
  ~ActiveLevels() {
#if defined(_OPENMP)
    omp_set_max_active_levels(previous_);
#endif
  }
  ActiveLevels(const ActiveLevels &) = delete;
  ActiveLevels &operator=(const ActiveLevels &) = delete;

private:
  int previous_ = 1;
};

struct MidEval {
  std::vector<double> energy;
  std::vector<double> force;
  std::uint64_t teams = 0;
};

void requireMid(const MidEval &batch, const MidEval &serial, std::size_t half,
                const char *label, int levels, bool pair_walk) {
  if (pair_walk) {
    WARN(label << " levels " << levels << " half_pairs " << half
               << " directed_team_visits " << (2 * half * kMidSystems)
               << " batch_teams " << batch.teams << " serial_teams " << serial.teams);
  } else {
    WARN(label << " levels " << levels << " half_pairs " << half << " batch_teams "
               << batch.teams << " serial_teams " << serial.teams);
  }
  REQUIRE(half > 0);
  requireBits(batch.energy, serial.energy);
  requireBits(batch.force, serial.force);
  // One in-system team per system. A spread of these sizes opens one
  // region and then each worker takes the two-visit pair walk.
#if defined(_OPENMP)
  REQUIRE(serial.teams == kMidSystems);
  REQUIRE(batch.teams == serial.teams);
#else
  REQUIRE(serial.teams == 0);
  REQUIRE(batch.teams == 0);
#endif
}

TEST_CASE("batches between the class floor and the shared threshold match serial",
          "[batch]") {
  REQUIRE(kMidAtoms > rgpot::LJPot::kTeamMinAtoms);
  REQUIRE(kMidAtoms > rgpot::StillingerWeberSi::kTeamMinAtoms);
  REQUIRE(kMidAtoms < rgpot::kTeamMinAtoms);

  rgpot::test_support::RestoreEnv threads("OMP_NUM_THREADS");
  threads.set("4");
  rgpot::test_support::RestoreEnv levels_env("OMP_MAX_ACTIVE_LEVELS");
#if defined(_OPENMP)
  REQUIRE(rgpot::threadsRequested() == 4);
  REQUIRE(rgpot::teamSize(kMidAtoms, rgpot::LJPot::kTeamMinAtoms, rgpot::LJPot::kTeamFrom) > 1);
  REQUIRE(rgpot::teamSize(kMidAtoms, rgpot::StillingerWeberSi::kTeamMinAtoms,
                          rgpot::StillingerWeberSi::kTeamFrom) > 1);
  // The shared threshold would still spread this size.
  REQUIRE(rgpot::teamSize(kMidAtoms, rgpot::kTeamMinAtoms) == 1);
#else
  REQUIRE(rgpot::threadsRequested() == 1);
#endif

  const std::vector<double> lj_pos = gridPositions(kMidSystems, 1.2);
  const std::vector<double> sw_pos = gridPositions(kMidSystems, 2.2);
  const std::size_t lj_half = halfPairs(lj_pos.data(), 15.0);
  const std::size_t sw_half =
      halfPairs(sw_pos.data(), rgpot::StillingerWeberSi::kCut * rgpot::StillingerWeberSi::kSigma);
  std::vector<int> lj_species(kMidAtoms, 18);
  std::vector<int> sw_species(kMidAtoms, 14);

  for (const int levels : {1, 2}) {
    levels_env.set(levels == 1 ? "1" : "2");
    ActiveLevels active(levels);

    auto lj_batch = [&]() {
      std::vector<double> forces(kMidSystems * kMidAtoms * 3, 0.0);
      std::vector<rgpot::ForceInput> inputs;
      std::vector<rgpot::ForceOut> outputs(kMidSystems);
      inputs.reserve(kMidSystems);
      for (std::size_t s = 0; s < kMidSystems; ++s) {
        inputs.push_back(rgpot::ForceInput{
            .nAtoms = kMidAtoms,
            .pos = lj_pos.data() + s * kMidAtoms * 3,
            .atmnrs = lj_species.data(),
            .box = kMidBox,
        });
        outputs[s] = rgpot::ForceOut{
            .F = forces.data() + s * kMidAtoms * 3,
            .energy = 0.0,
            .variance = 0.0,
            .stress = {},
            .has_stress = 0,
        };
      }
      rgpot::LJPot pot;
      pot.setNeighbourMode(rgpot::nlist::Mode::Stream);
      const auto before = rgpot::teamsOpened();
      pot.forceBatch(rgpot::ForceBatch{
          .nSystems = kMidSystems,
          .in = inputs.data(),
          .out = outputs.data(),
      });
      MidEval got;
      got.teams = rgpot::teamsOpened() - before;
      got.force = std::move(forces);
      got.energy.resize(kMidSystems);
      for (std::size_t s = 0; s < kMidSystems; ++s) {
        got.energy[s] = outputs[s].energy;
        REQUIRE(std::isfinite(got.energy[s]));
      }
      return got;
    };
    auto lj_serial = [&]() {
      MidEval got;
      got.energy.assign(kMidSystems, 0.0);
      got.force.assign(kMidSystems * kMidAtoms * 3, 0.0);
      const auto before = rgpot::teamsOpened();
      for (std::size_t s = 0; s < kMidSystems; ++s) {
        rgpot::LJPot pot;
        pot.setNeighbourMode(rgpot::nlist::Mode::Stream);
        rgpot::ForceOut out{
            .F = got.force.data() + s * kMidAtoms * 3,
            .energy = 0.0,
            .variance = 0.0,
            .stress = {},
            .has_stress = 0,
        };
        pot.forceImpl(rgpot::ForceInput{.nAtoms = kMidAtoms,
                                        .pos = lj_pos.data() + s * kMidAtoms * 3,
                                        .atmnrs = lj_species.data(),
                                        .box = kMidBox},
                      &out);
        got.energy[s] = out.energy;
      }
      got.teams = rgpot::teamsOpened() - before;
      return got;
    };
    requireMid(lj_batch(), lj_serial(), lj_half, "LJ", levels, true);

    auto sw_batch = [&]() {
      rgpot::StillingerWeberSi pot;
      pot.setNeighbourMode(rgpot::nlist::Mode::Stream);
      MidEval got;
      got.energy.assign(kMidSystems, 0.0);
      got.force.assign(kMidSystems * kMidAtoms * 3, 0.0);
      const auto before = rgpot::teamsOpened();
      pot.energyAndGradientBatch(kMidSystems, sw_pos.data(), sw_species.data(), kMidBox,
                                 kMidAtoms, got.energy.data(), got.force.data());
      got.teams = rgpot::teamsOpened() - before;
      for (double e : got.energy) {
        REQUIRE(std::isfinite(e));
      }
      return got;
    };
    auto sw_serial = [&]() {
      MidEval got;
      got.energy.assign(kMidSystems, 0.0);
      got.force.assign(kMidSystems * kMidAtoms * 3, 0.0);
      const auto before = rgpot::teamsOpened();
      for (std::size_t s = 0; s < kMidSystems; ++s) {
        rgpot::StillingerWeberSi pot;
        pot.setNeighbourMode(rgpot::nlist::Mode::Stream);
        pot.energyAndGradient(sw_pos.data() + s * kMidAtoms * 3, sw_species.data(), kMidBox,
                              kMidAtoms, &got.energy[s], got.force.data() + s * kMidAtoms * 3);
      }
      got.teams = rgpot::teamsOpened() - before;
      return got;
    };
    requireMid(sw_batch(), sw_serial(), sw_half, "SW", levels, false);
  }
}
