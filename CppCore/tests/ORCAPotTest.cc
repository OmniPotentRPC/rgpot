// MIT License
// Copyright 2023--present rgpot developers
//
// ORCAPot over the fake liborcac (orcac_h_fake_engine.c). RGPOT_ORCAC_ENGINE
// names the conforming fake; ORCAC_FAKE_C_ABI_MISMATCH,
// ORCAC_FAKE_ABI_MISMATCH, ORCAC_FAKE_SMALL_CONFIG, ORCAC_FAKE_SMALL_RESULT
// and ORCAC_FAKE_NO_COMPUTE name the variants the handshake must refuse
// before any session or evaluation.

#include "TestTempDir.hpp"
#include "rgpot/ORCAPot/ORCAPot.hpp"
#include "rgpot/abi/DynLib.hpp"
#include "rgpot/types/AtomMatrix.hpp"
#include "rgpot/units.hpp"
#ifdef RGPOT_HAS_CACHE
#include "rgpot/PotentialCache.hpp"
#endif

#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

using Catch::Approx;
using Catch::Matchers::ContainsSubstring;
using rgpot::types::AtomMatrix;

namespace {

// Mirrors orcac_fake_state_t in orcac_h_fake_engine.c.
struct FakeState {
  int sessions_created;
  int sessions_live;
  int evaluations;
  int null_probes;
  int in_call;
  int guess_policy;
  unsigned flags;
  int charge;
  int multiplicity;
  int nprocs;
  int maxcore_mb;
  double timeout_s;
  int used_guess;
  char method[256];
  char basis[256];
  char blocks[256];
  char executable[256];
};

AtomMatrix water() {
  return AtomMatrix{
      {0.0, 0.0, 0.117}, {0.0, 0.757, -0.469}, {0.0, -0.757, -0.469}};
}

const std::vector<int> kWaterZ{8, 1, 1};
constexpr const char *kExecutable = "/bin/sh";

using Box = std::array<std::array<double, 3>, 3>;

Box zeroBox() { return {}; }

/// The cubic box RgpotCalculator builds around a cluster with pbc=False.
Box paddedBox() {
  return {{{100.0, 0.0, 0.0}, {0.0, 100.0, 0.0}, {0.0, 0.0, 100.0}}};
}

std::string envPath(const char *name) {
  const char *value = std::getenv(name);
  REQUIRE(value != nullptr);
  REQUIRE(*value != '\0');
  return value;
}

rgpot::ORCAConfig baseConfig() {
  rgpot::ORCAConfig cfg;
  cfg.method = "B3LYP";
  cfg.basis = "def2-SVP";
  cfg.executable = kExecutable;
  return cfg;
}

FakeState *stateOf(rgpot::DynLib &lib) {
  using StateFn = FakeState *(*)();
  return lib.sym<StateFn>("orcac_fake_state")();
}

const FakeState &fakeState() {
  static rgpot::DynLib lib(envPath("RGPOT_ORCAC_ENGINE"));
  static FakeState *state = stateOf(lib);
  return *state;
}

std::string resolvedExecutable() {
  return std::filesystem::canonical(kExecutable).string();
}

double fakeEnergyEv(int charge, int multiplicity) {
  return (-1.0 + 0.01 * charge + 0.001 * (multiplicity - 1)) *
         rgpot::units::HARTREE_TO_EV;
}

} // namespace

TEST_CASE("ORCAPot loads the fake liborcac", "[orca]") {
  REQUIRE(rgpot::ORCAPot::probeAvailable());
  // The fake's default lookup finds no orca.
  REQUIRE_FALSE(rgpot::ORCAPot::abiAvailable());

  const int probes = fakeState().null_probes;
  rgpot::ORCAPot pot(baseConfig());
  REQUIRE(pot.available());
  REQUIRE(pot.loadError().empty());
  REQUIRE(fakeState().null_probes == probes + 1);
  REQUIRE(pot.getType() == rgpot::PotType::ORCA);
  REQUIRE(pot.libraryPath() == envPath("RGPOT_ORCAC_ENGINE"));
  REQUIRE(pot.engineVersion() == "orcac-fake/1.0");
  REQUIRE(pot.executableIdentity() == resolvedExecutable());
  const auto caps = pot.caps();
  REQUIRE(caps.reentrancy == rgpot::Reentrancy::PerInstance);
  REQUIRE(caps.perImageInstances);
  REQUIRE_FALSE(caps.periodic);
  REQUIRE_FALSE(caps.stress);
}

TEST_CASE("ORCAPot converts Eh and Eh/bohr", "[orca]") {
  rgpot::ORCAPot pot(baseConfig());
  REQUIRE(pot.available());
  auto [energy, forces, variance] = pot(water(), kWaterZ, zeroBox());
  REQUIRE(energy == Approx(fakeEnergyEv(0, 1)));
  REQUIRE(variance == 0.0);
  for (int i = 0; i < 3; ++i) {
    for (int k = 0; k < 3; ++k) {
      REQUIRE(forces(i, k) ==
              Approx(0.001 * (3 * i + k + 1) * rgpot::units::NEG_GRAD_TO_FORCE));
    }
  }
}

TEST_CASE("ORCAPot lowers ORCAConfig into OrcaCConfig", "[orca]") {
  rgpot::ORCAConfig cfg = baseConfig();
  cfg.blocks = "%scf\n  MaxIter 200\nend";
  cfg.nprocs = 4;
  cfg.maxcore_mb = 2000;
  cfg.timeout_s = 120.0;
  cfg.guess_policy = rgpot::ORCAConfig::GuessPolicy::Fresh;
  cfg.fresh_retry = false;
  rgpot::ORCAPot pot(cfg);
  REQUIRE(pot.available());
  const FakeState &s = fakeState();
  REQUIRE(std::string(s.method) == "B3LYP");
  REQUIRE(std::string(s.basis) == "def2-SVP");
  REQUIRE(std::string(s.blocks) == cfg.blocks);
  // The session runs the pinned file, not the name it was given.
  REQUIRE(std::string(s.executable) == resolvedExecutable());
  REQUIRE(s.nprocs == 4);
  REQUIRE(s.maxcore_mb == 2000);
  REQUIRE(s.timeout_s == 120.0);
  REQUIRE(s.guess_policy == 0);
  REQUIRE(s.flags == 1u);

  rgpot::ORCAConfig plain = baseConfig();
  plain.basis.clear();
  rgpot::ORCAPot defaults(plain);
  REQUIRE(defaults.available());
  REQUIRE(defaults.config().guess_policy ==
          rgpot::ORCAConfig::GuessPolicy::Fresh);
  REQUIRE(s.guess_policy == 0);
  REQUIRE(s.flags == 0u);
  REQUIRE(std::string(s.basis).empty());
  REQUIRE(s.timeout_s == rgpot::ORCAConfig::kDefaultTimeout);

  rgpot::ORCAConfig chained = baseConfig();
  chained.guess_policy = rgpot::ORCAConfig::GuessPolicy::MoRead;
  rgpot::ORCAPot moread(chained);
  REQUIRE(moread.available());
  REQUIRE(s.guess_policy == 1);
  REQUIRE(moread.paramsKey() != defaults.paramsKey());

  rgpot::ORCAConfig legacy = baseConfig();
  legacy.restart = true;
  rgpot::ORCAPot restarted(legacy);
  REQUIRE(restarted.available());
  REQUIRE(restarted.config().guess_policy ==
          rgpot::ORCAConfig::GuessPolicy::MoRead);
  REQUIRE(s.guess_policy == 1);
  REQUIRE(restarted.paramsKey() == moread.paramsKey());

  rgpot::ORCAConfig keep = baseConfig();
  keep.keep_files = true;
  REQUIRE_THROWS_WITH(rgpot::ORCAPot(keep), ContainsSubstring("keep_files"));
}

TEST_CASE("ORCAPot accepts the padded ASE box and refuses a wrapping cell",
          "[orca]") {
  rgpot::ORCAPot pot(baseConfig());
  auto padded = pot(water(), kWaterZ, paddedBox());
  auto open = pot(water(), kWaterZ, zeroBox());
  REQUIRE(std::get<0>(padded) == std::get<0>(open));

  // Exactly twice the extent along z is the smallest cell ASE pads to.
  const double z_extent = 0.117 + 0.469;
  Box tight = paddedBox();
  tight[2][2] = 2.0 * z_extent;
  REQUIRE_NOTHROW(pot(water(), kWaterZ, tight));

  Box wrapping = paddedBox();
  wrapping[2][2] = 1.5 * z_extent;
  REQUIRE_THROWS_WITH(pot(water(), kWaterZ, wrapping),
                      ContainsSubstring("ORCAPot: the atoms span") &&
                          ContainsSubstring("box vector 3") &&
                          ContainsSubstring("wrap across a periodic boundary"));

  // Long box vectors whose difference is short: the image along b - a
  // sits 1.1 Angstrom away, inside the 1.5 Angstrom span of the hydrogens.
  Box skewed = paddedBox();
  skewed[1] = {99.5, 1.0, 0.0};
  const int evaluations = fakeState().evaluations;
  REQUIRE_THROWS_WITH(pot(water(), kWaterZ, skewed),
                      ContainsSubstring("ORCAPot: the atoms span") &&
                          ContainsSubstring("wrap across a periodic boundary"));
  REQUIRE(fakeState().evaluations == evaluations);

  Box singular = paddedBox();
  singular[2] = singular[1];
  REQUIRE_THROWS_AS(pot(water(), kWaterZ, singular), std::invalid_argument);
  Box nonfinite = paddedBox();
  nonfinite[0][1] = std::numeric_limits<double>::quiet_NaN();
  REQUIRE_THROWS_AS(pot(water(), kWaterZ, nonfinite), std::invalid_argument);
}

TEST_CASE("ORCAPot charge and multiplicity go through orcac_configure",
          "[orca]") {
  rgpot::ORCAPot pot(baseConfig());
  const uint64_t neutral = pot.paramsKey();
  pot.setChargeMultiplicity(1, 2);
  REQUIRE(pot.paramsKey() != neutral);
  REQUIRE(pot.config().charge == 1);
  REQUIRE(pot.config().multiplicity == 2);
  REQUIRE(fakeState().multiplicity == 2);
  auto [energy, forces, variance] = pot(water(), kWaterZ, zeroBox());
  REQUIRE(energy == Approx(fakeEnergyEv(1, 2)));

  pot.setChargeMultiplicity(0, 1);
  REQUIRE(pot.paramsKey() == neutral);

  rgpot::ORCAConfig cation = baseConfig();
  cation.charge = 1;
  cation.multiplicity = 2;
  rgpot::ORCAPot fresh(cation);
  pot.setChargeMultiplicity(1, 2);
  REQUIRE(fresh.paramsKey() == pot.paramsKey());

  REQUIRE_THROWS_AS(pot.setChargeMultiplicity(0, 0), std::invalid_argument);
  REQUIRE_THROWS_WITH(pot.setChargeMultiplicity(0, 12),
                      ContainsSubstring("ORCAC_E_INVALID") &&
                          ContainsSubstring("multiplicity 1 to 9"));
  REQUIRE(pot.config().multiplicity == 2);
  REQUIRE(pot.paramsKey() == fresh.paramsKey());
}

TEST_CASE("ORCAPot key follows the method and not the resources", "[orca]") {
  rgpot::ORCAPot a(baseConfig());
  rgpot::ORCAConfig more = baseConfig();
  more.nprocs = 8;
  more.timeout_s = 10.0;
  rgpot::ORCAPot b(more);
  REQUIRE(a.paramsKey() == b.paramsKey());
  rgpot::ORCAConfig other = baseConfig();
  other.method = "PBE0";
  rgpot::ORCAPot c(other);
  REQUIRE(a.paramsKey() != c.paramsKey());
  rgpot::ORCAConfig blocks = baseConfig();
  blocks.blocks = "%scf\n  MaxIter 50\nend";
  rgpot::ORCAPot d(blocks);
  REQUIRE(a.paramsKey() != d.paramsKey());
}

TEST_CASE("ORCAPot chains guesses only when the policy is moread", "[orca]") {
  rgpot::ORCAConfig chained = baseConfig();
  chained.guess_policy = rgpot::ORCAConfig::GuessPolicy::MoRead;
  rgpot::ORCAPot pot(chained);
  (void)pot(water(), kWaterZ, zeroBox());
  REQUIRE_FALSE(fakeState().used_guess);
  (void)pot(water(), kWaterZ, zeroBox());
  REQUIRE(fakeState().used_guess);
  pot.setChargeMultiplicity(0, 3);
  (void)pot(water(), kWaterZ, zeroBox());
  REQUIRE_FALSE(fakeState().used_guess);

  rgpot::ORCAPot fresh(baseConfig());
  (void)fresh(water(), kWaterZ, zeroBox());
  (void)fresh(water(), kWaterZ, zeroBox());
  REQUIRE_FALSE(fakeState().used_guess);
}

TEST_CASE("ORCAPot pins the executable it resolved", "[orca]") {
  namespace fs = std::filesystem;
  rgpot::tests::TestTempDir dir("rgpot-orcapot-exec");
  const fs::path copy = dir.path() / "orca";
  fs::copy_file(kExecutable, copy);
  fs::permissions(copy, fs::perms::owner_all);
  const fs::path link = dir.path() / "orca-link";
  fs::create_symlink(kExecutable, link);

  rgpot::ORCAConfig cfg = baseConfig();
  cfg.executable = link.string();
  rgpot::ORCAPot pot(cfg);
  REQUIRE(pot.available());
  REQUIRE(pot.executableIdentity() == resolvedExecutable());
  REQUIRE(std::string(fakeState().executable) == resolvedExecutable());
  const uint64_t key = pot.paramsKey();
  REQUIRE(key == rgpot::ORCAPot(baseConfig()).paramsKey());
  (void)pot(water(), kWaterZ, zeroBox());

  // Retargeting the name changes the key and stops the runs.
  fs::remove(link);
  fs::create_symlink(copy, link);
  REQUIRE(pot.paramsKey() != key);
  int evaluations = fakeState().evaluations;
  REQUIRE_THROWS_AS(pot(water(), kWaterZ, zeroBox()), rgpot::ORCAError);
  REQUIRE_THROWS_WITH(pot(water(), kWaterZ, zeroBox()),
                      ContainsSubstring("changed since the session was opened"));
  REQUIRE(fakeState().evaluations == evaluations);

  fs::remove(link);
  fs::create_symlink(kExecutable, link);
  REQUIRE(pot.paramsKey() == key);
  (void)pot(water(), kWaterZ, zeroBox());

  // The same name and file with a new modification time is a new build.
  cfg.executable = copy.string();
  rgpot::ORCAPot local(cfg);
  REQUIRE(local.executableIdentity() == fs::canonical(copy).string());
  const uint64_t local_key = local.paramsKey();
  REQUIRE(local_key != key);
  (void)local(water(), kWaterZ, zeroBox());
  fs::last_write_time(copy, fs::last_write_time(copy) + std::chrono::seconds(5));
  REQUIRE(local.paramsKey() != local_key);
  evaluations = fakeState().evaluations;
  REQUIRE_THROWS_WITH(local(water(), kWaterZ, zeroBox()),
                      ContainsSubstring("ORCAPot: the executable") &&
                          ContainsSubstring("make a new ORCAPot"));
  REQUIRE(fakeState().evaluations == evaluations);
  rgpot::ORCAPot rebuilt(cfg);
  REQUIRE(rebuilt.paramsKey() != local_key);
  (void)rebuilt(water(), kWaterZ, zeroBox());
}

#ifdef RGPOT_HAS_CACHE
TEST_CASE("ORCAPot uses the cache only for fresh guesses", "[orca][cache]") {
  rgpot::tests::TestTempDir db("rgpot-orcapot-cache");
  rgpot::cache::PotentialCache cache((db.path() / "db").string());
  const auto run = [&](const rgpot::ORCAConfig &cfg) {
    rgpot::ORCAPot pot(cfg);
    pot.setCache(&cache);
    (void)pot(water(), kWaterZ, zeroBox());
  };

  const int before = fakeState().evaluations;
  run(baseConfig());
  REQUIRE(fakeState().evaluations == before + 1);
  run(baseConfig());
  REQUIRE(fakeState().evaluations == before + 1);

  // A chained guess makes the result depend on the previous geometry.
  rgpot::ORCAConfig chained = baseConfig();
  chained.guess_policy = rgpot::ORCAConfig::GuessPolicy::MoRead;
  run(chained);
  REQUIRE(fakeState().evaluations == before + 2);
  run(chained);
  REQUIRE(fakeState().evaluations == before + 3);
}
#endif

TEST_CASE("ORCAPot reports failed runs as ORCAError", "[orca]") {
  struct Case {
    const char *method;
    const char *status;
    const char *message;
  } cases[] = {
      {"FAIL", "ORCAC_E_ERROR", "method FAIL"},
      {"NOCONV", "ORCAC_E_NOT_CONVERGED", "did not converge"},
  };
  for (const auto &c : cases) {
    INFO(c.method);
    rgpot::ORCAConfig cfg = baseConfig();
    cfg.method = c.method;
    rgpot::ORCAPot pot(cfg);
    REQUIRE(pot.available());
    REQUIRE_THROWS_AS(pot(water(), kWaterZ, zeroBox()), rgpot::ORCAError);
    REQUIRE_THROWS_WITH(pot(water(), kWaterZ, zeroBox()),
                        ContainsSubstring(c.status) &&
                            ContainsSubstring(c.message));
  }
}

TEST_CASE("ORCAPot cancel stops the running call", "[orca]") {
  rgpot::ORCAConfig cfg = baseConfig();
  cfg.method = "SLOW";
  rgpot::ORCAPot pot(cfg);
  REQUIRE(pot.available());
  pot.cancel();
  std::string error;
  std::thread worker([&] {
    try {
      (void)pot(water(), kWaterZ, zeroBox());
    } catch (const rgpot::ORCAError &e) {
      error = e.what();
    }
  });
  const volatile int *in_call = &fakeState().in_call;
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(4);
  while (*in_call == 0 && std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  pot.cancel();
  worker.join();
  REQUIRE_THAT(error, ContainsSubstring("ORCAC_E_CANCELLED"));
}

TEST_CASE("ORCAPot runs an explicit executable without orcac_available",
          "[orca]") {
  rgpot::ORCAConfig cfg = baseConfig();
  rgpot::ORCAPot pot(cfg);
  REQUIRE(pot.available());

  cfg.executable = "/nonexistent/orca";
  rgpot::ORCAPot missing(cfg);
  REQUIRE_FALSE(missing.available());
  REQUIRE_THAT(missing.loadError(),
               ContainsSubstring("orcac_probe_executable(/nonexistent/orca)") &&
                   ContainsSubstring("ORCAC_E_NOT_FOUND"));
  REQUIRE_THROWS_AS(missing(water(), kWaterZ, zeroBox()), rgpot::ORCAError);

  cfg.executable.clear();
  ::unsetenv("ORCAC_ORCA");
  rgpot::ORCAPot searched(cfg);
  REQUIRE_FALSE(searched.available());
  REQUIRE_THAT(searched.loadError(), ContainsSubstring("orcac's search"));

  ::setenv("ORCAC_ORCA", kExecutable, 1);
  rgpot::ORCAPot found(cfg);
  ::unsetenv("ORCAC_ORCA");
  REQUIRE(found.available());
  REQUIRE(found.executableIdentity() == resolvedExecutable());
}

TEST_CASE("ORCAPot refuses a liborcac that disagrees with orcac.h before "
          "any evaluation",
          "[orca]") {
  // null_probes counts the session-less orcac_energy_gradient call of the
  // handshake. Only the result size needs it, so every other mismatch is
  // refused before the compute entry point is called at all.
  struct Variant {
    const char *env;
    const char *reason;
    int null_probes;
  } variants[] = {
      {"ORCAC_FAKE_C_ABI_MISMATCH",
       "reports C ABI version 2, but rgpot was built against 1", 0},
      {"ORCAC_FAKE_ABI_MISMATCH",
       "reports ABI version 2, but rgpot was built against 1", 0},
      {"ORCAC_FAKE_SMALL_CONFIG", "reports an OrcaCConfig struct_size of", 0},
      {"ORCAC_FAKE_SMALL_RESULT", "reports an OrcaCResult struct_size of", 1},
      {"ORCAC_FAKE_NO_COMPUTE", "lacks orcac_energy_gradient", 0},
  };
  for (const auto &v : variants) {
    INFO(v.env);
    rgpot::DynLib held(envPath(v.env));
    const FakeState *state = stateOf(held);
    rgpot::ORCAConfig cfg = baseConfig();
    cfg.library = envPath(v.env);
    rgpot::ORCAPot pot(cfg);
    REQUIRE_FALSE(pot.available());
    REQUIRE_THAT(pot.loadError(), ContainsSubstring(v.reason));
    REQUIRE_THROWS_WITH(pot(water(), kWaterZ, zeroBox()),
                        ContainsSubstring(v.reason));
    REQUIRE(state->sessions_created == 0);
    REQUIRE(state->evaluations == 0);
    REQUIRE(state->null_probes == v.null_probes);
  }
}

TEST_CASE("ORCAPot rejects a config that cannot run", "[orca]") {
  rgpot::ORCAConfig cfg = baseConfig();
  cfg.method.clear();
  REQUIRE_THROWS_AS(rgpot::ORCAPot(cfg), std::invalid_argument);
  cfg = baseConfig();
  cfg.multiplicity = 0;
  REQUIRE_THROWS_AS(rgpot::ORCAPot(cfg), std::invalid_argument);
  cfg = baseConfig();
  cfg.keywords = "TightSCF\nPrintBasis";
  REQUIRE_THROWS_AS(rgpot::ORCAPot(cfg), std::invalid_argument);
  cfg = baseConfig();
  cfg.timeout_s = -1.0;
  REQUIRE_THROWS_AS(rgpot::ORCAPot(cfg), std::invalid_argument);
}

TEST_CASE("ORCAPot names an explicit library it cannot open", "[orca]") {
  rgpot::ORCAConfig cfg = baseConfig();
  cfg.library = "/nonexistent/liborcac.so";
  rgpot::ORCAPot pot(cfg);
  REQUIRE_FALSE(pot.available());
  REQUIRE_THAT(pot.loadError(),
               ContainsSubstring("liborcac not loaded (tried "
                                 "/nonexistent/liborcac.so)") &&
                   ContainsSubstring("ORCAC_LIBRARY or RGPOT_ORCAC_ENGINE"));
}

// Hidden: meson runs it in a process of its own as ORCAPotAbsentLibrary,
// with RGPOT_ORCAC_ENGINE naming a file that does not exist. The exclusive
// environment rule must stop the search there, with no fallback.
TEST_CASE("ORCAPot without liborcac", "[.][orca][absent-library]") {
  const std::string missing = envPath("RGPOT_ORCAC_ENGINE");
  REQUIRE_FALSE(rgpot::ORCAPot::probeAvailable());
  rgpot::ORCAConfig cfg;
  cfg.method = "B3LYP";
  rgpot::ORCAPot pot(cfg);
  REQUIRE_FALSE(pot.available());
  REQUIRE_THAT(pot.loadError(), ContainsSubstring("tried " + missing + ")"));
  REQUIRE_THROWS_WITH(pot(water(), kWaterZ, zeroBox()),
                      ContainsSubstring("Install orcac"));
}
