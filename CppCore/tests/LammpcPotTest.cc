// MIT License
// Copyright 2023--present rgpot developers
//
// LammpsPot over lammpc_fake_engine, which RGPOT_LAMMPC_ENGINE names. The
// fake evaluates the unshifted lj/cut pair from PotentialConfig.lammps and
// returns a fixed stress, so each value here is checked against the analytic
// dimer and the fake's constants. RGPOT_LAMMPC_NOSTRESS_ENGINE names the
// variant without stress and RGPOT_LAMMPC_ABI2_ENGINE the one reporting an
// incompatible eindir bridge ABI major.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <capnp/message.h>
#include <capnp/serialize.h>
#include <kj/array.h>

#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "rgpot/ExternalPot/ExternalPot.hpp"
#include "rgpot/abi/DynLib.hpp"
#include "rgpot/ExternalPot/external_potentials.h"
#include "rgpot/rpc/Potentials.capnp.h"

using Catch::Matchers::ContainsSubstring;
using Catch::Matchers::WithinAbs;

namespace {

constexpr double kEps = 0.2;
constexpr double kSigma = 2.5;
constexpr double kSeparation = 3.0;
constexpr double kStress[9] = {0.1, 0.2, 0.3, 0.2, 0.4, 0.5, 0.3, 0.5, 0.6};
constexpr const char *kMissing = "./liblammpc_missing_for_test.so";

std::string env_path(const char *name) {
  const char *path = std::getenv(name);
  INFO(name);
  REQUIRE(path != nullptr);
  return path;
}

std::string fake_engine_path() { return env_path("RGPOT_LAMMPC_ENGINE"); }

std::vector<unsigned char> flat_bytes(::capnp::MallocMessageBuilder &msg) {
  auto words = ::capnp::messageToFlatArray(msg);
  const auto bytes = words.asBytes();
  return std::vector<unsigned char>(bytes.begin(), bytes.end());
}

std::vector<unsigned char> lammps_config(const char *pair_style = "lj/cut 10.0",
                                         const char *coeff = "1 1 0.2 2.5") {
  ::capnp::MallocMessageBuilder msg;
  auto lammps = msg.initRoot<::PotentialConfig>().initLammps();
  lammps.setPairStyle(pair_style);
  auto coeffs = lammps.initPairCoeffs(1);
  coeffs.set(0, coeff);
  auto types = lammps.initTypeToAtomicNumber(1);
  types.set(0, 29);
  lammps.setBoundary("f f f");
  return flat_bytes(msg);
}

std::vector<unsigned char> cpmd_config() {
  ::capnp::MallocMessageBuilder msg;
  msg.initRoot<::PotentialConfig>().initCpmd().setFunctional("BLYP");
  return flat_bytes(msg);
}

double lj_energy(double r) {
  const double sr6 = std::pow(kSigma / r, 6.0);
  return 4.0 * kEps * (sr6 * sr6 - sr6);
}

double lj_force_on_second(double r) {
  const double sr6 = std::pow(kSigma / r, 6.0);
  return 24.0 * kEps * (2.0 * sr6 * sr6 - sr6) / r;
}

struct Dimer {
  std::array<double, 6> pos{0.0, 0.0, 0.0, kSeparation, 0.0, 0.0};
  std::array<int, 2> atmnrs{29, 29};
  std::array<double, 9> box{20.0, 0.0, 0.0, 0.0, 20.0, 0.0, 0.0, 0.0, 20.0};
};

void require_dimer_forces(const double *f) {
  const double expected = lj_force_on_second(kSeparation);
  REQUIRE_THAT(f[0], WithinAbs(-expected, 1e-12));
  REQUIRE_THAT(f[1], WithinAbs(0.0, 1e-15));
  REQUIRE_THAT(f[2], WithinAbs(0.0, 1e-15));
  REQUIRE_THAT(f[3], WithinAbs(expected, 1e-12));
  REQUIRE_THAT(f[4], WithinAbs(0.0, 1e-15));
  REQUIRE_THAT(f[5], WithinAbs(0.0, 1e-15));
}

std::string construct_error(const std::vector<unsigned char> &bytes,
                            const std::string &path) {
  try {
    rgpot::LammpsPot pot(bytes.data(), bytes.size(), path);
  } catch (const std::exception &ex) {
    return ex.what();
  }
  return {};
}

} // namespace

TEST_CASE("LammpsPot lammpc constructor evaluates energy and forces",
          "[external][lammpc]") {
  const auto config = lammps_config();
  rgpot::LammpsPot pot(config.data(), config.size(), fake_engine_path());

  Dimer d;
  std::array<double, 6> forces{};
  rgpot::ForceOut out{};
  out.F = forces.data();
  const rgpot::ForceInput in{.nAtoms = 2,
                             .pos = d.pos.data(),
                             .atmnrs = d.atmnrs.data(),
                             .box = d.box.data()};
  pot.forceImpl(in, &out);

  REQUIRE_THAT(out.energy, WithinAbs(lj_energy(kSeparation), 1e-12));
  REQUIRE(out.variance == 0.0);
  require_dimer_forces(forces.data());
  REQUIRE(out.has_stress == 1);
  for (int i = 0; i < 9; ++i)
    REQUIRE(out.stress[i] == kStress[i]);

  const double moved = kSeparation + 0.25;
  d.pos[3] = moved;
  pot.forceImpl(in, &out);
  REQUIRE_THAT(out.energy, WithinAbs(lj_energy(moved), 1e-12));
  REQUIRE_THAT(forces[3], WithinAbs(lj_force_on_second(moved), 1e-12));
}

TEST_CASE("LammpsPot lammpc caps advertise stress", "[external][lammpc]") {
  const auto config = lammps_config();
  const rgpot::LammpsPot pot(config.data(), config.size(), fake_engine_path());
  const auto caps = pot.caps();
  REQUIRE(caps.reentrancy == rgpot::Reentrancy::PerInstance);
  REQUIRE(caps.periodic);
  REQUIRE(caps.stress);
}

TEST_CASE("rgpot_lammpc_potential searches RGPOT_LAMMPC_ENGINE when no path "
          "is given",
          "[external][lammpc]") {
  const auto config = lammps_config();
  auto pot = rgpot::rgpot_lammpc_potential(config.data(), config.size());
  REQUIRE(pot);
  Dimer d;
  std::array<double, 6> forces{};
  rgpot::ForceOut out{};
  out.F = forces.data();
  pot->forceImpl({.nAtoms = 2,
                  .pos = d.pos.data(),
                  .atmnrs = d.atmnrs.data(),
                  .box = d.box.data()},
                 &out);
  REQUIRE_THAT(out.energy, WithinAbs(lj_energy(kSeparation), 1e-12));
}

TEST_CASE("rgpot_lammpc_potential C entry point evaluates the dimer",
          "[external][lammpc][c_abi]") {
  const auto config = lammps_config();
  const std::string path = fake_engine_path();
  char err[512] = {0};
  rgpot_external_potential *pot = rgpot_lammpc_potential(
      path.c_str(), config.data(), config.size(), err, sizeof err);
  INFO(err);
  REQUIRE(pot != nullptr);

  Dimer d;
  std::array<double, 6> forces{};
  double energy = 0.0;
  const int status = rgpot_external_potential_energy_forces(
      pot, d.pos.data(), d.atmnrs.data(), 2, d.box.data(), forces.data(),
      &energy, err, sizeof err);
  rgpot_external_potential_free(pot);
  REQUIRE(status == 0);
  REQUIRE_THAT(energy, WithinAbs(lj_energy(kSeparation), 1e-12));
  require_dimer_forces(forces.data());

  rgpot_external_potential *searched = rgpot_lammpc_potential(
      nullptr, config.data(), config.size(), err, sizeof err);
  REQUIRE(searched != nullptr);
  rgpot_external_potential_free(searched);
}

TEST_CASE("An explicit lammpc engine_path that does not open is not replaced "
          "by the search",
          "[external][lammpc]") {
  REQUIRE(!fake_engine_path().empty());
  const auto config = lammps_config();
  const std::string what = construct_error(config, kMissing);
  REQUIRE_THAT(what, ContainsSubstring(kMissing));
  REQUIRE_THAT(what, ContainsSubstring("not loaded from"));

  char err[512] = {0};
  REQUIRE(rgpot_lammpc_potential(kMissing, config.data(), config.size(), err,
                                 sizeof err) == nullptr);
  REQUIRE_THAT(std::string(err), ContainsSubstring(kMissing));
}

TEST_CASE("LammpsPot lammpc constructor rejects bytes that are not a lammps "
          "PotentialConfig",
          "[external][lammpc]") {
  const std::string path = fake_engine_path();
  const auto config = lammps_config();
  const std::vector<unsigned char> misaligned(config.begin(), config.end() - 3);
  const std::vector<unsigned char> truncated(config.begin(),
                                             config.begin() + 8);
  const auto cpmd = cpmd_config();
  for (const auto *bytes : {&misaligned, &truncated, &cpmd}) {
    REQUIRE_THROWS_AS(rgpot::LammpsPot(bytes->data(), bytes->size(), path),
                      std::invalid_argument);
  }
  REQUIRE_THROWS_AS(rgpot::LammpsPot(nullptr, 0, path), std::invalid_argument);
  REQUIRE_THAT(construct_error(truncated, path),
               ContainsSubstring("not a flat PotentialConfig"));
  REQUIRE_THAT(construct_error(cpmd, path), ContainsSubstring("lammps arm"));

  char err[512] = {0};
  REQUIRE(rgpot_lammpc_potential(path.c_str(), misaligned.data(),
                                 misaligned.size(), err,
                                 sizeof err) == nullptr);
  REQUIRE_THAT(std::string(err), ContainsSubstring("word-aligned"));
}

TEST_CASE("LammpsPot lammpc constructor reports a refused session",
          "[external][lammpc]") {
  const auto config = lammps_config("eam/alloy");
  REQUIRE_THAT(construct_error(config, fake_engine_path()),
               ContainsSubstring("lammpc_session_create_from_config failed"));
}

TEST_CASE("LammpsPot lammpc finalizes the backend on every teardown path",
          "[external][lammpc]") {
  rgpot::DynLib fake;
  fake.open(fake_engine_path());
  using CountFn = int (*)(void);
  const auto init_calls = fake.sym<CountFn>("lammpc_fake_init_calls");
  const auto finalize_calls = fake.sym<CountFn>("lammpc_fake_finalize_calls");
  const auto config = lammps_config();

  SECTION("one pot") {
    const int inits = init_calls();
    const int finals = finalize_calls();
    {
      rgpot::LammpsPot pot(config.data(), config.size(), fake_engine_path());
      REQUIRE(init_calls() == inits + 1);
      REQUIRE(finalize_calls() == finals);
    }
    REQUIRE(finalize_calls() == finals + 1);
  }

  SECTION("a live peer keeps the backend until the last pot goes") {
    const int finals = finalize_calls();
    auto first = std::make_unique<rgpot::LammpsPot>(
        config.data(), config.size(), fake_engine_path());
    {
      rgpot::LammpsPot second(config.data(), config.size(),
                              fake_engine_path());
    }
    REQUIRE(finalize_calls() == finals);
    first.reset();
    REQUIRE(finalize_calls() == finals + 1);
  }

  SECTION("a move leaves one teardown") {
    const int finals = finalize_calls();
    {
      rgpot::LammpsPot pot(config.data(), config.size(), fake_engine_path());
      rgpot::LammpsPot moved(std::move(pot));
    }
    REQUIRE(finalize_calls() == finals + 1);
  }

  SECTION("a refused session") {
    const auto refused = lammps_config("eam/alloy");
    const int inits = init_calls();
    const int finals = finalize_calls();
    REQUIRE_THAT(construct_error(refused, fake_engine_path()),
                 ContainsSubstring("lammpc_session_create_from_config failed"));
    REQUIRE(init_calls() == inits + 1);
    REQUIRE(finalize_calls() == finals + 1);
  }
}

TEST_CASE("LammpsPot lammpc reports no stress when the engine declares none",
          "[external][lammpc]") {
  const auto config = lammps_config();
  rgpot::LammpsPot pot(config.data(), config.size(),
                       env_path("RGPOT_LAMMPC_NOSTRESS_ENGINE"));
  REQUIRE_FALSE(pot.caps().stress);

  Dimer d;
  std::array<double, 6> forces{};
  rgpot::ForceOut out{};
  out.F = forces.data();
  out.has_stress = 1;
  pot.forceImpl({.nAtoms = 2,
                 .pos = d.pos.data(),
                 .atmnrs = d.atmnrs.data(),
                 .box = d.box.data()},
                &out);
  REQUIRE(out.has_stress == 0);
  REQUIRE_THAT(out.energy, WithinAbs(lj_energy(kSeparation), 1e-12));
  require_dimer_forces(forces.data());
}

TEST_CASE("LammpsPot lammpc refuses an engine with an incompatible ABI",
          "[external][lammpc][compat]") {
  REQUIRE(!fake_engine_path().empty());
  const std::string path = env_path("RGPOT_LAMMPC_ABI2_ENGINE");
  const auto config = lammps_config();
  const std::string what = construct_error(config, path);
  REQUIRE_THAT(what, ContainsSubstring(path));
  REQUIRE_THAT(what, ContainsSubstring("lammpc backend refused"));
  REQUIRE_THAT(what, ContainsSubstring("eindir bridge ABI major 2 is not 1"));

  char err[512] = {0};
  REQUIRE(rgpot_lammpc_potential(path.c_str(), config.data(), config.size(),
                                 err, sizeof err) == nullptr);
  REQUIRE_THAT(std::string(err),
               ContainsSubstring("eindir bridge ABI major 2 is not 1"));
}

TEST_CASE("LammpsPot lammpc cache keys follow the engine path and config",
          "[external][lammpc][cache]") {
  const std::string path = fake_engine_path();
  const std::string other_path = env_path("RGPOT_LAMMPC_NOSTRESS_ENGINE");
  const auto config = lammps_config();
  const auto deeper = lammps_config("lj/cut 10.0", "1 1 0.3 2.5");

  const rgpot::LammpsPot a(config.data(), config.size(), path);
  const rgpot::LammpsPot a_again(config.data(), config.size(), path);
  const rgpot::LammpsPot b(deeper.data(), deeper.size(), path);
  const rgpot::LammpsPot a_elsewhere(config.data(), config.size(), other_path);

  REQUIRE(a.paramsKey() != 0);
  REQUIRE(a.paramsKey() == a_again.paramsKey());
  REQUIRE(a.paramsKey() != b.paramsKey());
  REQUIRE(a.paramsKey() != a_elsewhere.paramsKey());
  REQUIRE(b.paramsKey() != a_elsewhere.paramsKey());

  Dimer d;
  std::array<double, 6> forces{};
  rgpot::ForceOut out{};
  out.F = forces.data();
  const rgpot::ForceInput in{.nAtoms = 2,
                             .pos = d.pos.data(),
                             .atmnrs = d.atmnrs.data(),
                             .box = d.box.data()};
  a.forceImpl(in, &out);
  const double energy_a = out.energy;
  b.forceImpl(in, &out);
  REQUIRE(out.energy != energy_a);
  REQUIRE_THAT(out.energy, WithinAbs(lj_energy(kSeparation) * 1.5, 1e-12));
}

namespace {

struct ScratchDir {
  std::filesystem::path path;

  ScratchDir() {
    const auto stamp = std::chrono::steady_clock::now().time_since_epoch();
    path = std::filesystem::temp_directory_path() /
           ("rgpot_lammpc_key_" + std::to_string(stamp.count()));
    std::filesystem::create_directories(path);
  }
  ~ScratchDir() {
    std::error_code ignored;
    std::filesystem::remove_all(path, ignored);
  }
  ScratchDir(const ScratchDir &) = delete;
  ScratchDir &operator=(const ScratchDir &) = delete;

  std::string link(const std::string &name, const std::string &target) const {
    const auto at = path / name;
    std::error_code ignored;
    std::filesystem::remove(at, ignored);
    std::filesystem::create_symlink(target, at);
    return at.string();
  }

  std::string copy(const std::string &prefix, const std::string &source) const {
    const auto at =
        path / (prefix + std::filesystem::path(source).filename().string());
    std::filesystem::copy_file(
        source, at, std::filesystem::copy_options::overwrite_existing);
    return at.string();
  }
};

uint64_t key_for(const std::vector<unsigned char> &config,
                 const std::string &path) {
  const rgpot::LammpsPot pot(config.data(), config.size(), path);
  return pot.paramsKey();
}

} // namespace

TEST_CASE("LammpsPot lammpc requests that resolve to one backend share a key",
          "[external][lammpc][cache]") {
  const std::filesystem::path real =
      std::filesystem::canonical(fake_engine_path());
  const auto config = lammps_config();
  const ScratchDir dir;
  const std::string first = dir.link("first.so", real.string());
  const std::string second = dir.link("second.so", first);
  const std::string detour = (dir.path / ".." / dir.path.filename() /
                              "first.so").string();

  const uint64_t direct = key_for(config, real.string());
  REQUIRE(direct != 0);
  REQUIRE(key_for(config, first) == direct);
  REQUIRE(key_for(config, second) == direct);
  REQUIRE(key_for(config, detour) == direct);

  const rgpot::LammpsPot a(config.data(), config.size(), first);
  const rgpot::LammpsPot b(config.data(), config.size(), second);
  REQUIRE(a.paramsKey() == b.paramsKey());

  const auto deeper = lammps_config("lj/cut 10.0", "1 1 0.3 2.5");
  REQUIRE(key_for(deeper, first) != direct);
}

TEST_CASE("LammpsPot lammpc a re-pointed path gets a new key",
          "[external][lammpc][cache]") {
  const std::string stress = fake_engine_path();
  const std::string nostress = env_path("RGPOT_LAMMPC_NOSTRESS_ENGINE");
  const auto config = lammps_config();
  const ScratchDir dir;
  const std::string moving = dir.link("moving.so", stress);

  const uint64_t before = key_for(config, moving);
  REQUIRE(before == key_for(config, stress));

  dir.link("moving.so", nostress);
  const uint64_t after = key_for(config, moving);
  REQUIRE(after != before);
  REQUIRE(after == key_for(config, nostress));

  dir.link("moving.so", stress);
  REQUIRE(key_for(config, moving) == before);
}

namespace {

void set_env(const char *name, const std::string &value) {
#if defined(_WIN32)
  _putenv_s(name, value.c_str());
#else
  setenv(name, value.c_str(), 1);
#endif
}

void clear_env(const char *name) {
#if defined(_WIN32)
  _putenv_s(name, "");
#else
  unsetenv(name);
#endif
}

class EnvGuard {
public:
  explicit EnvGuard(const char *name) : name_(name) {
    if (const char *old = std::getenv(name)) {
      saved_ = old;
      had_ = true;
    }
  }
  ~EnvGuard() {
    if (had_) {
      set_env(name_, saved_);
    } else {
      clear_env(name_);
    }
  }
  EnvGuard(const EnvGuard &) = delete;
  EnvGuard &operator=(const EnvGuard &) = delete;

private:
  const char *name_;
  std::string saved_;
  bool had_ = false;
};

} // namespace

TEST_CASE("LammpsPot lammpc keys follow the engine the environment selects",
          "[external][lammpc][cache]") {
  const std::string stress = fake_engine_path();
  const std::string nostress = env_path("RGPOT_LAMMPC_NOSTRESS_ENGINE");
  const auto config = lammps_config();
  const EnvGuard restore_library("LAMMPC_LIBRARY");
  const EnvGuard restore_engine("RGPOT_LAMMPC_ENGINE");
  const ScratchDir dir;
  const std::string alias = dir.link("alias.so", stress);

  const uint64_t stress_key = key_for(config, stress);
  const uint64_t nostress_key = key_for(config, nostress);
  REQUIRE(stress_key != nostress_key);

  clear_env("LAMMPC_LIBRARY");
  set_env("RGPOT_LAMMPC_ENGINE", stress);
  REQUIRE(key_for(config, "") == stress_key);
  set_env("RGPOT_LAMMPC_ENGINE", nostress);
  REQUIRE(key_for(config, "") == nostress_key);
  set_env("RGPOT_LAMMPC_ENGINE", alias);
  REQUIRE(key_for(config, "") == stress_key);

  set_env("LAMMPC_LIBRARY", nostress);
  set_env("RGPOT_LAMMPC_ENGINE", stress);
  REQUIRE(key_for(config, "") == nostress_key);

  set_env("LAMMPC_LIBRARY", kMissing);
  REQUIRE(key_for(config, "") == stress_key);

  set_env("LAMMPC_LIBRARY", nostress);
  REQUIRE(key_for(config, stress) == stress_key);
}

TEST_CASE("LammpsPot lammpc environment engines keep separate results",
          "[external][lammpc][cache]") {
  const std::string stress = fake_engine_path();
  const std::string nostress = env_path("RGPOT_LAMMPC_NOSTRESS_ENGINE");
  const auto config = lammps_config();
  const EnvGuard restore_engine("RGPOT_LAMMPC_ENGINE");
  const EnvGuard restore_library("LAMMPC_LIBRARY");
  clear_env("LAMMPC_LIBRARY");

  set_env("RGPOT_LAMMPC_ENGINE", stress);
  const rgpot::LammpsPot with_stress(config.data(), config.size());
  set_env("RGPOT_LAMMPC_ENGINE", nostress);
  const rgpot::LammpsPot without_stress(config.data(), config.size());

  REQUIRE(with_stress.caps().stress);
  REQUIRE_FALSE(without_stress.caps().stress);
  REQUIRE(with_stress.paramsKey() != without_stress.paramsKey());
}

TEST_CASE("LammpsPot lammpc identical backends the environment selects get "
          "separate keys",
          "[external][lammpc][cache]") {
  const auto config = lammps_config();
  const EnvGuard restore_engine("RGPOT_LAMMPC_ENGINE");
  const EnvGuard restore_library("LAMMPC_LIBRARY");
  const ScratchDir dir;
  // Byte-identical copies report one version, ABI and capability set, so only
  // the file that loaded can tell them apart.
  const std::string first = dir.copy("first_", fake_engine_path());
  const std::string second = dir.copy("second_", fake_engine_path());
  clear_env("LAMMPC_LIBRARY");

  set_env("RGPOT_LAMMPC_ENGINE", first);
  const uint64_t first_key = key_for(config, "");
  set_env("RGPOT_LAMMPC_ENGINE", second);
  const uint64_t second_key = key_for(config, "");

  REQUIRE(first_key != 0);
  REQUIRE(first_key != second_key);
  REQUIRE(first_key == key_for(config, first));
  REQUIRE(second_key == key_for(config, second));
}

TEST_CASE("LammpsPot LJ keys follow the engine the environment selects",
          "[external][lammpc][cache]") {
  const std::string first = env_path("RGPOT_LAMMPS_LEGACY_ENGINE_A");
  const std::string second = env_path("RGPOT_LAMMPS_LEGACY_ENGINE_B");
  const EnvGuard restore_primary("RGPOT_LAMMPS_ENGINE");
  const EnvGuard restore_secondary("LAMMPS_ENGINE");
  const ScratchDir dir;
  const std::string alias = dir.link("legacy_alias.so", first);
  clear_env("LAMMPS_ENGINE");

  const auto key = [](const std::string &path) {
    rgpot::ExternalLjConfig config;
    config.engine_path = path;
    return rgpot::LammpsPot(config).paramsKey();
  };

  const uint64_t first_key = key(first);
  const uint64_t second_key = key(second);
  REQUIRE(first_key != second_key);
  REQUIRE(key(alias) == first_key);

  set_env("RGPOT_LAMMPS_ENGINE", first);
  REQUIRE(key("") == first_key);
  set_env("RGPOT_LAMMPS_ENGINE", second);
  REQUIRE(key("") == second_key);
  set_env("RGPOT_LAMMPS_ENGINE", alias);
  REQUIRE(key("") == first_key);

  set_env("RGPOT_LAMMPS_ENGINE", kMissing);
  set_env("LAMMPS_ENGINE", second);
  REQUIRE(key("") == second_key);

  set_env("RGPOT_LAMMPS_ENGINE", second);
  REQUIRE(key(first) == first_key);
}
