// MIT License
// Copyright 2023--present rgpot developers
//
// ProfileLoader resolves liblammpc and opens a session from
// PotentialConfig.lammps. Without the library the case skips. With it, one
// lj/cut dimer is checked against the unshifted analytic pair.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <capnp/message.h>
#include <capnp/serialize.h>
#include <kj/array.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

#include "rgpot/abi/ProfileLoader.hpp"
#include "rgpot/rpc/Potentials.capnp.h"

using Catch::Matchers::WithinAbs;

namespace {

struct Flat {
  kj::Array<::capnp::word> words;

  const void *data() const { return words.begin(); }
  size_t size() const { return words.asBytes().size(); }
};

Flat flatten(::capnp::MallocMessageBuilder &msg) {
  return Flat{::capnp::messageToFlatArray(msg)};
}

// The same lj/cut dimer lammpc checks in test_embed_lj_e2e.c.
constexpr double kEps = 0.2;
constexpr double kSigma = 2.5;
constexpr double kSeparation = 3.0;

Flat lammps_config() {
  ::capnp::MallocMessageBuilder msg;
  auto config = msg.initRoot<::PotentialConfig>();
  auto lammps = config.initLammps();
  lammps.setUnitsStyle("metal");
  lammps.setAtomStyle("atomic");
  lammps.setPairStyle("lj/cut 10.0");
  auto coeffs = lammps.initPairCoeffs(1);
  coeffs.set(0, "1 1 0.2 2.5");
  auto types = lammps.initTypeToAtomicNumber(1);
  types.set(0, 29);
  lammps.setNewtonPair(true);
  lammps.setBoundary("f f f");
  return flatten(msg);
}

Flat force_input(double separation) {
  ::capnp::MallocMessageBuilder msg;
  auto input = msg.initRoot<::ForceInput>();
  auto pos = input.initPos(6);
  pos.set(0, 0.0);
  pos.set(1, 0.0);
  pos.set(2, 0.0);
  pos.set(3, separation);
  pos.set(4, 0.0);
  pos.set(5, 0.0);
  auto atmnrs = input.initAtmnrs(2);
  atmnrs.set(0, 29);
  atmnrs.set(1, 29);
  input.setLengthUnit("angstrom");
  input.setEnergyUnit("eV");
  return flatten(msg);
}

double lj_energy(double r) {
  const double sr6 = std::pow(kSigma / r, 6.0);
  return 4.0 * kEps * (sr6 * sr6 - sr6);
}

// LAMMPS lj/cut: F = 24*eps*(2*(sigma/r)^12 - (sigma/r)^6)/r along +x on atom 1.
double lj_force_on_second(double r) {
  const double sr6 = std::pow(kSigma / r, 6.0);
  return 24.0 * kEps * (2.0 * sr6 * sr6 - sr6) / r;
}

std::string profile_message(const rgpot::abi::ProfileResult &result) {
  const size_t n = strnlen(result.message, sizeof(result.message));
  return std::string(result.message, n);
}

} // namespace

TEST_CASE("ProfileLoader names the lammpc shared library", "[abi][lammpc]") {
  const auto names = rgpot::abi::ProfileLoader::candidates("lammpc", "");
  const auto has = [&](const char *name) {
    return std::find(names.begin(), names.end(), name) != names.end();
  };
  REQUIRE(has("liblammpc.so"));
  REQUIRE(has("./liblammpc.so"));
  REQUIRE(has("liblammpc.dylib"));
  REQUIRE(has("./liblammpc.dylib"));
  REQUIRE(has("lammpc.dll"));
}

TEST_CASE("ProfileLoader opens a lammpc session from PotentialConfig.lammps",
          "[abi][lammpc]") {
  rgpot::abi::ProfileLoader loader;
  try {
    loader.load("lammpc");
  } catch (const std::runtime_error &ex) {
    const std::string msg = ex.what();
    if (msg.find("not loaded") != std::string::npos) {
      SKIP(msg);
    }
    throw;
  }
  if (loader.available == nullptr || loader.available() != 1) {
    SKIP("lammpc is not installed");
  }
  REQUIRE(loader.loaded());
  REQUIRE(loader.prefix() == "lammpc");

  const Flat config = lammps_config();
  void *session =
      loader.session_create_from_config(config.data(), config.size());
  struct Guard {
    rgpot::abi::ProfileLoader *loader;
    void *session;
    ~Guard() {
      if (session != nullptr && loader->session_destroy != nullptr) {
        loader->session_destroy(session);
      }
    }
  } guard{&loader, session};
  REQUIRE(session != nullptr);

  const Flat step = force_input(kSeparation);
  const size_t need = loader.potential_result_size_for_force_input(step.data(),
                                                                   step.size());
  REQUIRE(need > 0);
  REQUIRE(need % sizeof(::capnp::word) == 0);
  std::vector<::capnp::word> out(need / sizeof(::capnp::word));
  size_t wrote = 0;
  const auto result = loader.session_calculate_result(
      session, step.data(), step.size(), out.data(), need, &wrote);
  INFO(profile_message(result));
  REQUIRE(result.ok == 1);
  REQUIRE(wrote > 0);
  REQUIRE(wrote % sizeof(::capnp::word) == 0);

  auto words = kj::arrayPtr(out.data(), wrote / sizeof(::capnp::word));
  ::capnp::FlatArrayMessageReader reader(words);
  const auto decoded = reader.getRoot<::PotentialResult>();
  const auto forces = decoded.getForces();
  REQUIRE(forces.size() == 6);
  const double expected_f = lj_force_on_second(kSeparation);
  REQUIRE_THAT(decoded.getEnergy(), WithinAbs(lj_energy(kSeparation), 1e-9));
  REQUIRE_THAT(forces[0], WithinAbs(-expected_f, 1e-9));
  REQUIRE_THAT(forces[1], WithinAbs(0.0, 1e-12));
  REQUIRE_THAT(forces[2], WithinAbs(0.0, 1e-12));
  REQUIRE_THAT(forces[3], WithinAbs(expected_f, 1e-9));
  REQUIRE_THAT(forces[4], WithinAbs(0.0, 1e-12));
  REQUIRE_THAT(forces[5], WithinAbs(0.0, 1e-12));
}
