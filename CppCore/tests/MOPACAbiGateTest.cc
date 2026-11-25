// MIT License
// Copyright 2023--present rgpot developers

#include "rgpot/MOPACPot/MOPACPot.hpp"
#include "rgpot/MOPACPot/mopac_c_abi.h"
#include "rgpot/abi/DynLib.hpp"
#include "rgpot/types/AtomMatrix.hpp"
#include "rgpot/units.hpp"

#include <array>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using Catch::Approx;
using rgpot::types::AtomMatrix;

namespace {

struct Abi1MopacCResult {
  int ok;
  double energy_h;
  char message[512];
};

using GateCallsFn = int (*)(void);

std::string engine_from_env(const char *var) {
  const char *path = std::getenv(var);
  REQUIRE(path != nullptr);
  REQUIRE(*path != '\0');
  return path;
}

// Holds the library open across the MOPACPot lifetime so the call counter
// survives MOPACPot closing its own handle.
struct GateLib {
  explicit GateLib(const std::string &path) : lib(path.c_str()) {
    REQUIRE(lib.valid());
    calls = lib.sym_optional<GateCallsFn>("mopacc_gate_calls");
    REQUIRE(calls != nullptr);
  }
  rgpot::DynLib lib;
  GateCallsFn calls = nullptr;
};

std::string force_error(rgpot::MOPACPot &pot) {
  const AtomMatrix pos{{0.0, 0.0, 0.0}, {1.15, 0.0, 0.0}, {-1.06, 0.0, 0.0}};
  const std::vector<int> z{1, 6, 7};
  const std::array<std::array<double, 3>, 3> box = {
      {{100.0, 0.0, 0.0}, {0.0, 100.0, 0.0}, {0.0, 0.0, 100.0}}};
  try {
    (void)pot(pos, z, box);
  } catch (const std::runtime_error &e) {
    return e.what();
  }
  FAIL("MOPACPot computed against an incompatible libmopacc");
  return {};
}

void require_refused(const char *env_var, const std::string &expected) {
  const std::string path = engine_from_env(env_var);
  GateLib gate(path);
  {
    rgpot::MOPACPot::Config cfg;
    cfg.engine_path = path;
    rgpot::MOPACPot pot(cfg);
    REQUIRE_FALSE(pot.available());
    const std::string what = force_error(pot);
    INFO(what);
    REQUIRE(what.find(expected) != std::string::npos);
  }
  REQUIRE(gate.calls() == 0);
}

} // namespace

TEST_CASE("MOPACPot refuses an ABI 1 libmopacc before any call",
          "[mopac][abi][gate]") {
  require_refused("RGPOT_MOPACC_GATE_ABI1",
                  "incompatible mopacc C ABI: expected >= 2, found 1");
}

TEST_CASE("MOPACPot refuses ABI 2 with a different MopacCResult size",
          "[mopac][abi][gate]") {
  require_refused("RGPOT_MOPACC_GATE_ABI2_WRONG_SIZE",
                  "incompatible mopacc MopacCResult size: expected " +
                      std::to_string(sizeof(MopacCResult)) + ", found " +
                      std::to_string(sizeof(Abi1MopacCResult)));
}

TEST_CASE("MOPACPot refuses ABI 2 without mopacc_result_size",
          "[mopac][abi][gate]") {
  require_refused("RGPOT_MOPACC_GATE_ABI2_NO_SIZE",
                  "incompatible mopacc MopacCResult size: expected " +
                      std::to_string(sizeof(MopacCResult)) +
                      ", found no mopacc_result_size");
}

TEST_CASE("MOPACPot loads and computes against a matching ABI 2 libmopacc",
          "[mopac][abi][gate]") {
  const std::string path = engine_from_env("RGPOT_MOPACC_GATE_ABI2_OK");
  GateLib gate(path);
  rgpot::MOPACPot::Config cfg;
  cfg.engine_path = path;
  rgpot::MOPACPot pot(cfg);
  REQUIRE(pot.available());
  const int calls_after_load = gate.calls();

  const AtomMatrix pos{{0.0, 0.0, 0.0}, {1.15, 0.0, 0.0}, {-1.06, 0.0, 0.0}};
  const std::array<std::array<double, 3>, 3> box = {
      {{100.0, 0.0, 0.0}, {0.0, 100.0, 0.0}, {0.0, 0.0, 100.0}}};
  auto [energy, forces, variance] = pot(pos, {1, 6, 7}, box);
  (void)variance;
  REQUIRE(gate.calls() == calls_after_load + 1);
  REQUIRE(energy == Approx(-2.0 * rgpot::units::HARTREE_TO_EV));
  REQUIRE(forces(0, 0) == Approx(0.003 * rgpot::units::NEG_GRAD_TO_FORCE));
  REQUIRE(forces(2, 2) == Approx(0.027 * rgpot::units::NEG_GRAD_TO_FORCE));
}
