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

std::string force_error(const rgpot::MOPACPot &pot) {
  const AtomMatrix pos{{0.0, 0.0, 0.0}, {1.15, 0.0, 0.0}, {-1.06, 0.0, 0.0}};
  const std::vector<int> z{1, 6, 7};
  const std::array<double, 9> box = {100.0, 0.0, 0.0, 0.0, 100.0,
                                     0.0,   0.0, 0.0, 100.0};
  std::array<double, 9> forces{};
  const rgpot::ForceInput in{z.size(), pos.data(), z.data(), box.data()};
  rgpot::ForceOut out{};
  out.F = forces.data();
  try {
    pot.forceImpl(in, &out);
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
                  "incompatible mopacc C ABI: expected 2, found 1");
}

TEST_CASE("MOPACPot refuses a libmopacc without mopacc_c_abi_version",
          "[mopac][abi][gate]") {
  require_refused("RGPOT_MOPACC_GATE_NO_C_ABI",
                  "incompatible mopacc C ABI: expected 2, found no "
                  "mopacc_c_abi_version");
}

TEST_CASE("MOPACPot refuses C ABI 1 even with ABI 2 layouts and sizes",
          "[mopac][abi][gate]") {
  require_refused("RGPOT_MOPACC_GATE_C_ABI1",
                  "incompatible mopacc C ABI: expected 2, found 1");
}

TEST_CASE("MOPACPot refuses C ABI 3 even with ABI 2 sizes",
          "[mopac][abi][gate]") {
  require_refused("RGPOT_MOPACC_GATE_ABI3",
                  "incompatible mopacc C ABI: expected 2, found 3");
}

TEST_CASE("MOPACPot refuses ABI 2 without mopacc_abi_version",
          "[mopac][abi][gate]") {
  require_refused("RGPOT_MOPACC_GATE_ABI2_NO_LIB_ABI",
                  "incompatible mopacc library ABI: expected 2, found no "
                  "mopacc_abi_version");
}

TEST_CASE("MOPACPot refuses ABI 2 whose library ABI is 1",
          "[mopac][abi][gate]") {
  require_refused("RGPOT_MOPACC_GATE_ABI2_LIB_ABI1",
                  "incompatible mopacc library ABI: expected 2, found 1");
}

TEST_CASE("MOPACPot refuses C ABI 2 whose library ABI is 3",
          "[mopac][abi][gate]") {
  require_refused("RGPOT_MOPACC_GATE_ABI2_LIB_ABI3",
                  "incompatible mopacc library ABI: expected 2, found 3");
}

TEST_CASE("MOPACPot refuses ABI 2 reporting the ABI 1 MopacCResult size",
          "[mopac][abi][gate]") {
  require_refused("RGPOT_MOPACC_GATE_ABI2_OLD_LAYOUT",
                  "incompatible mopacc MopacCResult size: expected " +
                      std::to_string(sizeof(MopacCResult)) + ", found " +
                      std::to_string(sizeof(Abi1MopacCResult)));
}

TEST_CASE("MOPACPot refuses ABI 2 without mopacc_result_size",
          "[mopac][abi][gate]") {
  require_refused("RGPOT_MOPACC_GATE_ABI2_NO_RESULT_SIZE",
                  "incompatible mopacc MopacCResult size: expected " +
                      std::to_string(sizeof(MopacCResult)) +
                      ", found no mopacc_result_size");
}

TEST_CASE("MOPACPot refuses ABI 2 with a different MopacCResult size",
          "[mopac][abi][gate]") {
  require_refused("RGPOT_MOPACC_GATE_ABI2_BAD_RESULT_SIZE",
                  "incompatible mopacc MopacCResult size: expected " +
                      std::to_string(sizeof(MopacCResult)) + ", found " +
                      std::to_string(sizeof(MopacCResult) + 8));
}

TEST_CASE("MOPACPot refuses ABI 2 without mopacc_params_size",
          "[mopac][abi][gate]") {
  require_refused("RGPOT_MOPACC_GATE_ABI2_NO_PARAMS_SIZE",
                  "incompatible mopacc MopacCParams size: expected " +
                      std::to_string(sizeof(MopacCParams)) +
                      ", found no mopacc_params_size");
}

TEST_CASE("MOPACPot refuses ABI 2 with a different MopacCParams size",
          "[mopac][abi][gate]") {
  require_refused("RGPOT_MOPACC_GATE_ABI2_BAD_PARAMS_SIZE",
                  "incompatible mopacc MopacCParams size: expected " +
                      std::to_string(sizeof(MopacCParams)) + ", found " +
                      std::to_string(sizeof(MopacCParams) + 8));
}

TEST_CASE("MOPACPot stays unloaded when mopacc_set_params fails",
          "[mopac][abi][gate]") {
  const std::string path =
      engine_from_env("RGPOT_MOPACC_GATE_ABI2_SET_PARAMS_FAILS");
  GateLib gate(path);
  rgpot::MOPACPot::Config cfg;
  cfg.engine_path = path;
  rgpot::MOPACPot pot(cfg);
  REQUIRE_FALSE(pot.available());
  // mopacc_available and mopacc_set_params, nothing that computes.
  const int calls_after_load = gate.calls();
  REQUIRE(calls_after_load == 2);
  const std::string what = force_error(pot);
  INFO(what);
  REQUIRE(what.find("mopacc_set_params rejected MopacCParams: status 7") !=
          std::string::npos);
  REQUIRE(gate.calls() == calls_after_load);
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
