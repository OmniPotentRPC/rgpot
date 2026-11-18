// MIT License
// Copyright 2023--present rgpot developers

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <capnp/message.h>

#include <array>
#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>

#include "rgpot/NWChemPot/DynLib.hpp"
#include "rgpot/NWChemPot/NWChemPot.hpp"
#include "rgpot/rpc/Potentials.capnp.h"
#include "rgpot/types/AtomMatrix.hpp"
#include "rgpot/units.hpp"

using Catch::Matchers::WithinAbs;

namespace {

void set_env_var(const char *name, const std::string &value) {
#ifdef _WIN32
  _putenv_s(name, value.c_str());
#else
  setenv(name, value.c_str(), 1);
#endif
}

void clear_env_var(const char *name) {
#ifdef _WIN32
  _putenv_s(name, "");
#else
  unsetenv(name);
#endif
}

class RestoreEnv {
public:
  explicit RestoreEnv(const char *name) : name_(name) {
    if (const char *value = std::getenv(name)) {
      had_ = true;
      value_ = value;
    }
  }

  RestoreEnv(const RestoreEnv &) = delete;
  RestoreEnv &operator=(const RestoreEnv &) = delete;

  void clear() const { clear_env_var(name_); }

  ~RestoreEnv() {
    if (had_)
      set_env_var(name_, value_);
    else
      clear_env_var(name_);
  }

private:
  const char *name_;
  bool had_ = false;
  std::string value_;
};

std::string configured_engine_path() {
  const char *names[] = {"NWCHEMC_LIBRARY", "RGPOT_NWCHEMC_ENGINE",
                         "RGPOT_NWCHEM_ENGINE"};
  for (const char *name : names) {
    if (const char *value = std::getenv(name); value && *value)
      return value;
  }
  return {};
}

class RefuseGuard {
public:
  using Fn = void (*)(int);

  explicit RefuseGuard(Fn fn) : fn_(fn) {}
  RefuseGuard(const RefuseGuard &) = delete;
  RefuseGuard &operator=(const RefuseGuard &) = delete;
  ~RefuseGuard() {
    if (fn_)
      fn_(0);
  }

  void set(int refuse) const { fn_(refuse); }

private:
  Fn fn_;
};

} // namespace

TEST_CASE("NWChemPot passes serialized NWChemParams to nwchemc engine",
          "[nwchem][abi]") {
  if (!rgpot::NWChemPot::probeAvailable()) {
    SKIP("libnwchemc not available");
  }

  ::capnp::MallocMessageBuilder msg;
  auto p = msg.initRoot<::NWChemParams>();
  p.setBasis("6-31g");
  p.setTheory("dft");
  p.setScfType("blyp");
  p.setCharge(-1);
  p.setMultiplicity(2);

  rgpot::NWChemPot pot(p.asReader());
  REQUIRE(pot.available());

  rgpot::types::AtomMatrix positions(1, 3);
  positions(0, 0) = 0.0;
  positions(0, 1) = 0.0;
  positions(0, 2) = 0.0;
  std::vector<int> atmtypes{8};
  std::array<std::array<double, 3>, 3> box = {
      {{20.0, 0.0, 0.0}, {0.0, 20.0, 0.0}, {0.0, 0.0, 20.0}}};

  auto [energy, forces, variance] = pot(positions, atmtypes, box);
  (void)variance;

  REQUIRE(std::isfinite(energy));
  REQUIRE(energy != 0.0);
  for (int j = 0; j < 3; ++j)
    REQUIRE(std::isfinite(forces(0, j)));

  const double fake_e = 0.25 * rgpot::units::HARTREE_TO_EV;
  // In-tree fake returns fixed Ha/grad; real NWChem does not.
  if (std::abs(energy - fake_e) < 1e-9) {
    REQUIRE_THAT(energy, WithinAbs(fake_e, 1e-12));
    REQUIRE_THAT(forces(0, 0),
                 WithinAbs(0.001 * rgpot::units::NEG_GRAD_TO_FORCE, 1e-12));
    REQUIRE_THAT(forces(0, 1),
                 WithinAbs(0.002 * rgpot::units::NEG_GRAD_TO_FORCE, 1e-12));
    REQUIRE_THAT(forces(0, 2),
                 WithinAbs(0.003 * rgpot::units::NEG_GRAD_TO_FORCE, 1e-12));
  } else {
    // Real embed: O- / BLYP / 6-31G is bound (negative eV scale).
    REQUIRE(energy < 0.0);
  }
}

TEST_CASE("NWChemPot preserves typed DFT convergence parameters",
          "[nwchem][abi]") {
  if (!rgpot::NWChemPot::probeAvailable()) {
    SKIP("libnwchemc not available");
  }

  ::capnp::MallocMessageBuilder input_message;
  auto input = input_message.initRoot<::NWChemParams>();
  input.setTheory("dft");
  auto stanzas = input.initInputStanzas(1);
  stanzas[0].setKind(::NWChemInputStanza::Kind::DFT);
  auto dft = stanzas[0].initDft();
  dft.setXc("hyb_gga_xc_wb97x_v");
  dft.setEnergyConv(1.0e-6);

  rgpot::NWChemPot pot(input.asReader());
  REQUIRE(pot.available());

  ::capnp::MallocMessageBuilder output_message;
  auto output = output_message.initRoot<::NWChemParams>();
  pot.getParams(output);
  const auto restored_stanzas = output.asReader().getInputStanzas();

  REQUIRE(restored_stanzas.size() == 1);
  REQUIRE(restored_stanzas[0].getKind() == ::NWChemInputStanza::Kind::DFT);
  REQUIRE(restored_stanzas[0].getDft().getXc() == "hyb_gga_xc_wb97x_v");
  REQUIRE_THAT(restored_stanzas[0].getDft().getEnergyConv(),
               WithinAbs(1.0e-6, 1.0e-15));
}

TEST_CASE("NWChemPot reports a loaded engine configuration refusal",
          "[nwchem][abi]") {
  const char *path = std::getenv("RGPOT_NWCHEM_ENGINE");
  REQUIRE(path != nullptr);

  rgpot::NWChemPot pot;
  REQUIRE(pot.available());

  rgpot::DynLib engine;
  engine.open(path);
  using RejectParamsFn = void (*)(int);
  auto reject_params = engine.sym<RejectParamsFn>("nwchemc_fake_reject_params");
  reject_params(1);

  ::capnp::MallocMessageBuilder message;
  auto config = message.initRoot<::PotentialConfig>();
  auto params = config.initNwchem();
  params.setBasis("6-31g");
  params.setTheory("dft");
  params.setScfType("rhf");
  params.setCharge(0);
  params.setMultiplicity(1);

  std::string result;
  REQUIRE_FALSE(pot.setPotentialConfig(config.asReader(), &result));
  reject_params(0);

  REQUIRE(pot.available());
  REQUIRE(result == "rgpot params nwchem arm rejected by embed: "
                    "basis=6-31g theory=dft scfType=rhf charge=0 mult=1");
}

// NWChemPot has no session handle: available() checks only that libnwchemc
// loaded with nwchemc_energy_gradient and nwchemc_set_params, and setParams
// keeps both when nwchemc_set_params returns nonzero. A missing library
// reports engine not loaded while a parameter refusal reports rejected by
// embed.
TEST_CASE("NWChemPot setPotentialConfig distinguishes missing engine from "
          "refused params",
          "[nwchem][abi]") {
  const std::string engine_path = configured_engine_path();
  if (engine_path.empty()) {
    SKIP("explicit nwchemc engine path not set in environment");
  }

  rgpot::DynLib lib(engine_path);
  using VersionFn = const char *(*)(void);
  using RefuseFn = void (*)(int);
  using CountFn = int (*)(void);
  auto version = lib.sym_optional<VersionFn>("nwchemc_version");
  auto set_refuse =
      lib.sym_optional<RefuseFn>("nwchemc_fake_set_refuse_params");
  auto set_params_count =
      lib.sym_optional<CountFn>("nwchemc_fake_set_params_count");
  const bool is_fake =
      version && std::string(version()).rfind("nwchemc-fake/", 0) == 0;
  if (is_fake) {
    REQUIRE(set_refuse != nullptr);
    REQUIRE(set_params_count != nullptr);
  } else if (!set_refuse || !set_params_count) {
    SKIP("real libnwchemc does not expose fake refusal controls");
  }
  RefuseGuard refusal(set_refuse);

  RestoreEnv library_env("NWCHEMC_LIBRARY");
  RestoreEnv alt_env("RGPOT_NWCHEMC_ENGINE");
  RestoreEnv engine_env("RGPOT_NWCHEM_ENGINE");
  library_env.clear();
  alt_env.clear();
  engine_env.clear();

  rgpot::NWChemPot pot;
  if (pot.available()) {
    SKIP("default libnwchemc on loader search path masks missing engine path");
  }

  ::capnp::MallocMessageBuilder cfg_msg;
  auto cfg = cfg_msg.initRoot<::PotentialConfig>();
  auto nw = cfg.initNwchem();
  nw.setBasis("6-31g");
  nw.setTheory("dft");
  nw.setScfType("blyp");
  nw.setCharge(-1);
  nw.setMultiplicity(2);
  nw.setEnginePath("/nonexistent/libnwchemc.so");

  const int before = set_params_count();
  std::string message;
  REQUIRE_FALSE(pot.setPotentialConfig(cfg.asReader(), &message));
  REQUIRE_FALSE(pot.available());
  REQUIRE(set_params_count() == before);
  REQUIRE(message == "rgpot params nwchem arm failed (engine not loaded): "
                     "basis=6-31g theory=dft scfType=blyp charge=-1 mult=2 "
                     "enginePath=/nonexistent/libnwchemc.so");

  refusal.set(1);
  nw.setEnginePath(engine_path);
  REQUIRE_FALSE(pot.setPotentialConfig(cfg.asReader(), &message));
  REQUIRE(pot.available());
  REQUIRE(set_params_count() == before + 1);
  const std::string summary =
      "basis=6-31g theory=dft scfType=blyp charge=-1 mult=2 enginePath=" +
      engine_path;
  REQUIRE(message == "rgpot params nwchem arm rejected by embed: " + summary);

  REQUIRE_FALSE(pot.setPotentialConfig(cfg.asReader(), &message));
  REQUIRE(pot.available());
  REQUIRE(set_params_count() == before + 2);
  REQUIRE(message == "rgpot params nwchem arm rejected by embed: " + summary);

  refusal.set(0);
  REQUIRE(pot.setPotentialConfig(cfg.asReader(), &message));
  REQUIRE(pot.available());
  REQUIRE(set_params_count() == before + 3);
  REQUIRE(message == "rgpot params applied (nwchem arm): " + summary);

  ::capnp::MallocMessageBuilder none_msg;
  auto none_cfg = none_msg.initRoot<::PotentialConfig>();
  none_cfg.setNone();
  REQUIRE(pot.setPotentialConfig(none_cfg.asReader(), &message));
  REQUIRE(message == "no-op (rgpot params: none)");

  ::capnp::MallocMessageBuilder cpmd_msg;
  auto cpmd_cfg = cpmd_msg.initRoot<::PotentialConfig>();
  cpmd_cfg.initCpmd();
  REQUIRE_FALSE(pot.setPotentialConfig(cpmd_cfg.asReader(), &message));
  REQUIRE(message == "rgpot params arm cpmd not handled by NWChemPot "
                     "(use matching backend pot)");
}
