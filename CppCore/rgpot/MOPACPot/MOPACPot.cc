// MIT License
// Copyright 2023--present rgpot developers

#include "rgpot/MOPACPot/MOPACPot.hpp"

#include "rgpot/MOPACPot/mopac_c_abi.h"
#include "rgpot/abi/DynLib.hpp"
#include "rgpot/abi/EngineDlopen.hpp"
#include "rgpot/units.hpp"

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace rgpot {

using units::HARTREE_TO_EV;
using units::NEG_GRAD_TO_FORCE;

namespace {

constexpr std::size_t kExpectedMopacCParamsSize =
    (alignof(double) == 8) ? 48u : 36u;
constexpr std::size_t kExpectedMopacCResultSize =
    (alignof(double) == 8) ? 600u : 596u;
constexpr std::size_t kExpectedMopacCResultDipoleOffset =
    (alignof(double) == 8) ? 16u : 12u;
constexpr std::size_t kExpectedMopacCResultStressOffset =
    (alignof(double) == 8) ? 40u : 36u;
constexpr std::size_t kExpectedMopacCResultMessageOffset =
    (alignof(double) == 8) ? 88u : 84u;

static_assert(sizeof(MopacCParams) == kExpectedMopacCParamsSize,
              "MopacCParams size drifted from vendored mopacc ABI");
static_assert(sizeof(MopacCResult) == kExpectedMopacCResultSize,
              "MopacCResult size drifted from vendored mopacc ABI");
static_assert(offsetof(MopacCResult, dipole_debye) ==
                  kExpectedMopacCResultDipoleOffset,
              "MopacCResult::dipole_debye offset drifted from vendored ABI");
static_assert(offsetof(MopacCResult, stress_gpa) ==
                  kExpectedMopacCResultStressOffset,
              "MopacCResult::stress_gpa offset drifted from vendored ABI");
static_assert(offsetof(MopacCResult, message) ==
                  kExpectedMopacCResultMessageOffset,
              "MopacCResult::message offset drifted from vendored mopacc ABI");

static_assert(RGPOT_MOPACC_C_ABI_VERSION == 2 && MOPACC_ABI_VERSION == 2,
              "the layout constants above describe mopacc C ABI 2 only");

std::string expected_found(const char *what, std::size_t expected,
                           std::size_t found) {
  return std::string(what) + ": expected " + std::to_string(expected) +
         ", found " + std::to_string(found);
}

using EnergyGradientFn = MopacCResult (*)(int, const double *, const int *,
                                          const void *, size_t, double *);
using SetParamsFn = int (*)(const void *, size_t);
using AbiVersionFn = int (*)(void);
using SizeQueryFn = size_t (*)(void);
using AvailableFn = int (*)(void);

struct EngineBundle {
  DynLib engine_lib;
  EnergyGradientFn energy_gradient = nullptr;
  SetParamsFn set_params = nullptr;
  AbiVersionFn abi_version = nullptr;
  AbiVersionFn lib_abi_version = nullptr;
  SizeQueryFn result_size = nullptr;
  SizeQueryFn params_size = nullptr;
  AvailableFn available = nullptr;
  std::string load_error;
  bool loaded = false;
};

bool try_load_engine(EngineBundle &b, const std::string &engine_path) {
  b = EngineBundle{};
  const auto cands = abi::engine_candidates(
      "mopacc", engine_path, {"MOPACC_LIBRARY", "RGPOT_MOPACC_ENGINE"});
  std::string loaded_path;
  std::string eng_err;
  if (!abi::open_first_candidate(b.engine_lib, cands, &loaded_path, &eng_err)) {
    b.load_error = "libmopacc not loaded: " + eng_err;
    return false;
  }

  b.energy_gradient =
      b.engine_lib.sym_optional<EnergyGradientFn>("mopacc_energy_gradient");
  b.set_params = b.engine_lib.sym_optional<SetParamsFn>("mopacc_set_params");
  b.abi_version =
      b.engine_lib.sym_optional<AbiVersionFn>("mopacc_c_abi_version");
  b.lib_abi_version =
      b.engine_lib.sym_optional<AbiVersionFn>("mopacc_abi_version");
  b.result_size = b.engine_lib.sym_optional<SizeQueryFn>("mopacc_result_size");
  b.params_size = b.engine_lib.sym_optional<SizeQueryFn>("mopacc_params_size");
  b.available = b.engine_lib.sym_optional<AvailableFn>("mopacc_available");
  if (!b.energy_gradient) {
    b.load_error = "engine missing mopacc_energy_gradient";
    return false;
  }
  // ABI 1 libraries export the same compute symbols with a smaller
  // MopacCResult returned by value, and a later ABI may change the layout
  // without changing the sizes, so nothing that returns one may be called
  // until both versions equal this build's and the size queries agree.
  if (!b.abi_version) {
    b.load_error = "incompatible mopacc C ABI: expected " +
                   std::to_string(RGPOT_MOPACC_C_ABI_VERSION) +
                   ", found no mopacc_c_abi_version";
    return false;
  }
  if (const int found = b.abi_version(); found != RGPOT_MOPACC_C_ABI_VERSION) {
    b.load_error = "incompatible mopacc C ABI: expected " +
                   std::to_string(RGPOT_MOPACC_C_ABI_VERSION) + ", found " +
                   std::to_string(found);
    return false;
  }
  if (!b.lib_abi_version) {
    b.load_error = "incompatible mopacc library ABI: expected " +
                   std::to_string(MOPACC_ABI_VERSION) +
                   ", found no mopacc_abi_version";
    return false;
  }
  if (const int found = b.lib_abi_version(); found != MOPACC_ABI_VERSION) {
    b.load_error = "incompatible mopacc library ABI: expected " +
                   std::to_string(MOPACC_ABI_VERSION) + ", found " +
                   std::to_string(found);
    return false;
  }
  if (!b.result_size) {
    b.load_error = "incompatible mopacc MopacCResult size: expected " +
                   std::to_string(sizeof(MopacCResult)) +
                   ", found no mopacc_result_size";
    return false;
  }
  if (const std::size_t found = b.result_size();
      found != sizeof(MopacCResult)) {
    b.load_error = expected_found("incompatible mopacc MopacCResult size",
                                  sizeof(MopacCResult), found);
    return false;
  }
  if (!b.params_size) {
    b.load_error = "incompatible mopacc MopacCParams size: expected " +
                   std::to_string(sizeof(MopacCParams)) +
                   ", found no mopacc_params_size";
    return false;
  }
  if (const std::size_t found = b.params_size();
      found != sizeof(MopacCParams)) {
    b.load_error = expected_found("incompatible mopacc MopacCParams size",
                                  sizeof(MopacCParams), found);
    return false;
  }
  if (b.available && !b.available()) {
    b.load_error = "mopacc stub (libmopac not linked)";
    return false;
  }
  b.loaded = true;
  return true;
}

std::mutex g_probe_mu;
bool g_probe_done = false;
bool g_probe_ok = false;
bool g_abi_probe_done = false;
bool g_abi_probe_ok = false;

EngineBundle &probe_engine_bundle() {
  static EngineBundle retained;
  return retained;
}

} // namespace

struct MOPACPot::Impl {
  EngineBundle bundle;
  MopacCParams params{};
  std::string engine_path;
  mutable std::vector<double> grad_scratch;
};

MOPACPot::MOPACPot() : MOPACPot(Config{}) {}

MOPACPot::MOPACPot(const Config &cfg)
    : Potential(PotType::MOPAC), impl_(new Impl) {
  impl_->params.charge = cfg.charge;
  impl_->params.spin = cfg.spin;
  impl_->params.model = cfg.model;
  impl_->params.tolerance = cfg.tolerance;
  impl_->params.max_time = cfg.max_time;
  impl_->params.epsilon = 1.0;
  impl_->params.solver = MOPACC_SOLVER_MOPAC;
  impl_->engine_path = cfg.engine_path;
  EngineBundle &b = impl_->bundle;
  if (try_load_engine(b, impl_->engine_path) && b.set_params) {
    if (const int rc = b.set_params(&impl_->params, sizeof(impl_->params));
        rc != 0) {
      b.loaded = false;
      b.load_error = "mopacc_set_params rejected MopacCParams: status " +
                     std::to_string(rc);
    }
  }
}

MOPACPot::~MOPACPot() { delete impl_; }

bool MOPACPot::available() const {
  return impl_ && impl_->bundle.loaded && impl_->bundle.energy_gradient;
}

bool MOPACPot::probeAvailable() {
  return abi::probe_once(g_probe_mu, g_probe_done, g_probe_ok, [] {
    EngineBundle &tmp = probe_engine_bundle();
    return tmp.loaded || try_load_engine(tmp, "");
  });
}

bool MOPACPot::abiAvailable() {
  return abi::probe_once(g_probe_mu, g_abi_probe_done, g_abi_probe_ok, [] {
    EngineBundle &tmp = probe_engine_bundle();
    if (!tmp.loaded && !try_load_engine(tmp, "")) {
      return false;
    }
    return tmp.available != nullptr && tmp.available() != 0;
  });
}

void MOPACPot::forceImpl(const ForceInput &in, ForceOut *out) const {
  if (!available()) {
    throw std::runtime_error(std::string("MOPAC engine (libmopacc) not loaded: ") +
                             (impl_ ? impl_->bundle.load_error : "no impl"));
  }
  const int n = static_cast<int>(in.nAtoms);
  if (n <= 0)
    throw std::runtime_error("MOPACPot: nAtoms must be positive");
  if (!in.pos || !in.atmnrs || !out || !out->F)
    throw std::runtime_error("MOPACPot: null positions/atmnrs/forces buffer");

  std::vector<double> &grad = impl_->grad_scratch;
  grad.assign(static_cast<size_t>(n) * 3u, 0.0);
  MopacCResult res = impl_->bundle.energy_gradient(
      n, in.pos, in.atmnrs, &impl_->params, sizeof(impl_->params), grad.data());
  if (!res.ok) {
    const char *msg_end =
        std::find(res.message, res.message + sizeof(res.message), '\0');
    throw std::runtime_error(
        std::string("mopacc AM1 failed: ") +
        std::string(res.message,
                    static_cast<std::size_t>(msg_end - res.message)));
  }
  out->energy = res.energy_h * HARTREE_TO_EV;
  out->variance = 0.0;
  for (int i = 0; i < n * 3; ++i)
    out->F[static_cast<size_t>(i)] =
        grad[static_cast<size_t>(i)] * NEG_GRAD_TO_FORCE;
}

} // namespace rgpot
