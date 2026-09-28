// MIT License
// Copyright 2023--present rgpot developers

#include "rgpot/CPMDPot/CPMDPot.hpp"

#include <capnp/message.h>
#include <capnp/serialize.h>
#include <kj/array.h>

#include "rgpot/CPMDPot/cpmd_c_abi.h"
#include "rgpot/NWChemPot/DynLib.hpp"
#include "rgpot/units.hpp"

#include <cstdlib>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace rgpot {

using units::HARTREE_TO_EV;
using units::NEG_GRAD_TO_FORCE;

namespace {

using EnergyGradientFn = CPMDCResult (*)(int, const double *, const int *,
                                         const void *, size_t, double *);
using SetParamsFn = int (*)(const void *, size_t);
using SessionCreateFn = CPMDCSession *(*)(const void *, size_t);
using SessionDestroyFn = void (*)(CPMDCSession *);
using PotentialResultSizeFn = size_t (*)(const void *, size_t);
using SessionCalculateResultFn = CPMDCResult (*)(
    CPMDCSession *, const void *, size_t, void *, size_t, size_t *);
using VersionFn = const char *(*)(void);
using AvailableFn = int (*)(void);
using FeatureCountFn = size_t (*)(void);
using FeatureTableFn = const CPMDCFeatureEntry *(*)(void);
using FeatureFindFn = const CPMDCFeatureEntry *(*)(const char *);
using BindCalculatorsFn = int (*)(int);

struct ParamsView {
  const void *data = nullptr;
  size_t size = 0;
};

std::vector<std::string> engine_lib_candidates(const std::string &explicit_path) {
  std::vector<std::string> out;
  if (!explicit_path.empty())
    out.emplace_back(explicit_path);
  if (const char *e = std::getenv("CPMDC_LIBRARY"))
    out.emplace_back(e);
  if (const char *e = std::getenv("RGPOT_CPMDC_ENGINE"))
    out.emplace_back(e);
  if (const char *e = std::getenv("RGPOT_CPMD_ENGINE"))
    out.emplace_back(e);
  out.emplace_back("libcpmdc.so");
  out.emplace_back("./libcpmdc.so");
  out.emplace_back("libcpmdc.dylib");
  out.emplace_back("./libcpmdc.dylib");
  out.emplace_back("cpmdc.dll");
  return out;
}

void apply_env_hints(const std::string &cpmd_root) {
  if (!cpmd_root.empty()) {
#if !defined(_WIN32)
    setenv("CPMD_ROOT", cpmd_root.c_str(), 1);
#endif
  }
}

struct EngineBundle {
  DynLib engine_lib;
  EnergyGradientFn energy_gradient = nullptr;
  SetParamsFn set_params = nullptr;
  SessionCreateFn session_create = nullptr;
  SessionDestroyFn session_destroy = nullptr;
  PotentialResultSizeFn potential_result_size_for_force_input = nullptr;
  SessionCalculateResultFn session_calculate_result = nullptr;
  VersionFn version = nullptr;
  AvailableFn available = nullptr;
  FeatureCountFn feature_count = nullptr;
  FeatureTableFn feature_table = nullptr;
  FeatureFindFn feature_find = nullptr;
  std::string load_error;
  bool loaded = false;
};

bool try_load_engine(EngineBundle &b, const std::string &engine_path) {
  b.load_error.clear();
  b.loaded = false;
  b.energy_gradient = nullptr;
  b.set_params = nullptr;
  b.session_create = nullptr;
  b.session_destroy = nullptr;
  b.potential_result_size_for_force_input = nullptr;
  b.session_calculate_result = nullptr;
  b.version = nullptr;
  b.available = nullptr;
  b.feature_count = nullptr;
  b.feature_table = nullptr;
  b.feature_find = nullptr;

  bool eng_ok = false;
  std::string eng_err;
  for (const auto &cand : engine_lib_candidates(engine_path)) {
    try {
      b.engine_lib.open(cand);
      eng_ok = true;
      break;
    } catch (const std::exception &ex) {
      eng_err = ex.what();
    }
  }
  if (!eng_ok) {
    b.load_error = "libcpmdc not loaded: " + eng_err;
    return false;
  }

  b.energy_gradient =
      b.engine_lib.sym_optional<EnergyGradientFn>("cpmdc_energy_gradient");
  b.set_params = b.engine_lib.sym_optional<SetParamsFn>("cpmdc_set_params");
  b.session_create =
      b.engine_lib.sym_optional<SessionCreateFn>("cpmdc_session_create");
  b.session_destroy =
      b.engine_lib.sym_optional<SessionDestroyFn>("cpmdc_session_destroy");
  b.potential_result_size_for_force_input =
      b.engine_lib.sym_optional<PotentialResultSizeFn>(
          "cpmdc_potential_result_size_for_force_input");
  b.session_calculate_result =
      b.engine_lib.sym_optional<SessionCalculateResultFn>(
          "cpmdc_session_calculate_result");
  b.version = b.engine_lib.sym_optional<VersionFn>("cpmdc_version");
  b.available = b.engine_lib.sym_optional<AvailableFn>("cpmdc_available");
  b.feature_count =
      b.engine_lib.sym_optional<FeatureCountFn>("cpmdc_feature_count");
  b.feature_table =
      b.engine_lib.sym_optional<FeatureTableFn>("cpmdc_feature_table");
  b.feature_find =
      b.engine_lib.sym_optional<FeatureFindFn>("cpmdc_feature_find");

  const bool has_one_shot = b.energy_gradient && b.set_params;
  const bool has_session_result =
      b.session_create && b.session_destroy &&
      b.potential_result_size_for_force_input && b.session_calculate_result;
  const bool has_feature_discovery =
      b.feature_count && b.feature_table && b.feature_find;
  if (!has_feature_discovery) {
    b.load_error = "engine missing cpmdc feature discovery ABI";
    return false;
  }
  if (!has_one_shot && !has_session_result) {
    b.load_error =
        "engine missing cpmdc session result ABI or one-shot gradient ABI";
    return false;
  }
  b.loaded = true;
  return true;
}

bool has_session_result_abi(const EngineBundle &b) {
  return b.session_create && b.session_destroy &&
         b.potential_result_size_for_force_input && b.session_calculate_result;
}

std::vector<::capnp::word> serialize_params(const ::CPMDParams::Reader &params) {
  ::capnp::MallocMessageBuilder msg;
  msg.setRoot(params);
  auto words = ::capnp::messageToFlatArray(msg);
  std::vector<::capnp::word> out(words.size());
  std::memcpy(out.data(), words.begin(), words.size() * sizeof(::capnp::word));
  return out;
}

std::vector<::capnp::word> default_params() {
  ::capnp::MallocMessageBuilder msg;
  auto params = msg.initRoot<::CPMDParams>();
  return serialize_params(params.asReader());
}

ParamsView params_view(const std::vector<::capnp::word> &words) {
  ParamsView view;
  if (!words.empty()) {
    view.data = words.data();
    view.size = words.size() * sizeof(::capnp::word);
  }
  return view;
}

bool push_params_to_engine(EngineBundle &b,
                           const std::vector<::capnp::word> &params_words) {
  if (!b.set_params)
    return false;
  const ParamsView view = params_view(params_words);
  return b.set_params(view.data, view.size) == 0;
}

std::vector<::capnp::word> serialize_force_input(const ForceInput &in) {
  ::capnp::MallocMessageBuilder msg;
  auto force_input = msg.initRoot<::ForceInput>();
  const unsigned int coord_count = static_cast<unsigned int>(in.nAtoms * 3u);
  auto pos = force_input.initPos(coord_count);
  auto atmnrs = force_input.initAtmnrs(static_cast<unsigned int>(in.nAtoms));
  auto box = force_input.initBox(9);
  for (unsigned int i = 0; i < coord_count; ++i)
    pos.set(i, in.pos[i]);
  for (unsigned int i = 0; i < in.nAtoms; ++i)
    atmnrs.set(i, in.atmnrs[i]);
  for (unsigned int i = 0; i < 9; ++i)
    box.set(i, in.box[i]);
  force_input.setLengthUnit("angstrom");
  force_input.setEnergyUnit("eV");
  auto words = ::capnp::messageToFlatArray(msg);
  std::vector<::capnp::word> out(words.size());
  std::memcpy(out.data(), words.begin(), words.asBytes().size());
  return out;
}

std::string params_summary(const ::CPMDParams::Reader &params) {
  std::string out = "functional=" + std::string(params.getFunctional().cStr()) +
                    " cutoffRy=" + std::to_string(params.getCutOffRy()) +
                    " charge=" + std::to_string(params.getCharge()) +
                    " mult=" + std::to_string(params.getMultiplicity());
  if (params.getEnginePath().size() > 0)
    out += " enginePath=" + std::string(params.getEnginePath().cStr());
  if (params.getCpmdRoot().size() > 0)
    out += " cpmdRoot=" + std::string(params.getCpmdRoot().cStr());
  return out;
}

void copy_params_to_builder(const ::CPMDParams::Reader &params,
                            ::CPMDParams::Builder out) {
  out.setFunctional(params.getFunctional());
  out.setCutOffRy(params.getCutOffRy());
  out.setCharge(params.getCharge());
  out.setMultiplicity(params.getMultiplicity());
  out.setTask(params.getTask());
  out.setTitle(params.getTitle());
  out.setMemoryMb(params.getMemoryMb());
  out.setScratchDir(params.getScratchDir());
  out.setPermanentDir(params.getPermanentDir());
  out.setCpmdRoot(params.getCpmdRoot());
  out.setEnginePath(params.getEnginePath());
  const auto blocks = params.getInputBlocks();
  auto out_blocks = out.initInputBlocks(blocks.size());
  for (unsigned int i = 0; i < blocks.size(); ++i)
    out_blocks.set(i, blocks[i]);
  out.setInputSections(params.getInputSections());
}

::CPMDParams::Reader read_params_words(
    const std::vector<::capnp::word> &params_words,
    ::capnp::FlatArrayMessageReader &reader) {
  (void)params_words;
  return reader.getRoot<::CPMDParams>();
}

std::mutex g_probe_mu;
bool g_probe_done = false;
bool g_probe_ok = false;
bool g_abi_probe_done = false;
bool g_abi_probe_ok = false;

} // namespace

struct CPMDPot::Impl {
  EngineBundle bundle;
  std::vector<::capnp::word> params_words;
  std::string engine_path;
  std::string cpmd_root;
  CPMDCSession *session = nullptr;
  mutable std::vector<double> grad_scratch;

  void destroySession();
  bool configure();
  void forceSession(const ForceInput &in, ForceOut *out);
};

void CPMDPot::Impl::destroySession() {
  if (session && bundle.session_destroy)
    bundle.session_destroy(session);
  session = nullptr;
}

bool CPMDPot::Impl::configure() {
  destroySession();
  if (has_session_result_abi(bundle)) {
    const ParamsView view = params_view(params_words);
    session = bundle.session_create(view.data, view.size);
    return session != nullptr;
  }
  return push_params_to_engine(bundle, params_words);
}

CPMDPot::CPMDPot() : Potential(PotType::CPMD), impl_(new Impl) {
  impl_->params_words = default_params();
  apply_env_hints(impl_->cpmd_root);
  if (try_load_engine(impl_->bundle, impl_->engine_path))
    (void)impl_->configure();
}

CPMDPot::CPMDPot(const ::CPMDParams::Reader &params)
    : Potential(PotType::CPMD), impl_(new Impl) {
  impl_->params_words = serialize_params(params);
  impl_->engine_path = params.getEnginePath().cStr();
  impl_->cpmd_root = params.getCpmdRoot().cStr();
  apply_env_hints(impl_->cpmd_root);
  if (try_load_engine(impl_->bundle, impl_->engine_path))
    (void)impl_->configure();
}

CPMDPot::~CPMDPot() {
  if (impl_) {
    impl_->destroySession();
    delete impl_;
  }
}

bool CPMDPot::setParams(const ::CPMDParams::Reader &params) {
  if (!impl_)
    impl_ = new Impl;

  const std::string next_engine_path = params.getEnginePath().cStr();
  const std::string next_cpmd_root = params.getCpmdRoot().cStr();
  const bool need_reload =
      !impl_->bundle.loaded || next_engine_path != impl_->engine_path;

  impl_->params_words = serialize_params(params);
  impl_->engine_path = next_engine_path;
  impl_->cpmd_root = next_cpmd_root;
  apply_env_hints(impl_->cpmd_root);

  if (need_reload) {
    impl_->destroySession();
    impl_->bundle = EngineBundle{};
    if (!try_load_engine(impl_->bundle, impl_->engine_path))
      return false;
  } else if (!impl_->bundle.loaded) {
    if (!try_load_engine(impl_->bundle, impl_->engine_path))
      return false;
  }
  if (!impl_->bundle.loaded)
    return false;
  return impl_->configure();
}

void CPMDPot::getParams(::CPMDParams::Builder out) const {
  if (!impl_ || impl_->params_words.empty()) {
    ::capnp::MallocMessageBuilder msg;
    auto params = msg.initRoot<::CPMDParams>();
    copy_params_to_builder(params.asReader(), out);
    return;
  }

  auto words = kj::arrayPtr<const ::capnp::word>(impl_->params_words.data(),
                                                 impl_->params_words.size());
  ::capnp::FlatArrayMessageReader reader(words);
  auto params = read_params_words(impl_->params_words, reader);
  copy_params_to_builder(params, out);
}

bool CPMDPot::setPotentialConfig(const ::PotentialConfig::Reader &cfg,
                                 std::string *message_out) {
  switch (cfg.which()) {
  case ::PotentialConfig::NONE:
    if (message_out)
      *message_out = "no-op (rgpot params: none)";
    return true;
  case ::PotentialConfig::CPMD: {
    const auto cp = cfg.getCpmd();
    const bool ok = setParams(cp);
    if (message_out) {
      const std::string sum = params_summary(cp);
      if (ok)
        *message_out = "rgpot params applied (cpmd arm): " + sum;
      else if (!available())
        *message_out =
            "rgpot params cpmd arm failed (engine not loaded): " + sum;
      else
        *message_out = "rgpot params cpmd arm rejected by embed: " + sum;
    }
    return ok;
  }
  default:
    if (message_out)
      *message_out =
          "rgpot params arm not handled by CPMDPot (use matching backend pot)";
    return false;
  }
}

bool CPMDPot::available() const {
  if (!impl_ || !impl_->bundle.loaded)
    return false;
  if (has_session_result_abi(impl_->bundle))
    return impl_->session != nullptr;
  return impl_->bundle.energy_gradient && impl_->bundle.set_params;
}

int CPMDPot::bindCalculators(int ranks_per_calc) {
  EngineBundle bundle;
  if (!try_load_engine(bundle, ""))
    return -1;
  auto bind = bundle.engine_lib.sym_optional<BindCalculatorsFn>(
      "cpmdc_bind_calculator");
  if (!bind)
    return -1;
  return bind(ranks_per_calc);
}

bool CPMDPot::probe_available() {
  std::lock_guard<std::mutex> lock(g_probe_mu);
  if (g_probe_done)
    return g_probe_ok;
  EngineBundle tmp;
  g_probe_ok = try_load_engine(tmp, "");
  g_probe_done = true;
  return g_probe_ok;
}

bool CPMDPot::abi_available() {
  std::lock_guard<std::mutex> lock(g_probe_mu);
  if (g_abi_probe_done)
    return g_abi_probe_ok;
  EngineBundle tmp;
  if (!try_load_engine(tmp, "")) {
    g_abi_probe_ok = false;
  } else if (tmp.available) {
    g_abi_probe_ok = tmp.available() != 0;
  } else {
    g_abi_probe_ok = false;
  }
  g_abi_probe_done = true;
  return g_abi_probe_ok;
}

void CPMDPot::forceImpl(const ForceInput &in, ForceOut *out) const {
  if (!available()) {
    throw std::runtime_error(
        std::string("CPMD engine (libcpmdc) not loaded: ") +
        (impl_ ? impl_->bundle.load_error : "no impl"));
  }

  const int n = static_cast<int>(in.nAtoms);
  if (n <= 0)
    throw std::runtime_error("CPMDPot: nAtoms must be positive");
  if (!in.pos || !in.atmnrs || !in.box || !out || !out->F)
    throw std::runtime_error(
        "CPMDPot: null positions/atmnrs/box/forces buffer");

  if (has_session_result_abi(impl_->bundle) && impl_->session) {
    impl_->forceSession(in, out);
    return;
  }

  std::vector<double> &grad = impl_->grad_scratch;
  grad.assign(static_cast<size_t>(n) * 3u, 0.0);
  const ParamsView params = params_view(impl_->params_words);
  CPMDCResult res = impl_->bundle.energy_gradient(
      n, in.pos, in.atmnrs, params.data, params.size, grad.data());

  if (!res.ok) {
    throw std::runtime_error(std::string("CPMD engine failed: ") + res.message);
  }

  out->energy = res.energy_h * HARTREE_TO_EV;
  out->variance = 0.0;
  out->has_stress = 0;
  for (int i = 0; i < n * 3; ++i)
    out->F[static_cast<size_t>(i)] =
        grad[static_cast<size_t>(i)] * NEG_GRAD_TO_FORCE;
}

void CPMDPot::Impl::forceSession(const ForceInput &in, ForceOut *out) {
  const auto force_words = serialize_force_input(in);
  const ParamsView force_view = params_view(force_words);
  const size_t required =
      bundle.potential_result_size_for_force_input(force_view.data,
                                                   force_view.size);
  if (required == 0)
    throw std::runtime_error("CPMD engine rejected ForceInput sizing");

  std::vector<::capnp::word> result_words(
      (required + sizeof(::capnp::word) - 1u) / sizeof(::capnp::word));
  size_t written = 0;
  CPMDCResult res = bundle.session_calculate_result(
      session, force_view.data, force_view.size, result_words.data(),
      result_words.size() * sizeof(::capnp::word), &written);
  if (!res.ok)
    throw std::runtime_error(std::string("CPMD engine failed: ") + res.message);
  if (written == 0 || written > result_words.size() * sizeof(::capnp::word) ||
      (written % sizeof(::capnp::word)) != 0)
    throw std::runtime_error("CPMD engine returned invalid PotentialResult");

  auto words = kj::arrayPtr<const ::capnp::word>(
      result_words.data(), written / sizeof(::capnp::word));
  ::capnp::FlatArrayMessageReader reader(words);
  auto result = reader.getRoot<::PotentialResult>();
  const auto forces = result.getForces();
  const size_t expected_force_count = in.nAtoms * 3u;
  if (forces.size() != expected_force_count)
    throw std::runtime_error("CPMD engine returned wrong force count");
  out->energy = result.getEnergy();
  out->variance = 0.0;
  for (unsigned int i = 0; i < forces.size(); ++i)
    out->F[i] = forces[i];
  out->has_stress = 0;
  const auto stress = result.getStress();
  if (stress.size() == 9) {
    for (unsigned int i = 0; i < 9; ++i)
      out->stress[i] = stress[i];
    out->has_stress = 1;
  }
}

} // namespace rgpot
