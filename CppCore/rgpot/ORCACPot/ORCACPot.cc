// MIT License
// Copyright 2023--present rgpot developers

#include "rgpot/ORCACPot/ORCACPot.hpp"

#include "rgpot/ORCACPot/orcac_c_abi.h"
#include "rgpot/ParamHash.hpp"
#include "rgpot/abi/DynLib.hpp"
#include "rgpot/abi/EngineDlopen.hpp"
#include "rgpot/units.hpp"

#include <cstdlib>
#include <mutex>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace rgpot {

using units::HARTREE_BOHR_TO_EV_ANGSTROM;
using units::HARTREE_TO_EV;

namespace {

/// ORCAC_LIBRARY and RGPOT_ORCAC_ENGINE when set, then the soname with its
/// ABI generation and the default names. An explicit path is the only one.
std::vector<std::string> orcacCandidates(const std::string &engine_path) {
  if (!engine_path.empty()) {
    return {engine_path};
  }
  std::vector<std::string> out;
  for (const char *var : {"ORCAC_LIBRARY", "RGPOT_ORCAC_ENGINE"}) {
    if (const char *e = std::getenv(var); e != nullptr && *e != '\0') {
      out.emplace_back(e);
    }
  }
#if !defined(_WIN32) && !defined(__APPLE__)
  out.emplace_back("liborcac.so.1");
#endif
  for (auto &c : abi::engine_candidates("orcac", "", {})) {
    out.push_back(std::move(c));
  }
  return out;
}

/// NULL for an empty string, which orcac reads as "use the default".
const char *orNull(const std::string &s) { return s.empty() ? nullptr : s.c_str(); }

} // namespace

struct ORCACPot::Impl {
  ORCACConfig config;
  DynLib lib;
  std::string path;
  std::string load_error;
  orcac_config_init_fn config_init = nullptr;
  orcac_create_fn create = nullptr;
  orcac_configure_fn configure = nullptr;
  orcac_energy_gradient_fn energy_gradient = nullptr;
  orcac_cancel_fn cancel = nullptr;
  orcac_last_error_fn last_error = nullptr;
  orcac_destroy_fn destroy = nullptr;
  orcac_status_name_fn status_name = nullptr;
  orcac_version_fn version = nullptr;
  orcac_c_abi_version_fn c_abi_version = nullptr;
  orcac_abi_version_fn abi_version = nullptr;
  orcac_available_fn available = nullptr;
  orcac_probe_executable_fn probe_executable = nullptr;
  OrcaCSession *session = nullptr;
  mutable std::mutex mutex;
  mutable std::vector<double> gradient;

  /// Resolves the thirteen ABI 1 functions; returns the first one missing.
  const char *resolve() {
    const char *missing = nullptr;
    auto get = [&](auto &fn, const char *name) {
      fn = lib.sym_optional<std::remove_reference_t<decltype(fn)>>(name);
      if (fn == nullptr && missing == nullptr) {
        missing = name;
      }
    };
    get(config_init, "orcac_config_init");
    get(create, "orcac_create");
    get(configure, "orcac_configure");
    get(energy_gradient, "orcac_energy_gradient");
    get(cancel, "orcac_cancel");
    get(last_error, "orcac_last_error");
    get(destroy, "orcac_destroy");
    get(status_name, "orcac_status_name");
    get(version, "orcac_version");
    get(c_abi_version, "orcac_c_abi_version");
    get(abi_version, "orcac_abi_version");
    get(available, "orcac_available");
    get(probe_executable, "orcac_probe_executable");
    return missing;
  }

  std::string statusText(int st) const {
    const char *name = status_name(st);
    return name != nullptr ? name : "status " + std::to_string(st);
  }
};

ORCACPot::ORCACPot(ORCACConfig config)
    : Potential(PotType::ORCAC), impl_(new Impl) {
  impl_->config = std::move(config);
  const ORCACConfig &c = impl_->config;
  const auto candidates = orcacCandidates(c.engine_path);
  std::string open_error;
  if (!abi::open_first_candidate(impl_->lib, candidates, &impl_->path,
                                 &open_error, /*global=*/false)) {
    impl_->load_error = "liborcac not loaded (tried";
    for (const auto &cand : candidates) {
      impl_->load_error += " " + cand;
    }
    impl_->load_error += "): " + open_error;
    return;
  }
  const std::string where = impl_->path + ": ";
  if (const char *missing = impl_->resolve(); missing != nullptr) {
    impl_->load_error =
        where + "not an orcac C ABI " + std::to_string(RGPOT_ORCAC_C_ABI_VERSION) +
        " library (missing " + missing + ")";
    return;
  }
  if (impl_->c_abi_version() != RGPOT_ORCAC_C_ABI_VERSION ||
      impl_->abi_version() != ORCAC_ABI_VERSION) {
    impl_->load_error = where + "orcac C ABI " +
                        std::to_string(impl_->c_abi_version()) + " (ABI " +
                        std::to_string(impl_->abi_version()) + "), this rgpot speaks " +
                        std::to_string(RGPOT_ORCAC_C_ABI_VERSION);
    return;
  }
  if (c.method.empty()) {
    impl_->load_error = where + "ORCACConfig::method is required";
    return;
  }
  OrcaCConfig cfg;
  if (const int st = impl_->config_init(&cfg, sizeof(cfg)); st != ORCAC_OK) {
    impl_->load_error = where + "orcac_config_init: " + impl_->statusText(st);
    return;
  }
  cfg.method = c.method.c_str();
  cfg.basis = c.basis.c_str();
  cfg.keywords = orNull(c.keywords);
  cfg.blocks = orNull(c.blocks);
  cfg.executable = orNull(c.executable);
  cfg.scratch_dir = orNull(c.scratch_dir);
  cfg.charge = c.charge;
  cfg.multiplicity = c.multiplicity;
  cfg.nprocs = c.nprocs;
  cfg.maxcore_mb = c.maxcore_mb;
  cfg.timeout_s = c.timeout_s;
  cfg.guess_policy = c.restart ? ORCAC_GUESS_CHAIN : ORCAC_GUESS_FRESH;
  char errbuf[ORCAC_MESSAGE_SIZE] = "";
  OrcaCSession *session = nullptr;
  const int st = impl_->create(&cfg, &session, errbuf, sizeof(errbuf));
  if (st != ORCAC_OK || session == nullptr) {
    impl_->load_error = where + "orcac_create: " + impl_->statusText(st) + ": " +
                        (errbuf[0] != '\0' ? errbuf : "no reason given");
    return;
  }
  impl_->session = session;
}

ORCACPot::~ORCACPot() {
  if (impl_->session != nullptr) {
    impl_->destroy(impl_->session);
  }
  delete impl_;
}

bool ORCACPot::available() const noexcept { return impl_->session != nullptr; }

const std::string &ORCACPot::loadError() const noexcept { return impl_->load_error; }

const std::string &ORCACPot::libraryPath() const noexcept { return impl_->path; }

std::string ORCACPot::engineVersion() const {
  if (impl_->version == nullptr) {
    return {};
  }
  const char *v = impl_->version();
  return v != nullptr ? v : "";
}

const ORCACConfig &ORCACPot::config() const noexcept { return impl_->config; }

uint64_t ORCACPot::paramsKey() const noexcept {
  const ORCACConfig &c = impl_->config;
  Fnv1a fp;
  fp.str("orcac-abi-1");
  fp.str(impl_->path);
  for (const std::string *s : {&c.method, &c.basis, &c.keywords, &c.blocks, &c.executable}) {
    fp.u64(s->size());
    fp.str(*s);
  }
  fp.i64(c.charge);
  fp.i64(c.multiplicity);
  fp.b8(c.restart);
  return fp.h;
}

void ORCACPot::forceImpl(const ForceInput &in, ForceOut *out) const {
  if (!available()) {
    throw std::runtime_error("ORCACPot: " + impl_->load_error);
  }
  const int n = static_cast<int>(in.nAtoms);
  if (n <= 0) {
    throw std::invalid_argument("ORCACPot: nAtoms must be positive");
  }
  // ORCA has no periodic cell: liborcac takes no box, and the box is unused.
  std::lock_guard<std::mutex> lock(impl_->mutex);
  std::vector<double> &g = impl_->gradient;
  g.assign(static_cast<std::size_t>(n) * 3u, 0.0);
  double energy_h = 0.0;
  OrcaCResult meta{};
  meta.struct_size = sizeof(meta);
  const int st = impl_->energy_gradient(impl_->session, n, in.pos, in.atmnrs,
                                        ORCAC_WANT_ENERGY | ORCAC_WANT_GRADIENT,
                                        &energy_h, g.data(), &meta);
  if (st != ORCAC_OK) {
    const char *full = impl_->last_error(impl_->session);
    throw std::runtime_error("ORCACPot: orcac failed: " + impl_->statusText(st) + ": " +
                             (full != nullptr && *full != '\0' ? full : meta.message));
  }
  out->energy = energy_h * HARTREE_TO_EV;
  out->variance = 0.0;
  // orcac returns dE/dx; the force is its negative.
  for (std::size_t i = 0; i < g.size(); ++i) {
    out->F[i] = -g[i] * HARTREE_BOHR_TO_EV_ANGSTROM;
  }
}

} // namespace rgpot
