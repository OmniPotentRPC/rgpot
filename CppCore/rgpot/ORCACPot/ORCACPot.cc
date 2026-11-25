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
#include <utility>

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

} // namespace

struct ORCACPot::Impl {
  ORCACConfig config;
  DynLib lib;
  std::string path;
  std::string load_error;
  orcac_session_destroy_fn destroy = nullptr;
  orcac_session_energy_forces_fn energy_forces = nullptr;
  orcac_session_last_error_fn session_last_error = nullptr;
  orcac_session_run_directory_fn run_directory = nullptr;
  OrcaCSession *session = nullptr;
  mutable std::mutex mutex;
  mutable std::vector<double> forces;
};

ORCACPot::ORCACPot(ORCACConfig config)
    : Potential(PotType::ORCAC), impl_(new Impl) {
  impl_->config = std::move(config);
  const auto candidates = orcacCandidates(impl_->config.engine_path);
  std::string open_error;
  if (!abi::open_first_candidate(impl_->lib, candidates, &impl_->path,
                                 &open_error, /*global=*/false)) {
    impl_->load_error = "liborcac not loaded (tried";
    for (const auto &c : candidates) {
      impl_->load_error += " " + c;
    }
    impl_->load_error += "): " + open_error;
    return;
  }
  auto &lib = impl_->lib;
  const auto abi_version = lib.sym_optional<orcac_c_abi_version_fn>("orcac_c_abi_version");
  const auto create = lib.sym_optional<orcac_session_create_fn>("orcac_session_create");
  const auto create_from_config =
      lib.sym_optional<orcac_session_create_from_config_fn>("orcac_session_create_from_config");
  const auto last_error = lib.sym_optional<orcac_last_error_fn>("orcac_last_error");
  impl_->destroy = lib.sym_optional<orcac_session_destroy_fn>("orcac_session_destroy");
  impl_->energy_forces =
      lib.sym_optional<orcac_session_energy_forces_fn>("orcac_session_energy_forces");
  impl_->session_last_error =
      lib.sym_optional<orcac_session_last_error_fn>("orcac_session_last_error");
  impl_->run_directory =
      lib.sym_optional<orcac_session_run_directory_fn>("orcac_session_run_directory");
  const std::string where = impl_->path + ": ";
  if (abi_version == nullptr || create == nullptr || create_from_config == nullptr ||
      impl_->destroy == nullptr || impl_->energy_forces == nullptr) {
    impl_->load_error = where + "not a liborcac (missing orcac_c_abi_version, "
                                "orcac_session_create, orcac_session_create_from_config, "
                                "orcac_session_destroy or orcac_session_energy_forces)";
    return;
  }
  if (abi_version() != RGPOT_ORCAC_C_ABI_VERSION) {
    impl_->load_error = where + "orcac C ABI " + std::to_string(abi_version()) +
                        ", this rgpot speaks " +
                        std::to_string(RGPOT_ORCAC_C_ABI_VERSION);
    return;
  }
  // FlatArrayMessageReader in liborcac wants word alignment; vector storage
  // from operator new has it.
  const auto &msg = impl_->config.message;
  impl_->session = impl_->config.potential_config
                       ? create_from_config(msg.data(), msg.size())
                       : create(msg.data(), msg.size());
  if (impl_->session == nullptr) {
    impl_->load_error = where + "orcac session refused: " +
                        (last_error != nullptr ? last_error() : "no reason given");
  }
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

std::string ORCACPot::runDirectory() const {
  if (impl_->session == nullptr || impl_->run_directory == nullptr) {
    return {};
  }
  const char *dir = impl_->run_directory(impl_->session);
  return dir != nullptr ? dir : "";
}

uint64_t ORCACPot::paramsKey() const noexcept {
  Fnv1a fp;
  fp.str("orcac-1");
  fp.str(impl_->path);
  fp.b8(impl_->config.potential_config);
  fp.u64(impl_->config.message.size());
  fp.bytes(impl_->config.message.data(), impl_->config.message.size());
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
  std::vector<double> &f = impl_->forces;
  f.assign(static_cast<std::size_t>(n) * 3u, 0.0);
  const OrcaCResult res =
      impl_->energy_forces(impl_->session, n, in.pos, in.atmnrs, f.data());
  if (res.ok == 0) {
    const char *full = impl_->session_last_error != nullptr
                           ? impl_->session_last_error(impl_->session)
                           : nullptr;
    throw std::runtime_error(std::string("ORCACPot: orcac failed: ") +
                             (full != nullptr && *full != '\0' ? full : res.message));
  }
  out->energy = res.energy_h * HARTREE_TO_EV;
  out->variance = 0.0;
  for (std::size_t i = 0; i < f.size(); ++i) {
    out->F[i] = f[i] * HARTREE_BOHR_TO_EV_ANGSTROM;
  }
}

} // namespace rgpot
