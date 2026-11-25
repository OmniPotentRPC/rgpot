// MIT License
// Copyright 2023--present rgpot developers

#include "rgpot/ORCAPot/ORCAPot.hpp"

#include "rgpot/ORCAPot/ORCAGuards.hpp"
#include "rgpot/ORCAPot/orcac.h"
#include "rgpot/abi/DynLib.hpp"
#include "rgpot/abi/EngineDlopen.hpp"
#include "rgpot/engine/Sha256.hpp"
#include "rgpot/units.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace rgpot {

namespace {

static_assert(sizeof(OrcaCConfig) >= ORCAC_CONFIG_MIN_SIZE);
static_assert(sizeof(OrcaCResult) >= ORCAC_RESULT_MIN_SIZE);

using IntFn = int (*)(void);
using VersionFn = const char *(*)(void);
using ConfigInitFn = int (*)(OrcaCConfig *, size_t);
using CreateFn = int (*)(const OrcaCConfig *, OrcaCSession **, char *,
                         size_t);
using ConfigureFn = int (*)(OrcaCSession *, const OrcaCConfig *);
using EnergyGradientFn = int (*)(OrcaCSession *, int, const double *,
                                 const int *, uint32_t, double *, double *,
                                 OrcaCResult *);
using CancelFn = int (*)(OrcaCSession *);
using LastErrorFn = const char *(*)(const OrcaCSession *);
using DestroyFn = void (*)(OrcaCSession *);
using StatusNameFn = const char *(*)(int);
using ProbeExecutableFn = int (*)(const OrcaCConfig *, char *, size_t);

constexpr std::size_t kPathCapacity = 4096;

/// liborcac opened and checked against the vendored orcac.h.
struct OrcacLibrary {
  DynLib lib;
  std::string path;
  std::string error;
  IntFn c_abi_version = nullptr;
  IntFn abi_version = nullptr;
  VersionFn version = nullptr;
  ConfigInitFn config_init = nullptr;
  CreateFn create = nullptr;
  ConfigureFn configure = nullptr;
  EnergyGradientFn energy_gradient = nullptr;
  CancelFn cancel = nullptr;
  LastErrorFn last_error = nullptr;
  DestroyFn destroy = nullptr;
  StatusNameFn status_name = nullptr;
  IntFn available = nullptr;
  ProbeExecutableFn probe_executable = nullptr;
  bool loaded = false;
};

std::string join(const std::vector<std::string> &items) {
  std::string out;
  for (const auto &item : items) {
    out += (out.empty() ? "" : ", ") + item;
  }
  return out;
}

std::string bounded(const char *text, std::size_t cap) {
  return text ? std::string(text, strnlen(text, cap)) : std::string{};
}

/// The name orcac.h gives a status; the library's own table is not trusted
/// for codes this header defines.
std::string status_name(int status) {
  switch (status) {
  case ORCAC_OK:
    return "ORCAC_OK";
  case ORCAC_E_INVALID:
    return "ORCAC_E_INVALID";
  case ORCAC_E_ERROR:
    return "ORCAC_E_ERROR";
  case ORCAC_E_UNSUPPORTED:
    return "ORCAC_E_UNSUPPORTED";
  case ORCAC_E_TIMEOUT:
    return "ORCAC_E_TIMEOUT";
  case ORCAC_E_NOT_CONVERGED:
    return "ORCAC_E_NOT_CONVERGED";
  case ORCAC_E_BUSY:
    return "ORCAC_E_BUSY";
  case ORCAC_E_NOT_FOUND:
    return "ORCAC_E_NOT_FOUND";
  case ORCAC_E_CANCELLED:
    return "ORCAC_E_CANCELLED";
  default:
    return "status " + std::to_string(status);
  }
}

template <typename Fn>
void need(OrcacLibrary &b, Fn &slot, const char *name,
          std::vector<std::string> &missing) {
  slot = b.lib.sym_optional<Fn>(name);
  if (!slot) {
    missing.emplace_back(name);
  }
}

/// Empty when the opened library matches orcac.h, else the reason. Only the
/// version calls, orcac_config_init and a NULL-session
/// orcac_energy_gradient run here: no session is made and nothing is
/// evaluated.
std::string handshake(OrcacLibrary &b) {
  const std::string at = "liborcac at " + b.path;
  std::vector<std::string> missing;
  need(b, b.c_abi_version, "orcac_c_abi_version", missing);
  need(b, b.abi_version, "orcac_abi_version", missing);
  if (!missing.empty()) {
    return at + " lacks " + join(missing);
  }
  const int c_abi = b.c_abi_version();
  if (c_abi != RGPOT_ORCAC_C_ABI_VERSION) {
    return at + " reports C ABI version " + std::to_string(c_abi) +
           ", but rgpot was built against " +
           std::to_string(RGPOT_ORCAC_C_ABI_VERSION);
  }
  const int abi = b.abi_version();
  if (abi != ORCAC_ABI_VERSION) {
    return at + " reports ABI version " + std::to_string(abi) +
           ", but rgpot was built against " +
           std::to_string(ORCAC_ABI_VERSION);
  }

  need(b, b.version, "orcac_version", missing);
  need(b, b.config_init, "orcac_config_init", missing);
  need(b, b.create, "orcac_create", missing);
  need(b, b.configure, "orcac_configure", missing);
  need(b, b.energy_gradient, "orcac_energy_gradient", missing);
  need(b, b.cancel, "orcac_cancel", missing);
  need(b, b.last_error, "orcac_last_error", missing);
  need(b, b.destroy, "orcac_destroy", missing);
  need(b, b.status_name, "orcac_status_name", missing);
  need(b, b.available, "orcac_available", missing);
  need(b, b.probe_executable, "orcac_probe_executable", missing);
  if (!missing.empty()) {
    return at + " lacks " + join(missing);
  }

  OrcaCConfig cfg{};
  const int st = b.config_init(&cfg, sizeof(cfg));
  if (st != ORCAC_OK) {
    return at + ": orcac_config_init returned " + status_name(st);
  }
  if (cfg.struct_size < ORCAC_CONFIG_MIN_SIZE ||
      cfg.struct_size > sizeof(cfg)) {
    return at + " reports an OrcaCConfig struct_size of " +
           std::to_string(cfg.struct_size) + "; ABI 1 needs " +
           std::to_string(ORCAC_CONFIG_MIN_SIZE) + " to " +
           std::to_string(sizeof(cfg));
  }
  if (cfg.magic != ORCAC_CONFIG_MAGIC || cfg.version < 1u ||
      cfg.version > ORCAC_CONFIG_VERSION) {
    return at + " fills OrcaCConfig with magic " + std::to_string(cfg.magic) +
           " and version " + std::to_string(cfg.version) +
           "; rgpot's orcac.h expects " + std::to_string(ORCAC_CONFIG_MAGIC) +
           " and 1 to " + std::to_string(ORCAC_CONFIG_VERSION);
  }

  OrcaCResult meta{};
  meta.struct_size = sizeof(meta);
  const int probed = b.energy_gradient(nullptr, 0, nullptr, nullptr, 0,
                                       nullptr, nullptr, &meta);
  if (probed != ORCAC_E_INVALID) {
    return at + " answers orcac_energy_gradient without a session with " +
           status_name(probed) + " instead of ORCAC_E_INVALID";
  }
  if (meta.struct_size < ORCAC_RESULT_MIN_SIZE ||
      meta.struct_size > sizeof(meta)) {
    return at + " reports an OrcaCResult struct_size of " +
           std::to_string(meta.struct_size) + "; ABI 1 needs " +
           std::to_string(ORCAC_RESULT_MIN_SIZE) + " to " +
           std::to_string(sizeof(meta));
  }
  return {};
}

bool load_library(OrcacLibrary &b, const std::string &explicit_path) {
  b = OrcacLibrary{};
  const auto cands = abi::engine_candidates(
      "orcac", explicit_path, {"ORCAC_LIBRARY", "RGPOT_ORCAC_ENGINE"},
      abi::CandidateOptions{.exclusive_explicit = true,
                            .exclusive_env = true});
  std::string open_error;
  if (!abi::open_first_candidate(b.lib, cands, &b.path, &open_error)) {
    b.error = "liborcac not loaded (tried " + join(cands) + "): " +
              open_error +
              ". Install orcac and set ORCAC_LIBRARY or RGPOT_ORCAC_ENGINE "
              "to the path of liborcac";
    return false;
  }
  b.error = handshake(b);
  if (!b.error.empty()) {
    b.lib.close();
    return false;
  }
  b.loaded = true;
  return true;
}

void check_line(const std::string &value, const char *field) {
  if (value.find_first_of("\r\n") != std::string::npos) {
    throw std::invalid_argument(std::string("ORCAPot: ") + field +
                                " must not contain a line break");
  }
}

void check_multiplicity(int multiplicity) {
  if (multiplicity < 1) {
    throw std::invalid_argument("ORCAPot: multiplicity must be >= 1, got " +
                                std::to_string(multiplicity));
  }
}

void validate(const ORCAConfig &cfg) {
  if (cfg.method.empty()) {
    throw std::invalid_argument("ORCAPot: method is required");
  }
  check_line(cfg.method, "method");
  check_line(cfg.basis, "basis");
  check_line(cfg.keywords, "keywords");
  check_line(cfg.executable, "executable");
  check_line(cfg.library, "library");
  check_line(cfg.scratch_dir, "scratch_dir");
  check_multiplicity(cfg.multiplicity);
  if (cfg.nprocs < 1) {
    throw std::invalid_argument("ORCAPot: nprocs must be >= 1, got " +
                                std::to_string(cfg.nprocs));
  }
  if (cfg.maxcore_mb < 0) {
    throw std::invalid_argument("ORCAPot: maxcore_mb must be >= 0, got " +
                                std::to_string(cfg.maxcore_mb));
  }
  if (!std::isfinite(cfg.timeout_s) || cfg.timeout_s < 0.0) {
    throw std::invalid_argument(
        "ORCAPot: timeout_s must be finite and >= 0, got " +
        std::to_string(cfg.timeout_s));
  }
  if (cfg.guess_policy != ORCAConfig::GuessPolicy::MoRead &&
      cfg.guess_policy != ORCAConfig::GuessPolicy::Fresh) {
    throw std::invalid_argument("ORCAPot: unknown guess_policy");
  }
  if (cfg.keep_files) {
    throw std::invalid_argument(
        "ORCAPot: keep_files is not part of the orcac ABI 1 config; "
        "orcac_destroy always removes the run directory");
  }
}

const char *or_null(const std::string &s) {
  return s.empty() ? nullptr : s.c_str();
}

/// OrcaCConfig over cfg's strings with @p executable in place of
/// cfg.executable; both must outlive the result. An empty basis is passed
/// as "" so orcac omits it instead of using STO-3G.
OrcaCConfig to_c(const OrcacLibrary &b, const ORCAConfig &cfg,
                 const std::string &executable) {
  OrcaCConfig c{};
  (void)b.config_init(&c, sizeof(c));
  c.method = cfg.method.c_str();
  c.basis = cfg.basis.c_str();
  c.keywords = or_null(cfg.keywords);
  c.blocks = or_null(cfg.blocks);
  c.executable = or_null(executable);
  c.scratch_dir = or_null(cfg.scratch_dir);
  c.charge = cfg.charge;
  c.multiplicity = cfg.multiplicity;
  c.nprocs = cfg.nprocs;
  c.maxcore_mb = cfg.maxcore_mb;
  c.timeout_s = cfg.timeout_s;
  c.guess_policy = cfg.guess_policy == ORCAConfig::GuessPolicy::Fresh
                       ? ORCAC_GUESS_FRESH
                       : ORCAC_GUESS_CHAIN;
  c.flags = cfg.fresh_retry ? 0u : ORCAC_F_NO_FRESH_RETRY;
  return c;
}

void put_u32(engine::Sha256 &h, uint32_t v) {
  const uint8_t bytes[4] = {
      static_cast<uint8_t>(v), static_cast<uint8_t>(v >> 8),
      static_cast<uint8_t>(v >> 16), static_cast<uint8_t>(v >> 24)};
  h.update(bytes, sizeof(bytes));
}

void put_text(engine::Sha256 &h, const std::string &text) {
  put_u32(h, static_cast<uint32_t>(text.size()));
  h.update(text.data(), text.size());
}

/// Low 64 bits of SHA-256(method, basis, keywords, blocks, executable and
/// its stamp, ORCAC_ABI_VERSION, charge, multiplicity, guess policy,
/// changed): the last 16 hex digits.
uint64_t params_key(const ORCAConfig &cfg, const std::string &executable,
                    uint64_t stamp, bool changed) {
  engine::Sha256 h;
  put_text(h, cfg.method);
  put_text(h, cfg.basis);
  put_text(h, cfg.keywords);
  put_text(h, cfg.blocks);
  put_text(h, executable);
  put_u32(h, static_cast<uint32_t>(stamp));
  put_u32(h, static_cast<uint32_t>(stamp >> 32));
  put_u32(h, static_cast<uint32_t>(ORCAC_ABI_VERSION));
  put_u32(h, static_cast<uint32_t>(cfg.charge));
  put_u32(h, static_cast<uint32_t>(cfg.multiplicity));
  put_u32(h, static_cast<uint32_t>(cfg.guess_policy));
  put_u32(h, changed ? 1u : 0u);
  const std::string hex = h.hex_digest();
  return std::stoull(hex.substr(hex.size() - 16), nullptr, 16);
}

std::mutex g_probe_mu;
bool g_probe_done = false;
bool g_probe_ok = false;
bool g_abi_probe_done = false;
bool g_abi_probe_ok = false;

OrcacLibrary &probe_library() {
  static OrcacLibrary retained;
  return retained;
}

} // namespace

struct ORCAPot::Impl {
  OrcacLibrary lib;
  ORCAConfig config;
  OrcaCSession *session = nullptr;
  std::string error;
  std::string identity;
  uint64_t stamp = 0;
  mutable std::vector<double> gradient;

  ~Impl() {
    if (session && lib.destroy) {
      lib.destroy(session);
    }
  }

  [[nodiscard]] std::string session_error() const {
    std::string why = bounded(lib.last_error(session), ORCAC_MESSAGE_SIZE);
    return why.empty() ? std::string("no message") : why;
  }

  /// The executable config.executable names now, as orcac resolves it,
  /// or empty with the status in @p status.
  std::string resolve(int *status) const {
    const OrcaCConfig c = to_c(lib, config, config.executable);
    std::vector<char> resolved(kPathCapacity, '\0');
    *status = lib.probe_executable(&c, resolved.data(), resolved.size());
    return *status == ORCAC_OK ? bounded(resolved.data(), resolved.size())
                               : std::string{};
  }

  /// True while the executable resolves to identity with stamp, or when
  /// no session is open.
  [[nodiscard]] bool executable_unchanged() const noexcept {
    if (session == nullptr) {
      return true;
    }
    try {
      int status = ORCAC_OK;
      return resolve(&status) == identity &&
             orca::executableUnchanged(identity, identity, stamp);
    } catch (...) {
      return false;
    }
  }

  void check_executable(const char *when) const {
    if (!executable_unchanged()) {
      throw ORCAError(
          "ORCAPot: the executable " +
          (config.executable.empty() ? std::string("orcac found")
                                     : config.executable) +
          " changed " + when + ": it no longer resolves to " + identity +
          " with the size, modification time, change time, device, inode "
          "and content hash recorded when the session was opened; make a "
          "new ORCAPot to run the new file");
    }
  }

  void open() {
    if (!load_library(lib, config.library)) {
      error = lib.error;
      return;
    }
    int probed = ORCAC_OK;
    const std::string resolved = resolve(&probed);
    if (probed != ORCAC_OK) {
      error = "orcac_probe_executable(" +
              (config.executable.empty() ? std::string("orcac's search")
                                         : config.executable) +
              ") returned " + status_name(probed) + ": " +
              bounded(lib.last_error(nullptr), ORCAC_MESSAGE_SIZE);
      return;
    }
    if (!orca::identifyExecutable(resolved, &identity, &stamp)) {
      error = "could not resolve and read the executable " + resolved;
      identity.clear();
      return;
    }
    const OrcaCConfig c = to_c(lib, config, identity);
    std::vector<char> errbuf(ORCAC_MESSAGE_SIZE, '\0');
    const int created =
        lib.create(&c, &session, errbuf.data(), errbuf.size());
    if (created != ORCAC_OK || session == nullptr) {
      error = "orcac_create returned " + status_name(created) + ": " +
              bounded(errbuf.data(), errbuf.size());
      session = nullptr;
    }
  }
};

ORCAPot::ORCAPot(const ORCAConfig &config)
    : Potential(PotType::ORCA), impl_(nullptr) {
  validate(config);
  impl_ = new Impl;
  impl_->config = config;
  impl_->open();
}

ORCAPot::~ORCAPot() { delete impl_; }

bool ORCAPot::available() const noexcept {
  return impl_->lib.loaded && impl_->session != nullptr;
}

const std::string &ORCAPot::loadError() const noexcept { return impl_->error; }

const std::string &ORCAPot::libraryPath() const noexcept {
  return impl_->lib.path;
}

std::string ORCAPot::engineVersion() const {
  return impl_->lib.version ? bounded(impl_->lib.version(), 256)
                            : std::string{};
}

const std::string &ORCAPot::executableIdentity() const noexcept {
  return impl_->identity;
}

const ORCAConfig &ORCAPot::config() const noexcept { return impl_->config; }

uint64_t ORCAPot::paramsKey() const noexcept {
  return params_key(impl_->config, impl_->identity, impl_->stamp,
                    !impl_->executable_unchanged());
}

void ORCAPot::setChargeMultiplicity(int charge, int multiplicity) {
  check_multiplicity(multiplicity);
  if (!available()) {
    throw ORCAError("ORCAPot: " + impl_->error);
  }
  ORCAConfig next = impl_->config;
  next.charge = charge;
  next.multiplicity = multiplicity;
  const OrcaCConfig c = to_c(impl_->lib, next, impl_->identity);
  const int st = impl_->lib.configure(impl_->session, &c);
  if (st != ORCAC_OK) {
    throw ORCAError("ORCAPot: orcac_configure returned " + status_name(st) +
                    ": " + impl_->session_error());
  }
  impl_->config = next;
}

void ORCAPot::cancel() noexcept {
  if (impl_->session != nullptr && impl_->lib.cancel != nullptr) {
    (void)impl_->lib.cancel(impl_->session);
  }
}

void ORCAPot::forceImpl(const ForceInput &in, ForceOut *out) const {
  if (!available()) {
    throw ORCAError("ORCAPot: " + impl_->error);
  }
  if (in.nAtoms == 0) {
    throw std::invalid_argument("ORCAPot: nAtoms must be positive");
  }
  if (!in.pos || !in.atmnrs || !out || !out->F) {
    throw std::invalid_argument(
        "ORCAPot: null positions, atomic numbers or forces buffer");
  }
  orca::checkClusterBox("ORCAPot", in.box, in.nAtoms, in.pos);
  std::vector<int> z(in.atmnrs, in.atmnrs + in.nAtoms);
  std::vector<double> &g = impl_->gradient;
  g.assign(in.nAtoms * 3u, 0.0);
  double energy_h = 0.0;
  OrcaCResult meta{};
  meta.struct_size = sizeof(meta);
  // The file orcac runs is the one paramsKey fingerprints. It is checked
  // before the run and again after it, so a result is only returned, and
  // cached, for that file.
  impl_->check_executable("since the session was opened");
  const int st = impl_->lib.energy_gradient(
      impl_->session, static_cast<int>(in.nAtoms), in.pos, z.data(),
      ORCAC_WANT_ENERGY | ORCAC_WANT_GRADIENT, &energy_h, g.data(), &meta);
  impl_->check_executable("while it ran");
  if (st != ORCAC_OK) {
    std::string why = bounded(meta.message, sizeof(meta.message));
    if (why.empty()) {
      why = impl_->session_error();
    }
    throw ORCAError("ORCAPot: orcac_energy_gradient returned " +
                    status_name(st) + ": " + why);
  }
  out->energy = energy_h * units::HARTREE_TO_EV;
  out->variance = 0.0;
  for (std::size_t i = 0; i < g.size(); ++i) {
    out->F[i] = g[i] * units::NEG_GRAD_TO_FORCE;
  }
}

bool ORCAPot::probeAvailable() {
  return abi::probe_once(g_probe_mu, g_probe_done, g_probe_ok, [] {
    OrcacLibrary &b = probe_library();
    return b.loaded || load_library(b, "");
  });
}

bool ORCAPot::abiAvailable() {
  return abi::probe_once(g_probe_mu, g_abi_probe_done, g_abi_probe_ok, [] {
    OrcacLibrary &b = probe_library();
    if (!b.loaded && !load_library(b, "")) {
      return false;
    }
    return b.available != nullptr && b.available() != 0;
  });
}

} // namespace rgpot
