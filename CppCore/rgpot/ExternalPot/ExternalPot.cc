// MIT License
// Copyright 2023--present rgpot developers

#include "rgpot/ExternalPot/ExternalPot.hpp"
#include "rgpot/ExternalPot/detail/LoadedIdentity.hpp"

#include "rgpot/ParamHash.hpp"
#include "rgpot/abi/ProfileLoader.hpp"
#include "rgpot/ExternalPot/md_units.hpp"
#include "rgpot/LennardJones/LJPot.hpp"
#include "rgpot/MDSession/ProjectedLangevin.hpp"
#include "rgpot/engine_c_abi.h"
#include "rgpot/plugin/DynLib.hpp"
// Both build systems pass RGPOT_HAS_LAMMPC_PROFILE as 0 or 1. With 1 they also
// order the generated Potentials.capnp.h before this file, so a missing header
// is a compile error rather than a build without lammpc.
#if !defined(RGPOT_HAS_LAMMPC_PROFILE)
#error "RGPOT_HAS_LAMMPC_PROFILE must be defined to 0 or 1 by the build"
#elif RGPOT_HAS_LAMMPC_PROFILE
#include "rgpot/abi/Compat.hpp"
#include "rgpot/rpc/Potentials.capnp.h"
#include <capnp/message.h>
#include <capnp/serialize.h>
#include <kj/array.h>
#endif

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if !defined(_WIN32)
#include <dlfcn.h>
#include <sys/stat.h>
#endif

namespace rgpot {
namespace {

namespace dyn = plugin::dynlib;

using EngineCreate = RgpotEnginePot *(*)(const void *, size_t, char *, size_t);
using EngineDestroy = void (*)(RgpotEnginePot *);
using EngineForce = int (*)(RgpotEnginePot *, long, const double *, const int *,
                            double *, double *, double *, const double *,
                            rgpot_engine_coord_transform, void *);
using NativeForce = int (*)(void *, long, const double *, const int *, double *,
                            double *, double *, const double *);

void require_lj(const ExternalLjConfig &config) {
  const auto usable = [](double v) { return v > 0.0 && std::isfinite(v); };
  if (!usable(config.u0) || !usable(config.cutoff) || !usable(config.psi)) {
    throw std::invalid_argument(
        "u0, cutoff and psi must be positive and finite (eV and Angstrom)");
  }
}

void add_soname(std::vector<std::string> &out, const char *stem) {
#if defined(_WIN32)
  out.emplace_back(std::string(stem) + ".dll");
  out.emplace_back(std::string("lib") + stem + ".dll");
#elif defined(__APPLE__)
  out.emplace_back(std::string("lib") + stem + ".dylib");
  out.emplace_back(std::string("lib") + stem + ".so");
#else
  out.emplace_back(std::string("lib") + stem + ".so");
#endif
}

std::string last_dl() {
  const std::string text = dyn::error();
  return text.empty() ? "dlopen failed" : text;
}

// One dlclose per successful dlopen. glibc ignores dlclose once an object is
// RTLD_NODELETE and leaves the loader reference in place, so a constructor
// that opened with that flag on every call would raise the reference forever.
// An attempt is opened without the flag and closed if it does not become the
// resident engine. The resident open is pinned RTLD_NODELETE, which keeps the
// mapping and its code loaded, and a later constructor copies that handle.
void close_library(void *handle) noexcept {
  if (handle == nullptr) {
    return;
  }
#if defined(_WIN32)
  if (FreeLibrary(reinterpret_cast<HMODULE>(handle)) == 0) {
    std::fprintf(stderr, "FreeLibrary failed: %lu\n",
                 static_cast<unsigned long>(GetLastError()));
  }
#else
  dlerror();
  if (dlclose(handle) != 0) {
    const char *msg = dlerror();
    std::fprintf(stderr, "dlclose failed: %s\n",
                 msg != nullptr && msg[0] != '\0' ? msg : "dlclose failed");
  }
#endif
}

// Copies share one loader reference. A move leaves the source empty so it
// does not close. The last owner closes, from its destructor and not by
// throwing.
class LibraryHandle {
public:
  LibraryHandle() noexcept = default;
  explicit LibraryHandle(void *raw) : ref_(nullptr) {
    if (raw == nullptr) {
      return;
    }
    try {
      auto *created = new Ref;
      created->raw = raw;
      created->count.store(1, std::memory_order_relaxed);
      ref_ = created;
    } catch (...) {
      close_library(raw);
      throw;
    }
  }
  ~LibraryHandle() { release(); }

  LibraryHandle(const LibraryHandle &other) noexcept : ref_(other.ref_) {
    retain();
  }
  LibraryHandle &operator=(const LibraryHandle &other) noexcept {
    if (this != &other) {
      release();
      ref_ = other.ref_;
      retain();
    }
    return *this;
  }
  LibraryHandle(LibraryHandle &&other) noexcept : ref_(other.ref_) {
    other.ref_ = nullptr;
  }
  LibraryHandle &operator=(LibraryHandle &&other) noexcept {
    if (this != &other) {
      release();
      ref_ = other.ref_;
      other.ref_ = nullptr;
    }
    return *this;
  }

  void *get() const noexcept { return ref_ != nullptr ? ref_->raw : nullptr; }
  explicit operator bool() const noexcept { return get() != nullptr; }

private:
  struct Ref {
    void *raw = nullptr;
    std::atomic<unsigned> count{0};
  };

  void retain() noexcept {
    if (ref_ != nullptr) {
      ref_->count.fetch_add(1, std::memory_order_relaxed);
    }
  }
  void release() noexcept {
    Ref *held = ref_;
    ref_ = nullptr;
    if (held != nullptr &&
        held->count.fetch_sub(1, std::memory_order_acq_rel) == 1) {
      close_library(held->raw);
      delete held;
    }
  }

  Ref *ref_ = nullptr;
};

dyn::Handle as_dyn(void *handle) {
  return reinterpret_cast<dyn::Handle>(handle);
}

struct Loaded {
  LibraryHandle library;
  void *anchor = nullptr;
  EngineCreate create = nullptr;
  EngineDestroy destroy = nullptr;
  EngineForce force = nullptr;
  NativeForce native = nullptr;
  RgpotEnginePot *pot = nullptr;

  ~Loaded() {
    if (pot != nullptr && destroy != nullptr) {
      destroy(pot);
    }
  }

  Loaded() = default;
  Loaded(const Loaded &) = delete;
  Loaded &operator=(const Loaded &) = delete;
  Loaded(Loaded &&other) noexcept
      : library(std::move(other.library)), anchor(other.anchor),
        create(other.create), destroy(other.destroy), force(other.force),
        native(other.native), pot(other.pot) {
    other.anchor = nullptr;
    other.create = nullptr;
    other.destroy = nullptr;
    other.force = nullptr;
    other.native = nullptr;
    other.pot = nullptr;
  }
  Loaded &operator=(Loaded &&other) noexcept {
    if (this != &other) {
      if (pot != nullptr && destroy != nullptr) {
        destroy(pot);
      }
      library = std::move(other.library);
      anchor = other.anchor;
      create = other.create;
      destroy = other.destroy;
      force = other.force;
      native = other.native;
      pot = other.pot;
      other.anchor = nullptr;
      other.create = nullptr;
      other.destroy = nullptr;
      other.force = nullptr;
      other.native = nullptr;
      other.pot = nullptr;
    }
    return *this;
  }
};

struct ResidentEntry {
  LibraryHandle library;
  LibraryHandle pin;
  std::string canonical;
#if !defined(_WIN32)
  dev_t dev = 0;
  ino_t ino = 0;
#endif
};

struct ResidentTable {
  std::mutex mutex;
  std::map<std::string, ResidentEntry> by_canon;
  std::map<std::string, std::string> request_to_canon;
};

ResidentTable &residents() {
  static ResidentTable table;
  return table;
}

std::string canonical_file(const std::string &path) {
#if defined(_WIN32)
  return path;
#else
  char buf[PATH_MAX];
  if (realpath(path.c_str(), buf) == nullptr) {
    return {};
  }
  return buf;
#endif
}

using detail::FileIdentity;
using detail::identify_loaded;
using detail::path_prefix;

void ensure_current(const ResidentEntry &entry, const std::string &path) {
#if defined(_WIN32)
  (void)entry;
  (void)path;
#else
  struct stat st {};
  if (stat(entry.canonical.c_str(), &st) != 0) {
    throw std::runtime_error(path_prefix(path) +
                             "cannot identify the loaded file " +
                             entry.canonical + ": " + std::strerror(errno));
  }
  if (st.st_dev != entry.dev || st.st_ino != entry.ino) {
    throw std::runtime_error(
        path_prefix(path) + entry.canonical +
        " was replaced after it was loaded, and the loaded engine stays "
        "resident; load the new file from another path or restart the process");
  }
  if (!canonical_file(path).empty()) {
    struct stat requested {};
    if (stat(path.c_str(), &requested) != 0) {
      throw std::runtime_error(path_prefix(path) +
                               "cannot identify the requested file: " +
                               std::strerror(errno));
    }
    if (requested.st_dev != entry.dev || requested.st_ino != entry.ino) {
      throw std::runtime_error(
          path_prefix(path) +
          "the requested path was replaced or retargeted after it loaded "
          "another engine; use a new path or restart the process");
    }
  }
#endif
}

// `table.mutex` is held by the caller.
LibraryHandle find_resident(ResidentTable &table, const std::string &path) {
  std::string key = canonical_file(path);
  if (key.empty()) {
    key = path;
  }
  auto direct = table.by_canon.find(key);
  if (direct == table.by_canon.end()) {
    auto alias = table.request_to_canon.find(key);
    if (alias == table.request_to_canon.end()) {
      alias = table.request_to_canon.find(path);
    }
    if (alias == table.request_to_canon.end()) {
      return {};
    }
    direct = table.by_canon.find(alias->second);
    if (direct == table.by_canon.end()) {
      return {};
    }
  }
  ensure_current(direct->second, path);
  return direct->second.library;
}

LibraryHandle pin_resident(const std::string &path) {
#if defined(_WIN32)
  HMODULE pinned = nullptr;
  if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_PIN, path.c_str(),
                         &pinned) == 0 ||
      pinned == nullptr) {
    throw std::runtime_error(path_prefix(path) +
                             "the engine library could not be pinned: " +
                             dyn::error());
  }
  return LibraryHandle(reinterpret_cast<void *>(pinned));
#else
  dlerror();
  void *pin = dlopen(path.c_str(),
                     RTLD_NOW | RTLD_LOCAL | RTLD_NOLOAD | RTLD_NODELETE);
  if (pin == nullptr) {
    throw std::runtime_error(path_prefix(path) +
                             "RTLD_NODELETE could not be set: " +
                             last_dl());
  }
  return LibraryHandle(pin);
#endif
}

// `table.mutex` is held by the caller.
LibraryHandle publish_resident(ResidentTable &table, const std::string &path,
                               const LibraryHandle &opened,
                               const void *anchor) {
  const FileIdentity id = identify_loaded(anchor, path);
  auto existing = table.by_canon.find(id.canonical);
  if (existing != table.by_canon.end()) {
    ensure_current(existing->second, path);
    table.request_to_canon[path] = id.canonical;
    return existing->second.library;
  }
  ResidentEntry entry;
  entry.library = opened;
  entry.pin = pin_resident(path);
  entry.canonical = id.canonical;
#if !defined(_WIN32)
  entry.dev = id.dev;
  entry.ino = id.ino;
#endif
  auto [it, inserted] = table.by_canon.emplace(id.canonical, std::move(entry));
  if (!inserted) {
    throw std::runtime_error(path_prefix(path) +
                             "the engine file was registered twice");
  }
  table.request_to_canon[path] = id.canonical;
  const std::string via = canonical_file(path);
  if (!via.empty()) {
    table.request_to_canon[via] = id.canonical;
  }
  return it->second.library;
}

LibraryHandle open_attempt(const std::string &path) {
#if defined(_WIN32)
  void *raw = reinterpret_cast<void *>(dyn::open(path.c_str()));
#else
  void *raw = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
#endif
  if (raw == nullptr) {
    return {};
  }
  return LibraryHandle(raw);
}

void bind_symbols(Loaded &loaded, bool native) {
  void *handle = loaded.library.get();
  loaded.anchor = dyn::sym(as_dyn(handle), "rgpot_engine_create");
  loaded.create = reinterpret_cast<EngineCreate>(loaded.anchor);
  loaded.destroy =
      dyn::loadSym<EngineDestroy>(as_dyn(handle), "rgpot_engine_destroy");
  if (native) {
    loaded.native =
        dyn::loadSym<NativeForce>(as_dyn(handle), "rgpot_md_native_force");
  } else {
    loaded.force = dyn::loadSym<EngineForce>(as_dyn(handle), "rgpot_engine_force");
  }
}

bool have_symbols(const Loaded &loaded, bool native) {
  return loaded.create != nullptr && loaded.destroy != nullptr &&
         (native ? loaded.native != nullptr : loaded.force != nullptr);
}

Loaded open_library(const std::vector<std::string> &paths, bool native) {
  if (paths.empty()) {
    throw std::runtime_error("no engine library to open");
  }
  ResidentTable &table = residents();
  std::string tried;
  for (const std::string &path : paths) {
    Loaded loaded;
    {
      std::lock_guard<std::mutex> lock(table.mutex);
      loaded.library = find_resident(table, path);
    }
    if (loaded.library) {
      bind_symbols(loaded, native);
      if (!have_symbols(loaded, native)) {
        tried += path + ": missing " +
                 (native ? "rgpot_md_native_force" : "rgpot_engine_force") +
                 "; ";
        continue;
      }
      return loaded;
    }

    // dlopen runs library initializers. Keep them outside the registry lock:
    // an initializer may construct another external potential and consult
    // this table. Publish only after dlopen has completed and every required
    // symbol has been resolved.
    loaded.library = open_attempt(path);
    if (!loaded.library) {
      tried += path + ": " + last_dl() + "; ";
      continue;
    }
    bind_symbols(loaded, native);
    if (!have_symbols(loaded, native)) {
      tried += path + ": missing " +
               (native ? "rgpot_md_native_force" : "rgpot_engine_force") + "; ";
      continue;
    }
    // When another constructor published this file first, the handle opened
    // above is redundant and is closed on assignment. dlclose takes the
    // dynamic loader's lock, which a library initializer may hold while it
    // waits for the table mutex, so the assignment happens after unlocking.
    LibraryHandle resident;
    {
      std::lock_guard<std::mutex> lock(table.mutex);
      resident = publish_resident(table, path, loaded.library, loaded.anchor);
    }
    loaded.library = std::move(resident);
    return loaded;
  }
  throw std::runtime_error("engine library was not opened: " + tried);
}

std::vector<std::string> lammps_paths(const std::string &explicit_path) {
  std::vector<std::string> out;
  if (!explicit_path.empty()) {
    out.push_back(explicit_path);
    return out;
  }
  if (const char *env = std::getenv("RGPOT_LAMMPS_ENGINE")) {
    if (env[0] != '\0') {
      out.emplace_back(env);
    }
  }
  if (const char *env = std::getenv("LAMMPS_ENGINE")) {
    if (env[0] != '\0') {
      out.emplace_back(env);
    }
  }
  add_soname(out, "lammps_engine");
  add_soname(out, "lj_engine");
  return out;
}

void create_engine(Loaded &loaded, const void *config, size_t bytes) {
  char err[256] = {};
  loaded.pot = loaded.create(config, bytes, err, sizeof(err));
  if (loaded.pot == nullptr) {
    throw std::runtime_error(err[0] != '\0' ? err
                                            : "rgpot_engine_create failed");
  }
}

void check_geometry(const ForceInput &in, const ForceOut *out) {
  if (out == nullptr || out->F == nullptr || in.box == nullptr ||
      (in.nAtoms > 0 && (in.pos == nullptr || in.atmnrs == nullptr))) {
    throw std::invalid_argument("potential force buffers are incomplete");
  }
}

constexpr size_t kCapnpWordBytes = 8;

#if RGPOT_HAS_LAMMPC_PROFILE
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

void require_lammps_config(const std::vector<unsigned char> &bytes) {
  try {
    ::capnp::FlatArrayMessageReader reader(
        kj::arrayPtr(reinterpret_cast<const ::capnp::word *>(bytes.data()),
                     bytes.size() / sizeof(::capnp::word)));
    if (reader.getRoot<::PotentialConfig>().which() !=
        ::PotentialConfig::LAMMPS)
      throw std::invalid_argument(
          "lammpc config bytes are not a PotentialConfig with the lammps arm");
  } catch (const kj::Exception &ex) {
    throw std::invalid_argument(
        std::string("lammpc config bytes are not a flat PotentialConfig: ") +
        ex.getDescription().cStr());
  }
}

std::string profile_message(const rgpot::abi::ProfileResult &result) {
  const size_t n = strnlen(result.message, sizeof(result.message));
  return std::string(result.message, n);
}

bool declares_stress(const rgpot::abi::ProfileLoader &loader) {
  const auto bytes = loader.capabilities();
  std::vector<::capnp::word> storage(bytes.size() / sizeof(::capnp::word));
  std::memcpy(storage.data(), bytes.data(), bytes.size());
  ::capnp::FlatArrayMessageReader reader(
      kj::arrayPtr(storage.data(), storage.size()));
  for (const auto op : reader.getRoot<::Capabilities>().getOperations()) {
    if (op == ::Capabilities::Operation::STRESS)
      return true;
  }
  return false;
}
#endif

// Bump when the numerics either LammpsPot mode returns change.
constexpr uint64_t kLammpsKernelVersion = 1;

#if RGPOT_HAS_LAMMPC_PROFILE
// The key follows the backend that loaded, not the path that was asked for:
// the canonical path of the mapped file plus the version and ABI revision the
// library reports. Two requests that resolve to one file share a key, and a
// path that now points at another file gets a new one.
uint64_t lammpc_params_key(const rgpot::abi::ProfileLoader &loader,
                           const std::string &requested,
                           const std::vector<unsigned char> &config) {
  const std::string anchor_name = loader.prefix() + "_capabilities_result";
  void *anchor = loader.lib().sym_optional<void *>(anchor_name.c_str());
  if (anchor == nullptr)
    throw std::runtime_error("missing " + anchor_name);
  const FileIdentity backend = identify_loaded(
      anchor, requested.empty() ? loader.loaded_path() : requested);
  const char *version = loader.version ? loader.version() : nullptr;
  Fnv1a fp;
  fp.str("lammpc");
  fp.u64(kLammpsKernelVersion);
  fp.str(backend.canonical);
  fp.i64(loader.abi_version());
  fp.str(version != nullptr ? version : "");
  fp.u64(config.size());
  fp.bytes(config.data(), config.size());
  return fp.h;
}
#endif

// A lammpc backend keeps process-global state that finalize releases. Every
// LammpsPot that may have created a session holds its backend, keyed by the
// resolved finalize symbol, and the last holder to leave calls finalize, so
// destroying one pot leaves a live peer's backend initialized.
using ProfileFinalizeFn = rgpot::abi::ProfileLoader::FinalizeFn;

struct ProfileHolds {
  std::mutex mutex;
  std::map<ProfileFinalizeFn, size_t> count;
};

ProfileHolds &profile_holds() {
  // Leaked, so a pot destroyed during static teardown still finds it.
  static auto *holds = new ProfileHolds;
  return *holds;
}

void hold_profile(ProfileFinalizeFn finalize) {
  auto &holds = profile_holds();
  std::lock_guard<std::mutex> lock(holds.mutex);
  ++holds.count[finalize];
}

// finalize runs under the lock, so a new holder cannot create a session
// while the backend is still tearing down.
void release_profile(ProfileFinalizeFn finalize) {
  auto &holds = profile_holds();
  std::lock_guard<std::mutex> lock(holds.mutex);
  const auto it = holds.count.find(finalize);
  if (it == holds.count.end() || --it->second != 0)
    return;
  holds.count.erase(it);
  finalize();
}

} // namespace

struct LammpsPot::Impl {
  enum class Mode { LegacyLJ, LammpcProfile };
  Mode mode = Mode::LegacyLJ;
  Loaded legacy_engine;
  rgpot::abi::ProfileLoader profile_loader;
  std::vector<unsigned char> profile_config;
  void *profile_session = nullptr;
  bool profile_held = false;
  bool profile_stress = false;
  uint64_t params_key = 0;
  mutable std::mutex mutex;

  // Safe to call twice: the failed constructor and the destructor both do.
  void release_profile_session() {
    if (profile_session) {
      profile_loader.session_destroy(profile_session);
      profile_session = nullptr;
    }
    if (profile_held) {
      profile_held = false;
      release_profile(profile_loader.finalize);
    }
  }

  ~Impl() { release_profile_session(); }
};

LammpsPot::LammpsPot(const ExternalLjConfig &config)
    : Potential(PotType::Lammps), impl_(std::make_unique<Impl>()),
      config_(config) {
  require_lj(config_);
  impl_->mode = Impl::Mode::LegacyLJ;
  impl_->legacy_engine = open_library(lammps_paths(config_.engine_path), false);
  const double packed[3] = {config_.u0, config_.cutoff, config_.psi};
  create_engine(impl_->legacy_engine, packed, sizeof(packed));
  Fnv1a fp;
  fp.str("lammps-lj");
  fp.u64(kLammpsKernelVersion);
  fp.str(identify_loaded(impl_->legacy_engine.anchor, config_.engine_path)
             .canonical);
  fp.f64(config_.u0);
  fp.f64(config_.cutoff);
  fp.f64(config_.psi);
  impl_->params_key = fp.h;
}

LammpsPot::LammpsPot(const void *config_capnp, size_t config_capnp_size,
                     const std::string &engine_path)
    : Potential(PotType::Lammps), impl_(std::make_unique<Impl>()), config_{} {
  if (!config_capnp || config_capnp_size == 0)
    throw std::invalid_argument("lammpc config bytes are required");
  if (config_capnp_size % kCapnpWordBytes != 0)
    throw std::invalid_argument("lammpc config bytes must be word-aligned");
#if !RGPOT_HAS_LAMMPC_PROFILE
  (void)engine_path;
  throw std::runtime_error(
      "lammpc profile mode needs an RPC-capable build (Potentials.capnp.h)");
#else
  impl_->mode = Impl::Mode::LammpcProfile;
  impl_->profile_config.resize(config_capnp_size);
  std::memcpy(impl_->profile_config.data(), config_capnp, config_capnp_size);
  require_lammps_config(impl_->profile_config);
  const std::string backend =
      engine_path.empty() ? std::string("lammpc")
                          : "lammpc engine_path '" + engine_path + "'";
  try {
    rgpot::abi::checked_load(impl_->profile_loader, "lammpc", engine_path);
    impl_->profile_stress = declares_stress(impl_->profile_loader);
    impl_->params_key = lammpc_params_key(impl_->profile_loader, engine_path,
                                          impl_->profile_config);
  } catch (const std::exception &ex) {
    throw std::runtime_error(backend + ": " + ex.what());
  }
  if (impl_->profile_loader.available() != 1)
    throw std::runtime_error(backend + " reports unavailable");
  hold_profile(impl_->profile_loader.finalize);
  impl_->profile_held = true;
  impl_->profile_session = impl_->profile_loader.session_create_from_config(
      impl_->profile_config.data(), impl_->profile_config.size());
  if (!impl_->profile_session) {
    impl_->release_profile_session();
    throw std::runtime_error("lammpc_session_create_from_config failed");
  }
#endif
}

LammpsPot::~LammpsPot() = default;
LammpsPot::LammpsPot(LammpsPot &&) noexcept = default;
LammpsPot &LammpsPot::operator=(LammpsPot &&) noexcept = default;

uint64_t LammpsPot::paramsKey() const noexcept {
  return impl_ ? impl_->params_key : 0;
}

PotCaps LammpsPot::caps() const noexcept {
  const bool stress = impl_ && impl_->mode == Impl::Mode::LammpcProfile &&
                      impl_->profile_stress;
  return {.reentrancy = Reentrancy::PerInstance,
          .periodic = true,
          .stress = stress};
}

void LammpsPot::forceImpl(const ForceInput &in, ForceOut *out) const {
  check_geometry(in, out);
  std::lock_guard<std::mutex> lock(impl_->mutex);
  if (impl_->mode == Impl::Mode::LegacyLJ) {
    double variance = 0.0;
    const int status = impl_->legacy_engine.force(
        impl_->legacy_engine.pot, static_cast<long>(in.nAtoms), in.pos,
        in.atmnrs, out->F, &out->energy, &variance, in.box, nullptr, nullptr);
    if (status != 0)
      throw std::runtime_error("rgpot_engine_force failed");
    out->variance = variance;
    out->has_stress = 0;
    return;
  }

#if !RGPOT_HAS_LAMMPC_PROFILE
  throw std::runtime_error(
      "lammpc profile mode needs an RPC-capable build (Potentials.capnp.h)");
#else
  const auto force_input_words = serialize_force_input(in);
  const void *step_ptr = force_input_words.data();
  const size_t step_bytes = force_input_words.size() * sizeof(::capnp::word);
  const size_t result_size =
      impl_->profile_loader.potential_result_size_for_force_input(step_ptr,
                                                                  step_bytes);
  if (result_size == 0)
    throw std::runtime_error("lammpc refused PotentialResult sizing");

  std::vector<unsigned char> out_bytes(result_size);
  size_t written = 0;
  const auto result = impl_->profile_loader.session_calculate_result(
      impl_->profile_session, step_ptr, step_bytes, out_bytes.data(),
      out_bytes.size(), &written);
  if (!result.ok)
    throw std::runtime_error("lammpc failed: " + profile_message(result));
  if (written == 0 || written > out_bytes.size() ||
      (written % sizeof(::capnp::word)) != 0)
    throw std::runtime_error("lammpc returned invalid PotentialResult");

  auto words = kj::arrayPtr(
      reinterpret_cast<const ::capnp::word *>(out_bytes.data()),
      written / sizeof(::capnp::word));
  ::capnp::FlatArrayMessageReader reader(words);
  const auto potential_result = reader.getRoot<::PotentialResult>();
  const auto forces = potential_result.getForces();
  if (forces.size() != in.nAtoms * 3u)
    throw std::runtime_error("lammpc returned wrong force count");
  out->energy = potential_result.getEnergy();
  out->variance = 0.0;
  for (unsigned int i = 0; i < forces.size(); ++i)
    out->F[i] = forces[i];
  out->has_stress = 0;
  const auto stress = potential_result.getStress();
  if (stress.size() != 0 && stress.size() != 9)
    throw std::runtime_error("lammpc returned " +
                             std::to_string(stress.size()) +
                             " stress values, not 9");
  if (impl_->profile_stress) {
    if (stress.size() != 9)
      throw std::runtime_error(
          "lammpc declared the stress operation but returned no stress");
    for (unsigned int i = 0; i < 9; ++i)
      out->stress[i] = stress[i];
    out->has_stress = 1;
  }
#endif
}

struct NativeState {
  explicit NativeState(const ExternalLjConfig &config)
      : host(LJConfig{config.u0, config.cutoff, config.psi}) {}

  Loaded engine;
  LJPot host;
  bool use_native = false;
  mutable std::mutex mutex;
  mutable std::vector<double> positions;
  mutable std::vector<double> forces;
  mutable std::array<double, 9> box{};
};

void load_native(NativeState &state, const ExternalLjConfig &config,
                 double u0_kj, double cutoff_nm, double psi_nm) {
  if (config.engine_path.empty()) {
    return;
  }
  state.engine = open_library({config.engine_path}, true);
  const double packed[3] = {u0_kj, cutoff_nm, psi_nm};
  create_engine(state.engine, packed, sizeof(packed));
  state.use_native = true;
}

void eval_native(NativeState &state, const ForceInput &in, ForceOut *out,
                 double u0_host_unused) {
  (void)u0_host_unused;
  const size_t n3 = in.nAtoms * 3;
  state.positions.resize(n3);
  state.forces.resize(n3);
  for (size_t i = 0; i < n3; ++i) {
    state.positions[i] = in.pos[i] * md::kNmPerAngstrom;
  }
  for (int i = 0; i < 9; ++i) {
    state.box[static_cast<size_t>(i)] = in.box[i] * md::kNmPerAngstrom;
  }
  double energy_kj = 0.0;
  double variance = 0.0;
  const int status = state.engine.native(
      state.engine.pot, static_cast<long>(in.nAtoms), state.positions.data(),
      in.atmnrs, state.forces.data(), &energy_kj, &variance, state.box.data());
  if (status != 0) {
    throw std::runtime_error("rgpot_md_native_force failed");
  }
  out->energy = energy_kj * md::kEvPerKj;
  out->variance = variance * md::kEvPerKj * md::kEvPerKj;
  for (size_t i = 0; i < n3; ++i) {
    out->F[i] = state.forces[i] * md::kEvAPerKjNm;
  }
  out->has_stress = 0;
}

struct GromacsPot::Impl {
  explicit Impl(const ExternalLjConfig &config) : state(config) {}
  NativeState state;
};

GromacsPot::GromacsPot(const ExternalLjConfig &config)
    : Potential(PotType::Gromacs), impl_(std::make_unique<Impl>(config)),
      config_(config) {
  require_lj(config_);
  native_u0_ = config_.u0 * md::kKjPerEv;
  native_cutoff_ = config_.cutoff * md::kNmPerAngstrom;
  native_psi_ = config_.psi * md::kNmPerAngstrom;
  load_native(impl_->state, config_, native_u0_, native_cutoff_, native_psi_);
}

GromacsPot::~GromacsPot() = default;
GromacsPot::GromacsPot(GromacsPot &&) noexcept = default;
GromacsPot &GromacsPot::operator=(GromacsPot &&) noexcept = default;

PotCaps GromacsPot::caps() const noexcept {
  const bool native = impl_ && impl_->state.use_native;
  return {.reentrancy = Reentrancy::PerInstance,
          .periodic = true,
          .stress = !native};
}

void GromacsPot::forceImpl(const ForceInput &in, ForceOut *out) const {
  check_geometry(in, out);
  std::lock_guard<std::mutex> lock(impl_->state.mutex);
  if (impl_->state.use_native) {
    eval_native(impl_->state, in, out, config_.u0);
    return;
  }
  impl_->state.host.forceImpl(in, out);
}

struct OpenMMPot::Impl {
  explicit Impl(const ExternalLjConfig &config) : state(config) {}
  NativeState state;
};

OpenMMPot::OpenMMPot(const ExternalLjConfig &config)
    : Potential(PotType::OpenMM), impl_(std::make_unique<Impl>(config)),
      config_(config) {
  require_lj(config_);
  native_u0_ = config_.u0 * md::kKjPerEv;
  native_cutoff_ = config_.cutoff * md::kNmPerAngstrom;
  native_psi_ = config_.psi * md::kNmPerAngstrom;
  load_native(impl_->state, config_, native_u0_, native_cutoff_, native_psi_);
}

OpenMMPot::~OpenMMPot() = default;
OpenMMPot::OpenMMPot(OpenMMPot &&) noexcept = default;
OpenMMPot &OpenMMPot::operator=(OpenMMPot &&) noexcept = default;

PotCaps OpenMMPot::caps() const noexcept {
  const bool native = impl_ && impl_->state.use_native;
  return {.reentrancy = Reentrancy::PerInstance,
          .periodic = true,
          .stress = !native};
}

void OpenMMPot::forceImpl(const ForceInput &in, ForceOut *out) const {
  check_geometry(in, out);
  std::lock_guard<std::mutex> lock(impl_->state.mutex);
  if (impl_->state.use_native) {
    eval_native(impl_->state, in, out, config_.u0);
    return;
  }
  impl_->state.host.forceImpl(in, out);
}

std::unique_ptr<LammpsPot>
rgpot_lammps_potential(const ExternalLjConfig &config) {
  return std::make_unique<LammpsPot>(config);
}

std::unique_ptr<LammpsPot>
rgpot_lammpc_potential(const void *config_capnp, size_t config_capnp_size,
                       const std::string &engine_path) {
  return std::make_unique<LammpsPot>(config_capnp, config_capnp_size,
                                     engine_path);
}

std::unique_ptr<GromacsPot>
rgpot_gromacs_potential(const ExternalLjConfig &config) {
  return std::make_unique<GromacsPot>(config);
}

std::unique_ptr<OpenMMPot>
rgpot_openmm_potential(const ExternalLjConfig &config) {
  return std::make_unique<OpenMMPot>(config);
}

} // namespace rgpot

struct rgpot_external_potential {
  std::unique_ptr<rgpot::PotentialBase> pot;
};

struct rgpot_md_session {
  rgpot::plugin::dynlib::Handle library = nullptr;
  void *ctx = nullptr;
  rgpot_md_session_ops ops{};
  size_t n_atoms = 0;

  ~rgpot_md_session() {
    if (ctx != nullptr && ops.destroy != nullptr) {
      ops.destroy(ctx);
      ctx = nullptr;
    }
    if (library != nullptr) {
      rgpot::plugin::dynlib::close(library);
      library = nullptr;
    }
  }
};

namespace {

void clear_err(char *err, size_t err_len) {
  if (err != nullptr && err_len > 0) {
    err[0] = '\0';
  }
}

void write_err(char *err, size_t err_len, const char *text) {
  if (err == nullptr || err_len == 0) {
    return;
  }
  std::snprintf(err, err_len, "%s", text != nullptr ? text : "unknown error");
}

std::vector<std::string> md_engine_paths(const char *engine,
                                         const char *engine_path) {
  std::vector<std::string> out;
  if (engine_path != nullptr && engine_path[0] != '\0') {
    out.emplace_back(engine_path);
    return out;
  }
  if (engine == nullptr || engine[0] == '\0') {
    throw std::invalid_argument("MD session needs an engine name or path");
  }
  std::string lower;
  std::string upper;
  for (const unsigned char ch : std::string(engine)) {
    if (!std::isalnum(ch) && ch != '_') {
      throw std::invalid_argument("MD engine name must be alphanumeric");
    }
    lower.push_back(static_cast<char>(std::tolower(ch)));
    upper.push_back(static_cast<char>(std::toupper(ch)));
  }
  const std::string env_rgpot = "RGPOT_" + upper + "_ENGINE";
  if (const char *env = std::getenv(env_rgpot.c_str())) {
    if (env[0] != '\0') {
      out.emplace_back(env);
    }
  }
  const std::string env_plain = upper + "_ENGINE";
  if (const char *env = std::getenv(env_plain.c_str())) {
    if (env[0] != '\0') {
      out.emplace_back(env);
    }
  }
  rgpot::add_soname(out, (lower + "_engine").c_str());
  return out;
}

// Everything up to and including destroy is required; later members are
// optional, so a vtable built against an older header still loads.
bool valid_md_ops(const rgpot_md_session_ops *ops) {
  return ops != nullptr &&
         ops->struct_size >=
             offsetof(rgpot_md_session_ops, destroy) + sizeof(ops->destroy) &&
         ops->set_positions != nullptr && ops->set_velocities != nullptr &&
         ops->step_langevin != nullptr && ops->step_nve != nullptr &&
         ops->get_state != nullptr && ops->get_masses != nullptr &&
         ops->destroy != nullptr;
}

// A vtable from an older engine is shorter than ours: take the members it
// declares and leave the rest null, so RGPOT_MD_OPS_HAS sees them as absent.
rgpot_md_session_ops copy_md_ops(const rgpot_md_session_ops *ops) {
  rgpot_md_session_ops out{};
  std::memcpy(&out, ops, std::min(ops->struct_size, sizeof(out)));
  out.struct_size = std::min(ops->struct_size, sizeof(out));
  return out;
}

bool finite_array(const double *values, size_t count) {
  if (values == nullptr) {
    return false;
  }
  for (size_t i = 0; i < count; ++i) {
    if (!std::isfinite(values[i])) {
      return false;
    }
  }
  return true;
}

rgpot::ExternalLjConfig lj_config(const char *engine_path, double u0,
                                  double cutoff, double psi) {
  rgpot::ExternalLjConfig config;
  config.u0 = u0;
  config.cutoff = cutoff;
  config.psi = psi;
  if (engine_path != nullptr) {
    config.engine_path = engine_path;
  }
  return config;
}

template <typename Make>
rgpot_external_potential *adopt(Make &&make, char *err, size_t err_len) {
  try {
    std::unique_ptr<rgpot::PotentialBase> pot = make();
    return new rgpot_external_potential{std::move(pot)};
  } catch (const std::exception &ex) {
    write_err(err, err_len, ex.what());
    return nullptr;
  }
}

} // namespace

extern "C" {

double rgpot_md_kj_per_ev(void) { return rgpot::md::kKjPerEv; }

double rgpot_md_nm_per_angstrom(void) { return rgpot::md::kNmPerAngstrom; }

rgpot_external_potential *rgpot_lammps_potential(const char *engine_path,
                                                 double u0, double cutoff,
                                                 double psi, char *err,
                                                 size_t err_len) {
  return adopt(
      [&] {
        return rgpot::rgpot_lammps_potential(
            lj_config(engine_path, u0, cutoff, psi));
      },
      err, err_len);
}

rgpot_external_potential *
rgpot_lammpc_potential(const char *engine_path, const void *config_capnp,
                       size_t config_capnp_size, char *err, size_t err_len) {
  return adopt(
      [&] {
        const std::string path =
            (engine_path != nullptr) ? std::string(engine_path) : std::string{};
        return rgpot::rgpot_lammpc_potential(config_capnp, config_capnp_size,
                                             path);
      },
      err, err_len);
}

rgpot_external_potential *rgpot_gromacs_potential(const char *engine_path,
                                                  double u0, double cutoff,
                                                  double psi, char *err,
                                                  size_t err_len) {
  return adopt(
      [&] {
        return rgpot::rgpot_gromacs_potential(
            lj_config(engine_path, u0, cutoff, psi));
      },
      err, err_len);
}

rgpot_external_potential *rgpot_openmm_potential(const char *engine_path,
                                                 double u0, double cutoff,
                                                 double psi, char *err,
                                                 size_t err_len) {
  return adopt(
      [&] {
        return rgpot::rgpot_openmm_potential(
            lj_config(engine_path, u0, cutoff, psi));
      },
      err, err_len);
}

void rgpot_external_potential_free(rgpot_external_potential *pot) {
  delete pot;
}

int rgpot_external_potential_energy_forces(rgpot_external_potential *pot,
                                           const double *xyz,
                                           const int *atomic_numbers, size_t n,
                                           const double *box9, double *forces,
                                           double *energy, char *err,
                                           size_t err_len) {
  if (pot == nullptr || pot->pot == nullptr || energy == nullptr ||
      box9 == nullptr ||
      (n > 0 && (xyz == nullptr || atomic_numbers == nullptr ||
                 forces == nullptr))) {
    write_err(err, err_len, "energy_forces needs a potential and a geometry");
    return 1;
  }
  try {
    rgpot::types::AtomMatrix positions(n, 3);
    if (n > 0) {
      std::memcpy(positions.data(), xyz, n * 3 * sizeof(double));
    }
    std::vector<int> types(atomic_numbers, atomic_numbers + n);
    std::array<std::array<double, 3>, 3> box{};
    std::memcpy(&box, box9, sizeof(box));
    const auto result = (*pot->pot)(positions, types, box);
    *energy = std::get<0>(result);
    if (n > 0) {
      std::memcpy(forces, std::get<1>(result).data(), n * 3 * sizeof(double));
    }
    return 0;
  } catch (const std::exception &ex) {
    write_err(err, err_len, ex.what());
    return 1;
  } catch (...) {
    write_err(err, err_len, "unknown C++ exception");
    return 1;
  }
}

rgpot_md_session *rgpot_md_session_from_ops(void *ctx,
                                            const rgpot_md_session_ops *ops,
                                            size_t n_atoms, char *err,
                                            size_t err_len) {
  clear_err(err, err_len);
  try {
    if (ctx == nullptr || !valid_md_ops(ops) || n_atoms == 0 ||
        n_atoms > std::numeric_limits<size_t>::max() / 3) {
      throw std::invalid_argument("invalid MD session vtable or atom count");
    }
    auto out = std::make_unique<rgpot_md_session>();
    out->ctx = ctx;
    out->ops = copy_md_ops(ops);
    out->n_atoms = n_atoms;
    return out.release();
  } catch (const std::exception &ex) {
    write_err(err, err_len, ex.what());
    return nullptr;
  } catch (...) {
    write_err(err, err_len, "unknown C++ exception");
    return nullptr;
  }
}

rgpot_md_session *rgpot_md_session_create(const char *engine,
                                          const char *engine_path,
                                          const void *spec, size_t spec_len,
                                          const char *options, char *err,
                                          size_t err_len) {
  clear_err(err, err_len);
  try {
    if (spec == nullptr && spec_len > 0) {
      throw std::invalid_argument("MD session spec buffer is null");
    }
    const auto paths = md_engine_paths(engine, engine_path);
    std::string tried;
    for (const std::string &path : paths) {
      rgpot::plugin::dynlib::Handle handle =
          rgpot::plugin::dynlib::open(path.c_str());
      if (handle == nullptr) {
        tried += path + ": " + rgpot::last_dl() + "; ";
        continue;
      }
      auto create_fn =
          rgpot::plugin::dynlib::loadSym<rgpot_md_session_engine_create_fn>(
              handle, "rgpot_md_session_engine_create");
      if (create_fn == nullptr) {
        tried += path + ": missing rgpot_md_session_engine_create; ";
        rgpot::plugin::dynlib::close(handle);
        continue;
      }
      rgpot_md_session_ops ops{};
      void *ctx = nullptr;
      size_t n_atoms = 0;
      char inner_err[512] = {};
      if (create_fn(spec, spec_len, options, &ops, &ctx, &n_atoms, inner_err,
                    sizeof(inner_err)) != 0) {
        rgpot::plugin::dynlib::close(handle);
        throw std::runtime_error(inner_err[0] != '\0'
                                     ? inner_err
                                     : "rgpot_md_session_engine_create failed");
      }
      if (ctx == nullptr || !valid_md_ops(&ops) || n_atoms == 0 ||
          n_atoms > std::numeric_limits<size_t>::max() / 3) {
        if (ctx != nullptr && ops.destroy != nullptr) {
          ops.destroy(ctx);
        }
        rgpot::plugin::dynlib::close(handle);
        throw std::runtime_error("engine returned an invalid MD session");
      }
      auto out = std::make_unique<rgpot_md_session>();
      out->library = handle;
      out->ctx = ctx;
      out->ops = copy_md_ops(&ops);
      out->n_atoms = n_atoms;
      return out.release();
    }
    throw std::runtime_error("MD engine library was not opened: " + tried);
  } catch (const std::exception &ex) {
    write_err(err, err_len, ex.what());
    return nullptr;
  } catch (...) {
    write_err(err, err_len, "unknown C++ exception");
    return nullptr;
  }
}

size_t rgpot_md_session_n_atoms(const rgpot_md_session *session) {
  return session != nullptr ? session->n_atoms : 0;
}

int rgpot_md_session_get_masses(rgpot_md_session *session, double *masses,
                                size_t n_atoms, char *err, size_t err_len) {
  clear_err(err, err_len);
  try {
    if (session == nullptr || masses == nullptr || n_atoms != session->n_atoms) {
      throw std::invalid_argument("MD mass output buffer has wrong length");
    }
    return session->ops.get_masses(session->ctx, masses, n_atoms, err, err_len);
  } catch (const std::exception &ex) {
    write_err(err, err_len, ex.what());
    return 1;
  } catch (...) {
    write_err(err, err_len, "unknown C++ exception");
    return 1;
  }
}

int rgpot_md_session_set_positions(rgpot_md_session *session,
                                   const double *positions, size_t n3,
                                   char *err, size_t err_len) {
  clear_err(err, err_len);
  try {
    if (session == nullptr || n3 != 3 * session->n_atoms ||
        !finite_array(positions, n3)) {
      throw std::invalid_argument("MD positions must be finite 3*n_atoms data");
    }
    return session->ops.set_positions(session->ctx, positions, n3, err,
                                      err_len);
  } catch (const std::exception &ex) {
    write_err(err, err_len, ex.what());
    return 1;
  } catch (...) {
    write_err(err, err_len, "unknown C++ exception");
    return 1;
  }
}

int rgpot_md_session_set_velocities(rgpot_md_session *session,
                                    const double *velocities, size_t n3,
                                    char *err, size_t err_len) {
  clear_err(err, err_len);
  try {
    if (session == nullptr || n3 != 3 * session->n_atoms ||
        !finite_array(velocities, n3)) {
      throw std::invalid_argument(
          "MD velocities must be finite 3*n_atoms data");
    }
    return session->ops.set_velocities(session->ctx, velocities, n3, err,
                                       err_len);
  } catch (const std::exception &ex) {
    write_err(err, err_len, ex.what());
    return 1;
  } catch (...) {
    write_err(err, err_len, "unknown C++ exception");
    return 1;
  }
}

int rgpot_md_session_step_langevin(
    rgpot_md_session *session, double kT, double friction, double time_step,
    const double *plane_point, const double *plane_normal,
    const double *wall_points, const double *wall_normals, size_t n_walls,
    size_t n3, size_t steps, uint64_t seed, uint64_t *wall_hits, char *err,
    size_t err_len) {
  clear_err(err, err_len);
  try {
    if (session == nullptr || n3 != 3 * session->n_atoms) {
      throw std::invalid_argument("MD Langevin session or dimension mismatch");
    }
    if (!std::isfinite(kT) || kT <= 0.0 || !std::isfinite(friction) ||
        friction < 0.0 || !std::isfinite(time_step) || time_step <= 0.0) {
      throw std::invalid_argument(
          "MD Langevin kT, friction, and time_step must be valid");
    }
    if (!finite_array(plane_point, n3) || !finite_array(plane_normal, n3)) {
      throw std::invalid_argument(
          "MD Langevin plane_point and plane_normal must be finite");
    }
    if (n_walls > 0 &&
        (n_walls > std::numeric_limits<size_t>::max() / n3 ||
         !finite_array(wall_points, n_walls * n3) ||
         !finite_array(wall_normals, n_walls * n3))) {
      throw std::invalid_argument(
          "MD Langevin wall arrays must be finite n_walls*n3 buffers");
    }
    return session->ops.step_langevin(
        session->ctx, kT, friction, time_step, plane_point, plane_normal,
        wall_points, wall_normals, n_walls, n3, steps, seed, wall_hits, err,
        err_len);
  } catch (const std::exception &ex) {
    write_err(err, err_len, ex.what());
    return 1;
  } catch (...) {
    write_err(err, err_len, "unknown C++ exception");
    return 1;
  }
}

int rgpot_md_session_step_nve(rgpot_md_session *session, double time_step,
                              size_t steps, char *err, size_t err_len) {
  clear_err(err, err_len);
  try {
    if (session == nullptr || !std::isfinite(time_step) || time_step <= 0.0) {
      throw std::invalid_argument(
          "MD NVE needs a valid session and positive finite time_step");
    }
    return session->ops.step_nve(session->ctx, time_step, steps, err, err_len);
  } catch (const std::exception &ex) {
    write_err(err, err_len, ex.what());
    return 1;
  } catch (...) {
    write_err(err, err_len, "unknown C++ exception");
    return 1;
  }
}

int rgpot_md_session_get_state(rgpot_md_session *session, double *positions,
                               double *velocities, double *forces, size_t n3,
                               double *potential_energy, double *kinetic_energy,
                               char *err, size_t err_len) {
  clear_err(err, err_len);
  try {
    if (session == nullptr) {
      throw std::invalid_argument("null MD session");
    }
    if (n3 != 3 * session->n_atoms) {
      throw std::invalid_argument("MD state buffer length differs from 3*N");
    }
    return session->ops.get_state(session->ctx, positions, velocities, forces,
                                  n3, potential_energy, kinetic_energy, err,
                                  err_len);
  } catch (const std::exception &ex) {
    write_err(err, err_len, ex.what());
    return 1;
  } catch (...) {
    write_err(err, err_len, "unknown C++ exception");
    return 1;
  }
}

int rgpot_md_session_set_periodic_box(rgpot_md_session *session,
                                      const double *box9, char *err,
                                      size_t err_len) {
  clear_err(err, err_len);
  try {
    if (session == nullptr || !finite_array(box9, 9)) {
      throw std::invalid_argument("MD periodic box must be nine finite values");
    }
    if (!RGPOT_MD_OPS_HAS(&session->ops, set_periodic_box)) {
      throw std::runtime_error("MD engine does not support set_periodic_box");
    }
    return session->ops.set_periodic_box(session->ctx, box9, err, err_len);
  } catch (const std::exception &ex) {
    write_err(err, err_len, ex.what());
    return 1;
  } catch (...) {
    write_err(err, err_len, "unknown C++ exception");
    return 1;
  }
}

int rgpot_md_session_get_periodic_box(rgpot_md_session *session, double *box9,
                                      char *err, size_t err_len) {
  clear_err(err, err_len);
  try {
    if (session == nullptr || box9 == nullptr) {
      throw std::invalid_argument("MD periodic box output buffer is null");
    }
    if (!RGPOT_MD_OPS_HAS(&session->ops, get_periodic_box)) {
      throw std::runtime_error("MD engine does not support get_periodic_box");
    }
    return session->ops.get_periodic_box(session->ctx, box9, err, err_len);
  } catch (const std::exception &ex) {
    write_err(err, err_len, ex.what());
    return 1;
  } catch (...) {
    write_err(err, err_len, "unknown C++ exception");
    return 1;
  }
}

int rgpot_md_session_initialize(rgpot_md_session *session, char *err,
                                size_t err_len) {
  clear_err(err, err_len);
  try {
    if (session == nullptr) {
      throw std::invalid_argument("null MD session");
    }
    // An engine that evaluates forces on demand has nothing to prepare.
    if (!RGPOT_MD_OPS_HAS(&session->ops, initialize)) {
      return 0;
    }
    return session->ops.initialize(session->ctx, err, err_len);
  } catch (const std::exception &ex) {
    write_err(err, err_len, ex.what());
    return 1;
  } catch (...) {
    write_err(err, err_len, "unknown C++ exception");
    return 1;
  }
}

int rgpot_md_session_run_langevin(rgpot_md_session *session,
                                  const rgpot_md_langevin_request *request,
                                  rgpot_md_langevin_result *result, char *err,
                                  size_t err_len) {
  clear_err(err, err_len);
  try {
    if (session == nullptr || request == nullptr || result == nullptr) {
      throw std::invalid_argument("MD run_langevin needs a session, request "
                                  "and result");
    }
    if (request->struct_size < sizeof(rgpot_md_langevin_request) ||
        result->struct_size <
            offsetof(rgpot_md_langevin_result, constraint_force_len) +
                sizeof(size_t)) {
      throw std::invalid_argument(
          "MD run_langevin request or result struct_size is too small");
    }
    const size_t n3 = request->n3;
    if (n3 != 3 * session->n_atoms) {
      throw std::invalid_argument("MD Langevin dimension mismatch");
    }
    if (!std::isfinite(request->kT) || request->kT <= 0.0 ||
        !std::isfinite(request->friction) || request->friction < 0.0 ||
        !std::isfinite(request->time_step) || request->time_step <= 0.0) {
      throw std::invalid_argument(
          "MD Langevin kT, friction, and time_step must be valid");
    }
    if (request->wall_mode != RGPOT_MD_WALL_REFLECT &&
        request->wall_mode != RGPOT_MD_WALL_REJECT) {
      throw std::invalid_argument("MD Langevin wall_mode is not recognised");
    }
    const bool has_plane =
        request->plane_point != nullptr || request->plane_normal != nullptr;
    if (has_plane && (!finite_array(request->plane_point, n3) ||
                      !finite_array(request->plane_normal, n3))) {
      throw std::invalid_argument(
          "MD Langevin plane_point and plane_normal must both be finite "
          "3*n_atoms data, or both null");
    }
    if (request->n_walls > 0 &&
        (request->n_walls > std::numeric_limits<size_t>::max() / n3 ||
         !finite_array(request->wall_points, request->n_walls * n3) ||
         !finite_array(request->wall_normals, request->n_walls * n3))) {
      throw std::invalid_argument(
          "MD Langevin wall arrays must be finite n_walls*n3 buffers");
    }
    if (has_plane && result->constraint_force != nullptr &&
        result->constraint_force_len < request->steps) {
      throw std::invalid_argument(
          "MD Langevin constraint_force buffer is shorter than steps");
    }
    if (has_plane && RGPOT_MD_RESULT_HAS_METRIC(result) &&
        result->constraint_metric_len < request->steps) {
      throw std::invalid_argument(
          "MD Langevin constraint_metric buffer is shorter than steps");
    }
    if (!RGPOT_MD_OPS_HAS(&session->ops, run_langevin)) {
      throw std::runtime_error("MD engine does not support run_langevin");
    }
    return session->ops.run_langevin(session->ctx, request, result, err,
                                     err_len);
  } catch (const std::exception &ex) {
    write_err(err, err_len, ex.what());
    return 1;
  } catch (...) {
    write_err(err, err_len, "unknown C++ exception");
    return 1;
  }
}

int rgpot_md_session_reseed_velocities(rgpot_md_session *session, double kT,
                                       uint64_t seed,
                                       const double *plane_normal, size_t n3,
                                       char *err, size_t err_len) {
  clear_err(err, err_len);
  try {
    if (session == nullptr || n3 != 3 * session->n_atoms) {
      throw std::invalid_argument("MD reseed session or dimension mismatch");
    }
    if (!std::isfinite(kT) || kT < 0.0) {
      throw std::invalid_argument("MD reseed kT must be finite and not negative");
    }
    if (plane_normal != nullptr && !finite_array(plane_normal, n3)) {
      throw std::invalid_argument("MD reseed plane_normal must be finite");
    }
    // The draw is made here from the masses, so it is the same stream for
    // every engine and the engine only has to accept the velocities.
    std::vector<double> masses(session->n_atoms);
    if (session->ops.get_masses(session->ctx, masses.data(), session->n_atoms,
                                err, err_len) != 0) {
      return 1;
    }
    std::vector<double> per_coordinate(n3);
    for (size_t i = 0; i < session->n_atoms; ++i) {
      for (size_t d = 0; d < 3; ++d) {
        per_coordinate[3 * i + d] = masses[i];
      }
    }
    const std::vector<double> velocities = rgpot::md::maxwellBoltzmannVelocities(
        per_coordinate, kT * rgpot::md::kAmuA2Ps2PerEv, seed, plane_normal);
    return session->ops.set_velocities(session->ctx, velocities.data(), n3, err,
                                       err_len);
  } catch (const std::exception &ex) {
    write_err(err, err_len, ex.what());
    return 1;
  } catch (...) {
    write_err(err, err_len, "unknown C++ exception");
    return 1;
  }
}

int rgpot_md_session_get_positions(rgpot_md_session *session,
                                   double *positions, size_t n3, char *err,
                                   size_t err_len) {
  if (positions == nullptr) {
    write_err(err, err_len, "MD positions output buffer is null");
    return 1;
  }
  return rgpot_md_session_get_state(session, positions, nullptr, nullptr, n3,
                                    nullptr, nullptr, err, err_len);
}

int rgpot_md_session_get_velocities(rgpot_md_session *session,
                                    double *velocities, size_t n3, char *err,
                                    size_t err_len) {
  if (velocities == nullptr) {
    write_err(err, err_len, "MD velocities output buffer is null");
    return 1;
  }
  return rgpot_md_session_get_state(session, nullptr, velocities, nullptr, n3,
                                    nullptr, nullptr, err, err_len);
}

int rgpot_md_session_get_forces(rgpot_md_session *session, double *forces,
                                size_t n3, double *potential_energy,
                                char *err, size_t err_len) {
  if (forces == nullptr) {
    write_err(err, err_len, "MD forces output buffer is null");
    return 1;
  }
  return rgpot_md_session_get_state(session, nullptr, nullptr, forces, n3,
                                    potential_energy, nullptr, err, err_len);
}

int rgpot_md_session_get_energy(rgpot_md_session *session,
                                double *potential_energy,
                                double *kinetic_energy, char *err,
                                size_t err_len) {
  return rgpot_md_session_get_state(
      session, nullptr, nullptr, nullptr,
      session == nullptr ? 0 : 3 * session->n_atoms, potential_energy,
      kinetic_energy, err, err_len);
}

int rgpot_md_session_get_neighbors(rgpot_md_session *session,
                                   const rgpot_nlist_request *request,
                                   rgpot_nlist_soa *out, size_t *count,
                                   size_t *needed, char *err, size_t err_len) {
  clear_err(err, err_len);
  if (count != nullptr) {
    *count = 0;
  }
  if (needed != nullptr) {
    *needed = 0;
  }
  try {
    if (session == nullptr || request == nullptr) {
      throw std::invalid_argument("MD neighbor query needs a session and request");
    }
    if (!RGPOT_MD_OPS_HAS(&session->ops, get_neighbors)) {
      write_err(err, err_len, "MD engine does not expose neighbor queries");
      return RGPOT_MD_NLIST_UNSUPPORTED;
    }
    const int status =
        session->ops.get_neighbors(session->ctx, request, out, count, needed, err, err_len);
    if (status == RGPOT_MD_NLIST_OK || status == RGPOT_MD_NLIST_UNSUPPORTED ||
        status == RGPOT_MD_NLIST_ERROR) {
      return status;
    }
    return status == 0 ? RGPOT_MD_NLIST_OK : RGPOT_MD_NLIST_ERROR;
  } catch (const std::exception &ex) {
    write_err(err, err_len, ex.what());
    return RGPOT_MD_NLIST_ERROR;
  } catch (...) {
    write_err(err, err_len, "unknown C++ exception");
    return RGPOT_MD_NLIST_ERROR;
  }
}

int rgpot_md_session_set_neighbors(rgpot_md_session *session,
                                   const rgpot_nlist_request *request,
                                   const rgpot_nlist_soa *pairs, size_t count,
                                   char *err, size_t err_len) {
  clear_err(err, err_len);
  try {
    if (session == nullptr || request == nullptr) {
      throw std::invalid_argument("MD neighbor setter needs a session and request");
    }
    if (!RGPOT_MD_OPS_HAS(&session->ops, set_neighbors)) {
      write_err(err, err_len, "MD engine does not accept supplied neighbors");
      return RGPOT_MD_NLIST_UNSUPPORTED;
    }
    const int status = session->ops.set_neighbors(session->ctx, request, pairs, count, err, err_len);
    if (status == RGPOT_MD_NLIST_OK || status == RGPOT_MD_NLIST_UNSUPPORTED ||
        status == RGPOT_MD_NLIST_ERROR) {
      return status;
    }
    return status == 0 ? RGPOT_MD_NLIST_OK : RGPOT_MD_NLIST_ERROR;
  } catch (const std::exception &ex) {
    write_err(err, err_len, ex.what());
    return RGPOT_MD_NLIST_ERROR;
  } catch (...) {
    write_err(err, err_len, "unknown C++ exception");
    return RGPOT_MD_NLIST_ERROR;
  }
}

void rgpot_md_session_destroy(rgpot_md_session *session) { delete session; }

} // extern "C"
