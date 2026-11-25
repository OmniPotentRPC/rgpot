// MIT License — dlopen frontend for libmetatomic_engine.so
#include "rgpot/MetatomicPot/MetatomicDlopen.hpp"
#include "rgpot/Slices.hpp"
#include "rgpot/abi/DynLib.hpp"
#include "rgpot/abi/EngineDlopen.hpp"

#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace rgpot {

namespace {

using set_threads_fn = int (*)(RgpotMtaPot *, int);

// Thread state of each MetatomicDlopen, kept outside the object so the installed
// class layout stays as it was. An entry lives from a successful
// constructor to the destructor.
struct ThreadSlot {
  set_threads_fn set = nullptr;
  int threads = 0;
};

struct ThreadSlots {
  std::mutex mutex;
  std::unordered_map<const MetatomicDlopen *, ThreadSlot> slots;
};

ThreadSlots &thread_slots() {
  // Destructors may run during static teardown, so the map is never freed.
  static auto *slots = new ThreadSlots;
  return *slots;
}

ThreadSlot slot_of(const MetatomicDlopen *pot) noexcept {
  auto &s = thread_slots();
  std::lock_guard<std::mutex> lock(s.mutex);
  const auto it = s.slots.find(pot);
  return it == s.slots.end() ? ThreadSlot{} : it->second;
}

void store_slot(const MetatomicDlopen *pot, ThreadSlot slot) {
  auto &s = thread_slots();
  std::lock_guard<std::mutex> lock(s.mutex);
  s.slots.insert_or_assign(pot, slot);
}

void drop_slot(const MetatomicDlopen *pot) noexcept {
  auto &s = thread_slots();
  std::lock_guard<std::mutex> lock(s.mutex);
  s.slots.erase(pot);
}

RgpotMtaConfig to_c_config(const MetatomicConfig &c) {
  RgpotMtaConfig out{};
  out.model_path = c.model_path.c_str();
  out.device = c.device.c_str();
  out.length_unit = c.length_unit.c_str();
  out.extensions_directory = c.extensions_directory.c_str();
  out.check_consistency = c.check_consistency ? 1 : 0;
  out.uncertainty_threshold = c.uncertainty_threshold;
  out.dtype_override = c.dtype_override.c_str();
  out.random_rotation = c.random_rotation ? 1 : 0;
  out.n_symmetry_rotations = c.n_symmetry_rotations;
  out.so3_probe_scatter = c.so3_probe_scatter ? 1 : 0;
  out.torch_determinism_strict =
      (c.torch_determinism == TorchDeterminismPolicy::Strict) ? 1 : 0;
  return out;
}

} // namespace

MetatomicDlopen::MetatomicDlopen(const MetatomicConfig &config) {
  const auto cands = abi::engine_candidates(
      "metatomic_engine", config.engine_path,
      {"RGPOT_METATOMIC_ENGINE", "METATOMIC_ENGINE"},
      abi::CandidateOptions{.include_cwd_dot_slash = false,
                            .include_dll = false,
                            .include_dir_envs = true,
                            .exclusive_explicit = false,
                            .platform_only_soname = true});
  DynLib lib;
  std::string loaded_path;
  std::string open_err;
  if (!abi::open_first_candidate(lib, cands, &loaded_path, &open_err, true)) {
    std::string msg =
        "MetatomicDlopen: libmetatomic_engine.so not found "
        "(set RGPOT_METATOMIC_ENGINE / engine_path / EON_POTENTIALS_PATH)";
    if (!open_err.empty()) {
      msg += "; last dlerror: ";
      msg += open_err;
    }
    throw std::runtime_error(msg);
  }

  auto abi_fn = lib.sym_optional<int (*)(void)>("rgpot_mta_abi_version");
  m_create = lib.sym_optional<create_fn>("rgpot_mta_create");
  m_destroy = lib.sym_optional<destroy_fn>("rgpot_mta_destroy");
  auto set_threads =
      lib.sym_optional<set_threads_fn>("rgpot_mta_set_thread_count");
  m_force = lib.sym_optional<force_fn>("rgpot_mta_force");
  if (!abi_fn || !m_create || !m_destroy || !set_threads || !m_force) {
    throw std::runtime_error(
        "MetatomicDlopen: missing C ABI symbols in engine library");
  }
  if (abi_fn() != RGPOT_MTA_ABI_VERSION) {
    throw std::runtime_error("MetatomicDlopen: ABI version mismatch");
  }
  char err[1024]{};
  auto cfg = to_c_config(config);
  store_slot(this, ThreadSlot{set_threads, 0});
  m_pot = m_create(&cfg, err, sizeof err);
  if (!m_pot) {
    drop_slot(this);
    throw std::runtime_error(std::string("MetatomicDlopen: create failed: ") +
                             err);
  }
  m_lib = lib.native_handle();
  lib.release();
}

MetatomicDlopen::~MetatomicDlopen() {
  drop_slot(this);
  // Keep the engine mapped: torch/metatomic static teardown after dlclose
  // routinely SEGV at process exit.
  if (m_pot && m_destroy)
    m_destroy(m_pot);
  m_pot = nullptr;
  m_lib = nullptr;
}

void MetatomicDlopen::setThreadCount(int threads) {
  if (threads < 1) {
    throw std::invalid_argument("rgpot thread count must be at least 1");
  }
  ThreadSlot slot = slot_of(this);
  if (!slot.set || !m_pot || slot.set(m_pot, threads) != 0) {
    throw std::runtime_error("MetatomicDlopen: failed to set thread count");
  }
  slot.threads = threads;
  store_slot(this, slot);
}

void MetatomicDlopen::clearThreadCount() {
  ThreadSlot slot = slot_of(this);
  if (!slot.set || !m_pot || slot.set(m_pot, 0) != 0) {
    throw std::runtime_error("MetatomicDlopen: failed to clear thread count");
  }
  slot.threads = 0;
  store_slot(this, slot);
}

int MetatomicDlopen::threadCount() const noexcept {
  return slot_of(this).threads;
}

void MetatomicDlopen::forceImpl(const ForceInput &in, ForceOut *out) const {
  if (!m_pot || !m_force)
    throw std::runtime_error("MetatomicDlopen: engine not available");
  const ThreadSlot slot = slot_of(this);
  const ThreadBinding bind_threads{slot.threads};
  if (!slot.set || slot.set(m_pot, explicitThreadCount()) != 0)
    throw std::runtime_error(
        "MetatomicDlopen: failed to forward thread count");
  const int rc =
      m_force(m_pot, static_cast<long>(in.nAtoms), in.pos, in.atmnrs, out->F,
              &out->energy, &out->variance, in.box);
  if (rc != 0)
    throw std::runtime_error("MetatomicDlopen: force failed rc=" +
                             std::to_string(rc));
}

} // namespace rgpot
