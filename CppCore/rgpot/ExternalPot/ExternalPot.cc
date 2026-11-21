// MIT License
// Copyright 2023--present rgpot developers

#include "rgpot/ExternalPot/ExternalPot.hpp"

#include "rgpot/abi/ProfileLoader.hpp"
#include "rgpot/ExternalPot/md_units.hpp"
#include "rgpot/LennardJones/LJPot.hpp"
#include "rgpot/engine_c_abi.h"
#include "rgpot/plugin/DynLib.hpp"
#if __has_include("rgpot/rpc/Potentials.capnp.h")
#define RGPOT_HAS_LAMMPC_PROFILE 1
#include "rgpot/rpc/Potentials.capnp.h"
#include <capnp/message.h>
#include <capnp/serialize.h>
#include <kj/array.h>
#else
#define RGPOT_HAS_LAMMPC_PROFILE 0
#endif

#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <utility>
#include <vector>

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

struct Loaded {
  dyn::Handle library = nullptr;
  EngineCreate create = nullptr;
  EngineDestroy destroy = nullptr;
  EngineForce force = nullptr;
  NativeForce native = nullptr;
  RgpotEnginePot *pot = nullptr;

  ~Loaded() {
    if (pot != nullptr && destroy != nullptr) {
      destroy(pot);
    }
    if (library != nullptr) {
      dyn::close(library);
    }
  }

  Loaded() = default;
  Loaded(const Loaded &) = delete;
  Loaded &operator=(const Loaded &) = delete;
  Loaded(Loaded &&other) noexcept
      : library(other.library), create(other.create), destroy(other.destroy),
        force(other.force), native(other.native), pot(other.pot) {
    other.library = nullptr;
    other.pot = nullptr;
    other.destroy = nullptr;
  }
  Loaded &operator=(Loaded &&other) noexcept {
    if (this != &other) {
      if (pot != nullptr && destroy != nullptr) {
        destroy(pot);
      }
      if (library != nullptr) {
        dyn::close(library);
      }
      library = other.library;
      create = other.create;
      destroy = other.destroy;
      force = other.force;
      native = other.native;
      pot = other.pot;
      other.library = nullptr;
      other.pot = nullptr;
      other.destroy = nullptr;
    }
    return *this;
  }
};

Loaded open_library(const std::vector<std::string> &paths, bool native) {
  if (paths.empty()) {
    throw std::runtime_error("no engine library to open");
  }
  std::string tried;
  for (const std::string &path : paths) {
    dyn::Handle handle = dyn::open(path.c_str());
    if (handle == nullptr) {
      tried += path + ": " + last_dl() + "; ";
      continue;
    }
    Loaded loaded;
    loaded.library = handle;
    loaded.create = dyn::loadSym<EngineCreate>(handle, "rgpot_engine_create");
    loaded.destroy =
        dyn::loadSym<EngineDestroy>(handle, "rgpot_engine_destroy");
    if (native) {
      loaded.native =
          dyn::loadSym<NativeForce>(handle, "rgpot_md_native_force");
    } else {
      loaded.force = dyn::loadSym<EngineForce>(handle, "rgpot_engine_force");
    }
    const bool have =
        loaded.create != nullptr && loaded.destroy != nullptr &&
        (native ? loaded.native != nullptr : loaded.force != nullptr);
    if (!have) {
      tried += path + ": missing " +
               (native ? "rgpot_md_native_force" : "rgpot_engine_force") + "; ";
      dyn::close(handle);
      loaded.library = nullptr;
      continue;
    }
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

std::string profile_message(const rgpot::abi::ProfileResult &result) {
  const size_t n = strnlen(result.message, sizeof(result.message));
  return std::string(result.message, n);
}
#endif

} // namespace

struct LammpsPot::Impl {
  enum class Mode { LegacyLJ, LammpcProfile };
  Mode mode = Mode::LegacyLJ;
  Loaded legacy_engine;
  rgpot::abi::ProfileLoader profile_loader;
  std::vector<unsigned char> profile_config;
  void *profile_session = nullptr;
  mutable std::mutex mutex;

  ~Impl() {
    if (profile_session && profile_loader.session_destroy)
      profile_loader.session_destroy(profile_session);
  }
};

LammpsPot::LammpsPot(const ExternalLjConfig &config)
    : Potential(PotType::Lammps), impl_(std::make_unique<Impl>()),
      config_(config) {
  require_lj(config_);
  impl_->mode = Impl::Mode::LegacyLJ;
  impl_->legacy_engine = open_library(lammps_paths(config_.engine_path), false);
  const double packed[3] = {config_.u0, config_.cutoff, config_.psi};
  create_engine(impl_->legacy_engine, packed, sizeof(packed));
}

LammpsPot::LammpsPot(const void *config_capnp, size_t config_capnp_size,
                     const std::string &engine_path)
    : Potential(PotType::Lammps), impl_(std::make_unique<Impl>()), config_{} {
#if !RGPOT_HAS_LAMMPC_PROFILE
  (void)config_capnp;
  (void)config_capnp_size;
  (void)engine_path;
  throw std::runtime_error(
      "lammpc profile mode needs an RPC-capable build (Potentials.capnp.h)");
#else
  if (!config_capnp || config_capnp_size == 0)
    throw std::invalid_argument("lammpc config bytes are required");
  if (config_capnp_size % sizeof(::capnp::word) != 0)
    throw std::invalid_argument("lammpc config bytes must be word-aligned");
  impl_->mode = Impl::Mode::LammpcProfile;
  impl_->profile_config.resize(config_capnp_size);
  std::memcpy(impl_->profile_config.data(), config_capnp, config_capnp_size);
  impl_->profile_loader.load("lammpc", engine_path);
  if (impl_->profile_loader.available() != 1)
    throw std::runtime_error("lammpc backend reports unavailable");
  impl_->profile_session = impl_->profile_loader.session_create_from_config(
      impl_->profile_config.data(), impl_->profile_config.size());
  if (!impl_->profile_session)
    throw std::runtime_error("lammpc_session_create_from_config failed");
#endif
}

LammpsPot::~LammpsPot() = default;
LammpsPot::LammpsPot(LammpsPot &&) noexcept = default;
LammpsPot &LammpsPot::operator=(LammpsPot &&) noexcept = default;

PotCaps LammpsPot::caps() const noexcept {
  const bool stress = impl_ && impl_->mode == Impl::Mode::LammpcProfile;
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
  if (stress.size() == 9) {
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

namespace {

void write_err(char *err, size_t err_len, const char *text) {
  if (err == nullptr || err_len == 0) {
    return;
  }
  std::snprintf(err, err_len, "%s", text);
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
  }
}

} // extern "C"
