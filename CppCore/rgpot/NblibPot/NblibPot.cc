// MIT License
// Copyright 2023--present rgpot developers

#include "rgpot/NblibPot/NblibPot.hpp"

#include "rgpot/NblibPot/nblib_c_abi.h"
#include "rgpot/plugin/DynLib.hpp"
#include "rgpot/stress.hpp"

#include <array>
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace rgpot {
namespace {

namespace dyn = plugin::dynlib;

template <typename Fn> Fn symbol(dyn::Handle library, const char *name) {
  void *address = dyn::sym(library, name);
  if (address == nullptr) {
    throw std::runtime_error(std::string("librgpot_nblib has no ") + name);
  }
  return reinterpret_cast<Fn>(address);
}

std::vector<std::string> candidates(const std::string &explicitPath) {
  std::vector<std::string> out;
  if (!explicitPath.empty()) {
    out.push_back(explicitPath);
    return out;
  }
  if (const char *env = std::getenv("RGPOT_NBLIB_ENGINE")) {
    if (env[0] != '\0') {
      out.emplace_back(env);
    }
  }
  out.emplace_back(std::string("librgpot_nblib") + dyn::so_extension());
  return out;
}

std::string topologyText(const NblibConfig &config) {
  if (!config.topology.empty()) {
    return config.topology;
  }
  if (config.topology_file.empty()) {
    throw std::invalid_argument("NblibPot needs a topology or a topology file");
  }
  std::ifstream file(config.topology_file);
  if (!file) {
    throw std::runtime_error("cannot read " + config.topology_file);
  }
  std::ostringstream text;
  text << file.rdbuf();
  return text.str();
}

} // namespace

struct NblibPot::Impl {
  dyn::Handle library = nullptr;
  rgpot_nblib_potential *pot = nullptr;
  decltype(&rgpot_nblib_potential_free) release = nullptr;
  decltype(&rgpot_nblib_potential_energy_forces) evaluate = nullptr;
  decltype(&rgpot_nblib_potential_atom_count) count = nullptr;
  decltype(&rgpot_nblib_potential_cutoff) cutoff = nullptr;
  decltype(&rgpot_nblib_potential_masses) masses = nullptr;
  std::size_t atoms = 0;
  mutable std::mutex mutex;
  mutable std::array<double, 9> virial{};

  Impl() = default;
  Impl(const Impl &) = delete;
  Impl &operator=(const Impl &) = delete;
  ~Impl() {
    if (pot != nullptr && release != nullptr) {
      release(pot);
    }
    if (library != nullptr) {
      dyn::close(library);
    }
  }
};

NblibPot::NblibPot(const NblibConfig &config)
    : Potential(PotType::UNKNOWN), impl_(std::make_unique<Impl>()),
      config_(config) {
  const std::string text = topologyText(config_);
  std::string tried;
  for (const std::string &path : candidates(config_.engine_path)) {
    dyn::Handle handle = dyn::open(path.c_str());
    if (handle == nullptr) {
      tried += path + ": " + dyn::error() + "; ";
      continue;
    }
    impl_->library = handle;
    break;
  }
  if (impl_->library == nullptr) {
    throw std::runtime_error("librgpot_nblib was not opened: " + tried);
  }
  const dyn::Handle library = impl_->library;
  const auto version = symbol<decltype(&rgpot_nblib_abi_version)>(
      library, "rgpot_nblib_abi_version");
  if (version() != RGPOT_NBLIB_ABI_VERSION) {
    throw std::runtime_error("librgpot_nblib speaks ABI " +
                             std::to_string(version()) + ", this host ABI " +
                             std::to_string(RGPOT_NBLIB_ABI_VERSION));
  }
  const auto available = symbol<decltype(&rgpot_nblib_available)>(
      library, "rgpot_nblib_available");
  if (available() == 0) {
    throw std::runtime_error("librgpot_nblib was built without nblib");
  }
  const auto features =
      symbol<decltype(&rgpot_nblib_features)>(library, "rgpot_nblib_features");
  if ((features() & RGPOT_NBLIB_FEATURE_POTENTIAL) == 0) {
    throw std::runtime_error("librgpot_nblib has no potential");
  }
  const auto create = symbol<decltype(&rgpot_nblib_potential_create)>(
      library, "rgpot_nblib_potential_create");
  impl_->release = symbol<decltype(&rgpot_nblib_potential_free)>(
      library, "rgpot_nblib_potential_free");
  impl_->evaluate = symbol<decltype(&rgpot_nblib_potential_energy_forces)>(
      library, "rgpot_nblib_potential_energy_forces");
  impl_->count = symbol<decltype(&rgpot_nblib_potential_atom_count)>(
      library, "rgpot_nblib_potential_atom_count");
  impl_->cutoff = symbol<decltype(&rgpot_nblib_potential_cutoff)>(
      library, "rgpot_nblib_potential_cutoff");
  impl_->masses = symbol<decltype(&rgpot_nblib_potential_masses)>(
      library, "rgpot_nblib_potential_masses");
  char err[512] = {};
  impl_->pot = create(text.c_str(), err, sizeof(err));
  if (impl_->pot == nullptr) {
    throw std::runtime_error(err[0] != '\0' ? err : "nblib potential failed");
  }
  impl_->atoms = impl_->count(impl_->pot);
}

NblibPot::~NblibPot() = default;
NblibPot::NblibPot(NblibPot &&) noexcept = default;
NblibPot &NblibPot::operator=(NblibPot &&) noexcept = default;

PotCaps NblibPot::caps() const noexcept {
  return {.reentrancy = Reentrancy::PerInstance,
          .periodic = true,
          .stress = true};
}

std::size_t NblibPot::atomCount() const noexcept { return impl_->atoms; }

double NblibPot::cutoff() const noexcept {
  return impl_->cutoff(impl_->pot);
}

std::vector<double> NblibPot::masses() const {
  std::vector<double> out(impl_->atoms);
  if (impl_->masses(impl_->pot, out.data(), out.size()) != 0) {
    throw std::runtime_error("nblib masses failed");
  }
  return out;
}

std::vector<double> NblibPot::lastVirial() const {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  return {impl_->virial.begin(), impl_->virial.end()};
}

void NblibPot::forceImpl(const ForceInput &in, ForceOut *out) const {
  if (out == nullptr || out->F == nullptr || in.box == nullptr ||
      (in.nAtoms > 0 && in.pos == nullptr)) {
    throw std::invalid_argument("potential force buffers are incomplete");
  }
  std::lock_guard<std::mutex> lock(impl_->mutex);
  char err[512] = {};
  double energy = 0.0;
  std::array<double, 9> virial{};
  if (impl_->evaluate(impl_->pot, in.pos, in.nAtoms, in.box, out->F, &energy,
                      virial.data(), err, sizeof(err)) != 0) {
    throw std::runtime_error(err[0] != '\0' ? err : "nblib evaluation failed");
  }
  impl_->virial = virial;
  out->energy = energy;
  out->variance = 0.0;
  const double volume = cellVolume(in.box);
  std::array<double, 6> strain{};
  // stress = 2 Xi / V; publishCauchyStress divides the Voigt strain by V.
  strain[0] = 2.0 * virial[0];
  strain[1] = 2.0 * virial[4];
  strain[2] = 2.0 * virial[8];
  strain[3] = virial[5] + virial[7];
  strain[4] = virial[2] + virial[6];
  strain[5] = virial[1] + virial[3];
  publishCauchyStress(out, strain.data(), volume);
}

std::unique_ptr<NblibPot> make_nblib_potential(const NblibConfig &config) {
  return std::make_unique<NblibPot>(config);
}

} // namespace rgpot
