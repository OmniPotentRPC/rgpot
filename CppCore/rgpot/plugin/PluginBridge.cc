// MIT License
// Copyright 2023--present rgpot developers

#include "rgpot/plugin/PluginBridge.hpp"
#include "rgpot/ParamHash.hpp"
#include "rgpot/plugin/Descriptor.hpp"
#include "rgpot/plugin/PluginRegistry.hpp"
#include "rgpot/units.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <vector>

namespace rgpot::plugin {
namespace {
using SerialMutex = std::shared_ptr<std::recursive_mutex>;
SerialMutex serialPluginMutex() {
  static const auto mutex = std::make_shared<std::recursive_mutex>();
  return mutex;
}

std::string pluginError(const rgpot_plugin_descriptor_t &descriptor,
                        const rgpot_plugin_instance *instance,
                        const char *fallback) {
  if (descriptor.last_error && instance) {
    const char *message = descriptor.last_error(instance);
    if (message && message[0])
      return message;
  }
  return fallback;
}

struct InstanceDeleter {
  rgpot_plugin_destroy_fn destroy;
  SerialMutex mutex;
  void operator()(rgpot_plugin_instance *instance) const {
    if (instance) {
      std::lock_guard lock(*mutex);
      destroy(instance);
    }
  }
};
using Instance = std::unique_ptr<rgpot_plugin_instance, InstanceDeleter>;

class PluginPot : public Potential<PluginPot> {
public:
  PluginPot(rgpot_plugin_descriptor_t descriptor, Instance instance,
            std::shared_ptr<void> library, double length, double energy,
            uint32_t capabilities, uint64_t key)
      : Potential(PotType::UNKNOWN), m_library(std::move(library)),
        m_descriptor(descriptor), m_instance(std::move(instance)),
        m_length(length), m_energy(energy), m_force(energy * length),
        m_capabilities(capabilities), m_key(key) {}

  PotCaps caps() const noexcept override {
    return {.reentrancy = (m_capabilities & RGPOT_CAP_INSTANCE_SAFE)
                              ? Reentrancy::PerInstance
                              : Reentrancy::ProcessSerial,
            .periodic = (m_capabilities & RGPOT_CAP_PERIODIC) != 0};
  }

  uint64_t paramsKey() const noexcept override { return m_key; }

  void forceImpl(const ForceInput &input, ForceOut *output) const override {
    if (!output || !input.box || input.nAtoms > std::numeric_limits<size_t>::max() / 3 ||
        (input.nAtoms && (!input.pos || !input.atmnrs || !output->F)))
      throw std::invalid_argument("invalid plugin force buffers");
    const size_t count = 3 * input.nAtoms;
    std::unique_lock lock(*m_instance.get_deleter().mutex, std::defer_lock);
    if (!(m_capabilities & RGPOT_CAP_INSTANCE_SAFE))
      lock.lock();
    m_positions.resize(count);
    m_forces.assign(count, std::numeric_limits<double>::quiet_NaN());
    for (size_t i = 0; i < count; ++i)
      m_positions[i] = input.pos[i] * m_length;
    double cell[9];
    for (size_t i = 0; i < 9; ++i)
      cell[i] = input.box[i] * m_length;
    const rgpot_plugin_input_t request{
        input.nAtoms, m_positions.data(), input.atmnrs, cell};
    rgpot_plugin_output_t response{
        std::numeric_limits<double>::quiet_NaN(), 0.0, m_forces.data()};
    if (m_descriptor.calculate(m_instance.get(), &request, &response) != RGPOT_PLUGIN_OK)
      throw std::runtime_error(pluginError(m_descriptor, m_instance.get(),
                                          "plugin calculate failed"));
    const double energy = response.energy * m_energy;
    const double variance = response.variance * m_energy * m_energy;
    if (!std::isfinite(energy) || !std::isfinite(variance) || variance < 0.0)
      throw std::runtime_error("plugin returned invalid energy or variance");
    for (double &force : m_forces) {
      force *= m_force;
      if (!std::isfinite(force))
        throw std::runtime_error("plugin returned invalid forces");
    }
    output->energy = energy;
    output->variance = variance;
    output->has_stress = 0;
    std::fill(std::begin(output->stress), std::end(output->stress), 0.0);
    std::copy(m_forces.begin(), m_forces.end(), output->F);
  }

private:
  std::shared_ptr<void> m_library;
  rgpot_plugin_descriptor_t m_descriptor;
  Instance m_instance;
  double m_length;
  double m_energy;
  double m_force;
  uint32_t m_capabilities;
  uint64_t m_key;
  mutable std::vector<double> m_positions;
  mutable std::vector<double> m_forces;
};
} // namespace

std::unique_ptr<PotentialBase> create_from_plugin(
    const rgpot_plugin_descriptor_t *source, const std::string &config) {
  const auto descriptor = detail::checkedDescriptor(source);
  auto library = PluginRegistry::instance().retain_library(source);
  const std::string length_unit = descriptor.length_unit && descriptor.length_unit[0]
                                      ? descriptor.length_unit : "angstrom";
  const std::string energy_unit = descriptor.energy_unit && descriptor.energy_unit[0]
                                      ? descriptor.energy_unit : "eV";
  const double length = units::unit_conversion_factor("angstrom", length_unit);
  const double energy = units::unit_conversion_factor(energy_unit, "eV");
  if (!std::isfinite(length) || !std::isfinite(energy) || length <= 0.0 ||
      energy <= 0.0 || !std::isfinite(length * energy) ||
      !std::isfinite(energy * energy))
    throw std::invalid_argument("invalid plugin unit conversion");
  const auto mutex = serialPluginMutex();
  std::lock_guard lock(*mutex);
  Instance instance(nullptr, InstanceDeleter{descriptor.destroy, mutex});
  rgpot_plugin_instance *raw = nullptr;
  rgpot_plugin_status_t status;
  try {
    status = descriptor.create(config.c_str(), &raw);
  } catch (...) {
    instance.reset(raw);
    throw;
  }
  instance.reset(raw);
  if (status != RGPOT_PLUGIN_OK || !instance)
    throw std::runtime_error(pluginError(descriptor, raw, "plugin create failed"));
  const uint32_t capabilities = descriptor.capabilities
                                    ? descriptor.capabilities(raw)
                                    : RGPOT_CAP_ENERGY | RGPOT_CAP_FORCES;
  if ((capabilities & (RGPOT_CAP_ENERGY | RGPOT_CAP_FORCES)) !=
      (RGPOT_CAP_ENERGY | RGPOT_CAP_FORCES))
    throw std::invalid_argument("plugin must provide both energy and forces");
  Fnv1a key;
  key.str("rgpot-plugin-abi-1");
  key.str(descriptor.name);
  key.str(descriptor.version ? descriptor.version : "");
  key.str(config);
  key.str(length_unit);
  key.str(energy_unit);
  return std::make_unique<PluginPot>(descriptor, std::move(instance),
                                      std::move(library), length, energy,
                                      capabilities, key.h);
}

} // namespace rgpot::plugin
