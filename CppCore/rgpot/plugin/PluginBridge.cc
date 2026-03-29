// MIT License
// Copyright 2023--present rgpot developers

#include "rgpot/plugin/PluginBridge.hpp"
#include "rgpot/units.hpp"

#include <stdexcept>
#include <vector>

namespace rgpot::plugin {

namespace {

/// Adapter that wraps a plugin instance as a Potential<PluginPot>.
class PluginPot : public Potential<PluginPot> {
public:
  PluginPot(const rgpot_plugin_descriptor_t *desc,
            rgpot_plugin_instance *instance, double length_factor,
            double energy_factor)
      : Potential(PotType::UNKNOWN), m_desc(desc), m_instance(instance),
        m_length_factor(length_factor), m_energy_factor(energy_factor),
        m_force_factor(energy_factor / length_factor) {}

  ~PluginPot() {
    if (m_instance && m_desc->destroy) {
      m_desc->destroy(m_instance);
    }
  }

  PluginPot(const PluginPot &) = delete;
  PluginPot &operator=(const PluginPot &) = delete;

  void forceImpl(const ForceInput &in, ForceOut *out) const override {
    const size_t n3 = 3 * in.nAtoms;

    // Convert host positions (angstrom) to plugin native units
    m_pos_buf.resize(n3);
    for (size_t i = 0; i < n3; ++i) {
      m_pos_buf[i] = in.pos[i] * m_length_factor;
    }

    double cell_conv[9];
    for (int i = 0; i < 9; ++i) {
      cell_conv[i] = in.box[i] * m_length_factor;
    }

    rgpot_plugin_input_t pin = {in.nAtoms, m_pos_buf.data(), in.atmnrs,
                                cell_conv};
    rgpot_plugin_output_t pout = {0.0, 0.0, out->F};

    auto st = m_desc->calculate(m_instance, &pin, &pout);
    if (st != RGPOT_PLUGIN_OK) {
      std::string err = "plugin calculate failed";
      if (m_desc->last_error) {
        const char *msg = m_desc->last_error(m_instance);
        if (msg)
          err = msg;
      }
      throw std::runtime_error(err);
    }

    // Convert results from plugin units to host units (eV/angstrom)
    out->energy = pout.energy * m_energy_factor;
    out->variance = pout.variance * m_energy_factor * m_energy_factor;
    for (size_t i = 0; i < n3; ++i) {
      out->F[i] *= m_force_factor;
    }
  }

private:
  const rgpot_plugin_descriptor_t *m_desc;
  rgpot_plugin_instance *m_instance;
  double m_length_factor; // angstrom -> plugin length unit
  double m_energy_factor; // plugin energy unit -> eV
  double m_force_factor;  // plugin force unit -> eV/angstrom
  mutable std::vector<double> m_pos_buf;
};

} // namespace

std::unique_ptr<rgpot::PotentialBase>
create_from_plugin(const rgpot_plugin_descriptor_t *desc,
                   const std::string &config) {
  rgpot_plugin_instance *instance = nullptr;
  auto st = desc->create(config.c_str(), &instance);
  if (st != RGPOT_PLUGIN_OK || !instance) {
    std::string err = "failed to create plugin instance";
    if (desc->last_error) {
      const char *msg = desc->last_error(instance);
      if (msg)
        err = msg;
    }
    throw std::runtime_error(err);
  }

  // Compute unit conversion factors once
  double length_factor = 1.0;
  double energy_factor = 1.0;
  if (desc->length_unit && desc->length_unit[0] != '\0') {
    length_factor =
        rgpot::units::unit_conversion_factor("angstrom", desc->length_unit);
  }
  if (desc->energy_unit && desc->energy_unit[0] != '\0') {
    energy_factor =
        rgpot::units::unit_conversion_factor(desc->energy_unit, "eV");
  }

  return std::make_unique<PluginPot>(desc, instance, length_factor,
                                     energy_factor);
}

} // namespace rgpot::plugin
