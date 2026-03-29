// MIT License
// Copyright 2023--present rgpot developers
//
// Built-in plugin shim for the Lennard-Jones potential.

#include <rgpot/plugin.h>

#include "rgpot/ForceStructs.hpp"
#include "rgpot/LennardJones/LJPot.hpp"
#include "rgpot/plugin/PluginRegistry.hpp"

extern "C" {

static rgpot_plugin_status_t
lj_create(const char * /*config_json*/, rgpot_plugin_instance **out) {
  auto *pot = new (std::nothrow) rgpot::LJPot();
  if (!pot)
    return RGPOT_PLUGIN_ERROR;
  *out = reinterpret_cast<rgpot_plugin_instance *>(pot);
  return RGPOT_PLUGIN_OK;
}

static void lj_destroy(rgpot_plugin_instance *inst) {
  delete reinterpret_cast<rgpot::LJPot *>(inst);
}

static rgpot_plugin_status_t
lj_calculate(rgpot_plugin_instance *inst, const rgpot_plugin_input_t *input,
             rgpot_plugin_output_t *output) {
  auto *pot = reinterpret_cast<rgpot::LJPot *>(inst);
  rgpot::ForceInput fi{input->n_atoms, input->positions,
                       input->atomic_numbers, input->cell};
  rgpot::ForceOut fo{output->forces, 0.0, 0.0};
  pot->forceImpl(fi, &fo);
  output->energy = fo.energy;
  output->variance = fo.variance;
  return RGPOT_PLUGIN_OK;
}

static uint32_t lj_caps(const rgpot_plugin_instance *) {
  return RGPOT_CAP_ENERGY | RGPOT_CAP_FORCES;
}

static const rgpot_plugin_descriptor_t lj_descriptor = {
    sizeof(rgpot_plugin_descriptor_t),
    RGPOT_PLUGIN_API_VERSION,
    "lennard-jones",
    "1.0.0",
    nullptr, // angstrom (native)
    nullptr, // eV (native)
    lj_create,
    lj_destroy,
    lj_calculate,
    lj_caps,
    nullptr,
};

RGPOT_PLUGIN_EXPORT const rgpot_plugin_descriptor_t *rgpot_plugin_init(void) {
  return &lj_descriptor;
}

} // extern "C"
