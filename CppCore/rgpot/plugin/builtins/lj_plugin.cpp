// MIT License
// Copyright 2023--present rgpot developers

#include <rgpot/plugin.h>
#include "rgpot/LennardJones/LJPot.hpp"
#include <new>

namespace rgpot::plugin {
namespace {
rgpot_plugin_status_t create(const char *, rgpot_plugin_instance **output) {
  if (!output)
    return RGPOT_PLUGIN_INVALID_PARAM;
  *output = nullptr;
  try {
    *output = reinterpret_cast<rgpot_plugin_instance *>(new LJPot());
    return RGPOT_PLUGIN_OK;
  } catch (...) {
    return RGPOT_PLUGIN_ERROR;
  }
}
void destroy(rgpot_plugin_instance *instance) {
  delete reinterpret_cast<LJPot *>(instance);
}
rgpot_plugin_status_t calculate(rgpot_plugin_instance *instance,
                                const rgpot_plugin_input_t *input,
                                rgpot_plugin_output_t *output) {
  if (!instance || !input || !output)
    return RGPOT_PLUGIN_INVALID_PARAM;
  try {
    ForceInput request{input->n_atoms, input->positions, input->atomic_numbers,
                       input->cell};
    ForceOut result{.F = output->forces};
    reinterpret_cast<LJPot *>(instance)->forceImpl(request, &result);
    output->energy = result.energy;
    output->variance = result.variance;
    return RGPOT_PLUGIN_OK;
  } catch (...) {
    return RGPOT_PLUGIN_ERROR;
  }
}
uint32_t capabilities(const rgpot_plugin_instance *) {
  return RGPOT_CAP_ENERGY | RGPOT_CAP_FORCES | RGPOT_CAP_INSTANCE_SAFE |
         RGPOT_CAP_PERIODIC;
}
const rgpot_plugin_descriptor_t descriptor{
    sizeof(rgpot_plugin_descriptor_t), RGPOT_PLUGIN_API_VERSION,
    "lennard-jones", "1.0.0", nullptr, nullptr,
    create, destroy, calculate, capabilities, nullptr};
} // namespace
const rgpot_plugin_descriptor_t *builtinLennardJones() { return &descriptor; }
} // namespace rgpot::plugin
