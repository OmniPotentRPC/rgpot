// MIT License
// Copyright 2023--present rgpot developers

#include <rgpot/plugin.h>
#include "rgpot/LennardJones/LJPot.hpp"
#include "rgpot/Morse/MorsePot.hpp"
#include "rgpot/SuttonChen/SuttonChenPot.hpp"
#ifdef RGPOT_HAS_EAM_TABLE
#include "rgpot/EAMTable/EAMTablePot.hpp"
#endif

namespace rgpot::plugin {
namespace {
uint32_t capabilities(const rgpot_plugin_instance *) {
  return RGPOT_CAP_ENERGY | RGPOT_CAP_FORCES | RGPOT_CAP_INSTANCE_SAFE |
         RGPOT_CAP_PERIODIC;
}

const char *lastError(const rgpot_plugin_instance *) {
  return forceLastError();
}

template <class Pot>
rgpot_plugin_status_t calculatePot(rgpot_plugin_instance *instance,
                                   const rgpot_plugin_input_t *input,
                                   rgpot_plugin_output_t *output) {
  if (instance == nullptr || input == nullptr || output == nullptr) {
    return RGPOT_PLUGIN_INVALID_PARAM;
  }
  auto *pot = reinterpret_cast<Pot *>(instance);
  ForceOut result{};
  result.F = output->forces;
  const ForceInput request{input->n_atoms, input->positions,
                           input->atomic_numbers, input->cell};
  if (rejectForceCall(request, &result, pot->caps().periodic) != 0) {
    return RGPOT_PLUGIN_INVALID_PARAM;
  }
  try {
    pot->forceImpl(request, &result);
    output->energy = result.energy;
    output->variance = result.variance;
    return RGPOT_PLUGIN_OK;
  } catch (...) {
    return RGPOT_PLUGIN_ERROR;
  }
}

template <class Pot>
void destroyPot(rgpot_plugin_instance *instance) {
  delete reinterpret_cast<Pot *>(instance);
}

template <class Pot>
rgpot_plugin_status_t createDefault(const char *,
                                    rgpot_plugin_instance **output) {
  if (output == nullptr) {
    return RGPOT_PLUGIN_INVALID_PARAM;
  }
  *output = nullptr;
  try {
    *output = reinterpret_cast<rgpot_plugin_instance *>(new Pot());
    return RGPOT_PLUGIN_OK;
  } catch (...) {
    return RGPOT_PLUGIN_ERROR;
  }
}

rgpot_plugin_status_t createSuttonChen(const char *,
                                       rgpot_plugin_instance **output) {
  if (output == nullptr) {
    return RGPOT_PLUGIN_INVALID_PARAM;
  }
  *output = nullptr;
  try {
    SuttonChenConfig config;
    config.periodic = true;
    *output = reinterpret_cast<rgpot_plugin_instance *>(new SuttonChenPot(config));
    return RGPOT_PLUGIN_OK;
  } catch (...) {
    return RGPOT_PLUGIN_ERROR;
  }
}

#ifdef RGPOT_HAS_EAM_TABLE
rgpot_plugin_status_t createEam(const char *config, rgpot_plugin_instance **output,
                                EAMTableFormat format) {
  if (output == nullptr) {
    return RGPOT_PLUGIN_INVALID_PARAM;
  }
  *output = nullptr;
  if (config == nullptr || config[0] == '\0') {
    return RGPOT_PLUGIN_INVALID_PARAM;
  }
  try {
    EAMTableConfig table;
    table.path = config;
    table.format = format;
    table.periodic = true;
    *output = reinterpret_cast<rgpot_plugin_instance *>(new EAMTablePot(table));
    return RGPOT_PLUGIN_OK;
  } catch (...) {
    return RGPOT_PLUGIN_ERROR;
  }
}

rgpot_plugin_status_t createEamFuncfl(const char *config,
                                      rgpot_plugin_instance **output) {
  return createEam(config, output, EAMTableFormat::Funcfl);
}

rgpot_plugin_status_t createEamSetfl(const char *config,
                                    rgpot_plugin_instance **output) {
  return createEam(config, output, EAMTableFormat::Setfl);
}
#endif

rgpot_plugin_descriptor_t makeDescriptor(const char *name,
                                         rgpot_plugin_create_fn create,
                                         rgpot_plugin_destroy_fn destroy,
                                         rgpot_plugin_calculate_fn calculate) {
  return rgpot_plugin_descriptor_t{
      sizeof(rgpot_plugin_descriptor_t),
      RGPOT_PLUGIN_API_VERSION,
      name,
      "1.0.0",
      nullptr,
      nullptr,
      create,
      destroy,
      calculate,
      capabilities,
      lastError,
  };
}

const rgpot_plugin_descriptor_t ljDescriptor =
    makeDescriptor("lennard-jones", createDefault<LJPot>, destroyPot<LJPot>,
                   calculatePot<LJPot>);
const rgpot_plugin_descriptor_t morseDescriptor =
    makeDescriptor("classical-morse", createDefault<MorsePot>,
                   destroyPot<MorsePot>, calculatePot<MorsePot>);
const rgpot_plugin_descriptor_t suttonChenDescriptor =
    makeDescriptor("sutton-chen", createSuttonChen, destroyPot<SuttonChenPot>,
                   calculatePot<SuttonChenPot>);
#ifdef RGPOT_HAS_EAM_TABLE
const rgpot_plugin_descriptor_t eamFuncflDescriptor =
    makeDescriptor("eam-funcfl", createEamFuncfl, destroyPot<EAMTablePot>,
                   calculatePot<EAMTablePot>);
const rgpot_plugin_descriptor_t eamSetflDescriptor =
    makeDescriptor("eam-setfl", createEamSetfl, destroyPot<EAMTablePot>,
                   calculatePot<EAMTablePot>);
#endif
} // namespace

const rgpot_plugin_descriptor_t *builtinLennardJones() {
  return &ljDescriptor;
}
const rgpot_plugin_descriptor_t *builtinClassicalMorse() {
  return &morseDescriptor;
}
const rgpot_plugin_descriptor_t *builtinSuttonChen() {
  return &suttonChenDescriptor;
}
#ifdef RGPOT_HAS_EAM_TABLE
const rgpot_plugin_descriptor_t *builtinEamFuncfl() {
  return &eamFuncflDescriptor;
}
const rgpot_plugin_descriptor_t *builtinEamSetfl() {
  return &eamSetflDescriptor;
}
#endif
} // namespace rgpot::plugin
