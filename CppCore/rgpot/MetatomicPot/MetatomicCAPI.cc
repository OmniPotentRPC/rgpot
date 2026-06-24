// MIT License
// Copyright 2023--present rgpot developers

#include "rgpot/MetatomicPot/MetatomicPot.hpp"
#include "rgpot/metatomic.h"

#include <cstddef>
#include <vector>

namespace {

extern "C" rgpot_status_t
rgpot_metatomic_callback(void *user_data, const rgpot_force_input_t *input,
                         rgpot_force_out_t *output) {
  try {
    if (user_data == nullptr || input == nullptr || output == nullptr ||
        input->positions == nullptr) {
      return RGPOT_INVALID_PARAMETER;
    }

    auto *pot = static_cast<rgpot::MetatomicPot *>(user_data);

    auto *pos_tensor = &input->positions->dl_tensor;
    size_t n_atoms = static_cast<size_t>(pos_tensor->shape[0]);

    const double *pos = static_cast<const double *>(pos_tensor->data);
    const int *atmnrs =
        input->atomic_numbers
            ? static_cast<const int *>(input->atomic_numbers->dl_tensor.data)
            : nullptr;
    const double *box =
        input->box_matrix
            ? static_cast<const double *>(input->box_matrix->dl_tensor.data)
            : nullptr;

    std::vector<double> forces(n_atoms * 3, 0.0);
    rgpot::ForceInput force_input{
        .nAtoms = n_atoms, .pos = pos, .atmnrs = atmnrs, .box = box};
    rgpot::ForceOut force_out{
        .F = forces.data(), .energy = 0.0, .variance = 0.0};

    pot->forceImpl(force_input, &force_out);

    output->energy = force_out.energy;
    output->variance = force_out.variance;
    output->forces =
        rgpot_tensor_owned_cpu_f64_2d(forces.data(), static_cast<int64_t>(n_atoms), 3);
    return RGPOT_SUCCESS;
  } catch (...) {
    return RGPOT_INTERNAL_ERROR;
  }
}

extern "C" void rgpot_metatomic_free(void *user_data) {
  delete static_cast<rgpot::MetatomicPot *>(user_data);
}

} // namespace

extern "C" rgpot_potential_t *
rgpot_metatomic_potential_new(const rgpot_metatomic_config_t *config) {
  if (config == nullptr || config->model_path == nullptr) {
    return nullptr;
  }

  try {
    rgpot::MetatomicConfig cpp_config{};
    cpp_config.model_path = config->model_path;
    cpp_config.device = config->device == nullptr ? "" : config->device;
    cpp_config.length_unit =
        config->length_unit == nullptr ? "angstrom" : config->length_unit;
    cpp_config.extensions_directory =
        config->extensions_directory == nullptr ? ""
                                                : config->extensions_directory;
    cpp_config.check_consistency = config->check_consistency;
    cpp_config.uncertainty_threshold = config->uncertainty_threshold;
    cpp_config.dtype_override =
        config->dtype_override == nullptr ? "" : config->dtype_override;

    auto *pot = new rgpot::MetatomicPot(cpp_config);
    return rgpot_potential_new(rgpot_metatomic_callback, pot,
                               rgpot_metatomic_free);
  } catch (...) {
    return nullptr;
  }
}
