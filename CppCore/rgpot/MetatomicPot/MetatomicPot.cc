// MIT License
// Copyright 2023--present rgpot developers

#include "rgpot/MetatomicPot/MetatomicPot.hpp"
#include "vesin.h"

#include <torch/csrc/jit/runtime/graph_executor.h>

#include <cstdint>
#include <cstring>

using namespace std::string_literals;

namespace rgpot {

MetatomicPot::MetatomicPot(const MetatomicConfig &config)
    : Potential(PotType::Metatomic), m_config(config),
      m_model(torch::jit::Module()),
      m_device(torch::Device(c10::DeviceType::CPU)) {

  torch::jit::getProfilingMode() = false;

  // 1. Load model
  torch::optional<std::string> extensions_directory = torch::nullopt;
  if (!m_config.extensions_directory.empty()) {
    extensions_directory = m_config.extensions_directory;
  }

  m_model = metatomic_torch::load_atomistic_model(m_config.model_path,
                                                   extensions_directory);

  // 2. Extract capabilities and neighbor list requests
  m_capabilities =
      m_model.run_method("capabilities")
          .toCustomClass<metatomic_torch::ModelCapabilitiesHolder>();
  auto requests_ivalue = m_model.run_method("requested_neighbor_lists");
  for (const auto &request_ivalue : requests_ivalue.toList()) {
    auto request =
        request_ivalue.get()
            .toCustomClass<metatomic_torch::NeighborListOptionsHolder>();
    m_nl_requests.push_back(request);
  }

  // 3. Device selection
  torch::optional<std::string> desired = torch::nullopt;
  if (!m_config.device.empty()) {
    desired = m_config.device;
  }
  auto device_type =
      metatomic_torch::pick_device(m_capabilities->supported_devices, desired);
  m_device = torch::Device(device_type);

  // 4. Data type
  if (m_capabilities->dtype() == "float64") {
    m_dtype = torch::kFloat64;
  } else if (m_capabilities->dtype() == "float32") {
    m_dtype = torch::kFloat32;
  } else {
    throw std::runtime_error("Unsupported dtype: " + m_capabilities->dtype());
  }
  if (!m_config.dtype_override.empty()) {
    if (m_config.dtype_override == "float64") {
      m_dtype = torch::kFloat64;
    } else if (m_config.dtype_override == "float32") {
      m_dtype = torch::kFloat32;
    }
  }

  m_model.to(m_device);
  if (!m_config.dtype_override.empty() &&
      m_dtype != (m_capabilities->dtype() == "float64" ? torch::kFloat64
                                                       : torch::kFloat32)) {
    try {
      m_model.to(m_dtype);
    } catch (const std::exception &) {
      if (m_capabilities->dtype() == "float64") {
        m_dtype = torch::kFloat64;
      } else {
        m_dtype = torch::kFloat32;
      }
    }
  }

  // 5. Resolve energy output key
  auto outputs = m_capabilities->outputs();
  m_energy_key = metatomic_torch::pick_output("energy", outputs, torch::nullopt);
  if (!outputs.contains(m_energy_key)) {
    throw std::runtime_error("Missing energy output in metatomic model");
  }

  // 6. Evaluation options
  m_eval_options =
      torch::make_intrusive<metatomic_torch::ModelEvaluationOptionsHolder>();
  m_eval_options->set_length_unit(m_config.length_unit);

  auto model_output = outputs.at(m_energy_key);
  auto requested_output =
      torch::make_intrusive<metatomic_torch::ModelOutputHolder>();
  requested_output->per_atom = model_output->per_atom;
  requested_output->explicit_gradients = {};
  requested_output->set_quantity("energy");
  requested_output->set_unit("eV");
  m_eval_options->outputs.insert(m_energy_key, requested_output);

  m_check_consistency = m_config.check_consistency;
}

void MetatomicPot::forceImpl(const ForceInput &in, ForceOut *out) const {
  std::lock_guard<std::mutex> lock(m_mutex);

  long nAtoms = static_cast<long>(in.nAtoms);

  // 1. Create tensors
  auto f64_options =
      torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU);

  auto torch_positions =
      torch::from_blob(const_cast<double *>(in.pos), {nAtoms, 3}, f64_options)
          .to(m_dtype)
          .to(m_device)
          .set_requires_grad(true);

  auto torch_cell =
      torch::from_blob(const_cast<double *>(in.box), {3, 3}, f64_options)
          .to(m_dtype)
          .to(m_device);

  auto cell_norms = torch::norm(torch_cell, 2, /*dim=*/1);
  auto torch_pbc = cell_norms.abs() > 1e-9;
  bool periodic[3] = {torch_pbc[0].item<bool>(), torch_pbc[1].item<bool>(),
                      torch_pbc[2].item<bool>()};

  // Cache atomic types tensor (doesn't change between geometry steps)
  if (m_cached_natoms != in.nAtoms) {
    std::vector<int32_t> types_vec(in.atmnrs, in.atmnrs + nAtoms);
    m_cached_types =
        torch::tensor(types_vec, torch::TensorOptions().dtype(torch::kInt32))
            .to(m_device);
    m_cached_natoms = in.nAtoms;
  }
  auto atomic_types = m_cached_types;

  // 2. Create system
  auto system = torch::make_intrusive<metatomic_torch::SystemHolder>(
      atomic_types, torch_positions, torch_cell, torch_pbc);

  // 3. Compute neighbor lists
  for (const auto &request : m_nl_requests) {
    auto neighbors =
        computeNeighbors(request, nAtoms, in.pos, in.box, periodic);
    metatomic_torch::register_autograd_neighbors(system, neighbors,
                                                 m_check_consistency);
    system->add_neighbor_list(request, neighbors);
  }

  // 4. Run model
  auto ivalue_output = m_model.forward({
      std::vector<metatomic_torch::System>{system},
      m_eval_options,
      m_check_consistency,
  });
  auto dict_output = ivalue_output.toGenericDict();
  auto output_map = dict_output.at(m_energy_key)
                        .toCustomClass<metatensor_torch::TensorMapHolder>();

  // 5. Extract energy
  auto energy_block =
      metatensor_torch::TensorMapHolder::block_by_id(output_map, 0);
  auto energy_tensor = energy_block->values();
  out->energy = energy_tensor.sum().item<double>();

  // 6. Compute forces via autograd
  energy_tensor.backward(torch::ones_like(energy_tensor));
  auto positions_grad = system->positions().grad();
  auto forces_tensor = -positions_grad.to(torch::kCPU).to(torch::kFloat64);

  std::memcpy(out->F, forces_tensor.contiguous().data_ptr<double>(),
              in.nAtoms * 3 * sizeof(double));

  out->variance = 0.0;
}

metatensor_torch::TensorBlock MetatomicPot::computeNeighbors(
    metatomic_torch::NeighborListOptions request, long nAtoms,
    const double *positions, const double *box,
    const bool periodic[3]) const {

  auto cutoff = request->engine_cutoff(m_config.length_unit);

  VesinOptions options{};
  options.cutoff = cutoff;
  options.full = request->full_list();
  options.return_shifts = true;
  options.return_distances = false;
  options.return_vectors = true;

  VesinNeighborList *vesin_nl = new VesinNeighborList();

  // vesin <=0.2 / >=0.5: VesinDevice is a struct {type, device_id}.
  // vesin 0.3–0.4 briefly made VesinDevice an enum alias of VesinDeviceKind.
  // Brace-init works for the struct form shipped by current PyPI wheels.
  VesinDevice cpu{VesinCPU, 0};
  const char *error_message = nullptr;
  int status = vesin_neighbors(reinterpret_cast<const double(*)[3]>(positions),
                               static_cast<size_t>(nAtoms),
                               reinterpret_cast<const double(*)[3]>(box),
                               const_cast<bool *>(periodic), cpu, options,
                               vesin_nl, &error_message);

  if (status != EXIT_SUCCESS) {
    std::string err_str = "vesin_neighbors failed";
    if (error_message != nullptr) {
      err_str += ": " + std::string(error_message);
    }
    delete vesin_nl;
    throw std::runtime_error(err_str);
  }

  auto n_pairs = static_cast<int64_t>(vesin_nl->length);
  auto labels_options_cpu =
      torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU);

  auto pair_samples_values = torch::empty({n_pairs, 5}, labels_options_cpu);
  auto pair_samples_values_ptr = pair_samples_values.accessor<int32_t, 2>();
  for (int64_t i = 0; i < n_pairs; i++) {
    pair_samples_values_ptr[i][0] =
        static_cast<int32_t>(vesin_nl->pairs[i][0]);
    pair_samples_values_ptr[i][1] =
        static_cast<int32_t>(vesin_nl->pairs[i][1]);
    pair_samples_values_ptr[i][2] = vesin_nl->shifts[i][0];
    pair_samples_values_ptr[i][3] = vesin_nl->shifts[i][1];
    pair_samples_values_ptr[i][4] = vesin_nl->shifts[i][2];
  }

  auto deleter = [=](void *) {
    vesin_free(vesin_nl);
    delete vesin_nl;
  };

  auto pair_vectors = torch::from_blob(
      vesin_nl->vectors, {n_pairs, 3, 1}, deleter,
      torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

  auto neighbor_samples = torch::make_intrusive<metatensor_torch::LabelsHolder>(
      std::vector<std::string>{"first_atom", "second_atom", "cell_shift_a",
                               "cell_shift_b", "cell_shift_c"},
      pair_samples_values.to(m_device));

  auto labels_options_dev =
      torch::TensorOptions().dtype(torch::kInt32).device(m_device);
  auto neighbor_component =
      torch::make_intrusive<metatensor_torch::LabelsHolder>(
          "xyz", torch::tensor({0, 1, 2}, labels_options_dev).reshape({3, 1}));
  auto neighbor_properties =
      torch::make_intrusive<metatensor_torch::LabelsHolder>(
          "distance", torch::zeros({1, 1}, labels_options_dev));

  return torch::make_intrusive<metatensor_torch::TensorBlockHolder>(
      pair_vectors.to(m_dtype).to(m_device), neighbor_samples,
      std::vector<metatensor_torch::Labels>{neighbor_component},
      neighbor_properties);
}

} // namespace rgpot
