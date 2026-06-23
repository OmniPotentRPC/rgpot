#pragma once
// MIT License
// Copyright 2023--present rgpot developers

#include <mutex>
#include <string>
#include <vector>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wfloat-equal"
#pragma GCC diagnostic ignored "-Wfloat-conversion"
#pragma GCC diagnostic ignored "-Wsign-conversion"
#pragma GCC diagnostic ignored "-Wold-style-cast"

#include <torch/script.h>

#include "metatensor/torch.hpp"
#include "metatensor/torch/module.hpp"
#include "metatomic/torch.hpp"

#pragma GCC diagnostic pop

#include "rgpot/Potential.hpp"

namespace rgpot {

struct MetatomicConfig {
  std::string model_path;
  std::string device;
  std::string length_unit = "angstrom";
  std::string extensions_directory;
  bool check_consistency = false;
  double uncertainty_threshold = -1.0;
  std::string dtype_override;
};

class MetatomicPot : public Potential<MetatomicPot> {
public:
  explicit MetatomicPot(const MetatomicConfig &config);
  ~MetatomicPot() = default;

  MetatomicPot(const MetatomicPot &) = delete;
  MetatomicPot &operator=(const MetatomicPot &) = delete;

  void forceImpl(const ForceInput &in, ForceOut *out) const override;

private:
  MetatomicConfig m_config;

  mutable metatensor_torch::Module m_model;
  metatomic_torch::ModelCapabilities m_capabilities;
  std::vector<metatomic_torch::NeighborListOptions> m_nl_requests;
  metatomic_torch::ModelEvaluationOptions m_eval_options;

  torch::ScalarType m_dtype;
  torch::Device m_device;
  bool m_check_consistency;
  std::string m_energy_key;

  mutable std::mutex m_mutex;
  mutable torch::Tensor m_cached_types;  //!< Cached atomic types tensor.
  mutable size_t m_cached_natoms = 0;    //!< Atom count for cached types.

  metatensor_torch::TensorBlock
  computeNeighbors(metatomic_torch::NeighborListOptions request, long nAtoms,
                   const double *positions, const double *box,
                   const bool periodic[3]) const;
};

} // namespace rgpot
