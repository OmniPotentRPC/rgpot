#pragma once
// MIT License
// Copyright 2023--present rgpot developers
//
// Neighbor lists use vesin. Source is compatible with both vesin 0.3.x
// (enum VesinDevice, no Options.algorithm) and 0.5+ (struct VesinDevice
// {type, device_id}, Options.algorithm) via type traits in vesin_compat.hpp.

#include <atomic>
#include <map>
#include <utility>

#include "rgpot/nlist/VerletPairs.hpp"
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

#include "rgpot/MetatomicPot/MetatomicConfig.hpp"
#include "rgpot/Potential.hpp"

namespace rgpot {

class MetatomicPot : public Potential<MetatomicPot> {
public:
  explicit MetatomicPot(const MetatomicConfig &config);
  ~MetatomicPot() = default;

  MetatomicPot(const MetatomicPot &) = delete;
  MetatomicPot &operator=(const MetatomicPot &) = delete;

  void forceImpl(const ForceInput &in, ForceOut *out) const override;

  /// One model forward over the batch when no orientation average is on.
  /// Rotations stay on the per-system path, which owns that average.
  void forceBatchImpl(const ForceBatch &batch) const override;

  void setChargeSpin(int charge, int spin) {
    m_config.charge = charge;
    m_config.spin = spin;
  }

  /// The torch session and the Verlet pair caches are per instance and
  /// unlocked: concurrent callers keep one instance per thread or image.
  /// Entering one instance from two threads throws (ExclusiveCall).
  [[nodiscard]] PotCaps caps() const noexcept override {
    return {.reentrancy = Reentrancy::PerInstance,
            .perImageInstances = true,
            .batched = true};
  }


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
  std::string m_energy_uncertainty_key;
  double m_uncertainty_threshold = -1.0;

  mutable std::atomic<bool> m_busy{false};
  /// One Verlet cache per (system slot, neighbor request): slot 0 for the
  /// single-system path, slot s + 1 for system s of a batch.
  mutable std::map<std::pair<size_t, size_t>, nlist::VerletPairCache>
      m_nl_caches;
  nlist::VerletPairCache &neighborCache(size_t slot, size_t request) const;

public:
  /// Candidate rebuilds and calls summed over every neighbor cache.
  [[nodiscard]] std::pair<uint64_t, uint64_t> neighborCacheStats() const;

private:
  mutable torch::Tensor m_cached_types;
  mutable size_t m_cached_natoms = 0;

  metatensor_torch::TensorBlock
  computeNeighbors(metatomic_torch::NeighborListOptions request,
                   nlist::VerletPairCache &cache, long nAtoms,
                   const double *positions, const double *box,
                   const bool periodic[3]) const;
};

} // namespace rgpot
