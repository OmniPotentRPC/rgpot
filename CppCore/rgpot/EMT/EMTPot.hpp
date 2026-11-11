#pragma once
// MIT License
// Copyright 2023--present rgpot developers
//
// The Asap effective-medium kernel is the eOn vendored tree
// (client/potentials/EMT/Asap), BSD-3-Clause, copyright the eOn
// Development Team.

#include <cstdint>

#include "rgpot/ParamHash.hpp"
#include "rgpot/Potential.hpp"
#include "rgpot/nlist/Neighbours.hpp"

namespace rgpot {

/// Parameters for the Asap effective-medium potential.
/// `rasmussen` selects EMTRasmussenParameterProvider. The default is the
/// Asap copper parameter set, which is what `emt_rasmussen = false` means.
struct EMTConfig {
  bool rasmussen = false;
};

class EMTPot : public Potential<EMTPot> {
public:
  EMTPot() : EMTPot(EMTConfig{}) {}
  explicit EMTPot(const EMTConfig &config);
  ~EMTPot() override;

  EMTPot(const EMTPot &) = delete;
  EMTPot &operator=(const EMTPot &) = delete;

  [[nodiscard]] const EMTConfig &config() const noexcept { return m_config; }

  [[nodiscard]] PotCaps caps() const noexcept override {
    return {.reentrancy = Reentrancy::PerInstance, .perImageInstances = true};
  }

  [[nodiscard]] uint64_t paramsKey() const noexcept override {
    return m_paramsKey;
  }

  /// Stored Verlet list or a streaming cell walk.
  void setNeighbourMode(nlist::Mode mode) noexcept { mode_ = mode; }

  /// Partner indices kept by the last force call. Zero while streaming.
  [[nodiscard]] int neighbourSlots() const noexcept { return slots_; }

  void forceImpl(const ForceInput &in, ForceOut *out) const override;

private:
  void reset() const;

  EMTConfig m_config;
  nlist::Mode mode_ = nlist::Mode::Stored;
  mutable int slots_ = 0;
  uint64_t m_paramsKey = 0;
  mutable long m_nAtoms = 0;
  mutable void *m_cell = nullptr;
  mutable void *m_atoms = nullptr;
  mutable void *m_provider = nullptr;
  mutable void *m_emt = nullptr;
};

} // namespace rgpot
