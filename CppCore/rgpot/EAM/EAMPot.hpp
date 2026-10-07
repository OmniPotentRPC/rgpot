#pragma once
// MIT License
// Copyright 2023--present rgpot developers
//
// The cell-list EAM kernel is the eOn source compiled with EAM_STANDALONE,
// BSD-3-Clause, copyright the eOn Development Team.

#include <cstdint>

#include "rgpot/ParamHash.hpp"
#include "rgpot/Potential.hpp"

namespace rgpot {

/// The cell-list EAM has no free parameters. The element table is the
/// one shipped with the kernel.
struct EAMCellConfig {};

class EAMPot : public Potential<EAMPot> {
public:
  EAMPot();
  explicit EAMPot(const EAMCellConfig &);
  ~EAMPot() override;

  EAMPot(const EAMPot &) = delete;
  EAMPot &operator=(const EAMPot &) = delete;

  [[nodiscard]] PotCaps caps() const noexcept override {
    return {.reentrancy = Reentrancy::PerInstance, .perImageInstances = true};
  }

  [[nodiscard]] uint64_t paramsKey() const noexcept override {
    return m_paramsKey;
  }

  void forceImpl(const ForceInput &in, ForceOut *out) const override;

private:
  struct Impl;
  Impl *m_impl;
  uint64_t m_paramsKey = 0;
};

} // namespace rgpot
