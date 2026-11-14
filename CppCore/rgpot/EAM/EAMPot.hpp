#pragma once
// MIT License
// Copyright 2023--present rgpot developers
//
// The cell-list EAM kernel is the eOn source compiled with EAM_STANDALONE,
// BSD-3-Clause, copyright the eOn Development Team.

#include <cstddef>
#include <cstdint>

#include "rgpot/ParamHash.hpp"
#include "rgpot/Potential.hpp"
#include "rgpot/nlist/Neighbours.hpp"

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

  /// Neighbour-list builds. A second call on a geometry inside the Verlet
  /// skin does not increment this.
  [[nodiscard]] std::uint64_t listBuilds() const noexcept;

  /// Stored neighbour indices after the last build.
  [[nodiscard]] std::size_t neighborSlots() const noexcept;

  /// True when the last force call reused the Verlet list.
  [[nodiscard]] bool listReused() const noexcept;

  /// Partner indices kept between EAM stream passes. The density-only
  /// stream keeps none.
  [[nodiscard]] std::size_t streamPartners() const noexcept;

  /// Threads used by the most recent force call.
  [[nodiscard]] int threadsUsed() const noexcept;

  /// Stored Verlet list or a streaming cell walk. Auto follows the
  /// memory available to the process.
  void setNeighbourMode(nlist::Mode mode) noexcept;

  void forceImpl(const ForceInput &in, ForceOut *out) const override;

  /// forceImpl plus each atom's share of the energy: half of every pair
  /// term to each partner and the embedding term on its own atom. See
  /// Potential::forceEnergies.
  void forceEnergiesImpl(const ForceInput &in, ForceOut *out,
                         double *energies) const;

private:
  struct Impl;
  Impl *m_impl;
  uint64_t m_paramsKey = 0;
};

} // namespace rgpot
