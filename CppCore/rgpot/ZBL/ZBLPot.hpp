#pragma once
// MIT License
// Copyright 2023--present rgpot developers

/**
 * @brief Header file for the screened nuclear repulsion (ZBL) potential.
 *
 * This file defines the @c ZBLPot class, a universal screened Coulomb
 * repulsion with the LAMMPS switching function. Based on J. F. Ziegler,
 * J. P. Biersack and U. Littmark, "The Stopping and Range of Ions in
 * Matter", Pergamon (1985). The kernel is ported from eOn
 * (https://github.com/TheochemUI/eOn, client/potentials/ZBL),
 * BSD-3-Clause licensed, copyright the eOn Development Team, which in turn
 * adapts the GPL-licensed LAMMPS ``pair_zbl`` implementation.
 */

// clang-format off
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>
// clang-format on
#include "rgpot/Locality.hpp"
#include "rgpot/ParamHash.hpp"
#include "rgpot/PerAtom.hpp"
#include "rgpot/Potential.hpp"
#include "rgpot/Staged.hpp"
#include "rgpot/nlist/Neighbours.hpp"
#include "rgpot/pot_caps.hpp"
#include "rgpot/types/AtomMatrix.hpp"

namespace rgpot {

/**
 * @brief Parameters for the ZBL potential.
 *
 * Plain aggregate with inline defaults matching eOn's ``[ZBLPot]``
 * options: construct with designated initializers.
 */
struct ZBLConfig {
  double cut_inner = 2.0;  //!< Distance where switching starts (Angstrom).
  double cut_global = 2.5; //!< Truncation distance (Angstrom).
};

/// Screening and switching coefficients for one ordered pair of species.
struct ZblPairCoeffs {
  double d1 = 0.0;  //!< Screening exponent 1, scaled by the pair radius.
  double d2 = 0.0;  //!< Screening exponent 2, scaled by the pair radius.
  double d3 = 0.0;  //!< Screening exponent 3, scaled by the pair radius.
  double d4 = 0.0;  //!< Screening exponent 4, scaled by the pair radius.
  double zze = 0.0; //!< Z_i Z_j e^2 in eV Angstrom.
  double sw1 = 0.0; //!< Quadratic switching coefficient of the force.
  double sw2 = 0.0; //!< Cubic switching coefficient of the force.
  double sw3 = 0.0; //!< Cubic switching coefficient of the energy.
  double sw4 = 0.0; //!< Quartic switching coefficient of the energy.
  double sw5 = 0.0; //!< Constant energy offset at the cutoff.
};

/// Species-resolved coefficient tables for one set of atomic numbers.
struct ZblTables {
  std::vector<int> z;               //!< Sorted unique atomic numbers.
  std::vector<ZblPairCoeffs> pairs; //!< z.size() x z.size(), row-major.
};

/**
 * @class ZBLPot
 * @brief Universal screened nuclear repulsion with a switched cutoff.
 * @ingroup rgpot_potentials
 */
class ZBLPot : public Potential<ZBLPot> {
public:
  /// A call with no cell is an open system, not an error; see
  /// @c kernelCellOptional.
  static constexpr bool kCellOptional = true;

  ZBLPot() : ZBLPot(ZBLConfig{}) {}

  /**
   * @brief Constructs the potential from its cutoff pair.
   * @param c Configuration; requires 0 < cut_inner < cut_global.
   * @throws std::invalid_argument when the cutoff ordering is violated.
   */
  explicit ZBLPot(const ZBLConfig &c);

  [[nodiscard]] const ZBLConfig &config() const noexcept { return m_config; }

  [[nodiscard]] uint64_t paramsKey() const noexcept override {
    return m_paramsKey;
  }

  [[nodiscard]] PotCaps caps() const noexcept override {
    return {.reentrancy = Reentrancy::PerInstance, .stress = true};
  }

  /// Pairs count out to the global cutoff; a cutoff <= 0 has no pairs.
  [[nodiscard]] Locality locality() const noexcept {
    return {.cutoff = cut_global > 0.0 ? cut_global : 0.0, .pairwise = true};
  }

  /**
   * @brief Computes the forces and energy for a given configuration.
   * @param in Structure containing coordinates and cell info.
   * @param out Pointer to the results structure.
   * @return Void.
   */
  void forceImpl(const ForceInput &in, ForceOut *out) const override;

  /// forceImpl plus each pair energy and virial split half to each atom.
  void forcePerAtom(const ForceInput &in, ForceOut *out, const PerAtomOut &atoms) const;

  /// Staged.hpp: a pair potential has no densities.
  void densityStage(const HalfList &, double *) const {}

  /// Staged.hpp: the pair terms of forceImpl on a host's list.
  void forceStage(const HalfList &list, const double *densities, double *forces,
                  StageTally *tally) const;

  /// Stored list, streaming walk, or Auto (memory and the candidate bound).
  void setNeighbourMode(nlist::Mode mode) noexcept { mode_ = mode; }

  /// Atoms from which a call adds its terms atom by atom in index order,
  /// on any thread count: the least in kTeamFrom.
  static constexpr std::size_t kTeamMinAtoms = 192;
  /// Atoms from which a team of two threads opens, of three, and of four
  /// or more: the smallest clusters from which each took at most nine
  /// tenths of the one-thread time in thread_count_bench. A team evaluates
  /// each pair twice, and no team of two did.
  static constexpr TeamFrom kTeamFrom{kNoTeam, 768, kTeamMinAtoms};

  /// True when the last force call kept the Verlet list.
  [[nodiscard]] bool listReused() const noexcept { return neighbours_.reused(); }

  /// Pair indices copied out of the walk. Stream keeps none.
  [[nodiscard]] std::size_t retainedPairs() const noexcept {
    return neighbours_.streaming() ? 0 : neighbours_.slots();
  }

  [[nodiscard]] static double neighbourSkin(double cutoff) noexcept {
    return cutoff / 5.0;
  }

private:
  /// Bump when the kernel numerics change, so stale cache entries die.
  static constexpr uint64_t kKernelVersion = 2;

  void compute(const ForceInput &in, ForceOut *out, const PerAtomOut *atoms) const;

  /**
   * @brief Coefficient tables for the species present in a system.
   *
   * Tables are immutable once built and handed out as shared_ptr, so a
   * shared instance evaluated from several threads never races: the mutex
   * covers the lookup, never the physics. A system whose species set
   * differs from the cached one triggers a rebuild.
   */
  [[nodiscard]] std::shared_ptr<const ZblTables> tablesFor(const int *atomicNrs,
                                                           long N) const;

  /// Fills the switching coefficients for every species pair.
  [[nodiscard]] std::shared_ptr<const ZblTables>
  buildTables(std::vector<int> uniqueZ) const;

  double cut_inner;
  double cut_global;
  /// Switching turns on above this squared distance.
  double cut_inner_sq;
  ZBLConfig m_config;
  uint64_t m_paramsKey{0};

  mutable std::mutex m_tablesMtx;
  mutable std::shared_ptr<const ZblTables> m_tables;
  nlist::Mode mode_ = nlist::Mode::Auto;
  mutable nlist::Neighbours neighbours_;
};

} // namespace rgpot
