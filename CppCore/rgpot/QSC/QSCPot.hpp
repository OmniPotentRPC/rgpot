#pragma once
// MIT License
// Copyright 2023--present rgpot developers

/**
 * @file QSCPot.hpp
 * @author rgpot Developers
 * @date 2026-11-19
 * @brief Quantum Sutton-Chen multi-element alloy potential.
 *
 * Implements the Quantum Sutton-Chen (Q-SC) many-body potential of Y. Qi,
 * T. Cagin, Y. Kimura and W. A. Goddard III, Phys. Rev. B 59, 3527 (1999),
 * doi:10.1103/PhysRevB.59.3527, Eqs. (1a)-(1c) with the combination rules of
 * Eq. (2) (after H. Rafii-Tabar and A. P. Sutton, Philos. Mag. Lett. 63, 217
 * (1991), doi:10.1080/09500839108205994). The repulsive term and the density
 * both use the pair-mixed parameters: rho_i = sum_j (a_ij / r_ij)^m_ij. The
 * quantum correction of the Q-SC force field is carried by the fitted
 * parameters; the functional form is that of Sutton-Chen. The default table
 * holds eight metals (Ni, Cu, Rh, Pd, Ag, Ir, Pt, Au) as tabulated in eOn
 * client/potentials/QSC/Parameters.h, with the Cu length scale of Table I.
 */

// clang-format off
#include <cstddef>
#include <cstdint>
#include <memory>
// clang-format on

#include "rgpot/Locality.hpp"
#include "rgpot/PerAtom.hpp"
#include "rgpot/Potential.hpp"
#include "rgpot/Staged.hpp"
#include "rgpot/nlist/Neighbours.hpp"
#include "rgpot/pot_caps.hpp"
#include "rgpot/types/AtomMatrix.hpp"

namespace rgpot {

/**
 * @brief Parameters for the Quantum Sutton-Chen potential.
 *
 * Plain aggregate with inline defaults; cutoff and skin equal the defaults of
 * eOn's @c QSC class.
 */
struct QSCConfig {
  double cutoff =
      8.0; //!< Truncation distance (Angstrom); 0 or less keeps every pair.
  double verlet_skin = 0.5; //!< Verlet neighbour-list skin (Angstrom).
  bool periodic = true;     //!< Apply periodic minimum-image convention.
  bool stress =
      false; //!< Write the Cauchy stress when the cell volume is positive.
};

/**
 * @class QSCPot
 * @brief Quantum Sutton-Chen energy, analytic forces, stress and HVP.
 * @ingroup rgpot_potentials
 */
class QSCPot : public Potential<QSCPot> {
public:
  /**
   * @brief Constructs the potential with the default @c QSCConfig.
   */
  QSCPot() : QSCPot(QSCConfig{}) {}

  /**
   * @brief Constructs the potential from @a c and the default Q-SC table.
   * @param c Configuration parameters.
   */
  explicit QSCPot(const QSCConfig &c);

  /**
   * @brief Copies the potential with its own parameter table.
   *
   * The copy starts with the elements of @a other, and a later
   * setElementParameter() on either one does not reach the other.
   * @param other Potential to copy.
   */
  QSCPot(const QSCPot &other);

  /**
   * @brief Replaces this potential by a copy with its own parameter table.
   * @param other Potential to copy.
   * @return This potential.
   */
  QSCPot &operator=(const QSCPot &other);

  /**
   * @brief Moves the potential and its parameter table.
   *
   * A moved-from potential can be destroyed, assigned to, or copied.
   * @param other Potential to move from.
   */
  QSCPot(QSCPot &&other) = default;

  /**
   * @brief Move-assigns the potential and its parameter table.
   * @param other Potential to move from.
   * @return This potential.
   */
  QSCPot &operator=(QSCPot &&other) = default;

  /**
   * @brief Makes a replica that shares this potential's parameter table.
   *
   * A host that keeps one replica per thread uses this so that
   * setElementParameter() on any of them reaches all of them. Each replica
   * has its own neighbour list and counters.
   * @return A potential backed by the same parameter table.
   */
  [[nodiscard]] QSCPot sharedReplica();

  /**
   * @brief Returns the configuration struct used at construction.
   * @return Reference to the @c QSCConfig.
   */
  [[nodiscard]] const QSCConfig &config() const noexcept { return m_config; }

  /**
   * @brief Adds or replaces the Quantum Sutton-Chen parameters of element @a z.
   * @param z Atomic number (must be positive).
   * @param n Repulsive exponent (at least 1).
   * @param m Density exponent (at least 1).
   * @param epsilon Energy scale in eV (positive).
   * @param c Dimensionless embedding coefficient (positive).
   * @param a Lattice length scale in Angstrom (positive).
   * @return Void.
   */
  void setElementParameter(int z, double n, double m, double epsilon, double c,
                           double a);

  /**
   * @brief Reports the concurrency and derivative capabilities.
   * @return Capability descriptor.
   */
  [[nodiscard]] PotCaps caps() const noexcept override {
    return {.reentrancy = Reentrancy::PerInstance,
            .periodic = m_config.periodic,
            .stress = m_config.stress,
            .analyticHvp = true};
  }

  /**
   * @brief Returns the 64-bit FNV-1a fingerprint of the active parameters.
   * @return Parameter key for result caching.
   */
  [[nodiscard]] uint64_t paramsKey() const noexcept override;

  /**
   * @brief Declares the site cutoff and single scalar density per atom.
   * @return Locality descriptor.
   */
  [[nodiscard]] Locality locality() const noexcept {
    return {.cutoff = m_config.cutoff > 0.0 ? m_config.cutoff : 0.0,
            .densities = 1};
  }

  /**
   * @brief Computes energy, forces and optional Cauchy stress.
   * @param in Input geometry and species.
   * @param out Output energy, forces and stress.
   * @return Void.
   */
  void forceImpl(const ForceInput &in, ForceOut *out) const override;

  /**
   * @brief Computes energy, forces, stress and per-atom energies and virials.
   * @param in Input geometry and species.
   * @param out Output energy, forces and stress.
   * @param atoms Per-atom energy and Voigt virial buffers.
   * @return Void.
   */
  void forcePerAtom(const ForceInput &in, ForceOut *out,
                    const PerAtomOut &atoms) const;

  /**
   * @brief Computes the exact analytic Hessian-vector product @f$H v@f$.
   * @param in Input geometry and species.
   * @param v Direction vector of length @c 3*in.nAtoms.
   * @param hv Output product buffer of length @c 3*in.nAtoms.
   * @return Void.
   */
  void hessianVectorImpl(const ForceInput &in, const double *v,
                         double *hv) const override;

  /**
   * @brief Accumulates pair density contributions on a host's half list.
   * @param list Host half neighbour list.
   * @param densities Per-atom density array of length @c list.nAtoms.
   * @return Void.
   */
  void densityStage(const HalfList &list, double *densities) const;

  /**
   * @brief Evaluates embedding and pair forces on a host's half list.
   * @param list Host half neighbour list.
   * @param densities Completed per-atom densities.
   * @param forces Force accumulator of length @c 3*list.nAtoms.
   * @param tally Optional energy, virial and per-atom tally.
   * @return Void.
   */
  void forceStage(const HalfList &list, const double *densities, double *forces,
                  StageTally *tally) const;

  /**
   * @brief Selects the stored, streaming or automatic neighbour-list mode.
   * @param mode Neighbour mode.
   * @return Void.
   */
  void setNeighbourMode(nlist::Mode mode) noexcept { mode_ = mode; }

  /**
   * @brief Reports whether the last force call reused its stored Verlet list.
   * @return True when the stored list was reused.
   */
  [[nodiscard]] bool listReused() const noexcept {
    return neighbours_.reused();
  }

private:
  static constexpr uint64_t kKernelVersion = 1;

  struct TableState;
  struct SharedTable {};

  QSCPot(const QSCPot &other, SharedTable);

  void compute(const ForceInput &in, ForceOut *out,
               const PerAtomOut *atoms) const;

  QSCConfig m_config;
  std::shared_ptr<TableState> m_state;
  nlist::Mode mode_ = nlist::Mode::Auto;
  mutable nlist::Neighbours neighbours_;
};

} // namespace rgpot
