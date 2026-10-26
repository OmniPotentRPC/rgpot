#pragma once
// MIT License
// Copyright 2023--present rgpot developers

/**
 * @file TersoffSiC.hpp
 * @author rgpot Developers
 * @date 2026-10-26
 * @brief Tersoff Si-C potential, 1989 parameters and the 1990 erratum.
 *
 * J. Tersoff, Phys. Rev. B 39, 5566 (1989),
 * doi:10.1103/PhysRevB.39.5566, and the erratum Phys. Rev. B 41, 3248
 * (1990), doi:10.1103/PhysRevB.41.3248.2. The erratum sets carbon B to
 * 3.4674e2 eV (346.74 eV) and d to 4.3484. Carbon beta is dimensionless
 * and is the Table I value, not an erratum correction. This is not the
 * Fortran Tersoff silicon kernel.
 */

#include <cstddef>
#include <cstdint>

#include "rgpot/Potential.hpp"
#include "rgpot/nlist/HalfShell.hpp"

namespace rgpot {

/**
 * @brief Tersoff potential for silicon, carbon, and silicon-carbide.
 * @ingroup rgpot_potentials
 *
 * Atomic numbers are 14 and 6. One call evaluates a whole configuration.
 * @c energyAndGradientBatch evaluates many configurations that share a
 * cell and a species list.
 */
class TersoffSiC : public Potential<TersoffSiC> {
public:
  static constexpr double kChiSiC = 0.9776;

  /// Angular parameters of one element. Index 0 is carbon, 1 is silicon.
  struct Species {
    double c, d, h, n, beta, gamma;
    double c2, d2, beta_n;
  };
  /// Mixed pair. big_a and big_b are the repulsive and attractive amplitudes.
  struct Pair {
    double lambda1, lambda2, big_a, big_b, rin, rout;
  };

  TersoffSiC();

  [[nodiscard]] PotCaps caps() const noexcept override { return {}; }

  [[nodiscard]] uint64_t paramsKey() const noexcept override {
    return m_paramsKey;
  }

  void forceImpl(const ForceInput &in, ForceOut *out) const override;

  /**
   * @brief Energy and dE/dx for one configuration.
   * @param positions Row-major coordinates, Angstrom.
   * @param atomic_numbers Length n_atoms. 14 is silicon, 6 is carbon.
   * @param box Row-major cell, Angstrom. Periodic in every direction.
   * @param n_atoms Atom count.
   * @param energy Receives the energy, eV.
   * @param gradient Receives dE/dx, eV/Angstrom. The force is its negation.
   * @param neighbors Optional Verlet list. Null rebuilds the pairs on
   *        every call. A list is reused while each atom has moved by at
   *        most half its skin and the cell is unchanged.
   */
  void energyAndGradient(const double *positions, const int *atomic_numbers,
                         const double *box, std::size_t n_atoms, double *energy,
                         double *gradient,
                         nlist::NeighborList *neighbors = nullptr) const;

  /**
   * @brief Energy and dE/dx for configurations that share a cell.
   * @param n_config Number of configurations.
   * @param positions n_config blocks of n_atoms * 3 coordinates.
   * @param atomic_numbers Species shared by every configuration.
   * @param box Cell shared by every configuration.
   * @param n_atoms Atom count of each configuration.
   * @param energies Length n_config.
   * @param gradients Length n_config * n_atoms * 3.
   */
  void energyAndGradientBatch(std::size_t n_config, const double *positions,
                              const int *atomic_numbers, const double *box,
                              std::size_t n_atoms, double *energies,
                              double *gradients) const;

private:
  void evaluate(const double *positions, const int *atomic_numbers,
                const double *box, std::size_t n_atoms, double *energy,
                double *acc, double scale,
                nlist::NeighborList *neighbors) const;

  Species m_species[2]{};
  Pair m_pair[2][2]{};
  uint64_t m_paramsKey{0};
};

} // namespace rgpot
