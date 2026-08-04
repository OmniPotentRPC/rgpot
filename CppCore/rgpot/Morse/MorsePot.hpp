#pragma once
// MIT License
// Copyright 2023--present rgpot developers

/**
 * @brief Header file for the Morse potential class.
 *
 * This file defines the @c MorsePot class, which implements the Morse pair
 * potential in the form Doye and Wales use for cluster global optimization,
 * with a shifted cutoff.
 */

// clang-format off
#include <cmath>
#include <utility>
#include <vector>
#include <stdexcept>
// clang-format on
#include "rgpot/Potential.hpp"
#include "rgpot/types/AtomMatrix.hpp"
using rgpot::types::AtomMatrix;

namespace rgpot {

/**
 * @class MorsePot
 * @brief Morse pair potential with a shifted cutoff.
 * @ingroup rgpot_potentials
 *
 * The pair term is
 *
 * @f[
 *   V(r) = \epsilon\, e^{\rho_0 (1 - r/r_0)}
 *          \left( e^{\rho_0 (1 - r/r_0)} - 2 \right)
 * @f]
 *
 * which has depth @f$\epsilon@f$ at @f$r = r_0@f$. @f$\rho_0@f$ sets the range
 * of the force: larger values narrow the well without moving its minimum, so
 * one parameter sweeps a family of landscapes from long-ranged to short-ranged.
 *
 * This is the form and the reduced units of Doye and Wales, *Structural
 * consequences of the range of the interatomic potential* (J. Chem. Phys. 1996;
 * Cambridge Cluster Database), whose published global minima are what a
 * cluster search is checked against. Their tabulated energies are in units of
 * the pair well depth with @f$\epsilon = 1@f$ and @f$r_0 = 1@f$, which is what
 * the default constructor gives.
 */
class MorsePot : public Potential<MorsePot> {
public:
  /**
   * @brief Constructs the potential from depth, range, equilibrium separation
   *        and cutoff.
   * @param epsilon_ Pair well depth.
   * @param rho0_ Range parameter; larger is shorter ranged.
   * @param r0_ Separation at which the pair term is at its minimum.
   * @param cuttOffR_ Distance beyond which the potential is truncated.
   *
   * @c cuttOffU is the pair term evaluated at @c cuttOffR_ and is subtracted
   * from every pair, so the potential vanishes continuously at the cutoff.
   * Members initialize in declaration order, so @c cuttOffU may read the
   * parameters above it.
   */
  MorsePot(double epsilon_, double rho0_, double r0_, double cuttOffR_)
      : Potential(PotType::Morse), epsilon{epsilon_}, rho0{rho0_}, r0{r0_},
        cuttOffR{cuttOffR_},
        cuttOffU{epsilon * std::exp(rho0 * (1 - cuttOffR / r0)) *
                 (std::exp(rho0 * (1 - cuttOffR / r0)) - 2)} {}

  /**
   * @brief Default constructor, in the reduced units of the published tables.
   *
   * @f$\epsilon = 1@f$, @f$r_0 = 1@f$, @f$\rho_0 = 6@f$. The cutoff of 20
   * @f$r_0@f$ is far enough that the shift is below rounding for any
   * @f$\rho_0 \ge 3@f$, so a comparison against the tables is not affected by
   * it.
   */
  MorsePot() : MorsePot(1.0, 6.0, 1.0, 20.0) {}

  /**
   * @brief Computes the forces and energy for a given configuration.
   * @param in Structure containing coordinates and cell info.
   * @param out Pointer to the results structure.
   * @return Void.
   */
  void forceImpl(const ForceInput &in, ForceOut *out) const override;

private:
  double epsilon;  //!< Pair well depth.
  double rho0;     //!< Range parameter; larger is shorter ranged.
  double r0;       //!< Separation at the pair minimum.
  double cuttOffR; //!< Distance beyond which the potential is truncated.
  double cuttOffU; //!< Pair term evaluated at the cutoff.
};

} // namespace rgpot
