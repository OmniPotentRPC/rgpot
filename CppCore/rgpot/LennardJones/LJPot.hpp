#pragma once
// MIT License
// Copyright 2023--present rgpot developers

/**
 * @brief Header file for the Lennard-Jones potential class.
 *
 * This file defines the @c LJPot class, which implements a standard  12-6
 * Lennard-Jones potential with a shifted cutoff for use in  atomic simulations.
 */

// clang-format off
#include <utility>
#include <vector>
#include <stdexcept>
// clang-format on
#include "rgpot/Potential.hpp"
#include "rgpot/types/AtomMatrix.hpp"
using rgpot::types::AtomMatrix;

namespace rgpot {

/**
 * @class LJPot
 * @brief Implementation of a shifted 12-6 Lennard-Jones potential.
 * @ingroup rgpot_potentials
 */
class LJPot : public Potential<LJPot> {
public:
  /**
   * @brief Default constructor initializing parameters.
   */
  LJPot() : Potential(PotType::LJ), u0{1.0}, cuttOffR{15.0}, psi{1.0} {}

  /**
   * @brief Computes the forces and energy for a given configuration.
   * @param in Structure containing coordinates and cell info.
   * @param out Pointer to the results structure.
   * @return Void.
   */
  void forceImpl(const ForceInput &in, ForceOut *out) const override;

private:
  double u0;       //!< Well depth parameter.
  double cuttOffR; //!< Distance beyond which potential is truncated.
  double psi;      //!< Distance at which the inter-particle potential is zero.
  double cuttOffU; //!< Potential energy value at the cutoff distance.
};

} // namespace rgpot
