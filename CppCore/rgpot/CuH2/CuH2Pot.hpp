#pragma once
// MIT License
// Copyright 2023--present Rohit Goswami <HaoZeke>

/**
 * @brief Header for the Copper-Hydrogen (CuH2) EAM potential.
 *
 * This file defines the @c CuH2Pot class, which wraps a Fortran-based
 * Embedded Atom Method (EAM) implementation. It provides an interface
 * specifically for copper-hydrogen systems.
 */

// clang-format off
#include <utility>
#include <vector>
// clang-format on
#include "rgpot/Potential.hpp"
#include "rgpot/types/AtomMatrix.hpp"
using rgpot::types::AtomMatrix;

/**
 * @brief C-linkage bridge to the Fortran EAM implementation.
 * @param natms  Array containing counts of Cu and H atoms.
 * @param ndim   Total dimensions (3 * N).
 * @param box    Diagonal box vectors (assumes cubic).
 * @param R      Pointer to flat position array.
 * @param F      Pointer to output force array.
 * @param U      Pointer to output energy value.
 * @return Void.
 */
extern "C" void c_force_eam(int *natms, int ndim, double *box, double *R,
                            double *F, double *U);
// natms(2), ndim, U(1), R(ndim), F(ndim), box(3)

namespace rgpot {

/**
 * @class CuH2Pot
 * @brief Implementation of the CuH2 EAM potential.
 * @ingroup rgpot_potentials
 */
class CuH2Pot : public Potential<CuH2Pot> {
public:
  /**
   * @brief Constructor for CuH2Pot.
   */
  CuH2Pot() : Potential(PotType::CuH2) {}

  /**
   * @brief Computes forces and energy using the Fortran backend.
   * @param in   Structure containing coordinates and cell info.
   * @param out  Pointer to the results structure.
   * @return Void.
   */
  void forceImpl(const ForceInput &in, ForceOut *out) const override;

private:
  /**
   * @brief Legacy eOn-compatible force interface.
   * @param N          Number of atoms.
   * @param R          Flat array of positions.
   * @param atomicNrs  Array of atomic numbers.
   * @param F          Output force array.
   * @param U          Output energy pointer.
   * @param box        Box vector array.
   * @return Void.
   */
  void force(long N, const double *R, const int *atomicNrs, double *F,
             double *U, const double *box) const;
};

} // namespace rgpot
