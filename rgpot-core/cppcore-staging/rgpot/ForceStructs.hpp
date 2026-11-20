#pragma once
// MIT License
// Copyright 2023--present rgpot developers

/**
 * @brief POD structures for force and energy calculation interfaces.
 *
 * Defines the core data exchange structures used between the
 * high-level potential wrappers and the low-level physics engines.
 */

#include <cstddef>

namespace rgpot {

/**
 * @brief Data structure containing input configuration for force calls.
 * @ingroup rgpot
 */
typedef struct {
  const size_t nAtoms; //!< Total number of atoms in the system.
  const double *pos;   //!< Pointer to the flat array of atomic positions.
  const int *atmnrs;   //!< Pointer to the array of atomic numbers.
  const double *box;   //!< Pointer to the 3x3 simulation cell matrix.
} ForceInput;

/**
 * @brief Data structure to store results from force calculations.
 * @ingroup rgpot
 */
typedef struct {
  double *F;       //!< Pointer to the array where forces will be stored.
  double energy;   //!< Calculated potential energy of the configuration.
  double variance; //!< Variance or uncertainty of the calculation.
  // Variance here is 0 when not needed and that's OK.
  // stress is row-major xx,xy,xz,yx,yy,yz,zx,zy,zz in the result energy
  // and length units. has_stress is 0 when the engine did not compute it.
  double stress[9];
  int has_stress;
} ForceOut;

/**
 * @brief One batch of independent systems for a single evaluation call.
 * @ingroup rgpot
 *
 * Systems in a batch share nothing: each carries its own atom count, cell
 * and species list, so a caller may batch NEB images of one system or a
 * set of unrelated structures. `in` and `out` are parallel arrays of
 * length `nSystems`, and `out[i]` corresponds to `in[i]`.
 *
 * The caller owns both arrays and every buffer they point at, including
 * each `out[i].F`, which must have room for `3 * in[i].nAtoms` doubles.
 */
typedef struct {
  size_t nSystems;      //!< Number of independent systems in this batch.
  const ForceInput *in; //!< Array of nSystems inputs.
  ForceOut *out;        //!< Array of nSystems result slots.
} ForceBatch;

} // namespace rgpot
