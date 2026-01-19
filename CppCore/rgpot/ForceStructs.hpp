#pragma once
// MIT License
// Copyright 2023--present Rohit Goswami <HaoZeke>
#include <cstddef>

namespace rgpot {

typedef struct {
  // pointer to number of atoms, pointer to array of positions
  // address to supercell size
  const size_t nAtoms;
  const double *pos;
  const int *atmnrs;
  const double *box;
} ForceInput;

typedef struct {
  // pointer to array of forces
  double *F;
  // Internal energy
  double energy;
  // Variance here is 0 when not needed and that's OK
  double variance;
} ForceOut;

} // namespace rgpot
