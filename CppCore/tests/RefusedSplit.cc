// MIT License
// Copyright 2023--present rgpot developers

#include "rgpot/CalculatorGroup.hpp"

#include <cstdio>

#include <mpi.h>

// mpirun -n 12. Five ranks per calculator do not divide 12. Every rank
// reports that world size and a negative group index. calculatorsUseMpi
// follows MPI_Initialized.
int main() {
  int inited = 0;
  MPI_Initialized(&inited);
  if (rgpot::calculatorsUseMpi() != (inited ? 1 : 0))
    return 3;

  const rgpot::CalculatorGroup group = rgpot::bindCalculators(5);
  // mpirun treats an initialized process that skips MPI_Finalize as a failed job.
  rgpot::finalizeMpiAtExit();
  MPI_Initialized(&inited);
  if (!inited || rgpot::calculatorsUseMpi() != 1)
    return 4;

  int size = 0;
  int rank = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (size != 12)
    return 5;
  if (group.index >= 0 || rgpot::thisCalculator().index >= 0)
    return 6;
  if (group.world_size != 12 || rgpot::calculatorWorldSize() != 12)
    return 7;
  if (rgpot::calculatorCount() != 1)
    return 8;

  std::printf("refuse-split rank %d world %d index %d\n", rank,
              rgpot::calculatorWorldSize(), group.index);
  return 0;
}
