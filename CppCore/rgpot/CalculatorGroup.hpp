#pragma once
// MIT License
// Copyright 2023--present rgpot developers

#include <cstddef>

namespace rgpot {

// One independent calculator. A NEB image, a dimer end, or any other
// concurrent evaluation is one group. Ranks in a group share a
// subcommunicator. A second band is a second world, not a second
// split of the same one.
struct CalculatorGroup {
  int index = 0;
  int ranks = 1;
  int rank_in_group = 0;
  int world_size = 1;
};

// A backend that must own the subcommunicator (CPMD, and the same
// shape for any other MPI engine) registers this. It is called on
// every rank from bindCalculators. Return the group index, or -1.
using CalculatorHook = int (*)(int ranks_per_calculator);

void addCalculatorHook(CalculatorHook hook);

// Collective on MPI_COMM_WORLD when the process was started under MPI.
// ranks_per_calculator <= 0 means one group for the whole world.
// Returns index -1 when the world cannot be divided that way.
CalculatorGroup bindCalculators(int ranks_per_calculator);

const CalculatorGroup &thisCalculator();

// Copies the group's MPI_Comm into *comm_out when this build has MPI
// and a split exists. comm_bytes is sizeof(MPI_Comm) on the caller.
// Returns 0 when there is no communicator to give.
int calculatorComm(void *comm_out, std::size_t comm_bytes);

} // namespace rgpot
