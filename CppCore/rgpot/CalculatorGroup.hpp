#pragma once
// MIT License
// Copyright 2023--present rgpot developers

#include <cstddef>

/// One independent calculator is one group. A NEB image or a dimer end
/// is that group, and ranks in a group share a subcommunicator.
/// bindCalculators is collective on MPI_COMM_WORLD. Every rank calls it
/// before the first force.
/// shareFromCalculator broadcasts from the first rank of calculator
/// owner onto MPI_COMM_WORLD.
/// forceImpl fills ForceOut on the rank that calls it.
namespace rgpot {

/// One independent calculator. A NEB image, a dimer end, or any other
/// concurrent evaluation is one group. Ranks in a group share a
/// subcommunicator. A second band is a second world, not a second
/// split of the same one.
struct CalculatorGroup {
  int index = 0;
  int ranks = 1;
  int rank_in_group = 0;
  int world_size = 1;
};

// A backend that must own the subcommunicator (CPMD, and the same
// shape for any other MPI engine) registers this. It is called on
// every rank from bindCalculators. Return the group index, or -1
// to refuse the bind.
using CalculatorHook = int (*)(int ranks_per_calculator);

void addCalculatorHook(CalculatorHook hook);

/// Collective on MPI_COMM_WORLD when the process was started under MPI.
/// ranks_per_calculator <= 0 means one group for the whole world.
/// Returns index -1 when the world cannot be divided that way.
CalculatorGroup bindCalculators(int ranks_per_calculator);

const CalculatorGroup &thisCalculator();

// Copies the group's MPI_Comm into *comm_out when this build has MPI
// and a split exists. comm_bytes is sizeof(MPI_Comm) on the caller.
// Returns 0 when there is no communicator to give.
int calculatorComm(void *comm_out, std::size_t comm_bytes);

// 1 when MPI_Initialized reports that MPI is up. 0 when this build has
// no MPI, and 0 when MPI is not initialized.
int calculatorsUseMpi();

// MPI_COMM_WORLD size recorded by bindCalculators, including a refused
// split. 1 while calculators are unbound.
int calculatorWorldSize();

// Number of calculators the world is split into. 1 before a split and
// when the world could not be divided.
int calculatorCount();

/// Broadcasts bytes from the first rank of calculator `owner` to every
/// rank of MPI_COMM_WORLD, so all ranks hold that calculator's result.
/// Collective on MPI_COMM_WORLD. When more than one rank is bound, a
/// rank that cannot enter the broadcast aborts the world after every
/// rank prints the error. Returns 0 without MPI, before a split, or
/// for a bad owner in a single process. Returns 1 once the bytes are
/// in place.
int shareFromCalculator(int owner, void *data, std::size_t bytes);

// Registers, once per process, an exit handler that calls MPI_Finalize
// when bindCalculators called MPI_Init and MPI is not yet finalized.
// A host that called MPI_Init keeps that call and finalizes itself.
// No-op without MPI.
void finalizeMpiAtExit();

} // namespace rgpot
