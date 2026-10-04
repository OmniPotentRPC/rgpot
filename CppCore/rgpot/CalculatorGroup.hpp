#pragma once
// MIT License
// Copyright 2023--present rgpot developers

#include <cstddef>
#include <string>

/// One independent calculator is one group. A NEB image or a dimer end
/// is that group, and ranks in a group share a subcommunicator.
/// bindCalculators is collective on MPI_COMM_WORLD. Every rank calls it
/// before the first force.
/// shareFromCalculator broadcasts from the first rank of calculator
/// owner onto MPI_COMM_WORLD.
/// forceImpl fills ForceOut on the rank that calls it.
///
/// librgpot never links MPI. The MPI side lives in librgpot_mpi (meson
/// dependency and pkg-config name rgpot-mpi), which bindCalculators and
/// calculatorsUseMpi load with dlopen on first use through the C ABI in
/// calculator_mpi_abi.h. A process that never calls them never maps
/// libmpi; until the library is loaded every call below behaves as in a
/// build without MPI.
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

// A backend that must use the calculator subcommunicator registers this.
// It is called on every rank from bindCalculators. Return the group index,
// or -1 to refuse the bind.
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

// Passes the calculator's MPI_Comm to an engine without requiring MPI
// types in the caller. The callback receives the handle's address and
// byte size, then the ranks per calculator. It must borrow the handle,
// not free it. Returns -1 without a split or callback, else its result.
using CalculatorCommAdopter = int (*)(const void *, std::size_t, int);
int adoptCalculatorComm(CalculatorCommAdopter callback);

// Registers each callback once by address. Its code must stay loaded for
// the process lifetime. Registration before binding queues the callback;
// registration after binding gives it the existing communicator at once.
// Every rank registers the same callbacks in the same order. A callback
// must not reenter calculator binding or registration. Returns false for
// a null callback or a refused bind; a refusal remains in thisCalculator.
// Registration alone does not load MPI.
bool addCalculatorCommAdopter(CalculatorCommAdopter callback);

// Loads librgpot_mpi when needed, then 1 when MPI_Initialized reports
// that MPI is up. 0 when the library cannot be loaded, and 0 when MPI is
// not initialized.
int calculatorsUseMpi();

// Loads librgpot_mpi and keeps it for the process. path names the library
// file; nullptr searches RGPOT_MPI_LIBRARY, then RGPOT_MPI_LIBRARY_NAME
// next to librgpot, then the dynamic linker's path. Returns 1 when the
// MPI side is available (also when it was loaded already), else 0 with
// the reason in calculatorMpiLoadError().
int loadCalculatorMpi(const char *path = nullptr);

// 1 once librgpot_mpi is loaded in this process.
int calculatorMpiLoaded();

// Why the last loadCalculatorMpi failed; empty after a success.
std::string calculatorMpiLoadError();

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
// When abortMpiAtExit was called, the handler calls
// MPI_Abort(MPI_COMM_WORLD, 1) instead, whoever owns the MPI_Init.
// No-op without MPI.
void finalizeMpiAtExit();

// Marks this process as failed for the exit handler registered by
// finalizeMpiAtExit. An engine call that fails on one rank leaves the
// other ranks inside a collective (MPI_Bcast in shareFromCalculator, or
// the engine's own); the failed rank would then block in MPI_Finalize
// until the walltime kill. A backend calls this before throwing out of
// an engine call, so the exit handler aborts the world instead. Sets a
// process-wide flag; harmless without MPI and before finalizeMpiAtExit.
void abortMpiAtExit();

// True once abortMpiAtExit was called in this process.
bool mpiAbortRequested();

// Logical AND of flags[0..n) over the ranks of this process's calculator,
// in place: the scope of a groupCollective potential's force call.
// Collective on the calculator's communicator when it has more than one
// rank, and every rank of it must pass the same n; other calculators take
// no part. Returns 1 when the flags were combined, 0 (flags untouched)
// without MPI, before a split, or for a calculator of one rank.
int calculatorAgree(unsigned char *flags, std::size_t n);

// Error exchange at the end of a groupCollective force call. Every rank of
// the calculator calls it with its message (empty for success). When any
// rank has one, every message prints on every rank of the calculator and
// MPI_COMM_WORLD aborts; otherwise it returns. Without a split, a
// non-empty message on a calculator of one rank aborts the world; without
// librgpot_mpi it returns and the caller's exception carries the message.
void publishCalculatorError(const std::string &message);

} // namespace rgpot
