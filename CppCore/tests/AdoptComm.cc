// MIT License
// Copyright 2023--present rgpot developers

#include "rgpot/CPMDPot/CPMDPot.hpp"
#include "rgpot/CalculatorGroup.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <dlfcn.h>
#include <mpi.h>

namespace {

int split_count(bool *present) {
  void *sym = dlsym(RTLD_DEFAULT, "rgpot_test_mpi_split_count");
  if (!sym) {
    *present = false;
    return -1;
  }
  *present = true;
  return reinterpret_cast<int (*)()>(sym)();
}

} // namespace

// mpirun -n 2 missing ENGINE: both ranks return -1 and MPI_Comm_split
// does not run. mpirun -n 4 split ENGINE: two ranks per calculator, indexes
// 0, 0, 1, 1, the adopted communicator is MPI_IDENT with calculatorComm,
// and each rank sees one MPI_Comm_split.
int main(int argc, char **argv) {
  if (argc != 3)
    return 2;
  const char *mode = argv[1];
  const char *engine = argv[2];
  if (setenv("RGPOT_CPMD_ENGINE", engine, 1) != 0)
    return 2;

  // CPMDPot closes its own handle when bindCalculators returns. NODELETE
  // keeps the engine image, and the communicator it stored, mapped.
  void *keep = dlopen(engine, RTLD_NOW | RTLD_LOCAL | RTLD_NODELETE);
  if (!keep) {
    std::fprintf(stderr, "dlopen %s: %s\n", engine, dlerror());
    return 3;
  }

  MPI_Init(&argc, &argv);
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  bool profile = false;
  const int splits_at_init = split_count(&profile);

  if (std::strcmp(mode, "missing") == 0) {
    const int idx = rgpot::CPMDPot::bindCalculators(1);
    const int splits =
        profile ? split_count(&profile) - splits_at_init : -1;
    std::printf("adopt-missing rank %d %d splits %d\n", rank, idx, splits);
    const int rc = (profile && idx == -1 && splits == 0) ? 0 : 4;
    MPI_Finalize();
    return rc;
  }

  if (std::strcmp(mode, "split") == 0) {
    const int idx = rgpot::CPMDPot::bindCalculators(2);
    const int splits =
        profile ? split_count(&profile) - splits_at_init : -1;
    MPI_Comm mine = MPI_COMM_NULL;
    const int got = rgpot::calculatorComm(&mine, sizeof(mine));
    using AdoptedFn = int (*)(void *, std::size_t);
    auto *adopted =
        reinterpret_cast<AdoptedFn>(dlsym(keep, "cpmdc_adopted_comm"));
    MPI_Comm theirs = MPI_COMM_NULL;
    const int copied =
        adopted != nullptr ? adopted(&theirs, sizeof(theirs)) : -1;
    int cmp = 0;
    if (got == 1 && copied == 0)
      MPI_Comm_compare(mine, theirs, &cmp);
    const int ident = cmp == MPI_IDENT ? 1 : 0;
    const int expect = rank / 2;
    std::printf("adopt-split rank %d index %d ident %d splits %d\n", rank, idx,
                ident, splits);
    const int rc =
        (profile && idx == expect && ident == 1 && splits == 1) ? 0 : 5;
    MPI_Finalize();
    return rc;
  }

  MPI_Finalize();
  return 2;
}
