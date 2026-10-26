#include <mpi.h>

static int g_splits;

__attribute__((visibility("default"))) int
MPI_Comm_split(MPI_Comm comm, int color, int key, MPI_Comm *newcomm) {
  g_splits += 1;
  return PMPI_Comm_split(comm, color, key, newcomm);
}

__attribute__((visibility("default"))) int rgpot_test_mpi_split_count(void) {
  return g_splits;
}
