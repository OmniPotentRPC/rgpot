// MIT License
// Copyright 2023--present rgpot developers
//
// Cost of the calculator-group transport an MPI host pays per force call:
// bindCalculators (one MPI_Comm_split, once per process) and
// shareFromCalculator (validation plus broadcast of one calculator's
// energy and forces to the world, once per calculator per call). Not a
// meson test; run under mpirun, e.g.
//   mpirun -n 28 time_calculator_share --ranks-per-calc 4 --atoms 343
// Rank 0 prints one CSV row: the slowest rank's mean per call.

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>

#include "rgpot/CalculatorGroup.hpp"

int main(int argc, char **argv) {
  int rpc = 1;
  long atoms = 343;
  long calls = 2000;
  for (int i = 1; i + 1 < argc; i += 2) {
    if (std::strcmp(argv[i], "--ranks-per-calc") == 0)
      rpc = std::atoi(argv[i + 1]);
    else if (std::strcmp(argv[i], "--atoms") == 0)
      atoms = std::atol(argv[i + 1]);
    else if (std::strcmp(argv[i], "--calls") == 0)
      calls = std::atol(argv[i + 1]);
  }
  MPI_Init(&argc, &argv);
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  MPI_Barrier(MPI_COMM_WORLD);
  const double b0 = MPI_Wtime();
  const rgpot::CalculatorGroup g = rgpot::bindCalculators(rpc);
  const double bind = MPI_Wtime() - b0;
  if (g.index < 0) {
    if (rank == 0)
      std::fprintf(stderr, "world of %d cannot split into groups of %d\n",
                   size, rpc);
    MPI_Finalize();
    return 2;
  }
  const int ncalc = rgpot::calculatorCount();

  // One result: energy plus 3N forces, the bytes a NEB host shares.
  std::vector<double> buf(static_cast<std::size_t>(3 * atoms + 1), 1.0);
  const std::size_t bytes = buf.size() * sizeof(double);

  for (int w = 0; w < 20; ++w)
    for (int owner = 0; owner < ncalc; ++owner)
      rgpot::shareFromCalculator(owner, buf.data(), bytes);
  MPI_Barrier(MPI_COMM_WORLD);
  const double t0 = MPI_Wtime();
  for (long c = 0; c < calls; ++c)
    for (int owner = 0; owner < ncalc; ++owner)
      rgpot::shareFromCalculator(owner, buf.data(), bytes);
  const double per = (MPI_Wtime() - t0) / static_cast<double>(calls * ncalc);

  double worstPer = 0.0;
  double worstBind = 0.0;
  MPI_Reduce(&per, &worstPer, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  MPI_Reduce(&bind, &worstBind, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  if (rank == 0) {
    std::printf("world,ranks_per_calc,calculators,atoms,bytes,bind_us,share_us\n");
    std::printf("%d,%d,%d,%ld,%zu,%.2f,%.3f\n", size, rpc, ncalc, atoms, bytes,
                worstBind * 1e6, worstPer * 1e6);
  }
  MPI_Finalize();
  return 0;
}
