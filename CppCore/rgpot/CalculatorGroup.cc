// MIT License
// Copyright 2023--present rgpot developers

#include "rgpot/CalculatorGroup.hpp"

#include <mutex>
#include <vector>

#ifdef RGPOT_HAS_MPI
#include <mpi.h>
#endif

namespace rgpot {
namespace {
std::mutex g_mu;
std::vector<CalculatorHook> g_hooks;
CalculatorGroup g_group;
int g_bound_rpc = 0;
bool g_bound = false;
#ifdef RGPOT_HAS_MPI
MPI_Comm g_comm = MPI_COMM_NULL;
#endif
} // namespace

void addCalculatorHook(CalculatorHook hook) {
  if (!hook)
    return;
  std::lock_guard<std::mutex> lock(g_mu);
  for (CalculatorHook have : g_hooks) {
    if (have == hook)
      return;
  }
  g_hooks.push_back(hook);
}

CalculatorGroup bindCalculators(int ranks_per_calculator) {
  std::lock_guard<std::mutex> lock(g_mu);
  if (g_bound && g_bound_rpc == ranks_per_calculator)
    return g_group;
  if (g_bound) {
    CalculatorGroup refused;
    refused.index = -1;
    return refused;
  }

  int rpc = ranks_per_calculator;
  g_group = CalculatorGroup{};
#ifdef RGPOT_HAS_MPI
  int inited = 0;
  MPI_Initialized(&inited);
  if (!inited)
    MPI_Init(nullptr, nullptr);
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (rpc <= 0)
    rpc = size;
  if (rpc > size || size % rpc != 0) {
    g_group.index = -1;
  } else {
    MPI_Comm sub = MPI_COMM_NULL;
    MPI_Comm_split(MPI_COMM_WORLD, rank / rpc, rank % rpc, &sub);
    g_comm = sub;
    g_group.index = rank / rpc;
    g_group.ranks = rpc;
    g_group.rank_in_group = rank % rpc;
    g_group.world_size = size;
  }
#else
  if (rpc <= 0)
    rpc = 1;
  if (rpc != 1)
    g_group.index = -1;
  else {
    g_group.index = 0;
    g_group.ranks = 1;
    g_group.world_size = 1;
  }
#endif
  for (CalculatorHook hook : g_hooks) {
    int idx = hook(rpc);
    if (g_group.index < 0 && idx >= 0)
      g_group.index = idx;
  }
  g_bound = true;
  g_bound_rpc = rpc;
  return g_group;
}

const CalculatorGroup &thisCalculator() { return g_group; }

int calculatorComm(void *comm_out, std::size_t comm_bytes) {
#ifdef RGPOT_HAS_MPI
  if (!comm_out || comm_bytes != sizeof(MPI_Comm) || g_comm == MPI_COMM_NULL)
    return 0;
  *static_cast<MPI_Comm *>(comm_out) = g_comm;
  return 1;
#else
  (void)comm_out;
  (void)comm_bytes;
  return 0;
#endif
}

} // namespace rgpot
