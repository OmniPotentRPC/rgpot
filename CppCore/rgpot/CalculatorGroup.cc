// MIT License
// Copyright 2023--present rgpot developers

#include "rgpot/CalculatorGroup.hpp"

#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
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
bool g_mpi_owner = false;

int broadcastFromOwner(int owner, void *data, std::size_t bytes) {
  const int root = owner * g_group.ranks;
  auto *p = static_cast<std::uint8_t *>(data);
  while (bytes > 0) {
    const std::size_t chunk =
        bytes > static_cast<std::size_t>(INT_MAX) ? static_cast<std::size_t>(INT_MAX)
                                                   : bytes;
    MPI_Bcast(p, static_cast<int>(chunk), MPI_BYTE, root, MPI_COMM_WORLD);
    p += chunk;
    bytes -= chunk;
  }
  return 1;
}
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
  if (!inited) {
    MPI_Init(nullptr, nullptr);
    g_mpi_owner = true;
  }
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (rpc <= 0)
    rpc = size;
  if (rpc > size || size % rpc != 0) {
    g_group.index = -1;
    g_group.world_size = size;
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
  // A negative hook return refuses the bind. A non-negative return
  // fills an index the local split left negative.
  for (CalculatorHook hook : g_hooks) {
    int idx = hook(rpc);
    if (idx < 0)
      g_group.index = -1;
    else if (g_group.index < 0)
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

int calculatorsUseMpi() {
#ifdef RGPOT_HAS_MPI
  int inited = 0;
  MPI_Initialized(&inited);
  return inited ? 1 : 0;
#else
  return 0;
#endif
}

int calculatorWorldSize() {
  if (!g_bound)
    return 1;
  return g_group.world_size;
}

int calculatorCount() {
  if (!g_bound || g_group.index < 0 || g_group.ranks <= 0)
    return 1;
  return g_group.world_size / g_group.ranks;
}

int shareFromCalculator(int owner, void *data, std::size_t bytes) {
#ifdef RGPOT_HAS_MPI
  int inited = 0;
  MPI_Initialized(&inited);
  int size = 1;
  int rank = 0;
  if (inited) {
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  }
  const bool collective = inited && size > 1 && g_bound && g_group.index >= 0 &&
                          g_group.world_size > 1;
  if (!collective) {
    if (!g_bound || g_group.index < 0 || (bytes > 0 && !data))
      return 0;
    if (owner < 0 || owner >= calculatorCount())
      return 0;
    return broadcastFromOwner(owner, data, bytes);
  }

  int reason = 0;
  if (!g_bound || g_group.index < 0 || (bytes > 0 && !data))
    reason = 1;
  else if (owner < 0 || owner >= calculatorCount())
    reason = 2;
  if (bytes > static_cast<std::size_t>(LLONG_MAX))
    reason = 1;
  const long long local[3] = {
      reason,
      static_cast<long long>(owner),
      static_cast<long long>(bytes > static_cast<std::size_t>(LLONG_MAX)
                                 ? LLONG_MAX
                                 : bytes),
  };
  std::vector<long long> all(static_cast<std::size_t>(size) * 3, 0);
  if (MPI_Allgather(local, 3, MPI_LONG_LONG, all.data(), 3, MPI_LONG_LONG,
                    MPI_COMM_WORLD) != MPI_SUCCESS) {
    std::fprintf(stderr, "rgpot rank %d: shareFromCalculator allgather failed\n",
                 rank);
    std::fflush(stderr);
    MPI_Abort(MPI_COMM_WORLD, 1);
    return 0;
  }

  bool bad = false;
  for (int src = 0; src < size; ++src) {
    const std::size_t base = static_cast<std::size_t>(src) * 3;
    if (all[base] != 0 || all[base + 1] != all[1] || all[base + 2] != all[2])
      bad = true;
  }
  if (!bad)
    return broadcastFromOwner(owner, data, bytes);

  for (int src = 0; src < size; ++src) {
    const std::size_t base = static_cast<std::size_t>(src) * 3;
    const long long src_reason = all[base];
    const long long src_owner = all[base + 1];
    const long long src_bytes = all[base + 2];
    if (src_reason == 0 && src_owner == all[1] && src_bytes == all[2])
      continue;
    const char *why = "shareFromCalculator rejected";
    if (src_reason == 2)
      why = "shareFromCalculator owner out of range";
    else if (src_reason == 1)
      why = "shareFromCalculator missing buffer or group";
    else if (src_owner != all[1])
      why = "shareFromCalculator owner disagrees";
    else
      why = "shareFromCalculator byte count disagrees";
    std::fprintf(stderr, "rgpot rank %d: %s\n", src, why);
  }
  std::fflush(stderr);
  MPI_Abort(MPI_COMM_WORLD, 1);
  return 0;
#else
  (void)owner;
  (void)data;
  (void)bytes;
  return 0;
#endif
}

void finalizeMpiAtExit() {
#ifdef RGPOT_HAS_MPI
  static std::once_flag once;
  std::call_once(once, [] {
    std::atexit([] {
      int inited = 0;
      int finalized = 0;
      MPI_Initialized(&inited);
      MPI_Finalized(&finalized);
      if (!g_mpi_owner || !inited || finalized)
        return;
      if (const char *trace = std::getenv("RGPOT_MPI_FINALIZE_TRACE")) {
        if (trace[0] == '1' && trace[1] == '\0') {
          std::fprintf(stderr, "rgpot MPI_Finalize\n");
          std::fflush(stderr);
        }
      }
      MPI_Finalize();
    });
  });
#endif
}

} // namespace rgpot
