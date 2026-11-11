#pragma once
// Copyright 2023--present rgpot developers

#include "rgpot/ForceStructs.hpp"
#include "rgpot/UmaPot/UmaContract.hpp"
#include "vesin.h"

#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

namespace rgpot {

/// Directed edges in fairchem order: neighbor, center, then three shifts.
struct UmaEdgeList {
  std::vector<int64_t> neighbor;
  std::vector<int64_t> center;
  std::vector<float> offsets;
};

/// Buffers and the vesin list reused across force calls.
struct UmaEdgeSession {
  /// Threads for the molecular rows and the periodic list. 1 is serial.
  int threads = 1;
  /// 0 waits until one call has finished, then uses a 1 angstrom skin.
  /// 1 keeps the skin at 0. 2 applies that skin on every call.
  int skin_policy = 0;
  int calls = 0;
  double last_skin = 0.0;
  std::mutex mutex;
  VesinNeighborList held{};
  bool held_live = false;

  struct Row {
    std::vector<int64_t> neighbor;
    std::vector<double> dist;
    std::vector<std::size_t> order;
  };
  std::vector<Row> rows;
  /// Reused across calls. clear() keeps the allocation.
  UmaEdgeList edges;
  struct EdgeKey {
    double dist2 = 0;
    std::uint32_t neighbor = 0;
    std::int32_t s0 = 0;
    std::int32_t s1 = 0;
    std::int32_t s2 = 0;
  };
  std::vector<EdgeKey> keys;
  std::vector<std::size_t> bucket;
  struct CellCache;
  std::unique_ptr<CellCache> cell_cache;
  const void *list_ptr = nullptr;
  std::size_t list_len = 0;

  UmaEdgeSession();
  ~UmaEdgeSession();
  UmaEdgeSession(const UmaEdgeSession &) = delete;
  UmaEdgeSession &operator=(const UmaEdgeSession &) = delete;
};

/// The edge list UmaPot passes to the compiled graph.
/// A molecular package lists every ordered pair. Any other package
/// builds a periodic vesin list at the model cutoff.
const UmaEdgeList &build_system_edges(const ForceInput &local, double cutoff,
                                      double model_cutoff, double molecular_box,
                                      int max_neighbors, const UmaContract &contract,
                                      int64_t n_systems, UmaEdgeSession &session);

} // namespace rgpot
