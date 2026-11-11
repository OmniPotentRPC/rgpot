#pragma once
// Copyright (c) rgpot developers

#include "rgpot/ForceStructs.hpp"
#include "rgpot/UmaPot/UmaContract.hpp"
#include "vesin.h"

#include <cstdint>
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
  std::mutex mutex;
  VesinNeighborList held{};
  bool held_live = false;

  struct Row {
    std::vector<int64_t> neighbor;
    std::vector<double> dist;
    std::vector<std::size_t> order;
  };
  std::vector<Row> rows;

  UmaEdgeSession();
  ~UmaEdgeSession();
  UmaEdgeSession(const UmaEdgeSession &) = delete;
  UmaEdgeSession &operator=(const UmaEdgeSession &) = delete;
};

/// The edge list UmaPot passes to the compiled graph.
/// A molecular package lists every ordered pair. Any other package
/// builds a periodic vesin list at the model cutoff.
UmaEdgeList build_system_edges(const ForceInput &local, double cutoff,
                               double model_cutoff, double molecular_box,
                               int max_neighbors, const UmaContract &contract,
                               int64_t n_systems, UmaEdgeSession &session);

} // namespace rgpot
