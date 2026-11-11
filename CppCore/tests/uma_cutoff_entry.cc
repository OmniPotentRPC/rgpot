// Copyright 2023--present rgpot developers
//
// One timed build_system_edges call. Compiled once per version with
// -Drgpot=rgpot_vNNN and -DRGPOT_ENTRY=rgpot_time_vNNN.

#include "rgpot/UmaPot/UmaEdges.hpp"

#include <chrono>
#include <cmath>
#include <vector>

#ifndef RGPOT_ENTRY
#error RGPOT_ENTRY
#endif

namespace {

double time_call(const std::vector<double> &pos, const std::vector<int> &z, double side,
                 int threads, rgpot::UmaEdgeSession &session) {
  double box[9] = {side, 0, 0, 0, side, 0, 0, 0, side};
  rgpot::UmaContract contract;
  rgpot::ForceInput in{pos.size() / 3, pos.data(), z.data(), box};
  const auto t0 = std::chrono::steady_clock::now();
  const auto &list = rgpot::build_system_edges(in, 6.0, 6.0, 0.0, 300, contract, 1, session);
  const auto t1 = std::chrono::steady_clock::now();
  if (list.neighbor.empty())
    return -1.0;
  return std::chrono::duration<double, std::micro>(t1 - t0).count();
}

} // namespace

extern "C" double RGPOT_ENTRY(const double *pos, const int *atomic, int n, int threads,
                              int reuse) {
  std::vector<double> owned(pos, pos + static_cast<std::size_t>(n) * 3);
  std::vector<int> z(atomic, atomic + n);
  const double side = std::cbrt(static_cast<double>(n)) * 2.5;
  if (reuse == 0) {
    rgpot::UmaEdgeSession warm;
    warm.threads = threads;
    time_call(owned, z, side, threads, warm);
    rgpot::UmaEdgeSession session;
    session.threads = threads;
    return time_call(owned, z, side, threads, session);
  }
  rgpot::UmaEdgeSession session;
  session.threads = threads;
  time_call(owned, z, side, threads, session);
  time_call(owned, z, side, threads, session);
  return time_call(owned, z, side, threads, session);
}
