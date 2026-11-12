// Copyright (c) rgpot developers
//
// One UmaPot force call. The driver prints the wall time in seconds.
// Usage: uma_guard_bench <single|batch|periodic> <n> <package.pt2>

#include <chrono>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "rgpot/UmaPot/UmaPot.hpp"

namespace {

int grid_side(int n) {
  int side = 1;
  while (side * side * side < n)
    ++side;
  return side;
}

void place(std::vector<double> &pos, int n, double spacing, double origin) {
  const int side = grid_side(n);
  int filled = 0;
  for (int z = 0; z < side && filled < n; ++z) {
    for (int y = 0; y < side && filled < n; ++y) {
      for (int x = 0; x < side && filled < n; ++x) {
        pos[static_cast<size_t>(3 * filled)] = origin + spacing * x;
        pos[static_cast<size_t>(3 * filled + 1)] = origin + spacing * y;
        pos[static_cast<size_t>(3 * filled + 2)] = origin + spacing * z;
        ++filled;
      }
    }
  }
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 4) {
    std::cerr << "usage: uma_guard_bench <single|batch|periodic> <n> <pt2>\n";
    return 2;
  }
  const std::string mode = argv[1];
  const int n = std::stoi(argv[2]);
  if (n < 2)
    return 2;
  rgpot::UmaConfig cfg;
  cfg.model_path = argv[3];
  cfg.device = "cpu";
  cfg.task_name = "omol";
  cfg.charge = 0;
  cfg.spin = 1;
  rgpot::UmaPot pot(cfg);

  const bool periodic = mode == "periodic";
  const double spacing = periodic ? 2.0 : 0.5;
  const double origin = periodic ? 1.0 : 100.0;
  const double side = periodic ? grid_side(n) * spacing + spacing : 200.0;
  const double box[9] = {side, 0, 0, 0, side, 0, 0, 0, side};
  std::vector<double> pos(static_cast<size_t>(3 * n));
  std::vector<int> z(static_cast<size_t>(n), 6);
  place(pos, n, spacing, origin);
  std::vector<double> force(pos.size(), 0.0);
  rgpot::ForceOut slot{};
  slot.F = force.data();
  const rgpot::ForceInput in{static_cast<size_t>(n), pos.data(), z.data(), box};

  const auto t0 = std::chrono::steady_clock::now();
  if (mode == "batch") {
    std::vector<double> pos2 = pos;
    std::vector<double> force2(pos.size(), 0.0);
    rgpot::ForceOut slot2{};
    slot2.F = force2.data();
    const rgpot::ForceInput in2{static_cast<size_t>(n), pos2.data(), z.data(),
                                box};
    rgpot::ForceInput inputs[2] = {in, in2};
    rgpot::ForceOut outputs[2] = {slot, slot2};
    const rgpot::ForceBatch batch{2, inputs, outputs};
    pot.forceBatchImpl(batch);
    if (!std::isfinite(slot.energy) || !std::isfinite(slot2.energy))
      throw std::runtime_error("non-finite batch energy");
  } else {
    pot.forceImpl(in, &slot);
    if (!std::isfinite(slot.energy))
      throw std::runtime_error("non-finite energy");
  }
  const auto t1 = std::chrono::steady_clock::now();
  const double seconds = std::chrono::duration<double>(t1 - t0).count();
  std::cout << seconds << "\n";
  return 0;
}
