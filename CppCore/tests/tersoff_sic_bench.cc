// MIT License
// Copyright 2023--present rgpot developers
//
// Calls per second of TersoffSiC::energyAndGradient on a 216-atom silicon
// diamond cell. Three repeats, CPU model printed from /proc/cpuinfo.
// rebuild passes a null neighbour list. held reuses one list.

#include <chrono>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "rgpot/Tersoff/TersoffSiC.hpp"
#include "rgpot/nlist/HalfShell.hpp"

namespace {

std::string cpuModel() {
  std::ifstream in("/proc/cpuinfo");
  std::string line;
  while (std::getline(in, line)) {
    const std::string key = "model name";
    if (line.compare(0, key.size(), key) != 0) {
      continue;
    }
    const auto colon = line.find(':');
    if (colon == std::string::npos) {
      continue;
    }
    std::string name = line.substr(colon + 1);
    const auto first = name.find_first_not_of(" \t");
    if (first != std::string::npos) {
      name.erase(0, first);
    }
    return name;
  }
  return "unknown";
}

} // namespace

int main() {
  constexpr double kA0 = 5.432;
  constexpr int kRep = 3;
  const double basis[8][3] = {{0, 0, 0},          {0.5, 0.5, 0},
                              {0.5, 0, 0.5},      {0, 0.5, 0.5},
                              {0.25, 0.25, 0.25}, {0.75, 0.75, 0.25},
                              {0.75, 0.25, 0.75}, {0.25, 0.75, 0.75}};
  std::vector<double> pos;
  std::vector<int> types;
  for (int ix = 0; ix < kRep; ++ix) {
    for (int iy = 0; iy < kRep; ++iy) {
      for (int iz = 0; iz < kRep; ++iz) {
        for (const auto &b : basis) {
          pos.push_back((ix + b[0]) * kA0);
          pos.push_back((iy + b[1]) * kA0);
          pos.push_back((iz + b[2]) * kA0);
          types.push_back(14);
        }
      }
    }
  }
  const std::size_t n_atoms = types.size();
  const double box[9] = {kRep * kA0, 0, 0, 0, kRep * kA0, 0, 0, 0, kRep * kA0};
  std::vector<double> grad(pos.size());
  rgpot::TersoffSiC pot;
  double energy = 0.0;
  auto report = [&](const char *mode, int calls,
                    rgpot::nlist::NeighborList *list) {
    for (int i = 0; i < 5; ++i) {
      pot.energyAndGradient(pos.data(), types.data(), box, n_atoms, &energy,
                            grad.data(), list);
    }
    std::cout << "mode: " << mode << "\n";
    std::cout << "calls_per_repeat: " << calls << "\n";
    for (int rep = 1; rep <= 3; ++rep) {
      const auto t0 = std::chrono::steady_clock::now();
      for (int c = 0; c < calls; ++c) {
        pot.energyAndGradient(pos.data(), types.data(), box, n_atoms, &energy,
                              grad.data(), list);
      }
      const auto t1 = std::chrono::steady_clock::now();
      const double sec = std::chrono::duration<double>(t1 - t0).count();
      const double rate = static_cast<double>(calls) / sec;
      std::cout << "repeat " << rep << ": " << rate << " calls/s\n";
    }
  };
  std::cout << "cpu: " << cpuModel() << "\n";
  std::cout << "potential: TersoffSiC\n";
  std::cout << "entry: energyAndGradient\n";
  std::cout << "n_atoms: " << n_atoms << "\n";
  std::cout.setf(std::ios::fmtflags(0), std::ios::floatfield);
  std::cout.precision(6);
  report("rebuild", 100, nullptr);
  rgpot::nlist::NeighborList held;
  report("held", 200, &held);
  std::cout << "energy_eV: " << energy << "\n";
  return energy < 0.0 ? 0 : 1;
}
