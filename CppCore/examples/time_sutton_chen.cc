// MIT License
// Copyright 2023--present rgpot developers
//
// Energy and force calls per second for the reduced Sutton-Chen 10-8
// potential on the deposited Pt38 geometry. Prints the CPU model string
// from /proc/cpuinfo. Not a meson test.

#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <tuple>
#include <vector>

#include "rgpot/SuttonChen/SuttonChenPot.hpp"
#include "rgpot/types/AtomMatrix.hpp"

using rgpot::types::AtomMatrix;
using clock_type = std::chrono::steady_clock;

namespace {

AtomMatrix load_pt38() {
  std::ifstream in("CppCore/tests/data/sc108/pt38");
  if (!in) {
    std::fprintf(stderr, "cannot open CppCore/tests/data/sc108/pt38\n");
    std::exit(2);
  }
  double epsilon = 0.0;
  double c = 0.0;
  double a = 0.0;
  double nraw = 0.0;
  double mraw = 0.0;
  in >> epsilon >> c >> a >> nraw >> mraw;
  std::vector<double> xyz;
  double x = 0.0;
  double y = 0.0;
  double z = 0.0;
  while (in >> x >> y >> z) {
    xyz.push_back(x);
    xyz.push_back(y);
    xyz.push_back(z);
  }
  AtomMatrix pos(xyz.size() / 3, 3);
  for (size_t i = 0; i < xyz.size(); ++i) {
    pos(i / 3, i % 3) = xyz[i];
  }
  return pos;
}

std::string cpu_model() {
  std::ifstream in("/proc/cpuinfo");
  std::string line;
  while (std::getline(in, line)) {
    const std::string key = "model name";
    if (line.compare(0, key.size(), key) == 0) {
      const auto pos = line.find(':');
      if (pos != std::string::npos && pos + 2 < line.size()) {
        return line.substr(pos + 2);
      }
    }
  }
  return "unknown";
}

} // namespace

int main() {
  const AtomMatrix pos = load_pt38();
  const std::vector<int> types(pos.rows(), 78);
  const std::array<std::array<double, 3>, 3> box{
      {{80.0, 0.0, 0.0}, {0.0, 80.0, 0.0}, {0.0, 0.0, 80.0}}};
  rgpot::SuttonChenPot pot;

  const int warmup = 50;
  const int ncalls = 2000;
  for (int i = 0; i < warmup; ++i) {
    auto result = pot(pos, types, box);
    (void)result;
  }
  const auto t0 = clock_type::now();
  double energy = 0.0;
  for (int i = 0; i < ncalls; ++i) {
    auto result = pot(pos, types, box);
    energy = std::get<0>(result);
  }
  const auto t1 = clock_type::now();
  const double seconds = std::chrono::duration<double>(t1 - t0).count();
  const double per_second = static_cast<double>(ncalls) / seconds;

  std::printf("cpu: %s\n", cpu_model().c_str());
  std::printf("geometry: Pt38 Sutton-Chen 10-8, cutoff 40\n");
  std::printf("energy: %.10f\n", energy);
  std::printf("calls: %d\n", ncalls);
  std::printf("seconds: %.6f\n", seconds);
  std::printf("energy_force_calls_per_second: %.3f\n", per_second);
  return 0;
}
