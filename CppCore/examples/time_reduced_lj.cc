// MIT License
// Copyright 2023--present rgpot developers
//
// Energy and gradient calls per second for unshifted Lennard-Jones on
// 38 atoms. The geometry is the same Fibonacci shell the anneal bench
// uses. Prints the CPU model string from /proc/cpuinfo.

#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include "rgpot/ReducedPair/ReducedPair.hpp"

using clock_type = std::chrono::steady_clock;

namespace {

std::string cpu_model() {
  std::ifstream in("/proc/cpuinfo");
  std::string line;
  while (std::getline(in, line)) {
    const std::string key = "model name\t: ";
    if (line.rfind(key, 0) == 0) {
      return line.substr(key.size());
    }
  }
  return "unknown";
}

std::vector<double> fibonacci(int n) {
  std::vector<double> x(static_cast<std::size_t>(3 * n), 0.0);
  const double golden = std::acos(-1.0) * (3.0 - std::sqrt(5.0));
  for (int i = 0; i < n; ++i) {
    const double z = 1.0 - 2.0 * (static_cast<double>(i) + 0.5) / n;
    const double radius = std::sqrt(std::max(0.0, 1.0 - z * z));
    const double theta = golden * static_cast<double>(i);
    constexpr double shell = 2.5;
    x[3 * i] = shell * radius * std::cos(theta);
    x[3 * i + 1] = shell * radius * std::sin(theta);
    x[3 * i + 2] = shell * z;
  }
  return x;
}

} // namespace

int main() {
  constexpr int n = 38;
  const std::vector<double> x = fibonacci(n);
  double energy = 0.0;
  std::vector<double> gradient(static_cast<std::size_t>(3 * n), 0.0);
  if (rgpot_lj_gradient(1.0, 1.0, 0.0, n, x.data(), &energy, gradient.data()) !=
      0) {
    std::fprintf(stderr, "rgpot_lj_gradient failed\n");
    return 1;
  }
  constexpr int warmup = 50;
  constexpr int calls = 2000;
  for (int i = 0; i < warmup; ++i) {
    rgpot_lj_gradient(1.0, 1.0, 0.0, n, x.data(), &energy, gradient.data());
  }
  const auto start = clock_type::now();
  for (int i = 0; i < calls; ++i) {
    rgpot_lj_gradient(1.0, 1.0, 0.0, n, x.data(), &energy, gradient.data());
  }
  const double seconds =
      std::chrono::duration<double>(clock_type::now() - start).count();
  std::printf("cpu: %s\n", cpu_model().c_str());
  std::printf("atoms: %d\n", n);
  std::printf("energy: %.16f\n", energy);
  std::printf("calls: %d\n", calls);
  std::printf("seconds: %.6f\n", seconds);
  std::printf("energy_gradient_calls_per_second: %.3f\n", calls / seconds);
  return 0;
}
