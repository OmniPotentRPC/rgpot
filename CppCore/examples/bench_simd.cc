// MIT License
// Copyright 2023--present rgpot developers

// Steady-state force calls. The first call builds the pair list and is not
// timed. Print calls/sec for one process. A driver interleaves processes.

#include "rgpot/EAM/EAMPot.hpp"
#include "rgpot/LennardJones/LJPot.hpp"
#include "rgpot/Morse/MorsePot.hpp"

#ifndef RGPOT_BENCH_SCALAR_ONLY
#include "rgpot/simd/api.hpp"
#endif

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace {

void fcc(int ncell, double a, std::vector<double> &R, double box[9]) {
  static const double basis[4][3] = {
      {0, 0, 0}, {0, 0.5, 0.5}, {0.5, 0, 0.5}, {0.5, 0.5, 0}};
  R.clear();
  for (int i = 0; i < ncell; ++i) {
    for (int j = 0; j < ncell; ++j) {
      for (int k = 0; k < ncell; ++k) {
        for (int b = 0; b < 4; ++b) {
          R.push_back(a * (i + basis[b][0]));
          R.push_back(a * (j + basis[b][1]));
          R.push_back(a * (k + basis[b][2]));
        }
      }
    }
  }
  const double L = a * ncell;
  for (int i = 0; i < 9; ++i) {
    box[i] = 0;
  }
  box[0] = box[4] = box[8] = L;
}

template <class Pot>
double calls_per_sec(Pot &pot, const std::vector<double> &R,
                     const std::vector<int> &types, const double box[9],
                     int ncall) {
  std::vector<double> F(R.size(), 0.0);
  rgpot::ForceOut out{F.data(), 0.0, 0.0, {}, 0};
  const rgpot::ForceInput in{R.size() / 3, R.data(), types.data(), box};
  pot.forceImpl(in, &out);
  const auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < ncall; ++i) {
    pot.forceImpl(in, &out);
  }
  const auto t1 = std::chrono::steady_clock::now();
  const double sec = std::chrono::duration<double>(t1 - t0).count();
  return ncall / sec;
}

void copy_vs_force() {
  // The Eigen adapter copies a coordinate buffer. Time that copy against
  // one LJ force on the same 4000-atom cell.
  std::vector<double> R;
  double box[9];
  fcc(10, 3.615, R, box);
  std::vector<double> dst(R.size());
  const int ncopy = 200;
  const auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < ncopy; ++i) {
    std::memcpy(dst.data(), R.data(), R.size() * sizeof(double));
  }
  const auto t1 = std::chrono::steady_clock::now();
  std::vector<int> types(R.size() / 3, 29);
  rgpot::LJPot pot{rgpot::LJConfig{.u0 = 0.167, .cutoff = 6.0, .psi = 2.277}};
  const double cps = calls_per_sec(pot, R, types, box, 5);
  const double copy_sec =
      std::chrono::duration<double>(t1 - t0).count() / ncopy;
  const double force_sec = 1.0 / cps;
  std::cout << "copy_4000_atoms_sec " << copy_sec << "\n";
  std::cout << "lj_fcc_4000_sec " << force_sec << "\n";
  std::cout << "copy_over_force " << (copy_sec / force_sec) << "\n";
}

} // namespace

int main(int argc, char **argv) {
  const char *target = argc > 1 ? argv[1] : "default";
#ifndef RGPOT_BENCH_SCALAR_ONLY
  if (std::strcmp(target, "default") != 0) {
    rgpot::simd::set_target_for_test(target);
  }
  std::cout << "target " << rgpot::simd::chosen_target_name() << "\n";
#else
  std::cout << "target main\n";
  (void)target;
#endif
  if (argc > 2 && std::strcmp(argv[2], "copy") == 0) {
    copy_vs_force();
    return 0;
  }

  std::vector<double> R;
  double box[9];
  fcc(10, 3.615, R, box);
  std::vector<int> types(R.size() / 3, 29);
  rgpot::LJPot cu{rgpot::LJConfig{.u0 = 0.167, .cutoff = 6.0, .psi = 2.277}};
  std::cout << "lj_cu_fcc_4000 " << calls_per_sec(cu, R, types, box, 40)
            << "\n";

  fcc(6, 5.26, R, box);
  types.assign(R.size() / 3, 18);
  rgpot::LJPot ar{rgpot::LJConfig{.u0 = 0.0104, .cutoff = 8.5, .psi = 3.4}};
  std::cout << "lj_ar_fcc_864 " << calls_per_sec(ar, R, types, box, 40) << "\n";

  fcc(4, 2.9, R, box);
  types.assign(R.size() / 3, 78);
  rgpot::MorsePot morse;
  std::cout << "morse_fcc_256 " << calls_per_sec(morse, R, types, box, 20)
            << "\n";

  fcc(4, 4.05, R, box);
  types.assign(R.size() / 3, 13);
  rgpot::EAMPot eam;
  std::cout << "eam_al_fcc_256 " << calls_per_sec(eam, R, types, box, 8)
            << "\n";
  return 0;
}
