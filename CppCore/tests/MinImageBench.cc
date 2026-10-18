// MIT License
// Copyright 2023--present rgpot developers
//
// Micro-benchmark ported from the long-double fold comparison: 20000
// displacements on one triclinic cell, and one cached pair-list rebuild
// of 180 atoms. Prints one line per repeat so a caller can take the median.

#include "rgpot/nlist/MinimumImage.hpp"
#include "rgpot/nlist/PairListCache.hpp"

#include <chrono>
#include <cstdint>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {

template <typename Fn> double seconds(Fn &&fn) {
  const auto t0 = std::chrono::steady_clock::now();
  fn();
  const auto t1 = std::chrono::steady_clock::now();
  return std::chrono::duration<double>(t1 - t0).count();
}

} // namespace

int main(int argc, char **argv) {
  const std::string label = argc > 1 ? argv[1] : "fold";
  const double triclinic[9] = {4.0, 0.2, 0.1, 0.3, 5.0, -0.2, 0.1, -0.3, 6.0};
  const std::array<bool, 3> periodic{true, true, true};
  const rgpot::nlist::MinimumImage image(triclinic, periodic);
  std::mt19937_64 rng(1);
  std::uniform_real_distribution<double> u(-20.0, 20.0);
  constexpr int kN = 20000;
  std::vector<double> disp(3 * kN);
  for (double &v : disp)
    v = u(rng);

  constexpr int kAtoms = 180;
  std::vector<double> R(3 * kAtoms);
  std::uniform_real_distribution<double> p(0.0, 4.0);
  for (double &v : R)
    v = p(rng);
  rgpot::nlist::CachedPairList::Options opt;
  opt.cutoff = 3.0;
  opt.skin = 1.0;
  opt.periodic = periodic;

  if (label == "fold") {
    const double wall = seconds([&] {
      double sink = 0.0;
      for (int i = 0; i < kN; ++i) {
        double dx = disp[3 * i], dy = disp[3 * i + 1], dz = disp[3 * i + 2];
        image.fold(dx, dy, dz);
        sink += dx + dy + dz;
      }
      if (sink == 0.0)
        std::cerr << "sink\n";
    });
    std::cout << "fold_s " << wall << '\n';
    return 0;
  }
  const double wall = seconds([&] {
    rgpot::nlist::CachedPairList list;
    list.rebuild(R.data(), kAtoms, triclinic, opt);
    if (!list.valid(R.data(), kAtoms, triclinic, opt))
      std::cerr << "invalid\n";
  });
  std::cout << "rebuild_s " << wall << '\n';
  return 0;
}
