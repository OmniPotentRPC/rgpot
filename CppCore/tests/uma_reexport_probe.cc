// Copyright 2023--present rgpot developers
//
// Load one package on the three-atom line and print the dtypes.
// The coordinates are LINE_POSITIONS in scripts/export_uma_aoti.py.
// A refusal prints the exception text and exits 1.

#include <cstdio>
#include <exception>
#include <string>
#include <vector>

#include "rgpot/UmaPot/UmaPot.hpp"

namespace {

double energy_of(rgpot::UmaPot &pot, std::vector<double> &pos) {
  const std::vector<int> z = {6, 7, 1};
  const double box[9] = {25, 0, 0, 0, 25, 0, 0, 0, 25};
  std::vector<double> force(pos.size(), 0.0);
  rgpot::ForceOut slot{force.data(), 0.0, 0.0};
  const rgpot::ForceInput in{pos.size() / 3, pos.data(), z.data(), box};
  pot.forceImpl(in, &slot);
  return slot.energy;
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: uma_reexport_probe PACKAGE.pt2\n");
    return 2;
  }
  rgpot::UmaConfig cfg;
  cfg.model_path = argv[1];
  cfg.device = "cpu";
  cfg.task_name = "omol";
  cfg.charge = 0;
  cfg.spin = 1;
  try {
    rgpot::UmaPot pot(cfg);
    std::vector<double> base = {12.5, 12.5, 12.5, 12.5, 12.5, 13.5, 12.5, 12.5,
                                11.5};
    std::vector<double> shifted = base;
    shifted.back() += 1e-7;
    const double first = energy_of(pot, base);
    const double second = energy_of(pot, shifted);
    std::printf("pos_dtype=%s\n", pot.posDtypeName());
    std::printf("model_dtype=%s\n", pot.modelDtypeName());
    std::printf("energy=%.17g\n", first);
    std::printf("difference=%.17g\n", second - first);
    return 0;
  } catch (const std::exception &err) {
    std::fprintf(stderr, "%s\n", err.what());
    return 1;
  }
}
