// Print UmaPot energy/forces for Baker HCN so they can be compared to ASE.

#include <array>
#include <cstdio>
#include <vector>

#include "rgpot/UmaPot/UmaPot.hpp"
#include "rgpot/types/AtomMatrix.hpp"

using rgpot::types::AtomMatrix;

int main(int argc, char **argv) {
  const char *model = argc > 1 ? argv[1] : "data/lj38/lennard-jones.pt";

  rgpot::UmaConfig cfg;
  cfg.model_path = model;
  cfg.device = "cpu";
  cfg.task_name = "omol";
  cfg.charge = 0;
  cfg.spin = 1;
  rgpot::UmaPot pot(cfg);

  // Baker 01_hcn reactant.con, Angstrom, 25 A cubic cell.
  const AtomMatrix positions{
      {12.49734736216627162, 12.49892801474515913, 12.54059929828148512},
      {12.50115413363106498, 12.50036504272228832, 11.38209979880783251},
      {12.50149850420264563, 12.50069809648255514, 13.61514544631068446},
  };
  const std::vector<int> atmtypes{6, 7, 1};
  const std::array<std::array<double, 3>, 3> box{
      {{25.0, 0.0, 0.0}, {0.0, 25.0, 0.0}, {0.0, 0.0, 25.0}}};

  auto [energy, forces, variance] = pot(positions, atmtypes, box);
  (void)variance;
  std::printf("backend=UmaPot\n");
  std::printf("model=%s\n", model);
  std::printf("energy=%.17g\n", energy);
  for (size_t i = 0; i < forces.rows(); ++i) {
    std::printf("force %zu %.17g %.17g %.17g\n", i, forces(i, 0), forces(i, 1),
                forces(i, 2));
  }
  return 0;
}
