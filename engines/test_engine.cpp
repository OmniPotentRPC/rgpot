// MIT License
// Copyright 2023--present rgpot developers
//
// The LJ engine plugin speaks rgpot_engine_force, the same entry a
// metatomic or UMA engine exports. A host dlopens it and does not link
// the potential.

#include "lj_eval.hpp"

#include <cstdio>
#include <cstring>

int main(int argc, char **argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: test_engine liblj_engine.so\n");
    return 2;
  }
  const double xyz[6] = {0, 0, 0, 1.5, 0, 0};
  const int z[2] = {1, 1};
  const double box[9] = {20, 0, 0, 0, 20, 0, 0, 0, 20};
  const double config[3] = {1.0, 8.0, 1.0};
  double direct_f[6];
  double direct_e = 0;
  int has_stress = 0;
  double stress[9];
  if (rgpot_lj_eval(xyz, z, 2, box, config[0], config[1], config[2], direct_f, &direct_e,
                    &has_stress, stress) != 0) {
    return 1;
  }
  double plugin_f[6];
  double plugin_e = 0;
  char err[256] = {};
  if (rgpot_engine_eval(argv[1], config, sizeof(config), xyz, z, 2, box, plugin_f, &plugin_e, err,
                        sizeof(err)) != 0) {
    std::fprintf(stderr, "%s\n", err);
    return 1;
  }
  if (plugin_e != direct_e || std::memcmp(plugin_f, direct_f, sizeof(direct_f)) != 0) {
    std::fprintf(stderr, "engine %g direct %g\n", plugin_e, direct_e);
    return 1;
  }
  double again = 0;
  if (rgpot_engine_eval(argv[1], nullptr, 0, xyz, z, 2, box, plugin_f, &again, err,
                        sizeof(err)) != 0 || again != direct_e) {
    std::fprintf(stderr, "reopened engine %g\n", again);
    return 1;
  }
  std::printf("engine energy %.10f\n", plugin_e);
  return 0;
}
