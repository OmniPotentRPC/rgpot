#pragma once
// MIT License
// Copyright 2023--present rgpot developers

#include "rgpot/SuttonChen/SuttonChenPot.hpp"

namespace rgpot {

/// FCC sum of (r_nn / r)^power through cubic index 32. The sites are
/// (i, j, k) * a / 2 with i + j + k even, and r_nn = a / sqrt(2).
double fcc_power_sum(int power);

/// Energy per atom of the static fcc crystal at nearest-neighbor `radius`.
double sutton_chen_fcc_energy(const SuttonChenConfig &cfg, double radius);

/// Bulk nearest-neighbor distance and positive cohesive well depth.
struct SuttonChenScales {
  double nearest_neighbor = 0.0;
  double well_depth = 0.0;
};

SuttonChenScales sutton_chen_scales(const SuttonChenConfig &cfg);

/// Platinum 10-8. epsilon = 1.98330e-2 eV, a = 3.92 Angstrom.
SuttonChenConfig sutton_chen_platinum();

/// Gold 10-8. epsilon = 1.27930e-2 eV, a = 4.08 Angstrom.
SuttonChenConfig sutton_chen_gold();

} // namespace rgpot

extern "C" {

struct rgpot_sc_scales {
  double nearest_neighbor;
  double well_depth;
};

/// Bulk scales. Returns 0 on success, 1 on a bad argument.
RGPOT_SC_API int rgpot_sutton_chen_scales(const rgpot_sc_config *cfg,
                                          rgpot_sc_scales *out);

} // extern "C"
