#pragma once
// MIT License — config POD for UmaPot (vesin neighbor list + AOTI .pt2)

#include <string>

namespace rgpot {

/**
 * UMA / OMol calculator configuration.
 *
 * model_path is an AOTInductor package (.pt2) produced by
 * scripts/export_uma_aoti.py. Neighbor lists are built with vesin and
 * passed as tensors; the compiled graph does not call fairchem.
 */
struct UmaConfig {
  std::string model_path;
  std::string task_name = "omol";
  std::string device = "cpu";
  int charge = 0;
  int spin = 1;
  /// Cutoff in angstrom. Overridden by the sidecar JSON next to .pt2 when present.
  double cutoff = 6.0;
  int max_neighbors = 300;
};

} // namespace rgpot
