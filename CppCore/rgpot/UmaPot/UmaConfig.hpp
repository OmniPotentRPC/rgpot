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
  /// Cutoff in angstrom. The package's embedded `cutoff` metadata wins
  /// when present; no sidecar file is read. A molecular package does
  /// not truncate pairs at this value: the compiled list is every
  /// ordered pair, and the model envelope zeroes pairs past its own
  /// range (`model_cutoff` in the package, or this cutoff when that
  /// key is absent).
  double cutoff = 6.0;
  int max_neighbors = 300;
};

} // namespace rgpot
