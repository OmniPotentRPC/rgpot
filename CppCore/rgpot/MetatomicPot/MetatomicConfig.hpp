#pragma once
// MIT License — config POD shared by linked MetatomicPot and dlopen frontend
#include <string>

namespace rgpot {

enum class TorchDeterminismPolicy {
  Fast = 0,
  Strict = 1,
};

struct MetatomicConfig {
  std::string model_path;
  std::string device;
  std::string length_unit = "angstrom";
  std::string extensions_directory;
  bool check_consistency = false;
  double uncertainty_threshold = -1.0;
  std::string dtype_override;
  bool random_rotation = false;
  long n_symmetry_rotations = 0;
  bool so3_probe_scatter = false;
  TorchDeterminismPolicy torch_determinism = TorchDeterminismPolicy::Fast;
  /** Optional explicit path to libmetatomic_engine.so (dlopen path). */
  std::string engine_path;
  /** UMA / OMol task head baked into or selected by the exported model. */
  std::string task_name;
  int charge = 0;
  /** Spin multiplicity (1 = singlet). Attached as system extra data. */
  int spin = 1;
  /** Attach charge/spin TensorMaps on every System (UMA omol). */
  bool attach_system_extras = false;
};

void apply_torch_determinism_policy(TorchDeterminismPolicy policy);

} // namespace rgpot
