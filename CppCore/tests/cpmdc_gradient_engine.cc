#if defined(_WIN32) || defined(_WIN64)
#define RGPOT_CPMDC_BUILD
#endif
#include "rgpot/CPMDPot/cpmd_c_abi.h"

#include "cpmd_stress_oracle.hpp"

#include <cstring>

struct CPMDCStressTensor {
  int valid;
  double values[9];
};

namespace {

constexpr CPMDCFeatureEntry kFeatures[] = {
#include "cpmd_feature_table.inc"
};

CPMDCStressTensor g_stress{};

bool flat_message(const void *msg, size_t bytes) {
  return msg != nullptr && bytes >= 8 && (bytes % 8) == 0;
}

} // namespace

extern "C" {

int cpmdc_set_params(const void *params_capnp, size_t params_capnp_size_bytes) {
  return flat_message(params_capnp, params_capnp_size_bytes) ? 0 : -1;
}

CPMDCResult cpmdc_energy_gradient(int n_atoms, const double *positions_ang,
                                  const int *atomic_numbers,
                                  const void *params_capnp,
                                  size_t params_capnp_size_bytes,
                                  double *grad_h_bohr) {
  (void)positions_ang;
  (void)atomic_numbers;
  (void)params_capnp;
  (void)params_capnp_size_bytes;
  CPMDCResult result{};
  if (n_atoms <= 0 || grad_h_bohr == nullptr) {
    result.ok = 0;
    std::strncpy(result.message, "gradient engine needs atoms",
                 sizeof(result.message) - 1);
    return result;
  }
  g_stress.valid = 1;
  for (int i = 0; i < 9; ++i)
    g_stress.values[i] = cpmd_stress_oracle::nativeStress(i);
  const int ngrad = n_atoms * 3;
  for (int i = 0; i < ngrad; ++i)
    grad_h_bohr[i] = 0.02 + 0.001 * static_cast<double>(i);
  result.ok = 1;
  result.energy_h = cpmd_stress_oracle::kGradientHartree;
  std::strncpy(result.message, "gradient ok", sizeof(result.message) - 1);
  return result;
}

int cpmdc_last_stress(CPMDCStressTensor *out) {
  if (out == nullptr || !g_stress.valid)
    return -1;
  *out = g_stress;
  return 0;
}

size_t cpmdc_feature_count(void) {
  return sizeof(kFeatures) / sizeof(kFeatures[0]);
}

const CPMDCFeatureEntry *cpmdc_feature_table(void) { return kFeatures; }

const CPMDCFeatureEntry *cpmdc_feature_find(const char *feature_id) {
  if (feature_id == nullptr)
    return nullptr;
  for (const auto &feature : kFeatures) {
    if (std::strcmp(feature.feature_id, feature_id) == 0)
      return &feature;
  }
  return nullptr;
}

} // extern "C"
