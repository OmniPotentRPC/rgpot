/**
 * @file mopac_c_abi_stub.c
 * @brief Stub mopacc C ABI (always built; no OpenMOPAC link).
 */

#include "mopac_c_abi.h"

#include <stdio.h>
#include <string.h>

static const char *STUB_VERSION = "mopacc-stub/1.0.0";

int mopacc_set_params(const void *params, size_t params_size_bytes) {
  (void)params;
  (void)params_size_bytes;
  return -1;
}

MopacCResult mopacc_energy_gradient(int n_atoms, const double *positions_ang,
                                    const int *atomic_numbers,
                                    const void *params,
                                    size_t params_size_bytes,
                                    double *grad_h_bohr) {
  (void)n_atoms;
  (void)positions_ang;
  (void)atomic_numbers;
  (void)params;
  (void)params_size_bytes;
  (void)grad_h_bohr;
  MopacCResult r;
  memset(&r, 0, sizeof(r));
  r.ok = 0;
  r.energy_h = 0.0;
  snprintf(r.message, sizeof(r.message),
           "mopacc embed not available (stub). Provide libmopacc.so on the "
           "dlopen path.");
  return r;
}

int mopacc_c_abi_version(void) { return RGPOT_MOPACC_C_ABI_VERSION; }

int mopacc_abi_version(void) { return MOPACC_ABI_VERSION; }

int mopacc_available(void) { return 0; }

const char *mopacc_version(void) { return STUB_VERSION; }
