#if defined(_WIN32) || defined(_WIN64)
#define RGPOT_MOPACC_BUILD
#endif
#include "mopac_c_abi.h"

#include <stddef.h>
#include <stdio.h>
#include <string.h>

static MopacCParams g_params = {0,   0,   MOPACC_MODEL_AM1,   1.0,
                                3600, 1.0, MOPACC_SOLVER_MOPAC};

static int valid_params(const void *params, size_t n) {
  return params != NULL && n == sizeof(MopacCParams);
}

int mopacc_set_params(const void *params, size_t params_size_bytes) {
  if (!valid_params(params, params_size_bytes))
    return -1;
  memcpy(&g_params, params, sizeof(g_params));
  return 0;
}

MopacCResult mopacc_energy_gradient(int n_atoms, const double *positions_ang,
                                    const int *atomic_numbers,
                                    const void *params,
                                    size_t params_size_bytes,
                                    double *grad_h_bohr) {
  MopacCResult r;
  memset(&r, 0, sizeof(r));
  r.ok = 0;
  r.energy_h = 0.0;
  r.message[0] = '\0';
  if (n_atoms <= 0 || positions_ang == NULL || atomic_numbers == NULL ||
      grad_h_bohr == NULL || !valid_params(params, params_size_bytes)) {
    snprintf(r.message, sizeof(r.message), "invalid fake-engine arguments");
    return r;
  }
  memcpy(&g_params, params, sizeof(g_params));
  for (int i = 0; i < n_atoms * 3; ++i)
    grad_h_bohr[i] = 0.001 * (double)(i + 1);
  r.ok = 1;
  r.energy_h = 0.25 + 0.01 * (double)g_params.charge;
  r.dipole_debye[0] = 0.5;
  r.dipole_debye[1] = -0.25;
  r.dipole_debye[2] = 0.125;
  r.stress_gpa[0] = 0.01;
  r.stress_gpa[1] = 0.02;
  r.stress_gpa[2] = 0.03;
  snprintf(r.message, sizeof(r.message), "ok model=%d", g_params.model);
  return r;
}

int mopacc_c_abi_version(void) { return RGPOT_MOPACC_C_ABI_VERSION; }

int mopacc_abi_version(void) { return MOPACC_ABI_VERSION; }

size_t mopacc_result_size(void) { return sizeof(MopacCResult); }

size_t mopacc_params_size(void) { return sizeof(MopacCParams); }

int mopacc_available(void) { return 1; }

const char *mopacc_version(void) { return "mopacc-fake/0.1"; }
