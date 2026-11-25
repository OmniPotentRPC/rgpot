/* MIT License */
/* Copyright 2023--present rgpot developers */

#include <stddef.h>
#include <string.h>

#if defined(_WIN32) || defined(_WIN64)
#define UPSTREAM_STUB_API __declspec(dllexport)
#else
#define UPSTREAM_STUB_API
#endif

/*
 * Independent definition matching OmniPotentRPC/mopacc include/mopacc.h.
 * Intentionally does not include rgpot's vendored mopac_c_abi.h so any
 * caller-side layout drift is caught at runtime and under ASan.
 */
typedef struct UpstreamMopacCParams {
  int charge;
  int spin;
  int model;
  double tolerance;
  int max_time;
  double epsilon;
  int solver;
} UpstreamMopacCParams;

typedef struct UpstreamMopacCResult {
  int ok;
  double energy_h;
  double dipole_debye[3];
  double stress_gpa[6];
  char message[512];
} UpstreamMopacCResult;

static UpstreamMopacCParams g_params = {0, 0, 4, 1.0, 3600, 1.0, 0};

UPSTREAM_STUB_API int mopacc_set_params(const void *params,
                                        size_t params_size_bytes) {
  if (params == NULL || params_size_bytes != sizeof(UpstreamMopacCParams)) {
    return -1;
  }
  memcpy(&g_params, params, sizeof(g_params));
  return 0;
}

UPSTREAM_STUB_API UpstreamMopacCResult mopacc_energy_gradient(
    int n_atoms, const double *positions_ang, const int *atomic_numbers,
    const void *params, size_t params_size_bytes, double *grad_h_bohr) {
  UpstreamMopacCResult r;
  memset(&r, 0, sizeof(r));
  r.dipole_debye[0] = 1.25;
  r.dipole_debye[1] = -2.50;
  r.dipole_debye[2] = 3.75;
  r.stress_gpa[0] = 0.1;
  r.stress_gpa[1] = 0.2;
  r.stress_gpa[2] = 0.3;
  r.stress_gpa[3] = 0.4;
  r.stress_gpa[4] = 0.5;
  r.stress_gpa[5] = 0.6;

  if (n_atoms <= 0 || positions_ang == NULL || atomic_numbers == NULL ||
      grad_h_bohr == NULL || atomic_numbers[0] <= 0 || params == NULL ||
      params_size_bytes != sizeof(UpstreamMopacCParams)) {
    const char prefix[] = "upstream-err:";
    memset(r.message, 'E', sizeof(r.message) - 1);
    memcpy(r.message, prefix, sizeof(prefix) - 1);
    r.message[sizeof(r.message) - 1] = '\0';
    r.ok = 0;
    r.energy_h = 0.0;
    return r;
  }

  memcpy(&g_params, params, sizeof(g_params));
  for (int i = 0; i < n_atoms * 3; ++i) {
    grad_h_bohr[i] = 0.002 * (double)(i + 1);
  }
  r.ok = 1;
  r.energy_h = -1.5 + 0.01 * (double)g_params.charge + 0.001 * g_params.epsilon;

  const char ok_prefix[] = "upstream-ok:";
  memset(r.message, 'X', sizeof(r.message) - 1);
  memcpy(r.message, ok_prefix, sizeof(ok_prefix) - 1);
  r.message[sizeof(r.message) - 1] = '\0';
  return r;
}

UPSTREAM_STUB_API int mopacc_c_abi_version(void) { return 2; }

UPSTREAM_STUB_API int mopacc_abi_version(void) { return 2; }

UPSTREAM_STUB_API size_t mopacc_result_size(void) {
  return sizeof(UpstreamMopacCResult);
}

UPSTREAM_STUB_API size_t mopacc_params_size(void) {
  return sizeof(UpstreamMopacCParams);
}

UPSTREAM_STUB_API int mopacc_available(void) { return 1; }

UPSTREAM_STUB_API const char *mopacc_version(void) {
  return "mopacc-upstream-stub/1.0";
}
