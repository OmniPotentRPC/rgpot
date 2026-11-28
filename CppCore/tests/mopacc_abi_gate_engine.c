/* MIT License */
/* Copyright 2023--present rgpot developers */

/*
 * One source, several fake libmopacc builds for the MOPACPot load gate:
 *
 *   GATE_C_ABI         value of mopacc_c_abi_version, 0 to omit the symbol
 *   GATE_LIB_ABI       value of mopacc_abi_version, 0 to omit the symbol
 *   GATE_NEW_LAYOUT    1 for the ABI 2 structs, 0 for the ABI 1 ones
 *   GATE_RESULT_SIZE   mopacc_result_size: 0 omitted, 1 sizeof(GateResult),
 *                      2 a size off by 8
 *   GATE_PARAMS_SIZE   mopacc_params_size, same encoding with GateParams
 *   GATE_SET_PARAMS_STATUS  what mopacc_set_params returns for a valid blob
 *
 * Does not include the vendored header, so the layouts here are what a
 * library of that vintage really returns. mopacc_gate_calls counts every
 * call into the library other than the version and size queries.
 */

#include <stddef.h>
#include <string.h>

#if defined(_WIN32) || defined(_WIN64)
#define GATE_API __declspec(dllexport)
#else
#define GATE_API
#endif

#ifndef GATE_C_ABI
#error "GATE_C_ABI must be defined"
#endif
#ifndef GATE_LIB_ABI
#error "GATE_LIB_ABI must be defined"
#endif
#ifndef GATE_NEW_LAYOUT
#error "GATE_NEW_LAYOUT must be defined"
#endif
#ifndef GATE_RESULT_SIZE
#error "GATE_RESULT_SIZE must be defined"
#endif
#ifndef GATE_PARAMS_SIZE
#error "GATE_PARAMS_SIZE must be defined"
#endif
#ifndef GATE_SET_PARAMS_STATUS
#error "GATE_SET_PARAMS_STATUS must be defined"
#endif

#if GATE_NEW_LAYOUT
typedef struct GateParams {
  int charge;
  int spin;
  int model;
  double tolerance;
  int max_time;
  double epsilon;
  int solver;
} GateParams;

typedef struct GateResult {
  int ok;
  double energy_h;
  double dipole_debye[3];
  double stress_gpa[6];
  char message[512];
} GateResult;
#else
typedef struct GateParams {
  int charge;
  int spin;
  int model;
  double tolerance;
  int max_time;
} GateParams;

typedef struct GateResult {
  int ok;
  double energy_h;
  char message[512];
} GateResult;
#endif

static int g_calls = 0;

GATE_API int mopacc_gate_calls(void) { return g_calls; }

GATE_API int mopacc_set_params(const void *params, size_t params_size_bytes) {
  ++g_calls;
  if (params == NULL || params_size_bytes != sizeof(GateParams)) {
    return -1;
  }
  return GATE_SET_PARAMS_STATUS;
}

GATE_API GateResult mopacc_energy_gradient(
    int n_atoms, const double *positions_ang, const int *atomic_numbers,
    const void *params, size_t params_size_bytes, double *grad_h_bohr) {
  ++g_calls;
  GateResult r;
  memset(&r, 0, sizeof(r));
  (void)positions_ang;
  (void)atomic_numbers;
  if (n_atoms <= 0 || grad_h_bohr == NULL || params == NULL ||
      params_size_bytes != sizeof(GateParams)) {
    strcpy(r.message, "gate-err");
    return r;
  }
  for (int i = 0; i < n_atoms * 3; ++i) {
    grad_h_bohr[i] = 0.003 * (double)(i + 1);
  }
  r.ok = 1;
  r.energy_h = -2.0;
  strcpy(r.message, "gate-ok");
  return r;
}

#if GATE_C_ABI
GATE_API int mopacc_c_abi_version(void) { return GATE_C_ABI; }
#endif

#if GATE_LIB_ABI
GATE_API int mopacc_abi_version(void) { return GATE_LIB_ABI; }
#endif

GATE_API int mopacc_available(void) {
  ++g_calls;
  return 1;
}

GATE_API const char *mopacc_version(void) { return "mopacc-gate/0.1"; }

#if GATE_RESULT_SIZE
GATE_API size_t mopacc_result_size(void) {
  return sizeof(GateResult) + (GATE_RESULT_SIZE == 2 ? 8u : 0u);
}
#endif

#if GATE_PARAMS_SIZE
GATE_API size_t mopacc_params_size(void) {
  return sizeof(GateParams) + (GATE_PARAMS_SIZE == 2 ? 8u : 0u);
}
#endif
