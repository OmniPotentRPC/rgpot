/**
 * @file mopac_c_abi.h
 * @brief C ABI between MOPACPot frontend and libmopacc.so.
 *
 * Method options cross the ABI as a packed `MopacCParams` blob. NULL /
 * size 0 means AM1, charge 0, closed shell, vacuum. There is no Cap'n
 * Proto arm.
 *
 * Units at ABI: geometry Angstrom, energy Hartree, gradient Hartree/Bohr.
 */
#pragma once

#include <stddef.h>

/** Numeric ABI revision for the mopacc engine boundary. */
#define RGPOT_MOPACC_C_ABI_VERSION 1
#define MOPACC_ABI_VERSION 1

#ifdef __cplusplus
extern "C" {
#endif

#if defined(RGPOT_MOPACC_BUILD) && (defined(_WIN32) || defined(_WIN64))
#define RGPOT_MOPACC_API __declspec(dllexport)
#else
#define RGPOT_MOPACC_API
#endif

/** OpenMOPAC model index. AM1 is the default Hamiltonian. */
#define MOPACC_MODEL_PM7 0
#define MOPACC_MODEL_PM6_D3H4 1
#define MOPACC_MODEL_PM6_ORG 2
#define MOPACC_MODEL_PM6 3
#define MOPACC_MODEL_AM1 4
#define MOPACC_MODEL_RM1 5

/**
 * Sticky method options. Pass as the params buffer, or omit the buffer
 * to get AM1 / charge 0 / closed shell / vacuum.
 */
typedef struct MopacCParams {
  int charge;       /**< net charge */
  int spin;         /**< OpenMOPAC spin excitations */
  int model;        /**< 4 = AM1; 0 here still means AM1 when size is 0 */
  double tolerance; /**< GNORM/RELSCF scale; 1.0 is the OpenMOPAC default */
  int max_time;     /**< seconds; 0 uses 3600 */
} MopacCParams;

typedef struct MopacCResult {
  int ok;          /**< 1 success, 0 failure */
  double energy_h; /**< heat of formation converted to Hartree */
  char message[512];
} MopacCResult;

/**
 * Apply sticky method options from a packed `MopacCParams` blob.
 *
 * @return 0 on success, non-zero if embed unavailable or params are invalid
 */
RGPOT_MOPACC_API int mopacc_set_params(const void *params,
                                       size_t params_size_bytes);

/**
 * Energy + nuclear gradient.
 *
 * @param n_atoms            atom count
 * @param positions_ang      n_atoms*3 Angstrom, row-major xyz
 * @param atomic_numbers     n_atoms Z values
 * @param params             packed `MopacCParams`, or NULL for sticky/defaults
 * @param params_size_bytes  size of params in bytes
 * @param grad_h_bohr        out n_atoms*3 Hartree/Bohr, row-major
 */
RGPOT_MOPACC_API MopacCResult mopacc_energy_gradient(
    int n_atoms, const double *positions_ang, const int *atomic_numbers,
    const void *params, size_t params_size_bytes, double *grad_h_bohr);

/** Engine version string (static buffer). */
RGPOT_MOPACC_API const char *mopacc_version(void);

/** Numeric ABI revision for fail-closed frontend validation. */
RGPOT_MOPACC_API int mopacc_c_abi_version(void);

/** 1 if real embed (libmopac linked), 0 if stub. */
RGPOT_MOPACC_API int mopacc_available(void);

#ifdef __cplusplus
}
#endif
