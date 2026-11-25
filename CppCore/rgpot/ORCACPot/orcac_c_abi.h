/**
 * @file orcac_c_abi.h
 * @brief The part of liborcac's C ABI (OmniPotentRPC/orcac, include/orcac.h)
 * that ORCACPot calls.
 *
 * Parameters are unpacked flat Cap'n Proto messages of the rgconf schema:
 * ORCAParams, or a PotentialConfig whose union is the orca arm. Units:
 * positions in Angstrom; energy in Hartree and forces in Hartree/Bohr.
 * Keep this header in lockstep with orcac's include/orcac.h.
 */
#pragma once

#include <stddef.h>

#define RGPOT_ORCAC_C_ABI_VERSION 1

#ifdef __cplusplus
extern "C" {
#endif

typedef struct OrcaCResult {
  int ok;              /**< 1 success, 0 failure */
  double energy_h;     /**< total energy in Hartree */
  int reused_orbitals; /**< 1 when ORCA started from the previous orbitals */
  char message[1024];  /**< empty on success; truncated error otherwise */
} OrcaCResult;

typedef struct OrcaCSession OrcaCSession;

typedef int (*orcac_c_abi_version_fn)(void);
typedef int (*orcac_available_fn)(void);
typedef OrcaCSession *(*orcac_session_create_fn)(const void *, size_t);
typedef OrcaCSession *(*orcac_session_create_from_config_fn)(const void *,
                                                             size_t);
typedef int (*orcac_session_set_params_fn)(OrcaCSession *, const void *,
                                           size_t);
typedef void (*orcac_session_destroy_fn)(OrcaCSession *);
typedef OrcaCResult (*orcac_session_energy_forces_fn)(OrcaCSession *, int,
                                                      const double *,
                                                      const int *, double *);
typedef const char *(*orcac_session_last_error_fn)(const OrcaCSession *);
typedef const char *(*orcac_last_error_fn)(void);
typedef const char *(*orcac_session_run_directory_fn)(const OrcaCSession *);

#ifdef __cplusplus
}
#endif
