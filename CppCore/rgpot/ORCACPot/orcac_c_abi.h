/**
 * @file orcac_c_abi.h
 * @brief Function pointer types for the liborcac C ABI 1 that ORCACPot
 * resolves with dlsym.
 *
 * The structs, status codes and constants come from the vendored orcac.h,
 * pinned by orcac.h.pin. The exported surface is exactly these thirteen
 * functions; CppCore/tests/orcac_abi1_exports.txt lists them.
 */
#pragma once

#include "rgpot/ORCACPot/orcac.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef int (*orcac_config_init_fn)(OrcaCConfig *, size_t);
typedef int (*orcac_create_fn)(const OrcaCConfig *, OrcaCSession **, char *,
                               size_t);
typedef int (*orcac_configure_fn)(OrcaCSession *, const OrcaCConfig *);
typedef int (*orcac_energy_gradient_fn)(OrcaCSession *, int, const double *,
                                        const int *, uint32_t, double *,
                                        double *, OrcaCResult *);
typedef int (*orcac_cancel_fn)(OrcaCSession *);
typedef const char *(*orcac_last_error_fn)(const OrcaCSession *);
typedef void (*orcac_destroy_fn)(OrcaCSession *);
typedef const char *(*orcac_status_name_fn)(int);
typedef const char *(*orcac_version_fn)(void);
typedef int (*orcac_c_abi_version_fn)(void);
typedef int (*orcac_abi_version_fn)(void);
typedef int (*orcac_available_fn)(void);
typedef int (*orcac_probe_executable_fn)(const OrcaCConfig *, char *, size_t);

#ifdef __cplusplus
}
#endif
