/**
 * @file orca_c_abi.h
 * @brief C ABI for the ORCA subprocess potential.
 *
 * Host units match the rest of rgpot: positions in Angstrom, energy in
 * eV, forces in eV/Angstrom. The box is accepted and ignored.
 *
 * The calculate entry is `rgpot_orca_calculate`. It is not named
 * `rgpot_orca_force`: librgpot's Fortran symbol audit rejects an
 * exported `rgpot_*_force` symbol.
 */
#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifdef _WIN32
#ifdef RGPOT_ORCA_ENGINE_BUILD
#define RGPOT_ORCA_API __declspec(dllexport)
#else
#define RGPOT_ORCA_API __declspec(dllimport)
#endif
#else
#define RGPOT_ORCA_API __attribute__((visibility("default")))
#endif

typedef struct RgpotOrcaPot RgpotOrcaPot;

/**
 * Zero is not a set of defaults. Pass a null pointer to
 * `rgpot_orca_create` for `OrcaConfig`'s defaults. A null string keeps
 * that field's default (`HF`, `STO-3G`, empty keywords and executable).
 * `reuse_gbw` of 0 disables `.gbw` reuse. `nprocs` and `maxcore_mb` of
 * 0 omit the matching ORCA block.
 */
typedef struct RgpotOrcaConfig {
  int charge;
  int multiplicity;
  const char *method;
  const char *basis;
  const char *keywords;
  const char *executable;
  int reuse_gbw;
  int nprocs;
  int maxcore_mb;
} RgpotOrcaConfig;

#define RGPOT_ORCA_ABI_VERSION 1

RGPOT_ORCA_API int rgpot_orca_abi_version(void);

/// 1: this driver is compiled. The `orca` binary is resolved on calculate.
RGPOT_ORCA_API int rgpot_orca_available(void);

RGPOT_ORCA_API RgpotOrcaPot *rgpot_orca_create(const RgpotOrcaConfig *cfg,
                                               char *errbuf, size_t errlen);

RGPOT_ORCA_API void rgpot_orca_destroy(RgpotOrcaPot *pot);

RGPOT_ORCA_API void rgpot_orca_reset(RgpotOrcaPot *pot);

/// 1 when the last successful calculate reused the previous `.gbw`.
RGPOT_ORCA_API int rgpot_orca_reused_wavefunction(const RgpotOrcaPot *pot);

/**
 * Energy and forces. Returns 0 on success, 1 on a bad argument, 2 when
 * ORCA failed. The reason is in `rgpot_orca_last_error`.
 * `variance` may be null. `box` is nine doubles and is not used.
 */
RGPOT_ORCA_API int rgpot_orca_calculate(RgpotOrcaPot *pot, long nAtoms,
                                        const double *positions,
                                        const int *atomicNrs, double *forces,
                                        double *energy, double *variance,
                                        const double *box);

/// Empty string when the last call succeeded. Never null.
RGPOT_ORCA_API const char *rgpot_orca_last_error(const RgpotOrcaPot *pot);

#ifdef __cplusplus
}
#endif
