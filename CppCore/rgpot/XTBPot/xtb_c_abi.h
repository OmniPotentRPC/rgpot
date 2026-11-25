/**
 * @file xtb_c_abi.h
 * @brief C ABI between XTBDlopen frontend and libxtb_engine.so.
 *
 * Engine wraps libxtb's ISO_C_BINDING API (`xtb.h`) behind a small stable
 * plugin surface (create / force / destroy). Host units: positions Å, energy
 * eV, forces eV/Å. Mirrors the metatomic engine split: thin consumer,
 * optional heavy plugin.
 */
#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifdef _WIN32
#ifdef RGPOT_XTB_ENGINE_BUILD
#define RGPOT_XTB_API __declspec(dllexport)
#else
#define RGPOT_XTB_API __declspec(dllimport)
#endif
#else
#define RGPOT_XTB_API __attribute__((visibility("default")))
#endif

typedef struct RgpotXtbPot RgpotXtbPot;

/** GFN method codes (stable ABI). */
enum {
  RGPOT_XTB_METHOD_GFNFF = 0,
  RGPOT_XTB_METHOD_GFN0 = 1,
  RGPOT_XTB_METHOD_GFN1 = 2,
  RGPOT_XTB_METHOD_GFN2 = 3
};

typedef struct RgpotXtbConfig {
  int method; /**< RGPOT_XTB_METHOD_* */
  double accuracy;
  double electronic_temperature; /**< Kelvin */
  int max_iterations;
  double charge;
  int uhf;
} RgpotXtbConfig;

#define RGPOT_XTB_ABI_VERSION 1

RGPOT_XTB_API int rgpot_xtb_abi_version(void);

RGPOT_XTB_API RgpotXtbPot *rgpot_xtb_create(const RgpotXtbConfig *cfg,
                                            char *errbuf, size_t errlen);

RGPOT_XTB_API void rgpot_xtb_destroy(RgpotXtbPot *pot);

/** Return codes of rgpot_xtb_force. A frontend treats an unknown nonzero
 * code as a failure of unknown kind. */
enum {
  RGPOT_XTB_RC_OK = 0,
  RGPOT_XTB_RC_INVALID_ARGUMENT = -1,
  RGPOT_XTB_RC_FAILED = -2,
  RGPOT_XTB_RC_UNKNOWN = -3,
  /** The self-consistent field did not converge. */
  RGPOT_XTB_RC_SCC_NOT_CONVERGED = -4
};

/**
 * Energy + forces. Returns RGPOT_XTB_RC_OK (0) on success, otherwise one of
 * the RGPOT_XTB_RC_* codes. variance may be NULL.
 */
RGPOT_XTB_API int rgpot_xtb_force(RgpotXtbPot *pot, long nAtoms,
                                  const double *positions, const int *atomicNrs,
                                  double *forces, double *energy,
                                  double *variance, const double *box);

/**
 * Drop the stored SCC wavefunction. The calculator stays, and the next
 * force uses xtb's own guess.
 */
RGPOT_XTB_API void rgpot_xtb_reset(RgpotXtbPot *pot);

/** Displacement guard in Angstrom. Zero or negative disables reuse. */
RGPOT_XTB_API void rgpot_xtb_set_warm_move(RgpotXtbPot *pot, double angstrom);

RGPOT_XTB_API double rgpot_xtb_warm_move(const RgpotXtbPot *pot);

/** HOMO-LUMO gap guard in eV. Zero or negative reuses at any gap. */
RGPOT_XTB_API void rgpot_xtb_set_warm_min_gap(RgpotXtbPot *pot, double ev);

RGPOT_XTB_API double rgpot_xtb_warm_min_gap(const RgpotXtbPot *pot);

/** 1 when the last successful force reused the previous wavefunction. */
RGPOT_XTB_API int rgpot_xtb_reused_wavefunction(const RgpotXtbPot *pot);

RGPOT_XTB_API int rgpot_xtb_available(void);

/* ------------------------------------------------------------------ */
/* Optional symbols. An engine built before they were added lacks     */
/* them; a frontend probes them with dlsym and keeps working without. */
/* ------------------------------------------------------------------ */

/** Bits of rgpot_xtb_last_eval_flags; the values of
 * RGPOT_ENGINE_EVAL_SCC_RESCUED and RGPOT_ENGINE_EVAL_SCC_RETRIED. */
#define RGPOT_XTB_EVAL_SCC_RESCUED 0x1u
#define RGPOT_XTB_EVAL_SCC_RETRIED 0x2u

/**
 * Flags of the last successful force: RGPOT_XTB_EVAL_SCC_RESCUED when the
 * SCC converged only after the smeared-occupation rescue,
 * RGPOT_XTB_EVAL_SCC_RETRIED when a reused guess failed and xtb's own
 * guess converged. 0 for a plain SCC, before the first force, after
 * rgpot_xtb_reset and for a NULL pot.
 */
RGPOT_XTB_API unsigned long long
rgpot_xtb_last_eval_flags(const RgpotXtbPot *pot);

/**
 * Nonzero (the default) retries an SCC that does not converge with a
 * smeared occupation and cools back to the configured temperature. Zero
 * makes rgpot_xtb_force return RGPOT_XTB_RC_SCC_NOT_CONVERGED instead.
 */
RGPOT_XTB_API void rgpot_xtb_set_scc_rescue(RgpotXtbPot *pot, int enabled);

/** 1 when the SCC rescue is on, 0 when it is off or pot is NULL. */
RGPOT_XTB_API int rgpot_xtb_scc_rescue(const RgpotXtbPot *pot);

/** Source revision the engine was built from (a git short hash), or "".
 * The pointer refers to static storage. */
RGPOT_XTB_API const char *rgpot_xtb_source_revision(void);

#ifdef __cplusplus
}
#endif
