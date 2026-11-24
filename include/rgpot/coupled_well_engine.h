/* MIT License
 * Copyright 2023--present rgpot developers
 */
#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define RGPOT_COUPLED_WELL_CONFIG_MAGIC UINT32_C(0x43575054) /* "CWPT" */
#define RGPOT_COUPLED_WELL_CONFIG_VERSION UINT32_C(1)

/**
 * Version 1 create message for libcoupled_well_engine.
 *
 * The engine models one atom and acts on its x and y coordinates:
 *
 *   E(x, y) = V0 ((x / x0)^2 - 1)^2 + k / 2 (y - alpha x^2)^2
 *
 * with V0 = barrier_energy, x0 = well_position, k = transverse_spring and
 * alpha = coupling. The energy does not depend on z, so the z force and the z
 * row and column of the Hessian are zero. The cell is ignored: the model is
 * not periodic.
 *
 * Units are the generic engine ABI units: positions in Angstrom, energy in eV,
 * barrier_energy in eV, well_position in Angstrom, transverse_spring in
 * eV/Angstrom^2 and coupling in 1/Angstrom.
 *
 * create refuses a message whose magic, version or reserved field is wrong, a
 * non-positive atomic_number, a barrier_energy, well_position or
 * transverse_spring that is not finite and positive, and a coupling that is
 * not finite.
 */
typedef struct {
  uint32_t magic;           /**< RGPOT_COUPLED_WELL_CONFIG_MAGIC */
  uint32_t version;         /**< RGPOT_COUPLED_WELL_CONFIG_VERSION */
  int32_t atomic_number;    /**< the one species the engine accepts, positive */
  uint32_t reserved;        /**< must be zero */
  double barrier_energy;    /**< V0, eV, finite and positive */
  double well_position;     /**< x0, Angstrom, finite and positive */
  double transverse_spring; /**< k, eV/Angstrom^2, finite and positive */
  double coupling;          /**< alpha, 1/Angstrom, finite */
} RgpotCoupledWellConfigV1;

#ifdef __cplusplus
static_assert(sizeof(RgpotCoupledWellConfigV1) == 48,
              "RgpotCoupledWellConfigV1 is 48 bytes without padding");
#else
_Static_assert(sizeof(RgpotCoupledWellConfigV1) == 48,
               "RgpotCoupledWellConfigV1 is 48 bytes without padding");
#endif

#ifdef __cplusplus
}
#endif
