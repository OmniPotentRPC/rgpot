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
 * Version 1 parameters for the two-coordinate coupled-well engine.
 *
 * The model acts on the x and y coordinates of one atom:
 *
 * E = V0 ((x / x0)^2 - 1)^2
 *     + k / 2 (y - alpha x^2)^2.
 *
 * All parameters use the generic engine ABI units: eV and Angstrom.
 */
typedef struct {
  uint32_t magic;
  uint32_t version;
  int32_t atomic_number;
  uint32_t reserved;
  double barrier_energy;
  double well_position;
  double transverse_spring;
  double coupling;
} RgpotCoupledWellConfigV1;

#ifdef __cplusplus
}
#endif
