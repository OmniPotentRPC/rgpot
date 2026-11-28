/* MIT License
 * Copyright 2023--present rgpot developers
 */
#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define RGPOT_COUPLED_WELL_CONFIG_MAGIC UINT32_C(0x43575054) /* "CWPT" */
#define RGPOT_COUPLED_WELL_CONFIG_VERSION_V1 UINT32_C(1)
#define RGPOT_COUPLED_WELL_CONFIG_VERSION UINT32_C(2)
#define RGPOT_COUPLED_WELL_CONFIG_V1_SIZE UINT32_C(48)
#define RGPOT_COUPLED_WELL_CONFIG_MIN_SIZE UINT32_C(56)

/**
 * Version 1 create message for libcoupled_well_engine.
 *
 * A version-1 message is exactly
 * RGPOT_COUPLED_WELL_CONFIG_V1_SIZE bytes. magic is at offset 0, version
 * (RGPOT_COUPLED_WELL_CONFIG_VERSION_V1) at offset 4, atomic_number at offset
 * 8, reserved at offset 12, barrier_energy at offset 16, well_position at
 * offset 24, transverse_spring at offset 32 and coupling at offset 40.
 * RGPOT_COUPLED_WELL_CONFIG_VERSION is 2, so sources that recompile this V1
 * struct against current headers must use RGPOT_COUPLED_WELL_CONFIG_VERSION_V1.
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
 * The engine accepts this message and RgpotCoupledWellConfigV2. Any other
 * version is refused with the plain error string
 * "coupled-well config version is unknown". create also refuses a
 * nonzero reserved field, a non-positive atomic_number, a barrier_energy,
 * well_position or transverse_spring that is not finite and positive, and a
 * coupling that is not finite.
 */
typedef struct {
  uint32_t magic;           /**< RGPOT_COUPLED_WELL_CONFIG_MAGIC */
  uint32_t version;         /**< RGPOT_COUPLED_WELL_CONFIG_VERSION_V1 */
  int32_t atomic_number;    /**< the one species the engine accepts, positive */
  uint32_t reserved;        /**< must be zero */
  double barrier_energy;    /**< V0, eV, finite and positive */
  double well_position;     /**< x0, Angstrom, finite and positive */
  double transverse_spring; /**< k, eV/Angstrom^2, finite and positive */
  double coupling;          /**< alpha, 1/Angstrom, finite */
} RgpotCoupledWellConfigV1;

/**
 * Version 2 create message for libcoupled_well_engine.
 *
 * size carries the message size in bytes, which must be at least
 * RGPOT_COUPLED_WELL_CONFIG_MIN_SIZE. version is
 * RGPOT_COUPLED_WELL_CONFIG_VERSION. reserved0 and reserved1 must be zero.
 * Engines that know this version read the first
 * RGPOT_COUPLED_WELL_CONFIG_MIN_SIZE bytes and require any trailing bytes in a
 * larger message to be zero.
 *
 * The physical fields have the same meaning and units as
 * RgpotCoupledWellConfigV1.
 */
typedef struct {
  uint32_t magic;           /**< RGPOT_COUPLED_WELL_CONFIG_MAGIC */
  uint32_t size;            /**< message bytes, at least MIN_SIZE */
  uint32_t version;         /**< RGPOT_COUPLED_WELL_CONFIG_VERSION */
  uint32_t reserved0;       /**< must be zero */
  int32_t atomic_number;    /**< the one species the engine accepts, positive */
  uint32_t reserved1;       /**< must be zero */
  double barrier_energy;    /**< V0, eV, finite and positive */
  double well_position;     /**< x0, Angstrom, finite and positive */
  double transverse_spring; /**< k, eV/Angstrom^2, finite and positive */
  double coupling;          /**< alpha, 1/Angstrom, finite */
} RgpotCoupledWellConfigV2;

#ifdef __cplusplus
#define RGPOT_COUPLED_WELL_STATIC_ASSERT(condition, message)                   \
  static_assert(condition, message)
#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
#define RGPOT_COUPLED_WELL_STATIC_ASSERT(condition, message)                   \
  _Static_assert(condition, message)
#else
/* MSVC, compiling this header as C, does not treat _Static_assert as a
 * keyword and reports a syntax error on each check. A char array of
 * bound 1 is a valid type when the condition is true. A bound of -1 is
 * a constraint violation when it is false, so the same checks still
 * fail the translation. */
#define RGPOT_COUPLED_WELL_SA_JOIN(left, right) left##right
#define RGPOT_COUPLED_WELL_SA_JOIN2(left, right)                               \
  RGPOT_COUPLED_WELL_SA_JOIN(left, right)
#define RGPOT_COUPLED_WELL_STATIC_ASSERT(condition, message)                   \
  typedef char RGPOT_COUPLED_WELL_SA_JOIN2(rgpot_coupled_well_static_assert_,  \
                                           __LINE__)[((condition) ? 1 : -1)]
#endif

RGPOT_COUPLED_WELL_STATIC_ASSERT(sizeof(RgpotCoupledWellConfigV1) == 48,
                                 "RgpotCoupledWellConfigV1 is 48 bytes");
RGPOT_COUPLED_WELL_STATIC_ASSERT(offsetof(RgpotCoupledWellConfigV1, magic) == 0,
                                 "RgpotCoupledWellConfigV1.magic is at offset 0");
RGPOT_COUPLED_WELL_STATIC_ASSERT(
    offsetof(RgpotCoupledWellConfigV1, version) == 4,
    "RgpotCoupledWellConfigV1.version is at offset 4");
RGPOT_COUPLED_WELL_STATIC_ASSERT(
    offsetof(RgpotCoupledWellConfigV1, atomic_number) == 8,
    "RgpotCoupledWellConfigV1.atomic_number is at offset 8");
RGPOT_COUPLED_WELL_STATIC_ASSERT(
    offsetof(RgpotCoupledWellConfigV1, reserved) == 12,
    "RgpotCoupledWellConfigV1.reserved is at offset 12");
RGPOT_COUPLED_WELL_STATIC_ASSERT(
    offsetof(RgpotCoupledWellConfigV1, barrier_energy) == 16,
    "RgpotCoupledWellConfigV1.barrier_energy is at offset 16");
RGPOT_COUPLED_WELL_STATIC_ASSERT(
    offsetof(RgpotCoupledWellConfigV1, well_position) == 24,
    "RgpotCoupledWellConfigV1.well_position is at offset 24");
RGPOT_COUPLED_WELL_STATIC_ASSERT(
    offsetof(RgpotCoupledWellConfigV1, transverse_spring) == 32,
    "RgpotCoupledWellConfigV1.transverse_spring is at offset 32");
RGPOT_COUPLED_WELL_STATIC_ASSERT(
    offsetof(RgpotCoupledWellConfigV1, coupling) == 40,
    "RgpotCoupledWellConfigV1.coupling is at offset 40");
RGPOT_COUPLED_WELL_STATIC_ASSERT(RGPOT_COUPLED_WELL_CONFIG_V1_SIZE == 48,
                                 "RGPOT_COUPLED_WELL_CONFIG_V1_SIZE is 48");
RGPOT_COUPLED_WELL_STATIC_ASSERT(RGPOT_COUPLED_WELL_CONFIG_VERSION_V1 == 1,
                                 "coupled-well config version 1 is 1");
RGPOT_COUPLED_WELL_STATIC_ASSERT(sizeof(RgpotCoupledWellConfigV2) == 56,
                                 "RgpotCoupledWellConfigV2 is 56 bytes");
RGPOT_COUPLED_WELL_STATIC_ASSERT(RGPOT_COUPLED_WELL_CONFIG_MIN_SIZE == 56,
                                 "RGPOT_COUPLED_WELL_CONFIG_MIN_SIZE is the V2 size");
RGPOT_COUPLED_WELL_STATIC_ASSERT(RGPOT_COUPLED_WELL_CONFIG_VERSION == 2,
                                 "RGPOT_COUPLED_WELL_CONFIG_VERSION is 2");
RGPOT_COUPLED_WELL_STATIC_ASSERT(offsetof(RgpotCoupledWellConfigV2, size) == 4,
                                 "RgpotCoupledWellConfigV2.size is at offset 4");
RGPOT_COUPLED_WELL_STATIC_ASSERT(
    offsetof(RgpotCoupledWellConfigV2, version) == 8,
    "RgpotCoupledWellConfigV2.version is at offset 8");
RGPOT_COUPLED_WELL_STATIC_ASSERT(
    offsetof(RgpotCoupledWellConfigV2, reserved0) == 12,
    "RgpotCoupledWellConfigV2.reserved0 is at offset 12");
RGPOT_COUPLED_WELL_STATIC_ASSERT(
    offsetof(RgpotCoupledWellConfigV2, atomic_number) == 16,
    "RgpotCoupledWellConfigV2.atomic_number is at offset 16");
RGPOT_COUPLED_WELL_STATIC_ASSERT(
    offsetof(RgpotCoupledWellConfigV2, reserved1) == 20,
    "RgpotCoupledWellConfigV2.reserved1 is at offset 20");
RGPOT_COUPLED_WELL_STATIC_ASSERT(
    offsetof(RgpotCoupledWellConfigV2, barrier_energy) == 24,
    "RgpotCoupledWellConfigV2.barrier_energy is at offset 24");
RGPOT_COUPLED_WELL_STATIC_ASSERT(
    offsetof(RgpotCoupledWellConfigV2, coupling) == 48,
    "RgpotCoupledWellConfigV2.coupling is at offset 48");

#undef RGPOT_COUPLED_WELL_STATIC_ASSERT
#undef RGPOT_COUPLED_WELL_SA_JOIN
#undef RGPOT_COUPLED_WELL_SA_JOIN2

#ifdef __cplusplus
}
#endif
