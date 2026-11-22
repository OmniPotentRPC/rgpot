/* MIT License
 * Copyright 2023--present rgpot developers
 */
#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define RGPOT_POTLIBC_CONFIG_MAGIC UINT32_C(0x504F544C) /* "POTL" */
#define RGPOT_POTLIBC_CONFIG_VERSION UINT32_C(1)
#define RGPOT_POTLIBC_SURFACE_ID_CAPACITY 128
#define RGPOT_POTLIBC_CONFIG_MIN_SIZE UINT32_C(152)

/**
 * Version 1 create message for libpotlibc_engine.
 *
 * size carries the message size in bytes, which must be at least
 * RGPOT_POTLIBC_CONFIG_MIN_SIZE. Engines that know this version read the first
 * RGPOT_POTLIBC_CONFIG_MIN_SIZE bytes and require any trailing bytes in a
 * larger message to be zero.
 *
 * surface_id is a NUL-terminated canonical potlibc catalog id.
 * atom_count is zero for a fixed-size surface and the exact requested count
 * for a variable-size surface.
 *
 * Evaluation and capability calls on an RgpotEnginePot serialize per handle
 * with an error-checking mutex; if acquiring that mutex fails, the call
 * records a diagnostic for rgpot_engine_last_error and returns the fatal code
 * RGPOT_ENGINE_ERR_LOCK from rgpot/engine_c_abi.h. rgpot_engine_destroy must
 * not race an active call on the same handle and still releases the surface
 * and handle allocation if locking fails during teardown.
 */
typedef struct {
  uint32_t magic;
  uint32_t size;
  uint32_t version;
  uint32_t reserved;
  uint64_t atom_count;
  char surface_id[RGPOT_POTLIBC_SURFACE_ID_CAPACITY];
} RgpotPotlibcConfigV1;

#ifdef __cplusplus
static_assert(sizeof(RgpotPotlibcConfigV1) == RGPOT_POTLIBC_CONFIG_MIN_SIZE,
              "RgpotPotlibcConfigV1 has the documented wire size");
#else
_Static_assert(sizeof(RgpotPotlibcConfigV1) == RGPOT_POTLIBC_CONFIG_MIN_SIZE,
               "RgpotPotlibcConfigV1 has the documented wire size");
#endif

#ifdef __cplusplus
}
#endif
