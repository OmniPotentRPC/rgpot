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

/**
 * Version 1 create message for libpotlibc_engine.
 *
 * surface_id is a NUL-terminated canonical potlibc catalog id.
 * atom_count is zero for a fixed-size surface and the exact requested count
 * for a variable-size surface.
 */
typedef struct {
  uint32_t magic;
  uint32_t version;
  uint64_t atom_count;
  char surface_id[RGPOT_POTLIBC_SURFACE_ID_CAPACITY];
} RgpotPotlibcConfigV1;

#ifdef __cplusplus
}
#endif
