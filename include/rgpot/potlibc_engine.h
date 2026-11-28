/* MIT License
 * Copyright 2023--present rgpot developers
 */
#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define RGPOT_POTLIBC_CONFIG_MAGIC UINT32_C(0x504F544C) /* "POTL" */
#define RGPOT_POTLIBC_CONFIG_VERSION_V1 UINT32_C(1)
#define RGPOT_POTLIBC_CONFIG_VERSION UINT32_C(2)
#define RGPOT_POTLIBC_SURFACE_ID_CAPACITY 128
#define RGPOT_POTLIBC_CONFIG_V1_SIZE UINT32_C(144)
#define RGPOT_POTLIBC_CONFIG_MIN_SIZE UINT32_C(152)

/**
 * Version 1 create message for libpotlibc_engine.
 *
 * A version-1 message is exactly
 * RGPOT_POTLIBC_CONFIG_V1_SIZE bytes, with no size field and no reserved
 * field. magic is at offset 0, version (RGPOT_POTLIBC_CONFIG_VERSION_V1) at
 * offset 4, atom_count at offset 8 and surface_id at offset 16.
 * RGPOT_POTLIBC_CONFIG_VERSION is 2, so sources that recompile this V1 struct
 * against current headers must use RGPOT_POTLIBC_CONFIG_VERSION_V1.
 *
 * surface_id is a NUL-terminated canonical potlibc catalog id.
 * atom_count is zero for a fixed-size surface and the exact requested count
 * for a variable-size surface.
 *
 * The engine accepts this message and RgpotPotlibcConfigV2. Any other version
 * is refused with the plain error string "potlibc config version is unknown".
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
  uint32_t version;
  uint64_t atom_count;
  char surface_id[RGPOT_POTLIBC_SURFACE_ID_CAPACITY];
} RgpotPotlibcConfigV1;

/**
 * Version 2 create message for libpotlibc_engine.
 *
 * size carries the message size in bytes, which must be at least
 * RGPOT_POTLIBC_CONFIG_MIN_SIZE. version is RGPOT_POTLIBC_CONFIG_VERSION.
 * reserved must be zero. Engines that know this version read the first
 * RGPOT_POTLIBC_CONFIG_MIN_SIZE bytes and require any trailing bytes in a
 * larger message to be zero.
 *
 * surface_id and atom_count have the same meaning as in RgpotPotlibcConfigV1.
 */
typedef struct {
  uint32_t magic;
  uint32_t size;
  uint32_t version;
  uint32_t reserved;
  uint64_t atom_count;
  char surface_id[RGPOT_POTLIBC_SURFACE_ID_CAPACITY];
} RgpotPotlibcConfigV2;

#ifdef __cplusplus
#define RGPOT_POTLIBC_STATIC_ASSERT(condition, message)                        \
  static_assert(condition, message)
#else
#define RGPOT_POTLIBC_STATIC_ASSERT(condition, message)                        \
  _Static_assert(condition, message)
#endif

RGPOT_POTLIBC_STATIC_ASSERT(sizeof(RgpotPotlibcConfigV1) == 144,
                            "RgpotPotlibcConfigV1 is 144 bytes");
RGPOT_POTLIBC_STATIC_ASSERT(offsetof(RgpotPotlibcConfigV1, magic) == 0,
                            "RgpotPotlibcConfigV1.magic is at offset 0");
RGPOT_POTLIBC_STATIC_ASSERT(offsetof(RgpotPotlibcConfigV1, version) == 4,
                            "RgpotPotlibcConfigV1.version is at offset 4");
RGPOT_POTLIBC_STATIC_ASSERT(offsetof(RgpotPotlibcConfigV1, atom_count) == 8,
                            "RgpotPotlibcConfigV1.atom_count is at offset 8");
RGPOT_POTLIBC_STATIC_ASSERT(offsetof(RgpotPotlibcConfigV1, surface_id) == 16,
                            "RgpotPotlibcConfigV1.surface_id is at offset 16");
RGPOT_POTLIBC_STATIC_ASSERT(RGPOT_POTLIBC_CONFIG_V1_SIZE == 144,
                            "RGPOT_POTLIBC_CONFIG_V1_SIZE is 144");
RGPOT_POTLIBC_STATIC_ASSERT(RGPOT_POTLIBC_CONFIG_VERSION_V1 == 1,
                            "potlibc config version 1 is 1");
RGPOT_POTLIBC_STATIC_ASSERT(sizeof(RgpotPotlibcConfigV2) == 152,
                            "RgpotPotlibcConfigV2 is 152 bytes");
RGPOT_POTLIBC_STATIC_ASSERT(RGPOT_POTLIBC_CONFIG_MIN_SIZE == 152,
                            "RGPOT_POTLIBC_CONFIG_MIN_SIZE is the V2 size");
RGPOT_POTLIBC_STATIC_ASSERT(RGPOT_POTLIBC_CONFIG_VERSION == 2,
                            "RGPOT_POTLIBC_CONFIG_VERSION is 2");
RGPOT_POTLIBC_STATIC_ASSERT(offsetof(RgpotPotlibcConfigV2, size) == 4,
                            "RgpotPotlibcConfigV2.size is at offset 4");
RGPOT_POTLIBC_STATIC_ASSERT(offsetof(RgpotPotlibcConfigV2, version) == 8,
                            "RgpotPotlibcConfigV2.version is at offset 8");

#undef RGPOT_POTLIBC_STATIC_ASSERT

#ifdef __cplusplus
}
#endif
