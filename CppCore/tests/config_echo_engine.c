/* MIT License
 * Copyright 2023--present rgpot developers
 *
 * A generic engine (engine_c_abi.h) that keeps the configuration message it
 * was created from. Its force is zero and its energy is the length of that
 * message, and config_echo_last copies out the bytes the last create saw, so
 * a test can check what a host handed the engine without the backend the
 * message is meant for. PotentialFactoryTest loads it in place of liborcac.
 */

#include "rgpot/engine_c_abi.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct RgpotEnginePot {
  size_t len;
};

static unsigned char g_last[65536];
static size_t g_last_len;

RGPOT_ENGINE_API int rgpot_engine_abi_version(void) {
  return RGPOT_ENGINE_ABI_VERSION;
}

RGPOT_ENGINE_API int rgpot_engine_available(void) { return 1; }

RGPOT_ENGINE_API RgpotEnginePot *rgpot_engine_create(const void *config,
                                                     size_t config_len,
                                                     char *errbuf,
                                                     size_t errlen) {
  RgpotEnginePot *pot;
  if (config_len > sizeof(g_last)) {
    snprintf(errbuf, errlen, "config_echo: %zu bytes is too long", config_len);
    return NULL;
  }
  if (config_len > 0) {
    memcpy(g_last, config, config_len);
  }
  g_last_len = config_len;
  pot = calloc(1, sizeof(*pot));
  if (pot == NULL) {
    snprintf(errbuf, errlen, "config_echo: out of memory");
    return NULL;
  }
  pot->len = config_len;
  return pot;
}

RGPOT_ENGINE_API void rgpot_engine_destroy(RgpotEnginePot *pot) { free(pot); }

RGPOT_ENGINE_API int rgpot_engine_force(RgpotEnginePot *pot, long nAtoms,
                                        const double *positions,
                                        const int *atomicNrs, double *forces,
                                        double *energy, double *variance,
                                        const double *box,
                                        rgpot_engine_coord_transform transform,
                                        void *transform_user) {
  long i;
  (void)positions;
  (void)atomicNrs;
  (void)box;
  (void)transform;
  (void)transform_user;
  for (i = 0; i < 3 * nAtoms; ++i) {
    forces[i] = 0.0;
  }
  *energy = (double)pot->len;
  if (variance != NULL) {
    *variance = 0.0;
  }
  return RGPOT_ENGINE_OK;
}

/* Copies at most cap bytes of the last create's message; returns its length. */
RGPOT_ENGINE_API size_t config_echo_last(void *out, size_t cap) {
  memcpy(out, g_last, g_last_len < cap ? g_last_len : cap);
  return g_last_len;
}
