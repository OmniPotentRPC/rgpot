/* A harmonic engine behind the generic engine C ABI, for tests/engine.rs.
 * E = k/2 sum x^2 over all coordinates, F = -k x. The create message is one
 * native-endian double, k. An atomic number of zero fails the evaluation with
 * a recoverable code, a negative one with a fatal code. Compile definitions
 * drop symbols or change the ABI major. */
#include "rgpot/engine_c_abi.h"

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef FIXTURE_ABI
#define FIXTURE_ABI 1
#endif
#ifndef FIXTURE_MINOR
#define FIXTURE_MINOR RGPOT_ENGINE_ABI_MINOR
#endif
#ifndef FIXTURE_LAYOUT
#define FIXTURE_LAYOUT RGPOT_ENGINE_ABI_LAYOUT_REVISION
#endif
#ifndef FIXTURE_AVAILABLE
#define FIXTURE_AVAILABLE 1
#endif

struct RgpotEnginePot {
  double k;
  char error[128];
};

static int created = 0;
static int destroyed = 0;

int fixture_create_count(void) { return created; }
int fixture_destroy_count(void) { return destroyed; }
int fixture_unsupported_code(void) { return RGPOT_ENGINE_ERR_UNSUPPORTED; }
int fixture_lock_error_code(void) { return RGPOT_ENGINE_ERR_LOCK; }

int rgpot_engine_abi_version(void) { return FIXTURE_ABI; }

#ifndef NO_AVAILABLE
int rgpot_engine_available(void) { return FIXTURE_AVAILABLE; }
#endif

#ifndef NO_STAMP
RgpotEngineAbiStamp rgpot_engine_abi_stamp(void) {
  RgpotEngineAbiStamp stamp = {FIXTURE_ABI, FIXTURE_MINOR, FIXTURE_LAYOUT};
  return stamp;
}

int rgpot_engine_abi_compatible(const RgpotEngineAbiStamp *host) {
  return host != NULL && host->abi_major == FIXTURE_ABI &&
         host->layout_revision == FIXTURE_LAYOUT &&
         host->abi_minor <= FIXTURE_MINOR;
}
#endif

RgpotEnginePot *rgpot_engine_create(const void *config, size_t config_len,
                                    char *errbuf, size_t errlen) {
  RgpotEnginePot *pot;
  created += 1;
  if (config == NULL || config_len != sizeof(double)) {
    snprintf(errbuf, errlen, "harmonic engine: config is %zu bytes, need %zu",
             config_len, sizeof(double));
    return NULL;
  }
  if (((uintptr_t)config % 8) != 0) {
    snprintf(errbuf, errlen, "harmonic engine: config pointer %p is not 8-byte aligned",
             config);
    return NULL;
  }
  pot = calloc(1, sizeof *pot);
  memcpy(&pot->k, config, sizeof(double));
  return pot;
}

void rgpot_engine_destroy(RgpotEnginePot *pot) {
  destroyed += 1;
  free(pot);
}

#ifndef NO_FORCE
int rgpot_engine_force(RgpotEnginePot *pot, long n, const double *positions,
                       const int *atomic_nrs, double *forces, double *energy,
                       double *variance, const double *box,
                       rgpot_engine_coord_transform transform, void *user) {
  long i;
  double e = 0.0;
  (void)variance;
  (void)box;
  (void)transform;
  (void)user;
#ifdef FORCE_UNSUPPORTED
  snprintf(pot->error, sizeof pot->error, "force operation unavailable");
  return RGPOT_ENGINE_ERR_UNSUPPORTED;
#endif
#ifdef FORCE_LOCK_ERROR
  snprintf(pot->error, sizeof pot->error, "fixture handle lock failed");
  return RGPOT_ENGINE_ERR_LOCK;
#endif
  for (i = 0; i < n; i++) {
    if (atomic_nrs[i] == 0) {
      snprintf(pot->error, sizeof pot->error, "atom %ld has atomic number 0", i);
      return 1;
    }
    if (atomic_nrs[i] < 0) {
      snprintf(pot->error, sizeof pot->error, "atom %ld has a negative atomic number", i);
      return -1;
    }
  }
  for (i = 0; i < 3 * n; i++) {
    e += 0.5 * pot->k * positions[i] * positions[i];
    forces[i] = -pot->k * positions[i];
  }
  *energy = e;
  pot->error[0] = '\0';
  return 0;
}
#endif

#ifndef NO_BATCH
int rgpot_engine_force_batch(RgpotEnginePot *pot, long n_systems, long n,
                             const double *positions, const int *atomic_nrs,
                             double *forces, double *energies,
                             double *variances, const double *boxes,
                             rgpot_engine_coord_transform transform,
                             void *user) {
  long s;
  (void)variances;
  for (s = 0; s < n_systems; s++) {
    int rc = rgpot_engine_force(pot, n, positions + 3 * n * s, atomic_nrs,
                                forces + 3 * n * s, energies + s, NULL,
                                boxes + 9 * s, transform, user);
    if (rc != 0) return rc;
  }
  return 0;
}
#endif

#ifndef NO_LAST_ERROR
size_t rgpot_engine_last_error(const RgpotEnginePot *pot, char *buf, size_t len) {
  size_t full = strlen(pot->error);
  if (len > 0) {
    snprintf(buf, len, "%s", pot->error);
  }
  return full;
}
#endif

#ifndef NO_CAPS
int rgpot_engine_caps(const RgpotEnginePot *pot, RgpotEngineCaps *out) {
  RgpotEngineCaps filled;
  unsigned room;
  (void)pot;
#ifdef CAPS_ERROR
  return RGPOT_ENGINE_ERR_EVAL;
#endif
#ifdef CAPS_LOCK_ERROR
  snprintf(((RgpotEnginePot *)pot)->error, sizeof ((RgpotEnginePot *)pot)->error,
           "fixture caps lock failed");
  return RGPOT_ENGINE_ERR_LOCK;
#endif
  if (!out || out->size < sizeof(unsigned)) return 1;
  memset(&filled, 0, sizeof filled);
  filled.size = sizeof filled;
  filled.periodic = 1;
#if (!defined(NO_HVP) && !defined(NO_HVP_CAP)) || defined(FORCE_HVP_CAP)
  filled.analytic_hvp = 1;
#endif
#ifdef CAPS_SIZE_BEFORE_HVP
  filled.size = (unsigned)offsetof(RgpotEngineCaps, analytic_hvp);
#endif
  room = out->size < filled.size ? out->size : filled.size;
  memcpy(out, &filled, room);
  out->size = room;
  return 0;
}
#endif

#ifndef NO_HVP
int rgpot_engine_hvp(RgpotEnginePot *pot, long n, const double *positions,
                     const int *atomic_nrs, const double *direction,
                     double *hv_out, const double *box,
                     rgpot_engine_coord_transform transform, void *user) {
  (void)positions;
  (void)box;
  (void)transform;
  (void)user;
#if defined(NO_HVP_CAP) || defined(HVP_UNSUPPORTED)
  (void)n;
  (void)atomic_nrs;
  (void)direction;
  (void)hv_out;
  snprintf(pot->error, sizeof pot->error,
           "harmonic engine built without analytic HVP");
  return RGPOT_ENGINE_ERR_UNSUPPORTED;
#elif defined(HVP_LOCK_ERROR)
  (void)n;
  (void)atomic_nrs;
  (void)direction;
  (void)hv_out;
  snprintf(pot->error, sizeof pot->error, "fixture hvp lock failed");
  return RGPOT_ENGINE_ERR_LOCK;
#else
  long i;
  for (i = 0; i < n; i++) {
    if (atomic_nrs[i] == 0) {
      snprintf(pot->error, sizeof pot->error,
               "atom %ld has atomic number 0", i);
      return 1;
    }
    if (atomic_nrs[i] < 0) {
      snprintf(pot->error, sizeof pot->error,
               "atom %ld has a negative atomic number", i);
      return -1;
    }
  }
  for (i = 0; i < 3 * n; i++) {
    hv_out[i] = pot->k * direction[i];
  }
  pot->error[0] = '\0';
  return 0;
#endif
}
#endif
