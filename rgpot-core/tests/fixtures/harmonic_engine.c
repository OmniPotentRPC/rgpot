/* A harmonic engine behind the generic engine C ABI, for tests/engine.rs.
 * E = k/2 sum x^2 over all coordinates, F = -k x. The create message is one
 * native-endian double, k. An atomic number of zero fails the evaluation with
 * a recoverable code, a negative one with a fatal code. Compile definitions
 * drop symbols or change the ABI major. ACCEPT_CALLBACK_REPLACEMENT makes a
 * second callback registration succeed. RETAIN_REFUSED_CALLBACKS makes the
 * refused one stay registered although the call fails, as an engine written
 * before the header forbade it might. FIXTURE_UNLOAD_POLICY exports
 * rgpot_engine_unload_policy returning that value. REPLACE_SELF_IN_CREATE
 * makes the first create call rename "<its own file>.next" over its own file,
 * as a build that rewrites the plugin while it loads would. REENTER_IN_FINALIZER
 * calls the function whose address is in RGPOT_FIXTURE_FINI_CALLBACK from the
 * library destructor, which runs when the library is unloaded. */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE 1
#endif

#include "rgpot/engine_c_abi.h"

#include <dlfcn.h>

#include <errno.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifndef _WIN32
#include <time.h>
#endif

#ifndef FIXTURE_ABI
#define FIXTURE_ABI 1
#endif

#ifndef FIXTURE_STAMP_MAJOR
#define FIXTURE_STAMP_MAJOR FIXTURE_ABI
#endif

#ifndef FIXTURE_ABI_MINOR
#define FIXTURE_ABI_MINOR RGPOT_ENGINE_ABI_MINOR
#endif

#ifndef FIXTURE_JET_ORDER
#define FIXTURE_JET_ORDER 3
#endif

#ifndef FIXTURE_ABI_LAYOUT
#define FIXTURE_ABI_LAYOUT RGPOT_ENGINE_ABI_LAYOUT_REVISION
#endif
#ifndef FIXTURE_AVAILABLE
#define FIXTURE_AVAILABLE 1
#endif
#ifndef FIXTURE_REENTRANCY
#define FIXTURE_REENTRANCY 0
#endif

struct RgpotEnginePot {
  double k;
  double last_energy;
  int evaluated;
  unsigned long long model_state;
  RgpotEngineCallbacks callbacks;
#ifdef FIXTURE_CACHE
  int cache_attached;
#endif
  int active;
  char error[128];
};

static int created = 0;
static int destroyed = 0;
static int native_calls = 0;
static int batch_calls = 0;
static int null_callback_clears = 0;
static int last_oracle_rc = -1000000;
#ifdef REENTER_IN_INITIALIZER
static int initializer_reentry_failed = 0;
#endif
static int active_calls = 0;
static int max_concurrent_calls = 0;
static int max_handle_concurrent_calls = 0;
static RgpotEnginePot *last_pot = NULL;
#ifdef FATAL_ONCE
static int fatal_once_fired = 0;
#endif

#ifdef REENTER_IN_FINALIZER
__attribute__((destructor)) static void reenter_from_finalizer(void) {
  const char *address = getenv("RGPOT_FIXTURE_FINI_CALLBACK");
  uintptr_t raw;
  int (*callback)(void);
  if (address == NULL) {
    fputs("harmonic engine: RGPOT_FIXTURE_FINI_CALLBACK is not set\n", stderr);
    abort();
  }
  raw = (uintptr_t)strtoull(address, NULL, 16);
  memcpy(&callback, &raw, sizeof(callback));
  if (callback == NULL || callback() != 0) {
    fputs("harmonic engine: the finalizer could not load another engine\n",
          stderr);
    abort();
  }
}
#endif

#ifdef REENTER_IN_INITIALIZER
__attribute__((constructor)) static void reenter_from_initializer(void) {
  const char *address = getenv("RGPOT_FIXTURE_INIT_CALLBACK");
  uintptr_t raw;
  int (*callback)(void);
  if (address == NULL) {
    initializer_reentry_failed = 1;
    return;
  }
  raw = (uintptr_t)strtoull(address, NULL, 16);
  memcpy(&callback, &raw, sizeof(callback));
  if (callback == NULL || callback() != 0) {
    initializer_reentry_failed = 1;
  }
}
#endif

static void count_native_call(void) {
  __atomic_add_fetch(&native_calls, 1, __ATOMIC_SEQ_CST);
}

static void record_peak(int *peak, int candidate) {
  int prev = __atomic_load_n(peak, __ATOMIC_SEQ_CST);
  while (candidate > prev) {
    if (__atomic_compare_exchange_n(peak, &prev, candidate, 0,
                                    __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST)) {
      break;
    }
  }
}

static void enter_call(RgpotEnginePot *pot) {
  int global_now = __atomic_add_fetch(&active_calls, 1, __ATOMIC_SEQ_CST);
  int handle_now = __atomic_add_fetch(&pot->active, 1, __ATOMIC_SEQ_CST);
  record_peak(&max_concurrent_calls, global_now);
  record_peak(&max_handle_concurrent_calls, handle_now);
#if defined(CONCURRENCY_PROBE) && !defined(_WIN32)
  {
    struct timespec pause;
    pause.tv_sec = 0;
    pause.tv_nsec = 3000000L;
    nanosleep(&pause, NULL);
  }
#endif
}

static void leave_call(RgpotEnginePot *pot) {
  __atomic_sub_fetch(&pot->active, 1, __ATOMIC_SEQ_CST);
  __atomic_sub_fetch(&active_calls, 1, __ATOMIC_SEQ_CST);
}

int fixture_create_count(void) {
  return __atomic_load_n(&created, __ATOMIC_SEQ_CST);
}
int fixture_destroy_count(void) {
  return __atomic_load_n(&destroyed, __ATOMIC_SEQ_CST);
}
int fixture_native_call_count(void) {
  return __atomic_load_n(&native_calls, __ATOMIC_SEQ_CST);
}
int fixture_max_concurrent(void) {
  return __atomic_load_n(&max_concurrent_calls, __ATOMIC_SEQ_CST);
}
int fixture_max_handle_concurrent(void) {
  return __atomic_load_n(&max_handle_concurrent_calls, __ATOMIC_SEQ_CST);
}
int fixture_batch_call_count(void) { return batch_calls; }
int fixture_null_callback_clear_count(void) { return null_callback_clears; }
int fixture_last_oracle_rc(void) { return last_oracle_rc; }
void fixture_reset_concurrent(void) {
  __atomic_store_n(&active_calls, 0, __ATOMIC_SEQ_CST);
  __atomic_store_n(&max_concurrent_calls, 0, __ATOMIC_SEQ_CST);
  __atomic_store_n(&max_handle_concurrent_calls, 0, __ATOMIC_SEQ_CST);
}
int fixture_unsupported_code(void) { return RGPOT_ENGINE_ERR_UNSUPPORTED; }
int fixture_lock_error_code(void) { return RGPOT_ENGINE_ERR_LOCK; }

/* Invokes the registered oracle the way an engine thread would, outside any
 * call the host made. */
int fixture_call_oracle_outside_a_call(void) {
  double positions[3] = {1.0, 2.0, 3.0};
  double forces[3];
  double energy = 0.0;
  double box[9] = {10.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0};
  int number = 1;
  if (last_pot == NULL || last_pot->callbacks.oracle == NULL) return -1000000;
  return last_pot->callbacks.oracle(last_pot->callbacks.ctx, 1, positions,
                                    &number, forces, &energy, box);
}

int rgpot_engine_abi_version(void) { return FIXTURE_ABI; }

#ifndef NO_AVAILABLE
int rgpot_engine_available(void) { return FIXTURE_AVAILABLE; }
#endif

#ifndef NO_STAMP
RgpotEngineAbiStamp rgpot_engine_abi_stamp(void) {
  RgpotEngineAbiStamp stamp;
  stamp.abi_major = (unsigned short)FIXTURE_STAMP_MAJOR;
  stamp.abi_minor = (unsigned short)FIXTURE_ABI_MINOR;
  stamp.layout_revision = (unsigned short)FIXTURE_ABI_LAYOUT;
  return stamp;
}
#endif

RgpotEnginePot *rgpot_engine_create(const void *config, size_t config_len,
                                    char *errbuf, size_t errlen) {
  RgpotEnginePot *pot;
  __atomic_add_fetch(&created, 1, __ATOMIC_SEQ_CST);
#ifdef REPLACE_SELF_IN_CREATE
  {
    Dl_info info;
    char next[4096];
    if (dladdr((void *)&rgpot_engine_create, &info) == 0 ||
        info.dli_fname == NULL) {
      snprintf(errbuf, errlen, "harmonic engine: cannot locate its own file");
      return NULL;
    }
    snprintf(next, sizeof next, "%s.next", info.dli_fname);
    if (rename(next, info.dli_fname) != 0) {
      snprintf(errbuf, errlen, "harmonic engine: cannot replace %s: %s",
               info.dli_fname, strerror(errno));
      return NULL;
    }
  }
#endif
#ifdef REENTER_IN_INITIALIZER
  if (initializer_reentry_failed) {
    snprintf(errbuf, errlen,
             "harmonic engine: initializer could not load another engine");
    return NULL;
  }
#endif
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
  last_pot = pot;
  return pot;
}

#ifdef FIXTURE_CACHE
int rgpot_engine_set_cache(RgpotEnginePot *pot, const char *path) {
  if (path && strcmp(path, "cache/geometry results") != 0) {
    snprintf(pot->error, sizeof pot->error, "unexpected cache path");
    return RGPOT_ENGINE_ERR_INVALID;
  }
  pot->cache_attached = path != NULL;
  return RGPOT_ENGINE_OK;
}

int rgpot_engine_cache_counts(RgpotEnginePot *pot, uint64_t *computed,
                              uint64_t *served) {
  if (!pot->cache_attached) {
    snprintf(pot->error, sizeof pot->error, "cache detached");
    return RGPOT_ENGINE_ERR_UNSUPPORTED;
  }
  *computed = (uint64_t)UINT32_MAX + 1;
  *served = 7;
  return RGPOT_ENGINE_OK;
}
#endif

void rgpot_engine_destroy(RgpotEnginePot *pot) {
  __atomic_add_fetch(&destroyed, 1, __ATOMIC_SEQ_CST);
  if (last_pot == pot) last_pot = NULL;
  free(pot);
}

#ifndef NO_FORCE
int rgpot_engine_force(RgpotEnginePot *pot, long n, const double *positions,
                       const int *atomic_nrs, double *forces, double *energy,
                       double *variance, const double *box,
                       rgpot_engine_coord_transform transform, void *user) {
  long i;
  double e = 0.0;
  double *scratch = NULL;
  double scratch_box[9];
  count_native_call();
  (void)variance;
  if (transform != NULL) {
    int rc;
    scratch = malloc((size_t)(3 * n) * sizeof(double));
    memcpy(scratch, positions, (size_t)(3 * n) * sizeof(double));
    memcpy(scratch_box, box, sizeof scratch_box);
    rc = transform(user, n, scratch, scratch_box);
    if (rc != 0) {
      free(scratch);
#ifdef FATAL_AFTER_HOST_FAILURE
      snprintf(pot->error, sizeof pot->error, "host failure left the engine unusable");
      return -8;
#else
      return rc;
#endif
    }
    positions = scratch;
    box = scratch_box;
  }
#ifdef FATAL_ONCE
  if (!fatal_once_fired) {
    fatal_once_fired = 1;
    snprintf(pot->error, sizeof pot->error, "fatal once");
    free(scratch);
    return -7;
  }
#endif
#ifdef FORCE_LOCK_ERROR
  snprintf(pot->error, sizeof pot->error, "fixture handle lock failed");
  return RGPOT_ENGINE_ERR_LOCK;
#endif
  enter_call(pot);
  for (i = 0; i < n; i++) {
    if (atomic_nrs[i] == 0) {
      snprintf(pot->error, sizeof pot->error, "atom %ld has atomic number 0", i);
      free(scratch);
      leave_call(pot);
      return 1;
    }
    if (atomic_nrs[i] < 0) {
      snprintf(pot->error, sizeof pot->error, "atom %ld has a negative atomic number", i);
      free(scratch);
      leave_call(pot);
      return -1;
    }
  }
  if (pot->callbacks.oracle != NULL) {
    int rc;
    if (pot->callbacks.on_acquire != NULL)
      pot->callbacks.on_acquire(pot->callbacks.ctx, RGPOT_ACQUIRE_VARIANCE);
    rc = pot->callbacks.oracle(pot->callbacks.ctx, n, positions, atomic_nrs,
                               forces, energy, box);
    last_oracle_rc = rc;
#ifdef FATAL_AFTER_HOST_FAILURE
    if (rc != 0) {
      snprintf(pot->error, sizeof pot->error, "host failure left the engine unusable");
      free(scratch);
      leave_call(pot);
      return -8;
    }
#endif
    if (rc == 0) {
      pot->model_state += 1;
      if (pot->callbacks.on_model_update != NULL)
        pot->callbacks.on_model_update(pot->callbacks.ctx, pot->model_state);
    }
    free(scratch);
    leave_call(pot);
    return rc;
  }
#ifdef REJECT_COINCIDENT
  {
    long j;
    for (i = 0; i < n; i++) {
      for (j = i + 1; j < n; j++) {
        double dx = positions[3 * i] - positions[3 * j];
        double dy = positions[3 * i + 1] - positions[3 * j + 1];
        double dz = positions[3 * i + 2] - positions[3 * j + 2];
        if (!(dx * dx + dy * dy + dz * dz > 0.0)) {
          snprintf(pot->error, sizeof pot->error, "coincident atoms");
          free(scratch);
          leave_call(pot);
          return RGPOT_ENGINE_ERR_EVAL;
        }
      }
    }
  }
#endif
  for (i = 0; i < 3 * n; i++) {
    e += 0.5 * pot->k * positions[i] * positions[i];
    forces[i] = -pot->k * positions[i];
  }
  *energy = e;
  pot->last_energy = e;
  pot->evaluated = 1;
  pot->error[0] = '\0';
  free(scratch);
  leave_call(pot);
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
  int first = 0;
  count_native_call();
  batch_calls += 1;
  (void)variances;
  for (s = 0; s < n_systems; s++) {
    int rc = rgpot_engine_force(pot, n, positions + 3 * n * s, atomic_nrs,
                                forces + 3 * n * s, energies + s, NULL,
                                boxes + 9 * s, transform, user);
#ifdef BATCH_ALL_SYSTEMS
    if (rc != 0 && first == 0) first = rc;
#else
    if (rc != 0) return rc;
#endif
  }
  return first;
}
#endif

unsigned long long rgpot_engine_model_state(const RgpotEnginePot *pot) {
  return pot->model_state;
}

int rgpot_engine_set_callbacks(RgpotEnginePot *pot,
                               const RgpotEngineCallbacks *callbacks) {
  if (pot == NULL) return RGPOT_ENGINE_ERR_INVALID;
  if (callbacks == NULL) {
    null_callback_clears += 1;
    memset(&pot->callbacks, 0, sizeof pot->callbacks);
    return RGPOT_ENGINE_OK;
  }
#ifndef ACCEPT_CALLBACK_REPLACEMENT
  if (pot->callbacks.oracle != NULL) {
    snprintf(pot->error, sizeof pot->error, "callback replacement refused");
#ifdef RETAIN_REFUSED_CALLBACKS
    pot->callbacks = *callbacks;
#endif
    return RGPOT_ENGINE_ERR_EVAL;
  }
#endif
  pot->callbacks = *callbacks;
  return RGPOT_ENGINE_OK;
}

int rgpot_engine_hint(RgpotEnginePot *pot, const void *msg, size_t msg_len,
                      char *errbuf, size_t errlen) {
  const unsigned char *bytes = msg;
  if (pot == NULL || msg == NULL || msg_len == 0) {
    snprintf(errbuf, errlen, "invalid hint");
    return RGPOT_ENGINE_ERR_INVALID;
  }
  if (bytes[0] == 0xff) {
    snprintf(errbuf, errlen, "hint refused");
    return RGPOT_ENGINE_ERR_EVAL;
  }
  if (bytes[0] == 0xfe) return -5;
  pot->model_state += 1;
  if (pot->callbacks.on_model_update != NULL)
    pot->callbacks.on_model_update(pot->callbacks.ctx, pot->model_state);
  return RGPOT_ENGINE_OK;
}

#ifndef NO_LAST_ERROR
size_t rgpot_engine_last_error(const RgpotEnginePot *pot, char *buf, size_t len) {
  size_t full = strlen(pot->error);
  count_native_call();
  if (len > 0) {
    snprintf(buf, len, "%s", pot->error);
  }
  return full;
}
#endif

#ifndef NO_CAPS
#ifdef SHORT_CAPS
/* The layout before analytic_hessian was appended. The host must keep a
 * field it did not receive as the zero it wrote, not as a bit this plugin
 * never stored. */
typedef struct {
  unsigned size;
  int reentrancy;
  int per_image_instances;
  int batched;
  int periodic;
  int stress;
  int group_collective;
  int orbital_selection;
  int orbital_copy;
  int analytic_hvp;
} RgpotEngineCapsPrefix;
#endif

int rgpot_engine_caps(const RgpotEnginePot *pot, RgpotEngineCaps *out) {
#ifdef SHORT_CAPS
  RgpotEngineCapsPrefix filled;
#else
  RgpotEngineCaps filled;
#endif
  unsigned room;
  count_native_call();
#ifdef CAPS_LOCK_ERROR
  snprintf(((RgpotEnginePot *)pot)->error, sizeof ((RgpotEnginePot *)pot)->error,
           "fixture caps lock failed");
  return RGPOT_ENGINE_ERR_LOCK;
#endif
#ifdef CAPS_ERROR_CODE
  snprintf(((RgpotEnginePot *)pot)->error, sizeof ((RgpotEnginePot *)pot)->error,
           "fixture caps failed");
  return CAPS_ERROR_CODE;
#endif
  (void)pot;
  if (!out || out->size < sizeof(unsigned)) return 1;
  memset(&filled, 0, sizeof filled);
  filled.size = sizeof filled;
  filled.reentrancy = FIXTURE_REENTRANCY;
  filled.periodic = 1;
#if !defined(NO_HVP) && !defined(NO_HVP_CAP)
  filled.analytic_hvp = 1;
#if !defined(SHORT_CAPS) && (FIXTURE_ABI_MINOR >= 12)
  filled.analytic_hessian = 1;
#endif
#endif
#if !defined(NO_JET) && !defined(NO_JET_CAP) && (FIXTURE_ABI_MINOR >= 11)
  filled.jet_order = FIXTURE_JET_ORDER;
#ifdef JET_NEGATIVE_CAP
  filled.jet_order = -1;
#endif
#endif
  room = out->size < sizeof filled ? out->size : sizeof filled;
#if FIXTURE_ABI_MINOR < 12
  if (room > offsetof(RgpotEngineCaps, analytic_hessian)) {
    room = (unsigned)offsetof(RgpotEngineCaps, analytic_hessian);
  }
#endif
#ifdef FIXTURE_CAPS_MINOR4
  /* A minor-4 engine's caps struct ends before jet_order. */
  if (room > offsetof(RgpotEngineCaps, jet_order)) {
    room = (unsigned)offsetof(RgpotEngineCaps, jet_order);
  }
#endif
  memcpy(out, &filled, room);
#ifdef FIXTURE_CAPS_DIRTY_TAIL
  /* Bytes past the size the engine reports are not part of the answer. */
  out->jet_order = FIXTURE_JET_ORDER;
  out->analytic_hessian = 1;
#endif
  out->size = room;
  return 0;
}
#endif

#ifndef NO_HVP
int rgpot_engine_hvp(RgpotEnginePot *pot, long n, const double *positions,
                     const int *atomic_nrs, const double *direction,
                     double *hv_out, const double *box) {
  count_native_call();
  (void)positions;
  (void)box;
#if defined(NO_HVP_CAP) || defined(HVP_UNSUPPORTED)
  (void)n;
  (void)atomic_nrs;
  (void)direction;
  (void)hv_out;
  snprintf(pot->error, sizeof pot->error, "harmonic engine built without analytic HVP");
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
  uintptr_t db;
  uintptr_t hb;
  uintptr_t bytes;
#ifdef FATAL_ONCE
  if (!fatal_once_fired) {
    fatal_once_fired = 1;
    snprintf(pot->error, sizeof pot->error, "fatal once");
    return -7;
  }
#endif
  if (!pot || !positions || !atomic_nrs || !direction || !hv_out || !box || n <= 0) {
    if (pot) {
      snprintf(pot->error, sizeof pot->error, "invalid HVP argument");
    }
    return RGPOT_ENGINE_ERR_INVALID;
  }
  db = (uintptr_t)direction;
  hb = (uintptr_t)hv_out;
  bytes = (uintptr_t)n * 3u * sizeof(double);
  if (db < hb + bytes && hb < db + bytes) {
    snprintf(pot->error, sizeof pot->error, "direction and hv_out overlap");
    return RGPOT_ENGINE_ERR_INVALID;
  }
  enter_call(pot);
  for (i = 0; i < n; i++) {
    if (atomic_nrs[i] == 0) {
      snprintf(pot->error, sizeof pot->error, "atom %ld has atomic number 0", i);
      leave_call(pot);
      return 1;
    }
    if (atomic_nrs[i] < 0) {
      snprintf(pot->error, sizeof pot->error, "atom %ld has a negative atomic number", i);
      leave_call(pot);
      return -1;
    }
  }
#ifdef REJECT_COINCIDENT
  {
    long j;
    for (i = 0; i < n; i++) {
      for (j = i + 1; j < n; j++) {
        double dx = positions[3 * i] - positions[3 * j];
        double dy = positions[3 * i + 1] - positions[3 * j + 1];
        double dz = positions[3 * i + 2] - positions[3 * j + 2];
        if (!(dx * dx + dy * dy + dz * dz > 0.0)) {
          snprintf(pot->error, sizeof pot->error, "coincident atoms");
          return RGPOT_ENGINE_ERR_EVAL;
        }
      }
    }
  }
#endif
  for (i = 0; i < 3 * n; i++) {
#ifdef HVP_NONFINITE
    hv_out[i] = i == 1 ? NAN : pot->k * direction[i];
#else
    hv_out[i] = pot->k * direction[i];
#endif
  }
  pot->error[0] = '\0';
  leave_call(pot);
  return 0;
#endif
}
#endif

#ifndef NO_HESSIAN
int rgpot_engine_hessian(RgpotEnginePot *pot, long n, const double *positions,
                         const int *atomic_nrs, double *hessian_out,
                         const double *box) {
  count_native_call();
  (void)positions;
  (void)box;
#if defined(NO_HVP) || defined(NO_HVP_CAP) || defined(HESSIAN_UNSUPPORTED)
  (void)n;
  (void)atomic_nrs;
  (void)hessian_out;
  if (pot) {
    snprintf(pot->error, sizeof pot->error,
             "harmonic engine built without analytic Hessian");
  }
  return RGPOT_ENGINE_ERR_UNSUPPORTED;
#else
  long i;
  long dim;
  if (!pot || !positions || !atomic_nrs || !hessian_out || !box || n <= 0) {
    if (pot) {
      snprintf(pot->error, sizeof pot->error, "invalid Hessian argument");
    }
    return RGPOT_ENGINE_ERR_INVALID;
  }
  enter_call(pot);
  dim = 3 * n;
  for (i = 0; i < dim * dim; i++) {
    hessian_out[i] = 0.0;
  }
  for (i = 0; i < dim; i++) {
    hessian_out[i * dim + i] = pot->k;
  }
  pot->error[0] = '\0';
  leave_call(pot);
  return 0;
#endif
}
#endif

#if !defined(NO_COMPONENTS) && (FIXTURE_ABI_MINOR >= 5)
long rgpot_engine_last_components(const RgpotEnginePot *pot,
                                  RgpotEngineComponent *out, double *forces,
                                  long capacity) {
  native_calls += 1;
  (void)forces;
  if (!pot) return -1;
  if (!out || capacity <= 0) return 1;
  memset(&out[0], 0, sizeof out[0]);
  out[0].weight = 1.0;
  out[0].energy = pot->last_energy;
  out[0].enabled = 1;
  out[0].evaluated = pot->evaluated;
  out[0].has_stress = 0;
  return 1;
}

size_t rgpot_engine_component_name(const RgpotEnginePot *pot, long index,
                                   char *buf, size_t len) {
  static const char name[] = "harmonic";
  size_t full = sizeof(name) - 1;
  native_calls += 1;
  if (!pot || index != 0) {
    if (buf && len > 0) buf[0] = '\0';
    return 0;
  }
  if (buf && len > 0) {
    snprintf(buf, len, "%s", name);
  }
  return full;
}
#endif

#ifdef FIXTURE_UNLOAD_POLICY
int rgpot_engine_unload_policy(void) { return FIXTURE_UNLOAD_POLICY; }
#endif

#if !defined(NO_JET) && (FIXTURE_ABI_MINOR >= 11)
/* The gradient of the harmonic energy is k x, so along x + t v its first
 * derivative is k v and every higher one is zero. The report counts one
 * product per coordinate and system and the larger of |k x| and |k v| over
 * all systems as the scale. Layout of jet_out: [order][system][3 * n]. */
int rgpot_engine_taylor_jet(RgpotEnginePot *pot, long systems, long n,
                            const double *positions, const int *atomic_nrs,
                            const double *directions, long order,
                            double *jet_out,
                            RgpotEngineDerivativeReport *report,
                            const double *boxes) {
  count_native_call();
  if (!pot || !positions || !atomic_nrs || !directions || !jet_out || !boxes ||
      systems <= 0 || n <= 0 || order < 1) {
    if (pot) {
      snprintf(pot->error, sizeof pot->error, "invalid jet argument");
    }
    return RGPOT_ENGINE_ERR_INVALID;
  }
#ifdef JET_UNSUPPORTED
  (void)report;
  snprintf(pot->error, sizeof pot->error, "harmonic engine built without a jet");
  return RGPOT_ENGINE_ERR_UNSUPPORTED;
#else
  long i;
  long k;
  long s;
  double scale = 0.0;
  if (order > FIXTURE_JET_ORDER) {
    snprintf(pot->error, sizeof pot->error, "jet order %ld is not served", order);
    return RGPOT_ENGINE_ERR_UNSUPPORTED;
  }
  enter_call(pot);
  for (i = 0; i < n; i++) {
    if (atomic_nrs[i] == 0) {
      snprintf(pot->error, sizeof pot->error, "atom %ld has atomic number 0", i);
      leave_call(pot);
      return 1;
    }
  }
  if (pot->callbacks.oracle != NULL) {
    int rc;
    double energy = 0.0;
    double *scratch = malloc((size_t)(3 * n) * sizeof(double));
    rc = pot->callbacks.oracle(pot->callbacks.ctx, n, positions, atomic_nrs,
                               scratch, &energy, boxes);
    last_oracle_rc = rc;
    free(scratch);
    if (rc != 0) {
#ifdef FATAL_AFTER_HOST_FAILURE
      snprintf(pot->error, sizeof pot->error, "host failure left the engine unusable");
      leave_call(pot);
      return -8;
#else
      leave_call(pot);
      return rc;
#endif
    }
  }
  for (k = 0; k < order; k++) {
    for (i = 0; i < systems * 3 * n; i++) {
      jet_out[k * systems * 3 * n + i] = k == 0 ? pot->k * directions[i] : 0.0;
    }
  }
  for (s = 0; s < systems; s++) {
    for (i = 0; i < 3 * n; i++) {
      double a = fabs(pot->k * positions[s * 3 * n + i]);
      double b = fabs(pot->k * directions[s * 3 * n + i]);
      if (a > scale) scale = a;
      if (b > scale) scale = b;
    }
  }
#ifdef JET_UNTRACKED
  if (report) {
    report->tracked = 0;
    report->operations = 0;
    report->scale = 0.0;
  }
#else
  if (report) {
    report->tracked = 1;
    report->operations = systems * 3 * n;
    report->scale = scale;
#ifdef JET_NEGATIVE_OPERATIONS
    report->operations = -1;
#endif
#ifdef JET_NONFINITE_SCALE
    report->scale = NAN;
#endif
#ifdef JET_NEGATIVE_SCALE
    report->scale = -1.0;
#endif
  }
#endif
#ifdef JET_NONFINITE_OUTPUT
  jet_out[order * systems * 3 * n - 1] = JET_NONFINITE_OUTPUT;
#endif
  pot->error[0] = '\0';
  leave_call(pot);
  return 0;
#endif
}
#endif
