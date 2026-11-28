/* MIT License
 * Copyright 2023--present rgpot developers
 */
#include "rgpot/engine_c_abi.h"
#include "rgpot/potlibc_engine.h"

#include <potlibc.h>

#include <pthread.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ANGSTROM_PER_BOHR 0.529177210903
#define EV_PER_HARTREE 27.211386245988
#define FORCE_TO_EV_PER_ANGSTROM (EV_PER_HARTREE / ANGSTROM_PER_BOHR)

/* The most atoms for which 3 * n_atoms * sizeof(double) still fits size_t. */
#define MAX_ATOMS (SIZE_MAX / (3 * sizeof(double)))

#ifndef RGPOT_ENGINE_PROJECT_VERSION
#define RGPOT_ENGINE_PROJECT_VERSION "unknown"
#endif

#ifdef RGPOT_POTLIBC_TEST_LOCK_FAIL
#include <errno.h>
static int potlibc_test_fail_lock = 0;
static int potlibc_test_live_call_locks = 0;
RGPOT_ENGINE_API void rgpot_potlibc_test_set_lock_fail(int fail) {
  potlibc_test_fail_lock = fail;
}
RGPOT_ENGINE_API int rgpot_potlibc_test_live_call_locks(void) {
  return __atomic_load_n(&potlibc_test_live_call_locks, __ATOMIC_SEQ_CST);
}
static int potlibc_mutex_lock(pthread_mutex_t *mutex) {
  if (potlibc_test_fail_lock) {
    return EINVAL;
  }
  return pthread_mutex_lock(mutex);
}
#else
#define potlibc_mutex_lock pthread_mutex_lock
#endif

static int init_call_mutex(pthread_mutex_t *mutex) {
  pthread_mutexattr_t attr;
  int rc;
  if (pthread_mutexattr_init(&attr) != 0) {
    return -1;
  }
  rc = pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_ERRORCHECK);
  if (rc == 0) {
    rc = pthread_mutex_init(mutex, &attr);
  }
  pthread_mutexattr_destroy(&attr);
#ifdef RGPOT_POTLIBC_TEST_LOCK_FAIL
  if (rc == 0) {
    __atomic_add_fetch(&potlibc_test_live_call_locks, 1, __ATOMIC_SEQ_CST);
  }
#endif
  return rc;
}

/* The mutex is released whether or not the handle lock could be taken, so a
 * failed acquisition in rgpot_engine_destroy cannot leak it. */
static void release_call_mutex(pthread_mutex_t *mutex) {
  pthread_mutex_destroy(mutex);
#ifdef RGPOT_POTLIBC_TEST_LOCK_FAIL
  __atomic_sub_fetch(&potlibc_test_live_call_locks, 1, __ATOMIC_SEQ_CST);
#endif
}

struct RgpotEnginePot {
  potlibc_handle *surface;
  potlibc_metadata metadata;
  pthread_mutex_t call_lock;
  pthread_mutex_t error_lock;
  char error[512];
};

static void copy_message(char *destination, size_t capacity,
                         const char *message) {
  if (destination == NULL || capacity == 0) {
    return;
  }
  snprintf(destination, capacity, "%s", message);
}

static size_t bounded_length(const char *text, size_t capacity) {
  size_t length;
  for (length = 0; length < capacity; ++length) {
    if (text[length] == '\0') {
      return length;
    }
  }
  return capacity;
}

static void set_error(RgpotEnginePot *pot, const char *format, ...) {
  va_list arguments;
  if (pthread_mutex_lock(&pot->error_lock) != 0) {
    return;
  }
  va_start(arguments, format);
  vsnprintf(pot->error, sizeof pot->error, format, arguments);
  va_end(arguments);
  pthread_mutex_unlock(&pot->error_lock);
}

static int fail_potlibc(RgpotEnginePot *pot, const char *operation,
                        potlibc_status status) {
  set_error(pot, "%s: %s", operation, potlibc_status_string(status));
  return RGPOT_ENGINE_ERR_EVAL;
}

static int trailing_bytes_zero(const unsigned char *data, size_t count) {
  size_t index;
  for (index = 0; index < count; ++index) {
    if (data[index] != 0) {
      return 0;
    }
  }
  return 1;
}

typedef struct {
  uint64_t atom_count;
  char surface_id[RGPOT_POTLIBC_SURFACE_ID_CAPACITY];
} PotlibcCreateRequest;

/* Version 1 is the 144-byte layout: magic, version, atom_count, surface_id.
 * Version 2 carries size at offset 4 and version at offset 8. A u32 of 1 at
 * offset 4 is version 1, never a version-2 size (those are at least
 * RGPOT_POTLIBC_CONFIG_MIN_SIZE). A 144-byte message whose word at offset 4
 * is below that minimum, and is not 1, has an unknown version and is refused
 * before offset 8 (the low word of atom_count in version 1, which may be 2)
 * is read as a version-2 version. A word at offset 4 that is at least the
 * version-2 size is a truncated version-2 or later message, so decoding
 * follows the version-2 path and reports that the buffer is shorter than
 * config.size. */
static int decode_potlibc_config(PotlibcCreateRequest *out, const void *config,
                                 size_t config_len, char *errbuf,
                                 size_t errlen) {
  const unsigned char *bytes;
  uint32_t magic;
  uint32_t word4;
  RgpotPotlibcConfigV2 message;

  if (config == NULL || config_len < sizeof(uint32_t) * 2) {
    copy_message(errbuf, errlen,
                 "potlibc engine requires RgpotPotlibcConfigV1");
    return -1;
  }
  bytes = config;
  memcpy(&magic, bytes, sizeof magic);
  memcpy(&word4, bytes + sizeof magic, sizeof word4);
  if (magic != RGPOT_POTLIBC_CONFIG_MAGIC) {
    copy_message(errbuf, errlen,
                 "potlibc engine config magic, version, or reserved field");
    return -1;
  }
  if (word4 == RGPOT_POTLIBC_CONFIG_VERSION_V1) {
    RgpotPotlibcConfigV1 legacy;
    if (config_len != sizeof legacy) {
      copy_message(errbuf, errlen,
                   "potlibc engine requires RgpotPotlibcConfigV1");
      return -1;
    }
    memcpy(&legacy, config, sizeof legacy);
    out->atom_count = legacy.atom_count;
    memcpy(out->surface_id, legacy.surface_id, sizeof out->surface_id);
    return 0;
  }
  if (config_len == sizeof(RgpotPotlibcConfigV1) &&
      word4 < RGPOT_POTLIBC_CONFIG_MIN_SIZE) {
    copy_message(errbuf, errlen, "potlibc config version is unknown");
    return -1;
  }
  if (config_len < offsetof(RgpotPotlibcConfigV2, version) + sizeof(uint32_t)) {
    copy_message(errbuf, errlen,
                 "potlibc engine requires RgpotPotlibcConfigV2");
    return -1;
  }
  memcpy(&message.version,
         bytes + offsetof(RgpotPotlibcConfigV2, version),
         sizeof message.version);
  if (message.version != RGPOT_POTLIBC_CONFIG_VERSION) {
    copy_message(errbuf, errlen, "potlibc config version is unknown");
    return -1;
  }
  memset(&message, 0, sizeof message);
  memcpy(&message, config, sizeof(uint32_t) * 3);
  if (message.size < RGPOT_POTLIBC_CONFIG_MIN_SIZE) {
    copy_message(errbuf, errlen,
                 "potlibc config size is smaller than RgpotPotlibcConfigV2");
    return -1;
  }
  if (config_len < message.size) {
    copy_message(errbuf, errlen,
                 "potlibc config buffer is shorter than config.size");
    return -1;
  }
  memcpy(&message, config, sizeof message);
  if (message.size > sizeof message &&
      !trailing_bytes_zero(bytes + sizeof message,
                           message.size - sizeof message)) {
    copy_message(errbuf, errlen, "potlibc config trailing bytes must be zero");
    return -1;
  }
  if (message.reserved != 0) {
    copy_message(errbuf, errlen,
                 "potlibc engine config magic, version, or reserved field");
    return -1;
  }
  out->atom_count = message.atom_count;
  memcpy(out->surface_id, message.surface_id, sizeof out->surface_id);
  return 0;
}

static int checked_values(RgpotEnginePot *pot, size_t systems, size_t dimension,
                          size_t *values) {
  if (dimension != 0 && systems > SIZE_MAX / dimension) {
    set_error(pot, "invalid argument (batch): too many systems");
    return RGPOT_ENGINE_ERR_INVALID;
  }
  *values = systems * dimension;
  return RGPOT_ENGINE_OK;
}

static int checked_alloc_bytes(RgpotEnginePot *pot, size_t count,
                               size_t *bytes) {
  if (count > SIZE_MAX / sizeof(double)) {
    set_error(pot, "invalid argument");
    return RGPOT_ENGINE_ERR_INVALID;
  }
  *bytes = count * sizeof(double);
  return RGPOT_ENGINE_OK;
}

static int check_input(RgpotEnginePot *pot, long n_atoms,
                       const int *atomic_numbers, size_t *dimension) {
  size_t atom;
  if (n_atoms < 1 || atomic_numbers == NULL) {
    set_error(pot, "invalid argument");
    return RGPOT_ENGINE_ERR_INVALID;
  }
  if ((size_t)n_atoms > MAX_ATOMS) {
    set_error(pot, "atom count %ld is too large for the coordinate buffers",
              n_atoms);
    return RGPOT_ENGINE_ERR_INVALID;
  }
  *dimension = 3 * (size_t)n_atoms;
  if ((size_t)n_atoms != pot->metadata.atom_count) {
    set_error(pot, "potlibc surface %s requires %zu atoms, got %ld",
              pot->metadata.surface_id, pot->metadata.atom_count, n_atoms);
    return RGPOT_ENGINE_ERR_INVALID;
  }
  if (pot->metadata.atomic_numbers != NULL) {
    for (atom = 0; atom < (size_t)n_atoms; ++atom) {
      const int expected = pot->metadata.atomic_numbers[atom];
      if (expected != 0 && atomic_numbers[atom] != expected) {
        set_error(pot,
                  "potlibc surface %s atom %zu requires Z=%d, got Z=%d",
                  pot->metadata.surface_id, atom, expected,
                  atomic_numbers[atom]);
        return RGPOT_ENGINE_ERR_INVALID;
      }
    }
  }
  return RGPOT_ENGINE_OK;
}

static int ranges_overlap(const void *first, size_t first_bytes,
                          const void *second, size_t second_bytes) {
  const uintptr_t first_begin = (uintptr_t)first;
  const uintptr_t second_begin = (uintptr_t)second;
  if (first_begin < second_begin) {
    return second_begin - first_begin < first_bytes;
  }
  return first_begin - second_begin < second_bytes;
}

static int lock_handle(RgpotEnginePot *pot) {
  if (pot == NULL) {
    return RGPOT_ENGINE_ERR_INVALID;
  }
  if (potlibc_mutex_lock(&pot->call_lock) != 0) {
    set_error(pot, "potlibc engine handle lock failed");
    return RGPOT_ENGINE_ERR_LOCK;
  }
  return RGPOT_ENGINE_OK;
}

int rgpot_engine_abi_version(void) { return RGPOT_ENGINE_ABI_VERSION; }

int rgpot_engine_available(void) { return 1; }

RgpotEngineAbiStamp rgpot_engine_abi_stamp(void) {
  const RgpotEngineAbiStamp stamp = {
      RGPOT_ENGINE_ABI_VERSION,
      RGPOT_ENGINE_ABI_MINOR,
      RGPOT_ENGINE_ABI_LAYOUT_REVISION,
  };
  return stamp;
}

int rgpot_engine_abi_compatible(const RgpotEngineAbiStamp *host_stamp) {
  return host_stamp != NULL &&
         host_stamp->abi_major == RGPOT_ENGINE_ABI_VERSION &&
         host_stamp->layout_revision == RGPOT_ENGINE_ABI_LAYOUT_REVISION &&
         RGPOT_ENGINE_ABI_MINOR <= host_stamp->abi_minor;
}

RgpotEnginePot *rgpot_engine_create(const void *config, size_t config_len,
                                    char *errbuf, size_t errlen) {
  PotlibcCreateRequest message;
  RgpotEnginePot *pot;
  potlibc_status status;

  if (decode_potlibc_config(&message, config, config_len, errbuf, errlen) !=
      0) {
    return NULL;
  }
  if (message.surface_id[0] == '\0' ||
      bounded_length(message.surface_id,
                     RGPOT_POTLIBC_SURFACE_ID_CAPACITY) ==
          RGPOT_POTLIBC_SURFACE_ID_CAPACITY) {
    copy_message(errbuf, errlen,
                 "potlibc engine config needs a NUL-terminated surface id");
    return NULL;
  }
  if (message.atom_count != (uint64_t)(size_t)message.atom_count ||
      (size_t)message.atom_count > MAX_ATOMS) {
    copy_message(errbuf, errlen,
                 "potlibc engine atom count is too large for the coordinate "
                 "buffers");
    return NULL;
  }

  pot = calloc(1, sizeof *pot);
  if (pot == NULL) {
    copy_message(errbuf, errlen, "potlibc engine allocation failed");
    return NULL;
  }
  if (init_call_mutex(&pot->call_lock) != 0) {
    copy_message(errbuf, errlen, "potlibc engine mutex initialization failed");
    free(pot);
    return NULL;
  }
  if (pthread_mutex_init(&pot->error_lock, NULL) != 0) {
    copy_message(errbuf, errlen,
                 "potlibc engine error mutex initialization failed");
    release_call_mutex(&pot->call_lock);
    free(pot);
    return NULL;
  }
  if (message.atom_count == 0) {
    status = potlibc_create(message.surface_id, &pot->surface);
  } else {
    status = potlibc_create_sized(message.surface_id,
                                  (size_t)message.atom_count, &pot->surface);
  }
  if (status != POTLIBC_SUCCESS) {
    snprintf(pot->error, sizeof pot->error, "potlibc create: %s",
             potlibc_status_string(status));
    copy_message(errbuf, errlen, pot->error);
    pthread_mutex_destroy(&pot->error_lock);
    release_call_mutex(&pot->call_lock);
    free(pot);
    return NULL;
  }
  status = potlibc_get_metadata(pot->surface, &pot->metadata);
  if (status != POTLIBC_SUCCESS) {
    snprintf(pot->error, sizeof pot->error, "potlibc metadata: %s",
             potlibc_status_string(status));
    copy_message(errbuf, errlen, pot->error);
    potlibc_destroy(pot->surface);
    pthread_mutex_destroy(&pot->error_lock);
    release_call_mutex(&pot->call_lock);
    free(pot);
    return NULL;
  }
  if (pot->metadata.surface_id == NULL ||
      pot->metadata.atomic_numbers == NULL ||
      pot->metadata.length_unit == NULL ||
      pot->metadata.energy_unit == NULL ||
      pot->metadata.force_unit == NULL) {
    copy_message(errbuf, errlen, "potlibc surface metadata is incomplete");
    potlibc_destroy(pot->surface);
    pthread_mutex_destroy(&pot->error_lock);
    release_call_mutex(&pot->call_lock);
    free(pot);
    return NULL;
  }
  if (strcmp(pot->metadata.length_unit, "bohr") != 0 ||
      strcmp(pot->metadata.energy_unit, "hartree") != 0 ||
      strcmp(pot->metadata.force_unit, "hartree/bohr") != 0) {
    copy_message(errbuf, errlen, "potlibc surface has unsupported units");
    potlibc_destroy(pot->surface);
    pthread_mutex_destroy(&pot->error_lock);
    release_call_mutex(&pot->call_lock);
    free(pot);
    return NULL;
  }
  if ((pot->metadata.capabilities & POTLIBC_CAP_FORCES) == 0) {
    copy_message(errbuf, errlen, "potlibc surface has no analytic forces");
    potlibc_destroy(pot->surface);
    pthread_mutex_destroy(&pot->error_lock);
    release_call_mutex(&pot->call_lock);
    free(pot);
    return NULL;
  }
  return pot;
}

void rgpot_engine_destroy(RgpotEnginePot *pot) {
  int locked;
  if (pot == NULL) {
    return;
  }
  /* engine_c_abi.h forbids calling destroy concurrently with another call on
   * the same handle. If call_lock cannot be acquired, still release the
   * surface and free the handle so cleanup after RGPOT_ENGINE_ERR_LOCK does
   * not leak. */
  locked = (potlibc_mutex_lock(&pot->call_lock) == 0);
  if (pot->surface != NULL) {
    potlibc_destroy(pot->surface);
    pot->surface = NULL;
  }
  if (locked) {
    pthread_mutex_unlock(&pot->call_lock);
  }
  release_call_mutex(&pot->call_lock);
  pthread_mutex_destroy(&pot->error_lock);
  free(pot);
}

int rgpot_engine_force(RgpotEnginePot *pot, long n_atoms,
                       const double *positions, const int *atomic_numbers,
                       double *forces, double *energy, double *variance,
                       const double *box,
                       rgpot_engine_coord_transform transform,
                       void *transform_user) {
  size_t component;
  size_t dimension = 0;
  double local_box[9];
  double *coordinates;
  double *surface_forces;
  double surface_energy;
  potlibc_status status;
  int code = lock_handle(pot);
  if (code != RGPOT_ENGINE_OK) {
    return code;
  }
  code = check_input(pot, n_atoms, atomic_numbers, &dimension);
  if (code != RGPOT_ENGINE_OK || positions == NULL || forces == NULL ||
      energy == NULL || box == NULL) {
    if (code == RGPOT_ENGINE_OK) {
      set_error(pot, "invalid argument");
      code = RGPOT_ENGINE_ERR_INVALID;
    }
    pthread_mutex_unlock(&pot->call_lock);
    return code;
  }
  coordinates = malloc(dimension * sizeof *coordinates);
  surface_forces = malloc(dimension * sizeof *surface_forces);
  if (coordinates == NULL || surface_forces == NULL) {
    free(coordinates);
    free(surface_forces);
    set_error(pot, "potlibc force allocation failed");
    pthread_mutex_unlock(&pot->call_lock);
    return RGPOT_ENGINE_ERR_EVAL;
  }
  memcpy(coordinates, positions, dimension * sizeof *coordinates);
  memcpy(local_box, box, sizeof local_box);
  if (transform != NULL) {
    code = transform(transform_user, n_atoms, coordinates, local_box);
    if (code != RGPOT_ENGINE_OK) {
      set_error(pot, "coordinate transform returned %d", code);
      free(coordinates);
      free(surface_forces);
      pthread_mutex_unlock(&pot->call_lock);
      return code;
    }
  }
  for (component = 0; component < dimension; ++component) {
    coordinates[component] /= ANGSTROM_PER_BOHR;
  }
  status = potlibc_evaluate(pot->surface, 1, coordinates, &surface_energy,
                            surface_forces);
  free(coordinates);
  if (status != POTLIBC_SUCCESS) {
    code = fail_potlibc(pot, "potlibc force", status);
    free(surface_forces);
    pthread_mutex_unlock(&pot->call_lock);
    return code;
  }
  for (component = 0; component < dimension; ++component) {
    forces[component] =
        surface_forces[component] * FORCE_TO_EV_PER_ANGSTROM;
  }
  free(surface_forces);
  *energy = surface_energy * EV_PER_HARTREE;
  if (variance != NULL) {
    *variance = 0.0;
  }
  set_error(pot, "%s", "");
  pthread_mutex_unlock(&pot->call_lock);
  return RGPOT_ENGINE_OK;
}

int rgpot_engine_force_batch(
    RgpotEnginePot *pot, long n_systems, long n_atoms,
    const double *positions, const int *atomic_numbers, double *forces,
    double *energies, double *variances, const double *boxes,
    rgpot_engine_coord_transform transform, void *transform_user) {
  size_t component;
  size_t system;
  size_t systems;
  size_t dimension = 0;
  size_t values;
  size_t values_bytes;
  size_t systems_bytes;
  double *coordinates;
  double *surface_forces;
  double *surface_energies;
  potlibc_status status;
  int code = lock_handle(pot);
  if (code != RGPOT_ENGINE_OK) {
    return code;
  }
  code = check_input(pot, n_atoms, atomic_numbers, &dimension);
  if (code != RGPOT_ENGINE_OK || n_systems < 1 || positions == NULL ||
      forces == NULL || energies == NULL || boxes == NULL) {
    if (code == RGPOT_ENGINE_OK) {
      set_error(pot, "invalid argument (batch)");
      code = RGPOT_ENGINE_ERR_INVALID;
    }
    pthread_mutex_unlock(&pot->call_lock);
    return code;
  }
  systems = (size_t)n_systems;
  code = checked_values(pot, systems, dimension, &values);
  if (code != RGPOT_ENGINE_OK ||
      checked_alloc_bytes(pot, values, &values_bytes) != RGPOT_ENGINE_OK ||
      checked_alloc_bytes(pot, systems, &systems_bytes) != RGPOT_ENGINE_OK) {
    pthread_mutex_unlock(&pot->call_lock);
    return RGPOT_ENGINE_ERR_INVALID;
  }
  coordinates = malloc(values_bytes);
  surface_forces = malloc(values_bytes);
  surface_energies = malloc(systems_bytes);
  if (coordinates == NULL || surface_forces == NULL ||
      surface_energies == NULL) {
    free(coordinates);
    free(surface_forces);
    free(surface_energies);
    set_error(pot, "potlibc batch allocation failed");
    pthread_mutex_unlock(&pot->call_lock);
    return RGPOT_ENGINE_ERR_EVAL;
  }
  memcpy(coordinates, positions, values_bytes);
  for (system = 0; system < systems; ++system) {
    double local_box[9];
    memcpy(local_box, boxes + 9 * system, sizeof local_box);
    if (transform != NULL) {
      code = transform(transform_user, n_atoms,
                       coordinates + system * dimension, local_box);
      if (code != RGPOT_ENGINE_OK) {
        set_error(pot, "coordinate transform returned %d", code);
        free(coordinates);
        free(surface_forces);
        free(surface_energies);
        pthread_mutex_unlock(&pot->call_lock);
        return code;
      }
    }
  }
  for (component = 0; component < values; ++component) {
    coordinates[component] /= ANGSTROM_PER_BOHR;
  }
  status = potlibc_evaluate(pot->surface, systems, coordinates,
                            surface_energies, surface_forces);
  free(coordinates);
  if (status != POTLIBC_SUCCESS) {
    code = fail_potlibc(pot, "potlibc batch", status);
    free(surface_forces);
    free(surface_energies);
    pthread_mutex_unlock(&pot->call_lock);
    return code;
  }
  for (system = 0; system < systems; ++system) {
    energies[system] = surface_energies[system] * EV_PER_HARTREE;
    if (variances != NULL) {
      variances[system] = 0.0;
    }
  }
  for (component = 0; component < values; ++component) {
    forces[component] =
        surface_forces[component] * FORCE_TO_EV_PER_ANGSTROM;
  }
  free(surface_forces);
  free(surface_energies);
  set_error(pot, "%s", "");
  pthread_mutex_unlock(&pot->call_lock);
  return RGPOT_ENGINE_OK;
}

int rgpot_engine_hvp(RgpotEnginePot *pot, long n_atoms,
                     const double *positions, const int *atomic_numbers,
                     const double *direction, double *hv_out,
                     const double *box) {
  size_t component;
  size_t dimension = 0;
  double *coordinates;
  double *surface_direction;
  double *surface_product;
  potlibc_status status;
  int code = lock_handle(pot);
  if (code != RGPOT_ENGINE_OK) {
    return code;
  }
  code = check_input(pot, n_atoms, atomic_numbers, &dimension);
  if (code != RGPOT_ENGINE_OK || positions == NULL || direction == NULL ||
      hv_out == NULL || box == NULL) {
    if (code == RGPOT_ENGINE_OK) {
      set_error(pot, "invalid argument");
      code = RGPOT_ENGINE_ERR_INVALID;
    }
    pthread_mutex_unlock(&pot->call_lock);
    return code;
  }
  if (ranges_overlap(hv_out, dimension * sizeof *hv_out, direction,
                     dimension * sizeof *direction) ||
      ranges_overlap(hv_out, dimension * sizeof *hv_out, positions,
                     dimension * sizeof *positions) ||
      ranges_overlap(hv_out, dimension * sizeof *hv_out, atomic_numbers,
                     (size_t)n_atoms * sizeof *atomic_numbers) ||
      ranges_overlap(hv_out, dimension * sizeof *hv_out, box,
                     9 * sizeof *box)) {
    set_error(pot, "HVP output overlaps an input buffer");
    pthread_mutex_unlock(&pot->call_lock);
    return RGPOT_ENGINE_ERR_INVALID;
  }
  if ((pot->metadata.capabilities & POTLIBC_CAP_HVP) == 0) {
    set_error(pot, "potlibc surface %s has no analytic HVP",
              pot->metadata.surface_id);
    pthread_mutex_unlock(&pot->call_lock);
    return RGPOT_ENGINE_ERR_UNSUPPORTED;
  }
  coordinates = malloc(dimension * sizeof *coordinates);
  surface_direction = malloc(dimension * sizeof *surface_direction);
  surface_product = malloc(dimension * sizeof *surface_product);
  if (coordinates == NULL || surface_direction == NULL ||
      surface_product == NULL) {
    free(coordinates);
    free(surface_direction);
    free(surface_product);
    set_error(pot, "potlibc HVP allocation failed");
    pthread_mutex_unlock(&pot->call_lock);
    return RGPOT_ENGINE_ERR_EVAL;
  }
  memcpy(coordinates, positions, dimension * sizeof *coordinates);
  for (component = 0; component < dimension; ++component) {
    coordinates[component] /= ANGSTROM_PER_BOHR;
    surface_direction[component] =
        direction[component] / ANGSTROM_PER_BOHR;
  }
  status = potlibc_hessian_vector_product(
      pot->surface, 1, coordinates, surface_direction, surface_product);
  free(coordinates);
  free(surface_direction);
  if (status != POTLIBC_SUCCESS) {
    code = fail_potlibc(pot, "potlibc HVP", status);
    free(surface_product);
    pthread_mutex_unlock(&pot->call_lock);
    return code;
  }
  for (component = 0; component < dimension; ++component) {
    hv_out[component] =
        surface_product[component] * FORCE_TO_EV_PER_ANGSTROM;
  }
  free(surface_product);
  set_error(pot, "%s", "");
  pthread_mutex_unlock(&pot->call_lock);
  return RGPOT_ENGINE_OK;
}

int rgpot_engine_caps(const RgpotEnginePot *pot, RgpotEngineCaps *out) {
  RgpotEngineCaps result;
  RgpotEnginePot *mutable_pot = (RgpotEnginePot *)pot;
  size_t room;
  int code;
  if (out == NULL || out->size < sizeof(unsigned)) {
    return RGPOT_ENGINE_ERR_INVALID;
  }
  code = lock_handle(mutable_pot);
  if (code != RGPOT_ENGINE_OK) {
    return code;
  }
  memset(&result, 0, sizeof result);
  result.size = sizeof result;
  result.reentrancy = 1;
  result.per_image_instances = 1;
  result.batched = 1;
  result.analytic_hvp =
      (pot->metadata.capabilities & POTLIBC_CAP_HVP) != 0;
  room = out->size < sizeof result ? out->size : sizeof result;
  memcpy(out, &result, room);
  out->size = (unsigned)room;
  pthread_mutex_unlock(&mutable_pot->call_lock);
  return RGPOT_ENGINE_OK;
}

size_t rgpot_engine_last_error(const RgpotEnginePot *pot, char *buf,
                               size_t len) {
  RgpotEnginePot *mutable_pot = (RgpotEnginePot *)pot;
  size_t full;
  size_t copied;
  if (pot == NULL || pthread_mutex_lock(&mutable_pot->error_lock) != 0) {
    return 0;
  }
  full = strlen(pot->error);
  if (buf != NULL && len > 0) {
    copied = full < len - 1 ? full : len - 1;
    memcpy(buf, pot->error, copied);
    buf[copied] = '\0';
  }
  pthread_mutex_unlock(&mutable_pot->error_lock);
  return full;
}

const char *rgpot_engine_name(void) { return "potlibc"; }

const char *rgpot_engine_version(void) {
  return "potlibc/" RGPOT_ENGINE_PROJECT_VERSION;
}
