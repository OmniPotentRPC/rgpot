/* MIT License
 * Copyright 2023--present rgpot developers
 */
#include "rgpot/engine_c_abi.h"
#include "rgpot/potlibc_engine.h"

#include <dlfcn.h>
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define BOHR_TO_ANGSTROM 0.529177210903

typedef RgpotEnginePot *(*create_fn)(const void *, size_t, char *, size_t);
typedef void (*destroy_fn)(RgpotEnginePot *);
typedef int (*force_fn)(RgpotEnginePot *, long, const double *, const int *,
                        double *, double *, double *, const double *,
                        rgpot_engine_coord_transform, void *);
typedef int (*hvp_fn)(RgpotEnginePot *, long, const double *, const int *,
                      const double *, double *, const double *,
                      rgpot_engine_coord_transform, void *);
typedef int (*batch_fn)(RgpotEnginePot *, long, long, const double *,
                        const int *, double *, double *, double *,
                        const double *, rgpot_engine_coord_transform, void *);
typedef int (*abi_version_fn)(void);
typedef int (*available_fn)(void);
typedef RgpotEngineAbiStamp (*abi_stamp_fn)(void);
typedef int (*abi_compatible_fn)(const RgpotEngineAbiStamp *);
typedef const char *(*text_fn)(void);
typedef int (*caps_fn)(const RgpotEnginePot *, RgpotEngineCaps *);
typedef size_t (*last_error_fn)(const RgpotEnginePot *, char *, size_t);

typedef struct {
  abi_version_fn abi_version;
  available_fn available;
  abi_stamp_fn abi_stamp;
  abi_compatible_fn abi_compatible;
  text_fn name;
  text_fn version;
  create_fn create;
  destroy_fn destroy;
  force_fn force;
  hvp_fn hvp;
  batch_fn batch;
  caps_fn caps;
  last_error_fn last_error;
} Engine;

static uint64_t bits(double value) {
  uint64_t result;
  memcpy(&result, &value, sizeof result);
  return result;
}

static RgpotPotlibcConfigV2 config(const char *surface_id,
                                   uint64_t atom_count) {
  RgpotPotlibcConfigV2 result;
  memset(&result, 0, sizeof result);
  result.magic = RGPOT_POTLIBC_CONFIG_MAGIC;
  result.size = sizeof result;
  result.version = RGPOT_POTLIBC_CONFIG_VERSION;
  result.reserved = 0;
  result.atom_count = atom_count;
  snprintf(result.surface_id, sizeof result.surface_id, "%s", surface_id);
  return result;
}

static RgpotEnginePot *create(Engine *engine, const char *surface_id,
                              uint64_t atom_count) {
  RgpotPotlibcConfigV2 message = config(surface_id, atom_count);
  char error[512] = {0};
  RgpotEnginePot *pot =
      engine->create(&message, sizeof message, error, sizeof error);
  if (pot == NULL) {
    fprintf(stderr, "%s create failed: %s\n", surface_id, error);
  }
  return pot;
}

static int reference_energy(Engine *engine, const char *surface_id,
                            uint64_t atom_count, long n_atoms,
                            const double *positions_bohr,
                            const int *atomic_numbers, uint64_t expected_bits) {
  static const double box[9] = {
      20.0, 0.0, 0.0, 0.0, 20.0, 0.0, 0.0, 20.0,
  };
  RgpotEnginePot *pot = create(engine, surface_id, atom_count);
  double positions[24];
  double forces[24];
  double energy;
  long component;
  int code;
  if (pot == NULL) {
    return 1;
  }
  for (component = 0; component < 3 * n_atoms; ++component) {
    positions[component] = positions_bohr[component] * BOHR_TO_ANGSTROM;
  }
  code = engine->force(pot, n_atoms, positions, atomic_numbers, forces,
                       &energy, NULL, box, NULL, NULL);
  engine->destroy(pot);
  if (code != RGPOT_ENGINE_OK) {
    fprintf(stderr, "%s force returned %d\n", surface_id, code);
    return 1;
  }
  if (bits(energy) != expected_bits) {
    fprintf(stderr, "%s energy bits 0x%016llx, expected 0x%016llx\n",
            surface_id, (unsigned long long)bits(energy),
            (unsigned long long)expected_bits);
    return 1;
  }
  return 0;
}

typedef struct {
  double amount;
  last_error_fn last_error;
  RgpotEnginePot *pot;
} ShiftContext;

static int shift(void *user, long n_atoms, double *positions, double *box) {
  ShiftContext *context = (ShiftContext *)user;
  char error[512];
  (void)n_atoms;
  (void)box;
  if (context->last_error(context->pot, error, sizeof error) == 0) {
    return RGPOT_ENGINE_ERR_EVAL;
  }
  positions[0] += context->amount;
  return 0;
}

static int hvp_transform_is_applied(Engine *engine) {
  static const double box[9] = {
      20.0, 0.0, 0.0, 0.0, 20.0, 0.0, 0.0, 20.0,
  };
  static const int atomic_numbers[3] = {35, 1, 1};
  static const double positions[9] = {
      0.0, 0.0, 0.0,
      3.0802 * BOHR_TO_ANGSTROM, 0.0, 0.0,
      4.8202 * BOHR_TO_ANGSTROM, 0.0, 0.0,
  };
  static const double direction[9] = {
      0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.9,
  };
  RgpotEnginePot *pot = create(engine, "brh2leps", 0);
  double shifted_positions[9];
  double transformed[9];
  double direct[9];
  double forces[9];
  double energy;
  ShiftContext context;
  int code;
  if (pot == NULL) {
    return 1;
  }
  code = engine->force(pot, 2, positions, atomic_numbers, forces, &energy, NULL,
                       box, NULL, NULL);
  if (code == RGPOT_ENGINE_OK) {
    fprintf(stderr, "wrong atom count unexpectedly accepted\n");
    engine->destroy(pot);
    return 1;
  }
  context.amount = 0.125;
  context.last_error = engine->last_error;
  context.pot = pot;
  memcpy(shifted_positions, positions, sizeof shifted_positions);
  shifted_positions[0] += context.amount;
  code = engine->hvp(pot, 3, positions, atomic_numbers, direction, transformed,
                     box, shift, &context);
  if (code == RGPOT_ENGINE_OK) {
    code = engine->hvp(pot, 3, shifted_positions, atomic_numbers, direction,
                       direct, box, NULL, NULL);
  }
  engine->destroy(pot);
  if (code != RGPOT_ENGINE_OK ||
      memcmp(transformed, direct, sizeof transformed) != 0) {
    fprintf(stderr, "HVP coordinate transform was not applied identically\n");
    return 1;
  }
  return 0;
}

static int evaluate_message(Engine *engine, const void *message, size_t length,
                            double *energy, double *forces, double *product) {
  static const double box[9] = {
      20.0, 0.0, 0.0, 0.0, 20.0, 0.0, 0.0, 0.0, 20.0,
  };
  static const double positions_bohr[9] = {
      0.0, 0.0, 0.0, 3.0802, 0.0, 0.0, 4.8202, 0.0, 0.0,
  };
  static const int species[3] = {35, 1, 1};
  static const double direction[9] = {
      0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.9,
  };
  double positions[9];
  char error[512] = {0};
  long component;
  int code;
  RgpotEnginePot *pot =
      engine->create(message, length, error, sizeof error);
  if (pot == NULL) {
    fprintf(stderr, "create failed: %s\n", error);
    return 1;
  }
  for (component = 0; component < 9; ++component) {
    positions[component] = positions_bohr[component] * BOHR_TO_ANGSTROM;
  }
  code = engine->force(pot, 3, positions, species, forces, energy, NULL, box,
                       NULL, NULL);
  if (code == RGPOT_ENGINE_OK) {
    code = engine->hvp(pot, 3, positions, species, direction, product, box,
                       NULL, NULL);
  }
  engine->destroy(pot);
  if (code != RGPOT_ENGINE_OK) {
    fprintf(stderr, "evaluation returned %d\n", code);
    return 1;
  }
  return 0;
}

/* A client built against 41b87aa5 writes 144 bytes: magic, version 1,
 * atom_count, then the surface id. The tip must accept that message and
 * return the same energy, forces and Hessian-vector product as version 2. */
static int v1_client_matches_version_two(Engine *engine) {
  unsigned char legacy_bytes[144];
  RgpotPotlibcConfigV2 current;
  double legacy_energy = 0.0;
  double current_energy = 0.0;
  double legacy_forces[9];
  double current_forces[9];
  double legacy_product[9];
  double current_product[9];
  uint32_t magic = RGPOT_POTLIBC_CONFIG_MAGIC;
  uint32_t version = 1u;
  uint64_t atoms = 0;
  char error[512] = {0};
  memset(legacy_bytes, 0, sizeof legacy_bytes);
  memcpy(legacy_bytes + 0, &magic, sizeof magic);
  memcpy(legacy_bytes + 4, &version, sizeof version);
  memcpy(legacy_bytes + 8, &atoms, sizeof atoms);
  memcpy(legacy_bytes + 16, "brh2leps", 8);
  if (evaluate_message(engine, legacy_bytes, sizeof legacy_bytes,
                       &legacy_energy, legacy_forces, legacy_product) != 0) {
    fprintf(stderr, "version-1 potlibc message was refused\n");
    return 1;
  }
  current = config("brh2leps", 0);
  if (evaluate_message(engine, &current, sizeof current, &current_energy,
                       current_forces, current_product) != 0) {
    return 1;
  }
  if (bits(legacy_energy) != bits(current_energy) ||
      memcmp(legacy_forces, current_forces, sizeof legacy_forces) != 0 ||
      memcmp(legacy_product, current_product, sizeof legacy_product) != 0) {
    fprintf(stderr, "version-1 potlibc results differ from version 2\n");
    return 1;
  }

  version = 99u;
  memcpy(legacy_bytes + 4, &version, sizeof version);
  memset(error, 0, sizeof error);
  if (engine->create(legacy_bytes, sizeof legacy_bytes, error, sizeof error) !=
          NULL ||
      strstr(error, "version is unknown") == NULL) {
    fprintf(stderr, "version-1 unknown version: %s\n", error);
    return 1;
  }
  current = config("brh2leps", 0);
  current.version = 3u;
  memset(error, 0, sizeof error);
  if (engine->create(&current, sizeof current, error, sizeof error) != NULL ||
      strstr(error, "version is unknown") == NULL) {
    fprintf(stderr, "version-2 unknown version: %s\n", error);
    return 1;
  }
  return 0;
}

static int variable_size_requires_count(Engine *engine) {
  RgpotPotlibcConfigV2 message = config("vbo-beta1", 0);
  char error[512] = {0};
  RgpotEnginePot *pot =
      engine->create(&message, sizeof message, error, sizeof error);
  if (pot != NULL) {
    fprintf(stderr, "variable-size surface accepted a zero atom count\n");
    engine->destroy(pot);
    return 1;
  }
  if (error[0] == '\0') {
    fprintf(stderr, "variable-size surface failure had no diagnostic\n");
    return 1;
  }
  return 0;
}

static int config_size_is_extensible(Engine *engine) {
  RgpotPotlibcConfigV2 message = config("brh2leps", 0);
  unsigned char future[sizeof(RgpotPotlibcConfigV2) + 16];
  char error[512] = {0};
  RgpotEnginePot *pot;
  memset(future, 0, sizeof future);
  memcpy(future, &message, sizeof message);
  ((RgpotPotlibcConfigV2 *)future)->size = sizeof future;
  pot = engine->create(future, sizeof future, error, sizeof error);
  if (pot == NULL) {
    fprintf(stderr, "future-size config was refused: %s\n", error);
    return 1;
  }
  engine->destroy(pot);

  future[sizeof(message)] = 1;
  memset(error, 0, sizeof error);
  if (engine->create(future, sizeof future, error, sizeof error) != NULL ||
      error[0] == '\0') {
    fprintf(stderr, "nonzero trailing bytes were unexpectedly accepted\n");
    return 1;
  }

  message = config("brh2leps", 0);
  message.size = sizeof(message) - 8;
  memset(error, 0, sizeof error);
  if (engine->create(&message, sizeof message, error, sizeof error) != NULL ||
      error[0] == '\0') {
    fprintf(stderr, "too-small config size was unexpectedly accepted\n");
    return 1;
  }
  return 0;
}

typedef struct {
  Engine *engine;
  RgpotEnginePot *pot;
  const int *species;
  const double *box;
} ReenterContext;

static int reenter_on_locked_handle(void *user, long n_atoms, double *positions,
                                    double *box) {
  ReenterContext *ctx = (ReenterContext *)user;
  RgpotEngineCaps caps;
  double forces[9];
  double hv[9];
  double energy = 0.0;
  char error[512] = {0};
  int force_code;
  int hvp_code;
  int caps_code;
  (void)box;

  force_code = ctx->engine->force(ctx->pot, n_atoms, positions, ctx->species,
                                  forces, &energy, NULL, ctx->box, NULL, NULL);
  if (force_code != RGPOT_ENGINE_ERR_LOCK ||
      ctx->engine->last_error(ctx->pot, error, sizeof error) == 0) {
    fprintf(stderr, "reentrant force returned %d (error=%s)\n", force_code,
            error);
    return RGPOT_ENGINE_ERR_EVAL;
  }
  memset(error, 0, sizeof error);
  hvp_code = ctx->engine->hvp(ctx->pot, n_atoms, positions, ctx->species,
                              positions, hv, ctx->box, NULL, NULL);
  if (hvp_code != RGPOT_ENGINE_ERR_LOCK ||
      ctx->engine->last_error(ctx->pot, error, sizeof error) == 0) {
    fprintf(stderr, "reentrant hvp returned %d (error=%s)\n", hvp_code, error);
    return RGPOT_ENGINE_ERR_EVAL;
  }
  memset(&caps, 0, sizeof caps);
  caps.size = sizeof caps;
  caps_code = ctx->engine->caps(ctx->pot, &caps);
  if (caps_code != RGPOT_ENGINE_ERR_LOCK) {
    fprintf(stderr, "reentrant caps returned %d\n", caps_code);
    return RGPOT_ENGINE_ERR_EVAL;
  }
  return 0;
}

static int reentrant_lock_returns_named_code(Engine *engine) {
  static const double box[9] = {
      20.0, 0.0, 0.0, 0.0, 20.0, 0.0, 0.0, 20.0,
  };
  static const int species[3] = {35, 1, 1};
  static const double positions[9] = {
      0.0, 0.0, 0.0,
      3.0802 * BOHR_TO_ANGSTROM, 0.0, 0.0,
      4.8202 * BOHR_TO_ANGSTROM, 0.0, 0.0,
  };
  RgpotEnginePot *pot = create(engine, "brh2leps", 0);
  double forces[9];
  double energy = 0.0;
  ReenterContext ctx;
  int code;
  if (pot == NULL) {
    return 1;
  }
  ctx.engine = engine;
  ctx.pot = pot;
  ctx.species = species;
  ctx.box = box;
  code = engine->force(pot, 3, positions, species, forces, &energy, NULL, box,
                       reenter_on_locked_handle, &ctx);
  engine->destroy(pot);
  if (code != RGPOT_ENGINE_OK) {
    fprintf(stderr, "outer force around reentrant lock probe returned %d\n",
            code);
    return 1;
  }
  return 0;
}

static int destroy_frees_handle_when_lock_fails(const char *path) {
  static const double box[9] = {
      20.0, 0.0, 0.0, 0.0, 20.0, 0.0, 0.0, 20.0,
  };
  static const int species[3] = {35, 1, 1};
  static const double positions[9] = {
      0.0, 0.0, 0.0,
      3.0802 * BOHR_TO_ANGSTROM, 0.0, 0.0,
      4.8202 * BOHR_TO_ANGSTROM, 0.0, 0.0,
  };
  void *library = dlopen(path, RTLD_NOW | RTLD_LOCAL);
  create_fn create_sym;
  destroy_fn destroy_sym;
  force_fn force_sym;
  caps_fn caps_sym;
  last_error_fn last_error_sym;
  RgpotPotlibcConfigV2 message = config("brh2leps", 0);
  RgpotEngineCaps caps;
  RgpotEnginePot *pot;
  double forces[9];
  double energy = 0.0;
  char error[512] = {0};
  int force_code;
  int caps_code;

  if (library == NULL) {
    fprintf(stderr, "%s\n", dlerror());
    return 1;
  }
  if (dlsym(library, "potlibc_mutex_lock") != NULL ||
      dlsym(library, "potlibc_engine_fail_lock") != NULL ||
      dlsym(library, "rgpot_potlibc_test_set_lock_fail") != NULL ||
      dlsym(library, "lock_handle") != NULL) {
    fprintf(stderr, "lock-failure test library exported internal symbols\n");
    dlclose(library);
    return 1;
  }
  create_sym = (create_fn)dlsym(library, "rgpot_engine_create");
  destroy_sym = (destroy_fn)dlsym(library, "rgpot_engine_destroy");
  force_sym = (force_fn)dlsym(library, "rgpot_engine_force");
  caps_sym = (caps_fn)dlsym(library, "rgpot_engine_caps");
  last_error_sym = (last_error_fn)dlsym(library, "rgpot_engine_last_error");
  if (create_sym == NULL || destroy_sym == NULL || force_sym == NULL ||
      caps_sym == NULL || last_error_sym == NULL) {
    fprintf(stderr, "lock-failure test library is missing symbols\n");
    dlclose(library);
    return 1;
  }
  pot = create_sym(&message, sizeof message, error, sizeof error);
  if (pot == NULL) {
    fprintf(stderr, "lock-failure create failed: %s\n", error);
    dlclose(library);
    return 1;
  }
  force_code = force_sym(pot, 3, positions, species, forces, &energy, NULL, box,
                         NULL, NULL);
  memset(&caps, 0, sizeof caps);
  caps.size = sizeof caps;
  caps_code = caps_sym(pot, &caps);
  memset(error, 0, sizeof error);
  last_error_sym(pot, error, sizeof error);
  /* Destroy while potlibc_mutex_lock fails: must still release surface and pot. */
  destroy_sym(pot);
  dlclose(library);
  if (force_code != RGPOT_ENGINE_ERR_LOCK ||
      caps_code != RGPOT_ENGINE_ERR_LOCK || error[0] == '\0') {
    fprintf(stderr,
            "lock failure returned force=%d caps=%d error=%s\n",
            force_code, caps_code, error);
    return 1;
  }
  return 0;
}

#define BATCH_SYSTEMS 3
#define BATCH_ATOMS 3

static int refuse_second(void *user, long n_atoms, double *positions,
                         double *box) {
  int *calls = (int *)user;
  (void)n_atoms;
  (void)positions;
  (void)box;
  *calls += 1;
  return *calls == 2 ? 7 : 0;
}

static int batch_is_one_call_per_system(Engine *engine) {
  static const double box[9] = {
      20.0, 0.0, 0.0, 0.0, 20.0, 0.0, 0.0, 0.0, 20.0,
  };
  static const int atomic_numbers[BATCH_ATOMS] = {35, 1, 1};
  static const double stretch[BATCH_SYSTEMS] = {3.0802, 3.3, 2.9};
  RgpotEnginePot *pot = create(engine, "brh2leps", 0);
  double positions[BATCH_SYSTEMS * 9];
  double boxes[BATCH_SYSTEMS * 9];
  double forces[BATCH_SYSTEMS * 9];
  double energies[BATCH_SYSTEMS];
  double variances[BATCH_SYSTEMS];
  double sentinel_forces[BATCH_SYSTEMS * 9];
  double sentinel_energies[BATCH_SYSTEMS];
  double sentinel_variances[BATCH_SYSTEMS];
  RgpotEngineCaps caps;
  long system;
  long index;
  int calls = 0;
  int code;
  if (pot == NULL) {
    return 1;
  }
  memset(&caps, 0, sizeof caps);
  caps.size = sizeof caps;
  if (engine->caps(pot, &caps) != RGPOT_ENGINE_OK || caps.batched != 1 ||
      caps.analytic_hvp != 1) {
    fprintf(stderr, "caps do not report native batching and analytic HVP\n");
    engine->destroy(pot);
    return 1;
  }
  memset(positions, 0, sizeof positions);
  for (system = 0; system < BATCH_SYSTEMS; ++system) {
    positions[9 * system + 3] = stretch[system] * BOHR_TO_ANGSTROM;
    positions[9 * system + 6] = (stretch[system] + 1.74) * BOHR_TO_ANGSTROM;
    memcpy(boxes + 9 * system, box, sizeof box);
  }
  for (index = 0; index < BATCH_SYSTEMS * 9; ++index) {
    sentinel_forces[index] = -1.0;
  }
  for (index = 0; index < BATCH_SYSTEMS; ++index) {
    sentinel_energies[index] = -1.0;
    sentinel_variances[index] = -1.0;
  }
  memcpy(forces, sentinel_forces, sizeof forces);
  memcpy(energies, sentinel_energies, sizeof energies);
  memcpy(variances, sentinel_variances, sizeof variances);

  code = engine->batch(pot, BATCH_SYSTEMS, BATCH_ATOMS, positions,
                       atomic_numbers, forces, energies, variances, boxes,
                       NULL, NULL);
  if (code != RGPOT_ENGINE_OK) {
    fprintf(stderr, "batch returned %d\n", code);
    engine->destroy(pot);
    return 1;
  }
  for (system = 0; system < BATCH_SYSTEMS; ++system) {
    double single_forces[9];
    double single_energy;
    code = engine->force(pot, BATCH_ATOMS, positions + 9 * system,
                         atomic_numbers, single_forces, &single_energy, NULL,
                         box, NULL, NULL);
    if (code != RGPOT_ENGINE_OK ||
        bits(single_energy) != bits(energies[system]) ||
        memcmp(single_forces, forces + 9 * system, sizeof single_forces) != 0 ||
        variances[system] != 0.0) {
      fprintf(stderr, "batch system %ld differs from a single call\n", system);
      engine->destroy(pot);
      return 1;
    }
  }

  /* A failed batch writes nothing: the second transform refuses. */
  memcpy(forces, sentinel_forces, sizeof forces);
  memcpy(energies, sentinel_energies, sizeof energies);
  memcpy(variances, sentinel_variances, sizeof variances);
  code = engine->batch(pot, BATCH_SYSTEMS, BATCH_ATOMS, positions,
                       atomic_numbers, forces, energies, variances, boxes,
                       refuse_second, &calls);
  if (code != 7 || calls != 2 ||
      memcmp(forces, sentinel_forces, sizeof forces) != 0 ||
      memcmp(energies, sentinel_energies, sizeof energies) != 0 ||
      memcmp(variances, sentinel_variances, sizeof variances) != 0) {
    fprintf(stderr, "a refused batch transform wrote outputs (code %d)\n",
            code);
    engine->destroy(pot);
    return 1;
  }
  if (engine->batch(pot, 0, BATCH_ATOMS, positions, atomic_numbers, forces,
                    energies, variances, boxes, NULL,
                    NULL) != RGPOT_ENGINE_ERR_INVALID ||
      engine->batch(pot, BATCH_SYSTEMS, 2, positions, atomic_numbers, forces,
                    energies, variances, boxes, NULL,
                    NULL) != RGPOT_ENGINE_ERR_INVALID ||
      engine->batch(pot, BATCH_SYSTEMS, BATCH_ATOMS, positions, atomic_numbers,
                    forces, NULL, variances, boxes, NULL,
                    NULL) != RGPOT_ENGINE_ERR_INVALID ||
      engine->batch(pot, 0x7fffffffffffffffL, BATCH_ATOMS, positions,
                    atomic_numbers, forces, energies, variances, boxes, NULL,
                    NULL) != RGPOT_ENGINE_ERR_INVALID ||
      memcmp(energies, sentinel_energies, sizeof energies) != 0) {
    fprintf(stderr, "invalid batch arguments were not refused cleanly\n");
    engine->destroy(pot);
    return 1;
  }
  engine->destroy(pot);
  return 0;
}

static int caps_refuses_null_handle(Engine *engine) {
  RgpotEngineCaps caps;
  memset(&caps, 0, sizeof caps);
  caps.size = sizeof caps;
  if (engine->caps(NULL, &caps) != RGPOT_ENGINE_ERR_INVALID) {
    fprintf(stderr, "caps(NULL, out) was not rejected\n");
    return 1;
  }
  return 0;
}

static int evaluate_failure_preserves_outputs(Engine *engine) {
  static const double box[9] = {
      20.0, 0.0, 0.0, 0.0, 20.0, 0.0, 0.0, 0.0, 20.0,
  };
  static const int species[3] = {35, 1, 1};
  double positions[9] = {
      0.0, 0.0, 0.0,
      3.0802 * BOHR_TO_ANGSTROM, 0.0, 0.0,
      4.8202 * BOHR_TO_ANGSTROM, 0.0, 0.0,
  };
  double forces[9];
  double energy = -3.0;
  int index;
  RgpotEnginePot *pot = create(engine, "brh2leps", 0);
  int code;
  if (pot == NULL) {
    return 1;
  }
  for (index = 0; index < 9; ++index) {
    forces[index] = -5.0;
  }
  positions[0] = NAN; /* backend evaluate fails after argument checks. */
  code = engine->force(pot, 3, positions, species, forces, &energy, NULL, box,
                       NULL, NULL);
  engine->destroy(pot);
  if (code != RGPOT_ENGINE_ERR_EVAL || energy != -3.0) {
    fprintf(stderr, "evaluate failure did not preserve energy (code=%d)\n", code);
    return 1;
  }
  for (index = 0; index < 9; ++index) {
    if (forces[index] != -5.0) {
      fprintf(stderr, "evaluate failure overwrote force element %d\n", index);
      return 1;
    }
  }
  return 0;
}

typedef struct {
  Engine *engine;
  RgpotEnginePot *pot;
  int failures;
} ConcurrentCtx;

static void *concurrent_batch(void *user) {
  ConcurrentCtx *ctx = (ConcurrentCtx *)user;
  const int species[3] = {35, 1, 1};
  const double box[9] = {
      20.0, 0.0, 0.0, 0.0, 20.0, 0.0, 0.0, 0.0, 20.0,
  };
  const double positions[18] = {
      0.0, 0.0, 0.0, 3.0802 * BOHR_TO_ANGSTROM, 0.0, 0.0,
      4.8202 * BOHR_TO_ANGSTROM, 0.0, 0.0, 0.0, 0.0, 0.0,
      3.3 * BOHR_TO_ANGSTROM, 0.0, 0.0, 5.04 * BOHR_TO_ANGSTROM, 0.0, 0.0,
  };
  const double boxes[18] = {
      20.0, 0.0, 0.0, 0.0, 20.0, 0.0, 0.0, 0.0, 20.0,
      20.0, 0.0, 0.0, 0.0, 20.0, 0.0, 0.0, 0.0, 20.0,
  };
  double forces[18];
  double energies[2];
  int count;
  (void)box;
  for (count = 0; count < 200; ++count) {
    if (ctx->engine->batch(ctx->pot, 2, 3, positions, species, forces, energies,
                           NULL, boxes, NULL, NULL) != RGPOT_ENGINE_OK) {
      __sync_fetch_and_add(&ctx->failures, 1);
    }
  }
  return NULL;
}

static void *concurrent_hvp(void *user) {
  ConcurrentCtx *ctx = (ConcurrentCtx *)user;
  const int species[3] = {35, 1, 1};
  const double box[9] = {
      20.0, 0.0, 0.0, 0.0, 20.0, 0.0, 0.0, 0.0, 20.0,
  };
  const double positions[9] = {
      0.0, 0.0, 0.0,
      3.0802 * BOHR_TO_ANGSTROM, 0.0, 0.0,
      4.8202 * BOHR_TO_ANGSTROM, 0.0, 0.0,
  };
  const double direction[9] = {0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.9};
  double product[9];
  int count;
  for (count = 0; count < 200; ++count) {
    if (ctx->engine->hvp(ctx->pot, 3, positions, species, direction, product,
                         box, NULL, NULL) != RGPOT_ENGINE_OK) {
      __sync_fetch_and_add(&ctx->failures, 1);
    }
  }
  return NULL;
}

static void *concurrent_caps(void *user) {
  ConcurrentCtx *ctx = (ConcurrentCtx *)user;
  int count;
  for (count = 0; count < 200; ++count) {
    RgpotEngineCaps caps;
    memset(&caps, 0, sizeof caps);
    caps.size = sizeof caps;
    if (ctx->engine->caps(ctx->pot, &caps) != RGPOT_ENGINE_OK ||
        caps.batched != 1 || caps.reentrancy != 1) {
      __sync_fetch_and_add(&ctx->failures, 1);
    }
  }
  return NULL;
}

static int concurrent_batch_hvp_caps_calls(Engine *engine) {
  pthread_t threads[3];
  ConcurrentCtx ctx;
  int index;
  int created = 0;
  ctx.engine = engine;
  ctx.pot = create(engine, "brh2leps", 0);
  ctx.failures = 0;
  if (ctx.pot == NULL) {
    return 1;
  }
  if (pthread_create(&threads[0], NULL, concurrent_batch, &ctx) == 0) {
    created = 1;
  }
  if (created == 1 &&
      pthread_create(&threads[1], NULL, concurrent_hvp, &ctx) == 0) {
    created = 2;
  }
  if (created == 2 &&
      pthread_create(&threads[2], NULL, concurrent_caps, &ctx) == 0) {
    created = 3;
  }
  if (created != 3) {
    fprintf(stderr, "failed to create concurrency test threads (%d/3)\n",
            created);
    for (index = 0; index < created; ++index) {
      (void)pthread_join(threads[index], NULL);
    }
    engine->destroy(ctx.pot);
    return 1;
  }
  for (index = 0; index < created; ++index) {
    if (pthread_join(threads[index], NULL) != 0) {
      fprintf(stderr, "failed to join concurrency test thread %d\n", index);
      engine->destroy(ctx.pot);
      return 1;
    }
  }
  engine->destroy(ctx.pot);
  if (ctx.failures != 0) {
    fprintf(stderr, "concurrency calls had %d failures\n", ctx.failures);
    return 1;
  }
  return 0;
}

int main(int argc, char **argv) {
  static const double fh2[9] = {
      0.0, 0.0, 0.0, 2.6844, 0.0, 0.0, 4.4434, 0.0, 0.0,
  };
  static const double brh2[9] = {
      0.0, 0.0, 0.0, 3.0802, 0.0, 0.0, 4.8202, 0.0, 0.0,
  };
  static const double variable[12] = {
      0.0, 0.0, 0.0, 5.0, 0.0, 0.0,
      0.0, 5.0, 0.0, 0.0, 0.0, 5.0,
  };
  static const int fh2_species[3] = {9, 1, 1};
  static const int brh2_species[3] = {35, 1, 1};
  static const int variable_species[4] = {13, 13, 13, 13};
  Engine engine;
  void *library;
  if (argc != 2 && argc != 3) {
    return 2;
  }
  library = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
  if (library == NULL) {
    fprintf(stderr, "%s\n", dlerror());
    return 2;
  }
  engine.abi_version =
      (abi_version_fn)dlsym(library, "rgpot_engine_abi_version");
  engine.available = (available_fn)dlsym(library, "rgpot_engine_available");
  engine.abi_stamp = (abi_stamp_fn)dlsym(library, "rgpot_engine_abi_stamp");
  engine.abi_compatible =
      (abi_compatible_fn)dlsym(library, "rgpot_engine_abi_compatible");
  engine.name = (text_fn)dlsym(library, "rgpot_engine_name");
  engine.version = (text_fn)dlsym(library, "rgpot_engine_version");
  engine.create = (create_fn)dlsym(library, "rgpot_engine_create");
  engine.destroy = (destroy_fn)dlsym(library, "rgpot_engine_destroy");
  engine.force = (force_fn)dlsym(library, "rgpot_engine_force");
  engine.hvp = (hvp_fn)dlsym(library, "rgpot_engine_hvp");
  engine.batch = (batch_fn)dlsym(library, "rgpot_engine_force_batch");
  engine.caps = (caps_fn)dlsym(library, "rgpot_engine_caps");
  engine.last_error =
      (last_error_fn)dlsym(library, "rgpot_engine_last_error");
  if (engine.abi_version == NULL || engine.available == NULL ||
      engine.abi_stamp == NULL || engine.abi_compatible == NULL ||
      engine.name == NULL || engine.version == NULL || engine.create == NULL ||
      engine.destroy == NULL || engine.force == NULL || engine.hvp == NULL ||
      engine.batch == NULL || engine.caps == NULL ||
      engine.last_error == NULL) {
    fprintf(stderr, "potlibc engine symbols are incomplete\n");
    return 2;
  }
  if (dlsym(library, "potlibc_mutex_lock") != NULL ||
      dlsym(library, "potlibc_engine_fail_lock") != NULL ||
      dlsym(library, "rgpot_potlibc_test_set_lock_fail") != NULL ||
      dlsym(library, "lock_handle") != NULL) {
    fprintf(stderr, "potlibc engine exported internal symbols\n");
    return 2;
  }
  {
    const RgpotEngineAbiStamp host = {
        RGPOT_ENGINE_ABI_VERSION,
        RGPOT_ENGINE_ABI_MINOR,
        RGPOT_ENGINE_ABI_LAYOUT_REVISION,
    };
    const RgpotEngineAbiStamp stamp = engine.abi_stamp();
    if (engine.abi_version() != RGPOT_ENGINE_ABI_VERSION ||
        !engine.available() ||
        stamp.abi_major != host.abi_major ||
        stamp.abi_minor != host.abi_minor ||
        stamp.layout_revision != host.layout_revision ||
        !engine.abi_compatible(&host) ||
        strcmp(engine.name(), "potlibc") != 0 ||
        strncmp(engine.version(), "potlibc/", 8) != 0 ||
        engine.version()[8] == '\0') {
      fprintf(stderr, "potlibc engine identity is inconsistent\n");
      return 2;
    }
  }
  /* Direct POTLIB Fortran energies in Hartree, converted by
   * EV_PER_HARTREE: 0.005108137718625699 (FH2),
   * 0.017902457453951443 (BrH2), and
   * -0.18671325575959852 (variable-size VBO). */
  if (reference_energy(&engine, "fh25sec", 0, 3, fh2, fh2_species,
                       UINT64_C(0x3fc1cabc637f0856)) != 0 ||
      reference_energy(&engine, "brh2leps", 0, 3, brh2, brh2_species,
                       UINT64_C(0x3fdf2d7a109272ec)) != 0 ||
      reference_energy(&engine, "vbo-beta1", 4, 4, variable,
                       variable_species, UINT64_C(0xc01452a9f90879c8)) != 0 ||
      hvp_transform_is_applied(&engine) != 0 ||
      batch_is_one_call_per_system(&engine) != 0 ||
      config_size_is_extensible(&engine) != 0 ||
      v1_client_matches_version_two(&engine) != 0 ||
      caps_refuses_null_handle(&engine) != 0 ||
      evaluate_failure_preserves_outputs(&engine) != 0 ||
      concurrent_batch_hvp_caps_calls(&engine) != 0 ||
      variable_size_requires_count(&engine) != 0 ||
      reentrant_lock_returns_named_code(&engine) != 0 ||
      (argc == 3 && destroy_frees_handle_when_lock_fails(argv[2]) != 0)) {
    dlclose(library);
    return 1;
  }
  dlclose(library);
  return 0;
}
