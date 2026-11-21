/* MIT License
 * Copyright 2023--present rgpot developers
 */
#include "rgpot/engine_c_abi.h"
#include "rgpot/potlibc_engine.h"

#include <dlfcn.h>
#include <limits.h>
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
                      const double *, double *, const double *);
typedef int (*abi_version_fn)(void);
typedef int (*available_fn)(void);
typedef RgpotEngineAbiStamp (*abi_stamp_fn)(void);
typedef int (*abi_compatible_fn)(const RgpotEngineAbiStamp *);
typedef const char *(*text_fn)(void);
typedef int (*caps_fn)(const RgpotEnginePot *, RgpotEngineCaps *);
typedef size_t (*last_error_fn)(const RgpotEnginePot *, char *, size_t);
typedef void (*set_lock_fail_fn)(int);

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
  caps_fn caps;
  last_error_fn last_error;
} Engine;

static uint64_t bits(double value) {
  uint64_t result;
  memcpy(&result, &value, sizeof result);
  return result;
}

static RgpotPotlibcConfigV1 config(const char *surface_id,
                                   uint64_t atom_count) {
  RgpotPotlibcConfigV1 result;
  memset(&result, 0, sizeof result);
  result.magic = RGPOT_POTLIBC_CONFIG_MAGIC;
  result.version = RGPOT_POTLIBC_CONFIG_VERSION;
  result.atom_count = atom_count;
  snprintf(result.surface_id, sizeof result.surface_id, "%s", surface_id);
  return result;
}

static RgpotEnginePot *create(Engine *engine, const char *surface_id,
                              uint64_t atom_count) {
  RgpotPotlibcConfigV1 message = config(surface_id, atom_count);
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

static int hvp_arguments_are_checked(Engine *engine) {
  static const double box[9] = {
      20.0, 0.0, 0.0, 0.0, 20.0, 0.0, 0.0, 0.0, 20.0,
  };
  static const int atomic_numbers[3] = {35, 1, 1};
  static const double positions[9] = {
      0.0, 0.0, 0.0,
      3.0802 * BOHR_TO_ANGSTROM, 0.0, 0.0,
      4.8202 * BOHR_TO_ANGSTROM, 0.0, 0.0,
  };
  const double direction[9] = {
      0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.9,
  };
  RgpotEnginePot *pot = create(engine, "brh2leps", 0);
  double aliased[9];
  double product[9];
  double forces[9];
  double energy;
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
  memcpy(aliased, direction, sizeof aliased);
  code = engine->hvp(pot, 3, positions, atomic_numbers, aliased, aliased, box);
  if (code != RGPOT_ENGINE_ERR_INVALID) {
    fprintf(stderr, "HVP output overlapping the direction was accepted\n");
    engine->destroy(pot);
    return 1;
  }
  code = engine->hvp(pot, 3, positions, atomic_numbers, direction, product, box);
  engine->destroy(pot);
  if (code != RGPOT_ENGINE_OK) {
    fprintf(stderr, "HVP on disjoint buffers failed\n");
    return 1;
  }
  return 0;
}

static int variable_size_requires_count(Engine *engine) {
  RgpotPotlibcConfigV1 message = config("vbo-beta1", 0);
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

static int oversized_atom_counts_are_refused(Engine *engine) {
  static const double box[9] = {
      20.0, 0.0, 0.0, 0.0, 20.0, 0.0, 0.0, 0.0, 20.0,
  };
  static const int species[3] = {35, 1, 1};
  static const double positions[9] = {
      0.0, 0.0, 0.0,
      3.0802 * BOHR_TO_ANGSTROM, 0.0, 0.0,
      4.8202 * BOHR_TO_ANGSTROM, 0.0, 0.0,
  };
  const uint64_t too_many = (uint64_t)(SIZE_MAX / (3 * sizeof(double))) + 1;
  RgpotPotlibcConfigV1 message = config("vbo-beta1", too_many);
  char error[512] = {0};
  double product[9];
  double forces[9];
  double energy = 0.0;
  RgpotEnginePot *pot =
      engine->create(&message, sizeof message, error, sizeof error);
  int code;
  if (pot != NULL) {
    fprintf(stderr, "an atom count past SIZE_MAX / (3 * 8) was accepted\n");
    engine->destroy(pot);
    return 1;
  }
  if (strstr(error, "too large") == NULL) {
    fprintf(stderr, "oversized atom count create diagnostic: %s\n", error);
    return 1;
  }
  pot = create(engine, "brh2leps", 0);
  if (pot == NULL) {
    return 1;
  }
  code = engine->force(pot, LONG_MAX, positions, species, forces, &energy,
                       NULL, box, NULL, NULL);
  memset(error, 0, sizeof error);
  engine->last_error(pot, error, sizeof error);
  if (code != RGPOT_ENGINE_ERR_INVALID || strstr(error, "too large") == NULL) {
    fprintf(stderr, "oversized force returned %d (%s)\n", code, error);
    engine->destroy(pot);
    return 1;
  }
  code = engine->hvp(pot, LONG_MAX, positions, species, positions, product,
                     box);
  memset(error, 0, sizeof error);
  engine->last_error(pot, error, sizeof error);
  engine->destroy(pot);
  if (code != RGPOT_ENGINE_ERR_INVALID || strstr(error, "too large") == NULL) {
    fprintf(stderr, "oversized hvp returned %d (%s)\n", code, error);
    return 1;
  }
  return 0;
}

typedef enum {
  REENTER_FORCE = 0,
  REENTER_HVP = 1,
  REENTER_CAPS = 2,
} ReenterKind;

typedef struct {
  Engine *engine;
  RgpotEnginePot *pot;
  const int *species;
  const double *box;
  ReenterKind kind;
  int inner_failed;
} ReenterContext;

static int reenter_on_locked_handle(void *user, long n_atoms, double *positions,
                                    double *box) {
  ReenterContext *ctx = (ReenterContext *)user;
  RgpotEngineCaps caps;
  double forces[9];
  double hv[9];
  double energy = 0.0;
  char error[512] = {0};
  int code = RGPOT_ENGINE_OK;
  (void)box;

  if (ctx->kind == REENTER_FORCE) {
    code = ctx->engine->force(ctx->pot, n_atoms, positions, ctx->species,
                              forces, &energy, NULL, ctx->box, NULL, NULL);
  } else if (ctx->kind == REENTER_HVP) {
    code = ctx->engine->hvp(ctx->pot, n_atoms, positions, ctx->species,
                            positions, hv, ctx->box);
  } else {
    memset(&caps, 0, sizeof caps);
    caps.size = sizeof caps;
    code = ctx->engine->caps(ctx->pot, &caps);
  }
  if (code != RGPOT_ENGINE_ERR_LOCK ||
      ctx->engine->last_error(ctx->pot, error, sizeof error) == 0) {
    fprintf(stderr, "reentrant kind=%d returned %d (error=%s)\n",
            (int)ctx->kind, code, error);
    ctx->inner_failed = 1;
    return RGPOT_ENGINE_ERR_EVAL;
  }
  /* Return 0 from the callback so the outer force call must observe the
   * latched fatal state rather than a nonzero transform status. */
  return RGPOT_ENGINE_OK;
}

static int reentrant_lock_poisons_handle(Engine *engine) {
  static const double box[9] = {
      20.0, 0.0, 0.0, 0.0, 20.0, 0.0, 0.0, 20.0,
  };
  static const int species[3] = {35, 1, 1};
  static const double positions[9] = {
      0.0, 0.0, 0.0,
      3.0802 * BOHR_TO_ANGSTROM, 0.0, 0.0,
      4.8202 * BOHR_TO_ANGSTROM, 0.0, 0.0,
  };
  static const double direction[9] = {
      0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.9,
  };
  static const ReenterKind kinds[3] = {
      REENTER_FORCE,
      REENTER_HVP,
      REENTER_CAPS,
  };
  size_t idx;
  for (idx = 0; idx < 3; ++idx) {
    RgpotEnginePot *pot = create(engine, "brh2leps", 0);
    RgpotEngineCaps caps;
    double forces[9];
    double hv[9];
    double energy = 0.0;
    char error[512] = {0};
    ReenterContext ctx;
    int outer_code;
    int later_force;
    int later_hvp;
    int later_caps;
    if (pot == NULL) {
      return 1;
    }
    ctx.engine = engine;
    ctx.pot = pot;
    ctx.species = species;
    ctx.box = box;
    ctx.kind = kinds[idx];
    ctx.inner_failed = 0;
    outer_code = engine->force(pot, 3, positions, species, forces, &energy,
                               NULL, box, reenter_on_locked_handle, &ctx);
    if (ctx.inner_failed || outer_code != RGPOT_ENGINE_ERR_LOCK) {
      fprintf(stderr,
              "outer force after reentrant kind=%d returned %d, expected %d\n",
              (int)kinds[idx], outer_code, RGPOT_ENGINE_ERR_LOCK);
      engine->destroy(pot);
      return 1;
    }
    if (engine->last_error(pot, error, sizeof error) == 0) {
      fprintf(stderr,
              "outer force after reentrant kind=%d cleared last_error\n",
              (int)kinds[idx]);
      engine->destroy(pot);
      return 1;
    }
    later_force = engine->force(pot, 3, positions, species, forces, &energy,
                                NULL, box, NULL, NULL);
    later_hvp = engine->hvp(pot, 3, positions, species, direction, hv, box);
    memset(&caps, 0, sizeof caps);
    caps.size = sizeof caps;
    later_caps = engine->caps(pot, &caps);
    memset(error, 0, sizeof error);
    if (later_force != RGPOT_ENGINE_ERR_LOCK ||
        later_hvp != RGPOT_ENGINE_ERR_LOCK ||
        later_caps != RGPOT_ENGINE_ERR_LOCK ||
        engine->last_error(pot, error, sizeof error) == 0) {
      fprintf(stderr,
              "poisoned handle kind=%d returned force=%d hvp=%d caps=%d "
              "error=%s\n",
              (int)kinds[idx], later_force, later_hvp, later_caps, error);
      engine->destroy(pot);
      return 1;
    }
    engine->destroy(pot);
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
  set_lock_fail_fn set_lock_fail;
  RgpotPotlibcConfigV1 message = config("brh2leps", 0);
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
  create_sym = (create_fn)dlsym(library, "rgpot_engine_create");
  destroy_sym = (destroy_fn)dlsym(library, "rgpot_engine_destroy");
  force_sym = (force_fn)dlsym(library, "rgpot_engine_force");
  caps_sym = (caps_fn)dlsym(library, "rgpot_engine_caps");
  last_error_sym = (last_error_fn)dlsym(library, "rgpot_engine_last_error");
  set_lock_fail =
      (set_lock_fail_fn)dlsym(library, "rgpot_potlibc_test_set_lock_fail");
  if (create_sym == NULL || destroy_sym == NULL || force_sym == NULL ||
      caps_sym == NULL || last_error_sym == NULL || set_lock_fail == NULL) {
    fprintf(stderr, "lock-failure test library is missing symbols\n");
    dlclose(library);
    return 1;
  }
  set_lock_fail(0);
  pot = create_sym(&message, sizeof message, error, sizeof error);
  if (pot == NULL) {
    fprintf(stderr, "lock-failure create failed: %s\n", error);
    dlclose(library);
    return 1;
  }
  set_lock_fail(1);
  force_code = force_sym(pot, 3, positions, species, forces, &energy, NULL, box,
                         NULL, NULL);
  /* Even once locking succeeds again, the handle must stay fatally unusable. */
  set_lock_fail(0);
  memset(&caps, 0, sizeof caps);
  caps.size = sizeof caps;
  caps_code = caps_sym(pot, &caps);
  memset(error, 0, sizeof error);
  last_error_sym(pot, error, sizeof error);
  /* Destroy while potlibc_mutex_lock fails: must still release surface and pot. */
  set_lock_fail(1);
  destroy_sym(pot);
  set_lock_fail(0);
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

static int abi_compatibility_checks_host_minor(Engine *engine) {
  const RgpotEngineAbiStamp equal = {
      RGPOT_ENGINE_ABI_VERSION,
      RGPOT_ENGINE_ABI_MINOR,
      RGPOT_ENGINE_ABI_LAYOUT_REVISION,
  };
  const RgpotEngineAbiStamp older = {
      RGPOT_ENGINE_ABI_VERSION,
      RGPOT_ENGINE_ABI_MINOR - 1,
      RGPOT_ENGINE_ABI_LAYOUT_REVISION,
  };
  const RgpotEngineAbiStamp newer = {
      RGPOT_ENGINE_ABI_VERSION,
      RGPOT_ENGINE_ABI_MINOR + 1,
      RGPOT_ENGINE_ABI_LAYOUT_REVISION,
  };
  const RgpotEngineAbiStamp wrong_major = {
      RGPOT_ENGINE_ABI_VERSION + 1,
      RGPOT_ENGINE_ABI_MINOR,
      RGPOT_ENGINE_ABI_LAYOUT_REVISION,
  };
  const RgpotEngineAbiStamp wrong_layout = {
      RGPOT_ENGINE_ABI_VERSION,
      RGPOT_ENGINE_ABI_MINOR,
      RGPOT_ENGINE_ABI_LAYOUT_REVISION + 1,
  };
  if (engine->abi_compatible(NULL) != 0) {
    fprintf(stderr, "abi_compatible(NULL) returned nonzero\n");
    return 1;
  }
  if (!engine->abi_compatible(&equal)) {
    fprintf(stderr, "abi_compatible rejected equal host minor %u\n",
            (unsigned)equal.abi_minor);
    return 1;
  }
  if (engine->abi_compatible(&older)) {
    fprintf(stderr, "abi_compatible accepted older host minor %u\n",
            (unsigned)older.abi_minor);
    return 1;
  }
  if (!engine->abi_compatible(&newer)) {
    fprintf(stderr, "abi_compatible rejected newer host minor %u\n",
            (unsigned)newer.abi_minor);
    return 1;
  }
  if (engine->abi_compatible(&wrong_major) ||
      engine->abi_compatible(&wrong_layout)) {
    fprintf(stderr, "abi_compatible accepted mismatched major or layout\n");
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
  engine.caps = (caps_fn)dlsym(library, "rgpot_engine_caps");
  engine.last_error =
      (last_error_fn)dlsym(library, "rgpot_engine_last_error");
  if (engine.abi_version == NULL || engine.available == NULL ||
      engine.abi_stamp == NULL || engine.abi_compatible == NULL ||
      engine.name == NULL || engine.version == NULL || engine.create == NULL ||
      engine.destroy == NULL || engine.force == NULL || engine.hvp == NULL ||
      engine.caps == NULL || engine.last_error == NULL) {
    fprintf(stderr, "potlibc engine symbols are incomplete\n");
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
  if (abi_compatibility_checks_host_minor(&engine) != 0 ||
      reference_energy(&engine, "fh25sec", 0, 3, fh2, fh2_species,
                       UINT64_C(0x3fc1cabc637f0856)) != 0 ||
      reference_energy(&engine, "brh2leps", 0, 3, brh2, brh2_species,
                       UINT64_C(0x3fdf2d7a109272ec)) != 0 ||
      reference_energy(&engine, "vbo-beta1", 4, 4, variable,
                       variable_species, UINT64_C(0xc01452a9f90879c8)) != 0 ||
      hvp_arguments_are_checked(&engine) != 0 ||
      variable_size_requires_count(&engine) != 0 ||
      oversized_atom_counts_are_refused(&engine) != 0 ||
      reentrant_lock_poisons_handle(&engine) != 0 ||
      (argc == 3 && destroy_frees_handle_when_lock_fails(argv[2]) != 0)) {
    dlclose(library);
    return 1;
  }
  dlclose(library);
  return 0;
}
