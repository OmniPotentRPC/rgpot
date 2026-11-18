/* MIT License
 * Copyright 2023--present rgpot developers
 */
#include "rgpot/engine_c_abi.h"
#include "rgpot/potlibc_engine.h"

#include <dlfcn.h>
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
typedef int (*abi_version_fn)(void);
typedef int (*available_fn)(void);
typedef RgpotEngineAbiStamp (*abi_stamp_fn)(void);
typedef int (*abi_compatible_fn)(const RgpotEngineAbiStamp *);
typedef const char *(*text_fn)(void);
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
  if (argc != 2) {
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
  engine.last_error =
      (last_error_fn)dlsym(library, "rgpot_engine_last_error");
  if (engine.abi_version == NULL || engine.available == NULL ||
      engine.abi_stamp == NULL || engine.abi_compatible == NULL ||
      engine.name == NULL || engine.version == NULL || engine.create == NULL ||
      engine.destroy == NULL || engine.force == NULL || engine.hvp == NULL ||
      engine.last_error == NULL) {
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
  if (reference_energy(&engine, "fh25sec", 0, 3, fh2, fh2_species,
                       UINT64_C(0x3fc1cabc637f0856)) != 0 ||
      reference_energy(&engine, "brh2leps", 0, 3, brh2, brh2_species,
                       UINT64_C(0x3fdf2d7a109272ec)) != 0 ||
      reference_energy(&engine, "vbo-beta1", 4, 4, variable,
                       variable_species, UINT64_C(0xc01452a9f90879c8)) != 0 ||
      hvp_transform_is_applied(&engine) != 0 ||
      variable_size_requires_count(&engine) != 0) {
    dlclose(library);
    return 1;
  }
  dlclose(library);
  return 0;
}
