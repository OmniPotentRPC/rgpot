/* MIT License
 * Copyright 2023--present rgpot developers
 */

/* Observable behaviour of libcoupled_well_engine through its C ABI.
 *
 * E(x, y) = V0 ((x / x0)^2 - 1)^2 + k / 2 (y - alpha x^2)^2 with V0 = 3,
 * x0 = 2, k = 4 and alpha = 1/4. Every value below is an exact binary number,
 * so each check compares for equality with the closed form worked out by hand
 * from the expanded polynomial
 *   E = V0 (x^4 / x0^4 - 2 x^2 / x0^2 + 1) + k / 2 (y^2 - 2 alpha x^2 y
 *       + alpha^2 x^4),
 * not from the engine's factored expressions. */
#include "rgpot/coupled_well_engine.h"
#include "rgpot/engine_c_abi.h"

#include <math.h>
#include <pthread.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>

#ifndef RGPOT_ENGINE_PROJECT_VERSION
#define RGPOT_ENGINE_PROJECT_VERSION "unknown"
#endif

#define BARRIER 3.0
#define WELL 2.0
#define SPRING 4.0
#define COUPLING 0.25
#define NUMBER 6
#define THREADS 4
#define EVALUATIONS 2000

static const double BOX[9] = {10.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0};
static const int SPECIES[1] = {NUMBER};

static int failures = 0;

#define CHECK(condition)                                                       \
  do {                                                                         \
    if (!(condition)) {                                                        \
      fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__,         \
              #condition);                                                     \
      ++failures;                                                              \
    }                                                                          \
  } while (0)

static RgpotCoupledWellConfigV2 config(void) {
  RgpotCoupledWellConfigV2 result;
  memset(&result, 0, sizeof result);
  result.magic = RGPOT_COUPLED_WELL_CONFIG_MAGIC;
  result.size = sizeof result;
  result.version = RGPOT_COUPLED_WELL_CONFIG_VERSION;
  result.reserved0 = 0;
  result.atomic_number = NUMBER;
  result.reserved1 = 0;
  result.barrier_energy = BARRIER;
  result.well_position = WELL;
  result.transverse_spring = SPRING;
  result.coupling = COUPLING;
  return result;
}

static RgpotEnginePot *make(void) {
  const RgpotCoupledWellConfigV2 message = config();
  char error[256] = {0};
  RgpotEnginePot *pot =
      rgpot_engine_create(&message, sizeof message, error, sizeof error);
  if (pot == NULL) {
    fprintf(stderr, "create failed: %s\n", error);
  }
  return pot;
}

static int refused(const RgpotCoupledWellConfigV2 *message, size_t length) {
  char error[256] = {0};
  RgpotEnginePot *pot = rgpot_engine_create(message, length, error,
                                            sizeof error);
  if (pot != NULL) {
    rgpot_engine_destroy(pot);
    return 0;
  }
  return error[0] != '\0';
}

static int refused_bytes(const unsigned char *message, size_t length) {
  char error[256] = {0};
  RgpotEnginePot *pot = rgpot_engine_create(message, length, error,
                                            sizeof error);
  if (pot != NULL) {
    rgpot_engine_destroy(pot);
    return 0;
  }
  return error[0] != '\0';
}

static void identity_and_contract(void) {
  const RgpotEngineAbiStamp host = {RGPOT_ENGINE_ABI_VERSION,
                                    RGPOT_ENGINE_ABI_MINOR,
                                    RGPOT_ENGINE_ABI_LAYOUT_REVISION};
  const RgpotEngineAbiStamp stamp = rgpot_engine_abi_stamp();
  RgpotEngineAbiStamp other = host;
  CHECK(rgpot_engine_abi_version() == RGPOT_ENGINE_ABI_VERSION);
  CHECK(rgpot_engine_available() == 1);
  CHECK(stamp.abi_major == host.abi_major && stamp.abi_minor == host.abi_minor &&
        stamp.layout_revision == host.layout_revision);
  CHECK(strcmp(rgpot_engine_name(), "coupled-well") == 0);
  CHECK(strcmp(rgpot_engine_version(),
               "coupled-well/" RGPOT_ENGINE_PROJECT_VERSION) == 0);

  /* The documented rule: equal major, equal layout, engine minor at most the
   * host's. A host built against a newer minor is served; an older one is not. */
  CHECK(rgpot_engine_abi_compatible(&host) == 1);
  other.abi_minor = (unsigned short)(host.abi_minor + 1);
  CHECK(rgpot_engine_abi_compatible(&other) == 1);
  other = host;
  other.abi_minor = (unsigned short)(host.abi_minor - 1);
  CHECK(rgpot_engine_abi_compatible(&other) == 0);
  other = host;
  other.abi_major = (unsigned short)(host.abi_major + 1);
  CHECK(rgpot_engine_abi_compatible(&other) == 0);
  other = host;
  other.layout_revision = (unsigned short)(host.layout_revision + 1);
  CHECK(rgpot_engine_abi_compatible(&other) == 0);
  CHECK(rgpot_engine_abi_compatible(NULL) == 0);
}

static void create_validation(void) {
  RgpotCoupledWellConfigV2 message = config();
  RgpotEnginePot *pot = make();
  CHECK(pot != NULL);
  rgpot_engine_destroy(pot);
  rgpot_engine_destroy(NULL);

  CHECK(refused(NULL, sizeof message));
  CHECK(refused(&message, offsetof(RgpotCoupledWellConfigV2, version) + 3));
  CHECK(refused(&message, sizeof message - 1));
  {
    unsigned char future[sizeof message + 8];
    memset(future, 0, sizeof future);
    memcpy(future, &message, sizeof message);
    ((RgpotCoupledWellConfigV2 *)future)->size = sizeof future;
    CHECK(!refused_bytes(future, sizeof future));
    future[sizeof message] = 7u;
    CHECK(refused_bytes(future, sizeof future));
  }
  message = config();
  message.size = sizeof(message) - 8;
  CHECK(refused(&message, sizeof message));
  message = config();
  message.magic ^= 1u;
  CHECK(refused(&message, sizeof message));
  message = config();
  message.version += 1u;
  CHECK(refused(&message, sizeof message));
  message = config();
  message.reserved0 = 1u;
  CHECK(refused(&message, sizeof message));
  message = config();
  message.reserved1 = 1u;
  CHECK(refused(&message, sizeof message));
  message = config();
  message.atomic_number = 0;
  CHECK(refused(&message, sizeof message));
  message = config();
  message.atomic_number = -1;
  CHECK(refused(&message, sizeof message));
  message = config();
  message.barrier_energy = 0.0;
  CHECK(refused(&message, sizeof message));
  message.barrier_energy = -1.0;
  CHECK(refused(&message, sizeof message));
  message.barrier_energy = NAN;
  CHECK(refused(&message, sizeof message));
  message = config();
  message.well_position = 0.0;
  CHECK(refused(&message, sizeof message));
  message.well_position = INFINITY;
  CHECK(refused(&message, sizeof message));
  message = config();
  message.transverse_spring = -4.0;
  CHECK(refused(&message, sizeof message));
  message.transverse_spring = NAN;
  CHECK(refused(&message, sizeof message));
  message = config();
  message.coupling = NAN;
  CHECK(refused(&message, sizeof message));
  message.coupling = -INFINITY;
  CHECK(refused(&message, sizeof message));
  message = config();
  message.coupling = -COUPLING;
  {
    char error[256] = {0};
    pot = rgpot_engine_create(&message, sizeof message, error, sizeof error);
    CHECK(pot != NULL);
    rgpot_engine_destroy(pot);
  }
  message.coupling = NAN;
  CHECK(rgpot_engine_create(&message, sizeof message, NULL, 0) == NULL);
  CHECK(rgpot_engine_create(NULL, 0, NULL, 0) == NULL);
}

typedef struct {
  double x, y, energy, fx, fy, hxx, hxy, hyy;
} Reference;

/* Closed forms from the expanded polynomial, worked by hand. */
static const Reference POINTS[3] = {
    /* minimum of both wells: x = x0, y = alpha x0^2 */
    {2.0, 1.0, 0.0, 0.0, 0.0, 10.0, -4.0, 4.0},
    /* barrier top */
    {0.0, 0.0, 3.0, 0.0, 0.0, -3.0, 0.0, 4.0},
    /* generic point */
    {1.0, 0.5, 1.8125, 2.75, -1.0, -0.25, -2.0, 4.0},
};

static void analytic_force_and_hessian(void) {
  RgpotEnginePot *pot = make();
  size_t index;
  if (pot == NULL) {
    ++failures;
    return;
  }
  for (index = 0; index < 3; ++index) {
    const Reference *want = &POINTS[index];
    const double positions[3] = {want->x, want->y, 7.0};
    double forces[3] = {-1.0, -1.0, -1.0};
    double energy = -1.0;
    double variance = -1.0;
    double product[3] = {-1.0, -1.0, -1.0};
    const double along_x[3] = {1.0, 0.0, 9.0};
    const double along_y[3] = {0.0, 1.0, 9.0};
    CHECK(rgpot_engine_force(pot, 1, positions, SPECIES, forces, &energy,
                             &variance, BOX, NULL, NULL) == RGPOT_ENGINE_OK);
    CHECK(energy == want->energy);
    CHECK(forces[0] == want->fx);
    CHECK(forces[1] == want->fy);
    CHECK(forces[2] == 0.0);
    CHECK(variance == 0.0);
    CHECK(rgpot_engine_hvp(pot, 1, positions, SPECIES, along_x, product, BOX) ==
          RGPOT_ENGINE_OK);
    CHECK(product[0] == want->hxx && product[1] == want->hxy &&
          product[2] == 0.0);
    CHECK(rgpot_engine_hvp(pot, 1, positions, SPECIES, along_y, product, BOX) ==
          RGPOT_ENGINE_OK);
    CHECK(product[0] == want->hxy && product[1] == want->hyy &&
          product[2] == 0.0);
  }
  {
    /* Generic point, direction (1, 2, 5): H d = (-0.25 - 4, -2 + 8, 0). */
    const double positions[3] = {1.0, 0.5, 0.0};
    const double direction[3] = {1.0, 2.0, 5.0};
    double product[3];
    CHECK(rgpot_engine_hvp(pot, 1, positions, SPECIES, direction, product,
                           BOX) == RGPOT_ENGINE_OK);
    CHECK(product[0] == -4.25 && product[1] == 6.0 && product[2] == 0.0);
  }
  rgpot_engine_destroy(pot);
}

static int shift_x(void *user, long n_atoms, double *positions, double *box) {
  (void)n_atoms;
  box[0] += 1.0;
  positions[0] += *(const double *)user;
  return 0;
}

static int refuse(void *user, long n_atoms, double *positions, double *box) {
  (void)n_atoms;
  (void)positions;
  (void)box;
  return *(const int *)user;
}

static int refuse_second(void *user, long n_atoms, double *positions,
                         double *box) {
  int *calls = (int *)user;
  (void)n_atoms;
  (void)positions;
  (void)box;
  *calls += 1;
  return *calls == 2 ? 7 : 0;
}

static void transform_is_scratch_only(void) {
  RgpotEnginePot *pot = make();
  const double amount = 1.0;
  const double positions[3] = {0.0, 0.5, 0.0};
  const double box[9] = {10.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0};
  double forces[3];
  double energy;
  int code = 11;
  if (pot == NULL) {
    ++failures;
    return;
  }
  /* x -> 1 reproduces the generic point (1, 0.5). */
  CHECK(rgpot_engine_force(pot, 1, positions, SPECIES, forces, &energy, NULL,
                           box, shift_x, (void *)&amount) == RGPOT_ENGINE_OK);
  CHECK(energy == POINTS[2].energy && forces[0] == POINTS[2].fx &&
        forces[1] == POINTS[2].fy);
  CHECK(positions[0] == 0.0 && box[0] == 10.0);

  energy = -1.0;
  forces[0] = -1.0;
  CHECK(rgpot_engine_force(pot, 1, positions, SPECIES, forces, &energy, NULL,
                           box, refuse, &code) == code);
  CHECK(energy == -1.0 && forces[0] == -1.0);
  CHECK(rgpot_engine_last_error(pot, NULL, 0) > 0);
  rgpot_engine_destroy(pot);
}

static void rejects_bad_requests(void) {
  RgpotEnginePot *pot = make();
  const double positions[3] = {1.0, 0.5, 0.0};
  const double not_finite[3] = {NAN, 0.5, 0.0};
  const double huge[3] = {1.0e200, 0.5, 0.0};
  const double finite_extreme[3] = {1.0, 0.0, 0.0};
  const int wrong_species[1] = {NUMBER + 1};
  const double direction[3] = {1.0, 0.0, 0.0};
  const double bad_direction[3] = {INFINITY, 0.0, 0.0};
  double forces[3];
  double energy;
  double product[3];
  char message[256];
  if (pot == NULL) {
    ++failures;
    return;
  }
  CHECK(rgpot_engine_force(NULL, 1, positions, SPECIES, forces, &energy, NULL,
                           BOX, NULL, NULL) == RGPOT_ENGINE_ERR_INVALID);
  CHECK(rgpot_engine_force(pot, 2, positions, SPECIES, forces, &energy, NULL,
                           BOX, NULL, NULL) == RGPOT_ENGINE_ERR_INVALID);
  CHECK(rgpot_engine_force(pot, 0, positions, SPECIES, forces, &energy, NULL,
                           BOX, NULL, NULL) == RGPOT_ENGINE_ERR_INVALID);
  CHECK(rgpot_engine_force(pot, 1, NULL, SPECIES, forces, &energy, NULL, BOX,
                           NULL, NULL) == RGPOT_ENGINE_ERR_INVALID);
  CHECK(rgpot_engine_force(pot, 1, positions, NULL, forces, &energy, NULL, BOX,
                           NULL, NULL) == RGPOT_ENGINE_ERR_INVALID);
  CHECK(rgpot_engine_force(pot, 1, positions, SPECIES, NULL, &energy, NULL,
                           BOX, NULL, NULL) == RGPOT_ENGINE_ERR_INVALID);
  CHECK(rgpot_engine_force(pot, 1, positions, SPECIES, forces, NULL, NULL, BOX,
                           NULL, NULL) == RGPOT_ENGINE_ERR_INVALID);
  CHECK(rgpot_engine_force(pot, 1, positions, SPECIES, forces, &energy, NULL,
                           NULL, NULL, NULL) == RGPOT_ENGINE_ERR_INVALID);
  CHECK(rgpot_engine_force(pot, 1, positions, wrong_species, forces, &energy,
                           NULL, BOX, NULL, NULL) == RGPOT_ENGINE_ERR_INVALID);
  CHECK(rgpot_engine_last_error(pot, message, sizeof message) > 0 &&
        strstr(message, "atomic number") != NULL);
  CHECK(rgpot_engine_force(pot, 1, not_finite, SPECIES, forces, &energy, NULL,
                           BOX, NULL, NULL) == RGPOT_ENGINE_ERR_INVALID);
  forces[0] = -1.0;
  energy = -1.0;
  CHECK(rgpot_engine_force(pot, 1, huge, SPECIES, forces, &energy, NULL, BOX,
                           NULL, NULL) == RGPOT_ENGINE_ERR_EVAL);
  CHECK(energy == -1.0 && forces[0] == -1.0);
  CHECK(rgpot_engine_hvp(pot, 1, positions, SPECIES, bad_direction, product,
                         BOX) == RGPOT_ENGINE_ERR_INVALID);
  CHECK(rgpot_engine_hvp(pot, 1, positions, SPECIES, direction, NULL, BOX) ==
        RGPOT_ENGINE_ERR_INVALID);
  CHECK(rgpot_engine_hvp(pot, 1, positions, wrong_species, direction, product,
                         BOX) == RGPOT_ENGINE_ERR_INVALID);
  CHECK(rgpot_engine_hvp(pot, 1, huge, SPECIES, direction, product, BOX) ==
        RGPOT_ENGINE_ERR_EVAL);
  {
    RgpotCoupledWellConfigV2 tiny = config();
    RgpotEnginePot *small;
    const double scaled_min[3] = {1.0e-155, 0.0, 0.0};
    const double along_y[3] = {0.0, 1.0, 0.0};
    tiny.barrier_energy = 1.0e-320;
    tiny.well_position = 1.0e-155;
    tiny.transverse_spring = 1.0;
    tiny.coupling = 0.0;
    small = rgpot_engine_create(&tiny, sizeof tiny, message, sizeof message);
    CHECK(small != NULL);
    forces[0] = -1.0;
    energy = -1.0;
    CHECK(rgpot_engine_force(small, 1, finite_extreme, SPECIES, forces, &energy,
                             NULL, BOX, NULL, NULL) == RGPOT_ENGINE_OK);
    CHECK(isfinite(energy) && isfinite(forces[0]) && isfinite(forces[1]));
    forces[0] = -1.0;
    forces[1] = -1.0;
    energy = -1.0;
    CHECK(rgpot_engine_force(small, 1, scaled_min, SPECIES, forces, &energy,
                             NULL, BOX, NULL, NULL) == RGPOT_ENGINE_OK);
    CHECK(energy == 0.0 && forces[0] == 0.0 && forces[1] == 0.0);
    CHECK(rgpot_engine_hvp(small, 1, scaled_min, SPECIES, direction, product,
                           BOX) == RGPOT_ENGINE_OK);
    CHECK(isfinite(product[0]) && product[1] == 0.0 && product[2] == 0.0);
    CHECK(rgpot_engine_hvp(small, 1, scaled_min, SPECIES, along_y, product,
                           BOX) == RGPOT_ENGINE_OK);
    CHECK(product[0] == 0.0 && product[1] == 1.0 && product[2] == 0.0);
    rgpot_engine_destroy(small);
  }
  {
    RgpotCoupledWellConfigV2 huge_alpha = config();
    RgpotEnginePot *stiff;
    const double point[3] = {1.0, 1.0e200, 0.0};
    const double along_y[3] = {0.0, 1.0, 0.0};
    huge_alpha.barrier_energy = 1.0;
    huge_alpha.well_position = 1.0;
    huge_alpha.transverse_spring = 1.0;
    huge_alpha.coupling = 1.0e200;
    stiff = rgpot_engine_create(&huge_alpha, sizeof huge_alpha, message,
                                sizeof message);
    CHECK(stiff != NULL);
    forces[0] = 9.0;
    forces[1] = 9.0;
    energy = 9.0;
    CHECK(rgpot_engine_force(stiff, 1, point, SPECIES, forces, &energy, NULL,
                             BOX, NULL, NULL) == RGPOT_ENGINE_OK);
    CHECK(energy == 0.0 && forces[0] == 0.0 && forces[1] == 0.0);
    CHECK(rgpot_engine_hvp(stiff, 1, point, SPECIES, along_y, product, BOX) ==
          RGPOT_ENGINE_OK);
    CHECK(product[0] == -2.0e200 && product[1] == 1.0 && product[2] == 0.0);
    rgpot_engine_destroy(stiff);
  }
  CHECK(rgpot_engine_force(pot, 1, positions, SPECIES, forces, &energy, NULL,
                           BOX, NULL, NULL) == RGPOT_ENGINE_OK);
  CHECK(rgpot_engine_last_error(pot, message, sizeof message) == 0);
  rgpot_engine_destroy(pot);
}

static void batch_matches_the_analytic_values(void) {
  RgpotEnginePot *pot = make();
  double positions[9];
  double boxes[27];
  double forces[9];
  double energies[3];
  double variances[3] = {-1.0, -1.0, -1.0};
  size_t index;
  if (pot == NULL) {
    ++failures;
    return;
  }
  for (index = 0; index < 3; ++index) {
    positions[3 * index] = POINTS[index].x;
    positions[3 * index + 1] = POINTS[index].y;
    positions[3 * index + 2] = (double)index;
    memcpy(boxes + 9 * index, BOX, sizeof BOX);
  }
  CHECK(rgpot_engine_force_batch(pot, 3, 1, positions, SPECIES, forces,
                                 energies, variances, boxes, NULL,
                                 NULL) == RGPOT_ENGINE_OK);
  for (index = 0; index < 3; ++index) {
    CHECK(energies[index] == POINTS[index].energy);
    CHECK(forces[3 * index] == POINTS[index].fx);
    CHECK(forces[3 * index + 1] == POINTS[index].fy);
    CHECK(forces[3 * index + 2] == 0.0);
    CHECK(variances[index] == 0.0);
  }
  CHECK(rgpot_engine_force_batch(pot, 3, 1, positions, SPECIES, forces,
                                 energies, NULL, boxes, NULL,
                                 NULL) == RGPOT_ENGINE_OK);
  rgpot_engine_destroy(pot);
}

static void batch_failure_writes_nothing(void) {
  RgpotEnginePot *pot = make();
  double positions[9] = {1.0, 0.5, 0.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0};
  double boxes[27];
  double forces[9];
  double energies[3];
  double variances[3];
  int calls = 0;
  int refusal = 5;
  size_t index;
  if (pot == NULL) {
    ++failures;
    return;
  }
  for (index = 0; index < 3; ++index) {
    memcpy(boxes + 9 * index, BOX, sizeof BOX);
  }
  for (index = 0; index < 9; ++index) {
    forces[index] = -1.0;
  }
  for (index = 0; index < 3; ++index) {
    energies[index] = -1.0;
    variances[index] = -1.0;
  }
  /* The second system's transform fails: the first system is not written. */
  CHECK(rgpot_engine_force_batch(pot, 3, 1, positions, SPECIES, forces,
                                 energies, variances, boxes, refuse_second,
                                 &calls) == 7);
  CHECK(calls == 2);
  CHECK(forces[0] == -1.0 && forces[8] == -1.0);
  CHECK(energies[0] == -1.0 && variances[0] == -1.0);
  CHECK(rgpot_engine_force_batch(pot, 3, 1, positions, SPECIES, forces,
                                 energies, variances, boxes, refuse,
                                 &refusal) == refusal);
  CHECK(forces[0] == -1.0 && energies[0] == -1.0);

  /* The last system overflows: the finite ones before it are not written. */
  positions[6] = 1.0e200;
  CHECK(rgpot_engine_force_batch(pot, 3, 1, positions, SPECIES, forces,
                                 energies, variances, boxes, NULL,
                                 NULL) == RGPOT_ENGINE_ERR_EVAL);
  CHECK(forces[0] == -1.0 && energies[0] == -1.0 && variances[0] == -1.0);
  positions[6] = 0.0;

  CHECK(rgpot_engine_force_batch(pot, 0, 1, positions, SPECIES, forces,
                                 energies, variances, boxes, NULL,
                                 NULL) == RGPOT_ENGINE_ERR_INVALID);
  CHECK(rgpot_engine_force_batch(pot, -3, 1, positions, SPECIES, forces,
                                 energies, variances, boxes, NULL,
                                 NULL) == RGPOT_ENGINE_ERR_INVALID);
  CHECK(rgpot_engine_force_batch(pot, 3, 0, positions, SPECIES, forces,
                                 energies, variances, boxes, NULL,
                                 NULL) == RGPOT_ENGINE_ERR_INVALID);
  CHECK(rgpot_engine_force_batch(pot, 3, 2, positions, SPECIES, forces,
                                 energies, variances, boxes, NULL,
                                 NULL) == RGPOT_ENGINE_ERR_INVALID);
  CHECK(rgpot_engine_force_batch(pot, 3, 1, NULL, SPECIES, forces, energies,
                                 variances, boxes, NULL,
                                 NULL) == RGPOT_ENGINE_ERR_INVALID);
  CHECK(rgpot_engine_force_batch(pot, 3, 1, positions, NULL, forces, energies,
                                 variances, boxes, NULL,
                                 NULL) == RGPOT_ENGINE_ERR_INVALID);
  CHECK(rgpot_engine_force_batch(pot, 3, 1, positions, SPECIES, NULL, energies,
                                 variances, boxes, NULL,
                                 NULL) == RGPOT_ENGINE_ERR_INVALID);
  CHECK(rgpot_engine_force_batch(pot, 3, 1, positions, SPECIES, forces, NULL,
                                 variances, boxes, NULL,
                                 NULL) == RGPOT_ENGINE_ERR_INVALID);
  CHECK(rgpot_engine_force_batch(pot, 3, 1, positions, SPECIES, forces,
                                 energies, variances, NULL, NULL,
                                 NULL) == RGPOT_ENGINE_ERR_INVALID);
  CHECK(rgpot_engine_force_batch(NULL, 3, 1, positions, SPECIES, forces,
                                 energies, variances, boxes, NULL,
                                 NULL) == RGPOT_ENGINE_ERR_INVALID);
  CHECK(rgpot_engine_force_batch(pot, 0x7fffffffffffffffL, 1, positions,
                                 SPECIES, forces, energies, variances, boxes,
                                 NULL, NULL) == RGPOT_ENGINE_ERR_INVALID);
  CHECK(forces[0] == -1.0 && energies[0] == -1.0 && variances[0] == -1.0);
  rgpot_engine_destroy(pot);
}

static void caps_and_last_error(void) {
  RgpotEnginePot *pot = make();
  RgpotEngineCaps caps;
  RgpotEngineCaps small;
  const double positions[3] = {1.0, 0.5, 0.0};
  double forces[3];
  double energy;
  char buffer[8];
  char full[256];
  size_t length;
  if (pot == NULL) {
    ++failures;
    return;
  }
  memset(&caps, 0xff, sizeof caps);
  caps.size = sizeof caps;
  CHECK(rgpot_engine_caps(pot, &caps) == RGPOT_ENGINE_OK);
  CHECK(caps.size == sizeof caps);
  CHECK(caps.reentrancy == 0 && caps.batched == 1 && caps.analytic_hvp == 1);
  CHECK(caps.periodic == 0 && caps.stress == 0 && caps.group_collective == 0);
  memset(&small, 0xff, sizeof small);
  small.size = (unsigned)offsetof(RgpotEngineCaps, per_image_instances);
  CHECK(rgpot_engine_caps(pot, &small) == RGPOT_ENGINE_OK);
  CHECK(small.size == offsetof(RgpotEngineCaps, per_image_instances));
  CHECK(small.reentrancy == 0);
  /* Fields past the reported size keep the 0xff fill. */
  CHECK(small.batched != 0 && small.analytic_hvp != 0);
  small.size = 1;
  CHECK(rgpot_engine_caps(pot, &small) == RGPOT_ENGINE_ERR_INVALID);
  CHECK(rgpot_engine_caps(pot, NULL) == RGPOT_ENGINE_ERR_INVALID);
  CHECK(rgpot_engine_caps(NULL, &caps) == RGPOT_ENGINE_ERR_INVALID);

  CHECK(rgpot_engine_last_error(NULL, buffer, sizeof buffer) == 0);
  CHECK(rgpot_engine_last_error(pot, buffer, sizeof buffer) == 0);
  CHECK(rgpot_engine_force(pot, 2, positions, SPECIES, forces, &energy, NULL,
                           BOX, NULL, NULL) == RGPOT_ENGINE_ERR_INVALID);
  length = rgpot_engine_last_error(pot, full, sizeof full);
  CHECK(length > sizeof buffer && strlen(full) == length);
  memset(buffer, 'x', sizeof buffer);
  CHECK(rgpot_engine_last_error(pot, buffer, sizeof buffer) == length);
  CHECK(strlen(buffer) == sizeof buffer - 1 &&
        strncmp(buffer, full, sizeof buffer - 1) == 0);
  CHECK(rgpot_engine_last_error(pot, NULL, 0) == length);
  CHECK(rgpot_engine_last_error(pot, full, sizeof full) == length);
  rgpot_engine_destroy(pot);
}

typedef struct {
  RgpotEnginePot *pot;
  int bad;
} Worker;

static void *evaluate_repeatedly(void *argument) {
  Worker *worker = (Worker *)argument;
  const double positions[3] = {1.0, 0.5, 0.0};
  double forces[3];
  double energy;
  int count;
  for (count = 0; count < EVALUATIONS; ++count) {
    if (rgpot_engine_force(worker->pot, 1, positions, SPECIES, forces, &energy,
                           NULL, BOX, NULL, NULL) != RGPOT_ENGINE_OK ||
        energy != POINTS[2].energy || forces[0] != POINTS[2].fx ||
        forces[1] != POINTS[2].fy) {
      worker->bad += 1;
    }
  }
  return NULL;
}

static void *fail_repeatedly(void *argument) {
  Worker *worker = (Worker *)argument;
  const double positions[3] = {1.0, 0.5, 0.0};
  double forces[3];
  double energy;
  char message[256];
  int count;
  for (count = 0; count < EVALUATIONS; ++count) {
    (void)rgpot_engine_force(worker->pot, 2, positions, SPECIES, forces,
                             &energy, NULL, BOX, NULL, NULL);
    (void)rgpot_engine_last_error(worker->pot, message, sizeof message);
  }
  return NULL;
}

static void *batch_repeatedly(void *argument) {
  Worker *worker = (Worker *)argument;
  const double positions[6] = {1.0, 0.5, 0.0, 2.0, 1.0, 0.0};
  const double boxes[18] = {10.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0,
                            10.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0};
  double forces[6];
  double energies[2];
  int count;
  for (count = 0; count < EVALUATIONS; ++count) {
    if (rgpot_engine_force_batch(worker->pot, 2, 1, positions, SPECIES, forces,
                                 energies, NULL, boxes, NULL, NULL) !=
        RGPOT_ENGINE_OK) {
      worker->bad += 1;
    }
  }
  return NULL;
}

static void *hvp_repeatedly(void *argument) {
  Worker *worker = (Worker *)argument;
  const double positions[3] = {1.0, 0.5, 0.0};
  const double direction[3] = {1.0, 2.0, 0.0};
  double product[3];
  int count;
  for (count = 0; count < EVALUATIONS; ++count) {
    if (rgpot_engine_hvp(worker->pot, 1, positions, SPECIES, direction, product,
                         BOX) != RGPOT_ENGINE_OK) {
      worker->bad += 1;
    }
  }
  return NULL;
}

static void *caps_repeatedly(void *argument) {
  Worker *worker = (Worker *)argument;
  RgpotEngineCaps caps;
  int count;
  for (count = 0; count < EVALUATIONS; ++count) {
    memset(&caps, 0, sizeof caps);
    caps.size = sizeof caps;
    if (rgpot_engine_caps(worker->pot, &caps) != RGPOT_ENGINE_OK ||
        caps.analytic_hvp != 1 || caps.batched != 1) {
      worker->bad += 1;
    }
  }
  return NULL;
}

static void one_handle_serves_concurrent_calls(void) {
  RgpotEnginePot *pot = make();
  pthread_t threads[THREADS + 1];
  Worker workers[THREADS + 1];
  int index;
  if (pot == NULL) {
    ++failures;
    return;
  }
  for (index = 0; index <= THREADS; ++index) {
    workers[index].pot = pot;
    workers[index].bad = 0;
    CHECK(pthread_create(&threads[index], NULL,
                         index == THREADS ? fail_repeatedly
                                          : evaluate_repeatedly,
                         &workers[index]) == 0);
  }
  for (index = 0; index <= THREADS; ++index) {
    CHECK(pthread_join(threads[index], NULL) == 0);
    CHECK(workers[index].bad == 0);
  }
  rgpot_engine_destroy(pot);
}

static void one_handle_serves_concurrent_batch_hvp_and_caps(void) {
  RgpotEnginePot *pot = make();
  pthread_t threads[THREADS];
  Worker workers[THREADS];
  int index;
  void *(*jobs[THREADS])(void *) = {batch_repeatedly, hvp_repeatedly,
                                    caps_repeatedly, evaluate_repeatedly};
  if (pot == NULL) {
    ++failures;
    return;
  }
  for (index = 0; index < THREADS; ++index) {
    workers[index].pot = pot;
    workers[index].bad = 0;
    CHECK(pthread_create(&threads[index], NULL, jobs[index], &workers[index]) ==
          0);
  }
  for (index = 0; index < THREADS; ++index) {
    CHECK(pthread_join(threads[index], NULL) == 0);
    CHECK(workers[index].bad == 0);
  }
  rgpot_engine_destroy(pot);
}

static void put_u32(unsigned char *bytes, size_t offset, uint32_t value) {
  memcpy(bytes + offset, &value, sizeof value);
}

static void put_i32(unsigned char *bytes, size_t offset, int32_t value) {
  memcpy(bytes + offset, &value, sizeof value);
}

static void put_f64(unsigned char *bytes, size_t offset, double value) {
  memcpy(bytes + offset, &value, sizeof value);
}

static int same_geometry(RgpotEnginePot *legacy, RgpotEnginePot *current,
                         const double *positions, const double *direction) {
  double legacy_forces[3];
  double current_forces[3];
  double legacy_energy;
  double current_energy;
  double legacy_product[3];
  double current_product[3];
  int legacy_code;
  int current_code;
  legacy_code =
      rgpot_engine_force(legacy, 1, positions, SPECIES, legacy_forces,
                         &legacy_energy, NULL, BOX, NULL, NULL);
  current_code =
      rgpot_engine_force(current, 1, positions, SPECIES, current_forces,
                         &current_energy, NULL, BOX, NULL, NULL);
  if (legacy_code != RGPOT_ENGINE_OK || current_code != RGPOT_ENGINE_OK ||
      legacy_energy != current_energy ||
      memcmp(legacy_forces, current_forces, sizeof legacy_forces) != 0) {
    return 0;
  }
  legacy_code = rgpot_engine_hvp(legacy, 1, positions, SPECIES, direction,
                                 legacy_product, BOX);
  current_code = rgpot_engine_hvp(current, 1, positions, SPECIES, direction,
                                  current_product, BOX);
  return legacy_code == RGPOT_ENGINE_OK && current_code == RGPOT_ENGINE_OK &&
         memcmp(legacy_product, current_product, sizeof legacy_product) == 0;
}

/* A version-1 client writes 48 bytes: magic, version 1,
 * atomic_number, reserved, then the four parameters. The tip must accept
 * that message and return the same energy, forces and Hessian-vector product
 * as the version-2 message for the same parameters. */
static void v1_client_matches_version_two(void) {
  unsigned char legacy_bytes[48];
  unsigned char oversized_v1[sizeof(RgpotCoupledWellConfigV2)];
  RgpotCoupledWellConfigV2 current_message;
  RgpotEnginePot *legacy;
  RgpotEnginePot *current;
  char error[256];
  const double direction[3] = {1.0, 2.0, 5.0};
  size_t index;
  memset(legacy_bytes, 0, sizeof legacy_bytes);
  put_u32(legacy_bytes, 0, RGPOT_COUPLED_WELL_CONFIG_MAGIC);
  put_u32(legacy_bytes, 4, 1u);
  put_i32(legacy_bytes, 8, NUMBER);
  put_u32(legacy_bytes, 12, 0u);
  put_f64(legacy_bytes, 16, BARRIER);
  put_f64(legacy_bytes, 24, WELL);
  put_f64(legacy_bytes, 32, SPRING);
  put_f64(legacy_bytes, 40, COUPLING);

  memset(error, 0, sizeof error);
  legacy = rgpot_engine_create(legacy_bytes, sizeof legacy_bytes, error,
                               sizeof error);
  CHECK(legacy != NULL);
  current_message = config();
  memset(error, 0, sizeof error);
  current = rgpot_engine_create(&current_message, sizeof current_message, error,
                                sizeof error);
  CHECK(current != NULL);
  if (legacy != NULL && current != NULL) {
    for (index = 0; index < 3; ++index) {
      const double positions[3] = {POINTS[index].x, POINTS[index].y, 7.0};
      CHECK(same_geometry(legacy, current, positions, direction));
    }
  }
  rgpot_engine_destroy(legacy);
  rgpot_engine_destroy(current);

  memset(error, 0, sizeof error);
  CHECK(rgpot_engine_create(legacy_bytes, sizeof legacy_bytes - 1, error,
                            sizeof error) == NULL);
  CHECK(strstr(error, "RgpotCoupledWellConfigV1") != NULL);
  memset(oversized_v1, 0, sizeof oversized_v1);
  memcpy(oversized_v1, legacy_bytes, sizeof legacy_bytes);
  memset(error, 0, sizeof error);
  CHECK(rgpot_engine_create(oversized_v1, sizeof legacy_bytes + 1, error,
                            sizeof error) == NULL);
  CHECK(strstr(error, "RgpotCoupledWellConfigV1") != NULL);
  memset(error, 0, sizeof error);
  CHECK(rgpot_engine_create(oversized_v1, sizeof oversized_v1, error,
                            sizeof error) == NULL);
  CHECK(strstr(error, "RgpotCoupledWellConfigV1") != NULL);

  put_u32(legacy_bytes, 4, 99u);
  memset(error, 0, sizeof error);
  CHECK(rgpot_engine_create(legacy_bytes, sizeof legacy_bytes, error,
                            sizeof error) == NULL);
  CHECK(strcmp(error, "coupled-well config version is unknown") == 0);

  /* Overlapping word at offset 8 is 2 (atomic_number == 2, matching
   * RGPOT_COUPLED_WELL_CONFIG_VERSION at V2's version offset). A word at
   * offset 4 below the version-2 size is still an unknown version. A word
   * at least that size, with version 2 at offset 8, is a short buffer. */
  put_i32(legacy_bytes, 8, 2);
  put_u32(legacy_bytes, 4, 99u);
  memset(error, 0, sizeof error);
  CHECK(rgpot_engine_create(legacy_bytes, sizeof legacy_bytes, error,
                            sizeof error) == NULL);
  CHECK(strcmp(error,
               "coupled-well config buffer is shorter than config.size") == 0);

  put_u32(legacy_bytes, 4, RGPOT_COUPLED_WELL_CONFIG_MIN_SIZE - 1u);
  memset(error, 0, sizeof error);
  CHECK(rgpot_engine_create(legacy_bytes, sizeof legacy_bytes, error,
                            sizeof error) == NULL);
  CHECK(strcmp(error, "coupled-well config version is unknown") == 0);

  put_u32(legacy_bytes, 4, RGPOT_COUPLED_WELL_CONFIG_VERSION);
  memset(error, 0, sizeof error);
  CHECK(rgpot_engine_create(legacy_bytes, sizeof legacy_bytes, error,
                            sizeof error) == NULL);
  CHECK(strcmp(error, "coupled-well config version is unknown") == 0);

  current_message = config();
  current_message.version = RGPOT_COUPLED_WELL_CONFIG_VERSION_V1;
  memset(error, 0, sizeof error);
  CHECK(rgpot_engine_create(&current_message, sizeof current_message, error,
                            sizeof error) == NULL);
  CHECK(strcmp(error, "coupled-well config version is unknown") == 0);

  current_message = config();
  current_message.version = 3u;
  memset(error, 0, sizeof error);
  CHECK(rgpot_engine_create(&current_message, sizeof current_message, error,
                            sizeof error) == NULL);
  CHECK(strcmp(error, "coupled-well config version is unknown") == 0);
}

/* A version-2 message cut to the 48-byte version-1 length still carries a
 * plausible size at offset 4 (56, 152 or 200) and version 2 at offset 8.
 * That buffer is shorter than config.size. A genuine version-1 message still
 * decodes. */
static void truncated_v2_reports_short_buffer(void) {
  unsigned char legacy[48];
  unsigned char truncated[48];
  RgpotCoupledWellConfigV2 full;
  RgpotEnginePot *pot;
  char error[256];
  const uint32_t sizes[3] = {RGPOT_COUPLED_WELL_CONFIG_MIN_SIZE, 152u, 200u};
  size_t index;

  memset(legacy, 0, sizeof legacy);
  put_u32(legacy, 0, RGPOT_COUPLED_WELL_CONFIG_MAGIC);
  put_u32(legacy, 4, RGPOT_COUPLED_WELL_CONFIG_VERSION_V1);
  put_i32(legacy, 8, NUMBER);
  put_u32(legacy, 12, 0u);
  put_f64(legacy, 16, BARRIER);
  put_f64(legacy, 24, WELL);
  put_f64(legacy, 32, SPRING);
  put_f64(legacy, 40, COUPLING);
  memset(error, 0, sizeof error);
  pot = rgpot_engine_create(legacy, sizeof legacy, error, sizeof error);
  CHECK(pot != NULL);
  rgpot_engine_destroy(pot);

  full = config();
  for (index = 0; index < 3; ++index) {
    full.size = sizes[index];
    memset(truncated, 0, sizeof truncated);
    memcpy(truncated, &full, sizeof truncated);
    memset(error, 0, sizeof error);
    CHECK(rgpot_engine_create(truncated, sizeof truncated, error,
                              sizeof error) == NULL);
    CHECK(strcmp(error,
                 "coupled-well config buffer is shorter than config.size") ==
          0);
  }
}

static void hvp_rejects_overlapping_buffers(void) {
  RgpotEnginePot *pot = make();
  const double positions[3] = {1.0, 0.5, 0.0};
  const double direction[3] = {1.0, 2.0, 0.0};
  const double expected[3] = {-4.25, 6.0, 0.0};
  double aliased[3] = {1.0, 2.0, 0.0};
  double window[4] = {1.0, 2.0, 0.0, 7.5};
  double pos_alias[3] = {1.0, 0.5, 0.0};
  double box_alias[9];
  double adjacent[6] = {1.0, 2.0, 0.0, -9.0, -9.0, -9.0};
  union {
    int species[1];
    double product[3];
    unsigned char raw[3 * sizeof(double)];
  } shared;
  unsigned char saved_shared[3 * sizeof(double)];
  char error[256];
  size_t index;
  if (pot == NULL) {
    ++failures;
    return;
  }

  /* Exact alias: product == direction. */
  CHECK(rgpot_engine_hvp(pot, 1, positions, SPECIES, aliased, aliased, BOX) ==
        RGPOT_ENGINE_ERR_INVALID);
  rgpot_engine_last_error(pot, error, sizeof error);
  CHECK(strstr(error, "overlap") != NULL);
  CHECK(memcmp(aliased, direction, sizeof aliased) == 0);

  /* Partial overlap: direction[0..2] vs product[1..3] and reverse. */
  CHECK(rgpot_engine_hvp(pot, 1, positions, SPECIES, window, window + 1, BOX) ==
        RGPOT_ENGINE_ERR_INVALID);
  rgpot_engine_last_error(pot, error, sizeof error);
  CHECK(strstr(error, "overlap") != NULL);
  CHECK(window[0] == 1.0 && window[1] == 2.0 && window[2] == 0.0 &&
        window[3] == 7.5);

  CHECK(rgpot_engine_hvp(pot, 1, positions, SPECIES, window + 1, window, BOX) ==
        RGPOT_ENGINE_ERR_INVALID);
  rgpot_engine_last_error(pot, error, sizeof error);
  CHECK(strstr(error, "overlap") != NULL);
  CHECK(window[0] == 1.0 && window[1] == 2.0 && window[2] == 0.0 &&
        window[3] == 7.5);

  /* Overlap with positions. */
  CHECK(rgpot_engine_hvp(pot, 1, pos_alias, SPECIES, direction, pos_alias,
                         BOX) == RGPOT_ENGINE_ERR_INVALID);
  rgpot_engine_last_error(pot, error, sizeof error);
  CHECK(strstr(error, "overlap") != NULL);
  CHECK(memcmp(pos_alias, positions, sizeof pos_alias) == 0);

  /* Overlap with box. */
  memcpy(box_alias, BOX, sizeof box_alias);
  CHECK(rgpot_engine_hvp(pot, 1, positions, SPECIES, direction, box_alias + 2,
                         box_alias) == RGPOT_ENGINE_ERR_INVALID);
  rgpot_engine_last_error(pot, error, sizeof error);
  CHECK(strstr(error, "overlap") != NULL);
  CHECK(memcmp(box_alias, BOX, sizeof box_alias) == 0);

  /* Overlap with atomicNumbers. */
  memset(&shared, 0x5a, sizeof shared);
  shared.species[0] = NUMBER;
  memcpy(saved_shared, shared.raw, sizeof saved_shared);
  CHECK(rgpot_engine_hvp(pot, 1, positions, shared.species, direction,
                         shared.product, BOX) == RGPOT_ENGINE_ERR_INVALID);
  rgpot_engine_last_error(pot, error, sizeof error);
  CHECK(strstr(error, "overlap") != NULL);
  CHECK(memcmp(shared.raw, saved_shared, sizeof saved_shared) == 0);

  /* Adjacent, non-overlapping buffers succeed in both orders. */
  CHECK(rgpot_engine_hvp(pot, 1, positions, SPECIES, adjacent, adjacent + 3,
                         BOX) == RGPOT_ENGINE_OK);
  for (index = 0; index < 3; ++index) {
    CHECK(adjacent[index] == direction[index]);
    CHECK(adjacent[3 + index] == expected[index]);
  }
  adjacent[0] = -9.0;
  adjacent[1] = -9.0;
  adjacent[2] = -9.0;
  adjacent[3] = direction[0];
  adjacent[4] = direction[1];
  adjacent[5] = direction[2];
  CHECK(rgpot_engine_hvp(pot, 1, positions, SPECIES, adjacent + 3, adjacent,
                         BOX) == RGPOT_ENGINE_OK);
  for (index = 0; index < 3; ++index) {
    CHECK(adjacent[index] == expected[index]);
    CHECK(adjacent[3 + index] == direction[index]);
  }

  rgpot_engine_destroy(pot);
}

int main(void) {
  identity_and_contract();
  create_validation();
  v1_client_matches_version_two();
  truncated_v2_reports_short_buffer();
  analytic_force_and_hessian();
  transform_is_scratch_only();
  rejects_bad_requests();
  hvp_rejects_overlapping_buffers();
  batch_matches_the_analytic_values();
  batch_failure_writes_nothing();
  caps_and_last_error();
  one_handle_serves_concurrent_calls();
  one_handle_serves_concurrent_batch_hvp_and_caps();
  if (failures != 0) {
    fprintf(stderr, "%d coupled-well engine checks failed\n", failures);
    return 1;
  }
  return 0;
}
