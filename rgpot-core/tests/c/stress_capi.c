/* MIT License
 * Copyright 2023--present rgpot developers
 *
 * The stress entry points as a C host calls them, and the proof that a host
 * built against C ABI minor 6 keeps working.
 *
 * The minor-6 structs are frozen below as that header spelled them. A call
 * through them must evaluate, must leave their canary words alone, and the
 * header of this build must still give them the same size and offsets.
 */
/* rgpot.h names eindir_objective_t; the first header defines it for C hosts. */
/* clang-format off */
#include "rgpot/eindir_abi.hpp"
#include "rgpot.h"
/* clang-format on */

#include <stddef.h>
#include <stdio.h>
#include <string.h>

#if RGPOT_ABI_VERSION_MAJOR != 1 || RGPOT_ABI_VERSION_MINOR < 7
#error "the stress entry points need C ABI 1.7 or newer"
#endif

typedef struct {
  DLManagedTensorVersioned *positions;
  DLManagedTensorVersioned *atomic_numbers;
  DLManagedTensorVersioned *box_matrix;
} Minor6Input;

typedef struct {
  DLManagedTensorVersioned *forces;
  double energy;
  double variance;
} Minor6Out;

#define CANARY 0xA5A55A5ADEADBEEFULL
#define PADDING 16

typedef struct {
  Minor6Input input;
  unsigned long long after[PADDING];
} GuardedInput;

typedef struct {
  Minor6Out out;
  unsigned long long after[PADDING];
} GuardedOut;

static int failures = 0;

#define CHECK(cond, ...)                                                       \
  do {                                                                         \
    if (!(cond)) {                                                             \
      fprintf(stderr, "%s:%d: ", __FILE__, __LINE__);                          \
      fprintf(stderr, __VA_ARGS__);                                            \
      fprintf(stderr, "\n");                                                   \
      ++failures;                                                              \
    }                                                                          \
  } while (0)

static int forces_calls = 0;
static int stress_calls = 0;

static void write_forces(const rgpot_force_input_t *input,
                         rgpot_force_out_t *output) {
  const int64_t n = 2;
  double forces[6] = {0.5, -0.5, 0.25, -0.25, 0.125, -0.125};
  (void)input;
  output->energy = -1.75;
  output->variance = 0.0;
  output->forces = rgpot_tensor_owned_cpu_f64_2d(forces, n, 3);
}

static rgpot_status_t force_callback(void *user_data,
                                     const rgpot_force_input_t *input,
                                     rgpot_force_out_t *output) {
  (void)user_data;
  ++forces_calls;
  write_forces(input, output);
  return RGPOT_SUCCESS;
}

static rgpot_status_t stress_callback(void *user_data,
                                      const rgpot_force_input_t *input,
                                      rgpot_force_out_t *output,
                                      rgpot_stress_out_t *stress) {
  int i;
  (void)user_data;
  ++stress_calls;
  write_forces(input, output);
  for (i = 0; i < 9; ++i) {
    stress->stress[i] = 0.5 * (double)(i + 1);
  }
  stress->has_stress = 1;
  return RGPOT_SUCCESS;
}

static void check_minor_6_layout(void) {
  CHECK(sizeof(rgpot_force_input_t) == sizeof(Minor6Input),
        "rgpot_force_input_t is %zu bytes, minor 6 had %zu",
        sizeof(rgpot_force_input_t), sizeof(Minor6Input));
  CHECK(sizeof(rgpot_force_out_t) == sizeof(Minor6Out),
        "rgpot_force_out_t is %zu bytes, minor 6 had %zu",
        sizeof(rgpot_force_out_t), sizeof(Minor6Out));
  CHECK(offsetof(rgpot_force_input_t, positions) ==
            offsetof(Minor6Input, positions),
        "positions moved");
  CHECK(offsetof(rgpot_force_input_t, atomic_numbers) ==
            offsetof(Minor6Input, atomic_numbers),
        "atomic_numbers moved");
  CHECK(offsetof(rgpot_force_input_t, box_matrix) ==
            offsetof(Minor6Input, box_matrix),
        "box_matrix moved");
  CHECK(offsetof(rgpot_force_out_t, forces) == offsetof(Minor6Out, forces),
        "forces moved");
  CHECK(offsetof(rgpot_force_out_t, energy) == offsetof(Minor6Out, energy),
        "energy moved");
  CHECK(offsetof(rgpot_force_out_t, variance) == offsetof(Minor6Out, variance),
        "variance moved");
}

static void check_minor_6_caller_is_accepted(void) {
  rgpot_abi_stamp_t stamp;
  stamp.abi_major = 1;
  stamp.abi_minor = 6;
  stamp.layout_revision = 1;
  CHECK(rgpot_abi_compatible(&stamp) != 0,
        "a caller built against minor 6 layout 1 is refused");
  stamp.abi_minor = RGPOT_ABI_VERSION_MINOR;
  CHECK(rgpot_abi_compatible(&stamp) != 0, "the current stamp is refused");
  stamp.abi_minor = RGPOT_ABI_VERSION_MINOR + 1;
  CHECK(rgpot_abi_compatible(&stamp) == 0, "a newer minor is accepted");
  CHECK(RGPOT_ABI_LAYOUT_REVISION == 1,
        "the layout revision moved to %d, minor-6 callers would be refused",
        RGPOT_ABI_LAYOUT_REVISION);
}

int main(void) {
  double positions[6] = {0.0, 0.0, 0.0, 2.0, 0.0, 0.0};
  int numbers[2] = {29, 29};
  double cell[9] = {9.0, 0.0, 0.0, 0.0, 9.0, 0.0, 0.0, 0.0, 9.0};
  rgpot_force_input_t built;
  GuardedInput guarded_input;
  GuardedOut guarded_out;
  rgpot_potential_t *plain;
  rgpot_potential_t *capable;
  rgpot_stress_out_t block;
  rgpot_force_out_t out;
  rgpot_status_t status;
  const double *forces;
  int i;

  check_minor_6_layout();
  check_minor_6_caller_is_accepted();

  built = rgpot_force_input_create(2, positions, numbers, cell);
  memcpy(&guarded_input.input, &built, sizeof built);
  for (i = 0; i < PADDING; ++i) {
    guarded_input.after[i] = CANARY;
    guarded_out.after[i] = CANARY;
  }
  memset(&guarded_out.out, 0, sizeof guarded_out.out);

  plain = rgpot_potential_new(force_callback, NULL, NULL);
  capable = rgpot_potential_new(force_callback, NULL, NULL);
  CHECK(plain != NULL && capable != NULL, "constructor failed");
  CHECK(rgpot_potential_supports_stress(plain) == 0,
        "a potential with no stress callback advertises stress");
  CHECK(rgpot_potential_set_stress_callback(capable, stress_callback) ==
            RGPOT_SUCCESS,
        "registering the stress callback failed");
  CHECK(rgpot_potential_supports_stress(capable) != 0,
        "a potential with a stress callback does not advertise it");

  /* A minor-6 caller: its structs, the original entry point. */
  status = rgpot_potential_calculate(
      capable, (const rgpot_force_input_t *)&guarded_input.input,
      (rgpot_force_out_t *)&guarded_out.out);
  CHECK(status == RGPOT_SUCCESS, "minor-6 evaluation failed: %s",
        rgpot_last_error());
  CHECK(guarded_out.out.energy == -1.75, "minor-6 energy %g",
        guarded_out.out.energy);
  CHECK(guarded_out.out.forces != NULL, "minor-6 forces missing");
  if (guarded_out.out.forces != NULL) {
    forces = (const double *)rgpot_tensor_data(guarded_out.out.forces);
    CHECK(forces[0] == 0.5 && forces[5] == -0.125, "minor-6 forces differ");
    rgpot_tensor_free(guarded_out.out.forces);
  }
  for (i = 0; i < PADDING; ++i) {
    CHECK(guarded_out.after[i] == CANARY, "output canary %d overwritten", i);
    CHECK(guarded_input.after[i] == CANARY, "input canary %d overwritten", i);
  }
  CHECK(forces_calls == 1 && stress_calls == 0,
        "a force-only call ran the stress callback (%d force, %d stress)",
        forces_calls, stress_calls);

  /* The stress entry point. */
  memset(&out, 0, sizeof out);
  block = rgpot_stress_out_create();
  CHECK(block.struct_size == sizeof(rgpot_stress_out_t),
        "struct_size is %u, sizeof is %zu", block.struct_size,
        sizeof(rgpot_stress_out_t));
  status = rgpot_potential_calculate_stress(capable, &built, &out, &block);
  CHECK(status == RGPOT_SUCCESS, "stress call failed: %s", rgpot_last_error());
  CHECK(block.has_stress == 1, "has_stress is %d", block.has_stress);
  CHECK(block.stress[0] == 0.5 && block.stress[8] == 4.5, "stress differs");
  CHECK(stress_calls == 1, "stress callback ran %d times", stress_calls);
  rgpot_tensor_free(out.forces);

  /* A potential with no stress callback refuses, with a message, and the
     block stays zero. */
  memset(&out, 0, sizeof out);
  block = rgpot_stress_out_create();
  status = rgpot_potential_calculate_stress(plain, &built, &out, &block);
  CHECK(status == RGPOT_INVALID_PARAMETER, "status %d for a missing capability",
        (int)status);
  CHECK(strstr(rgpot_last_error(), "stress") != NULL,
        "message does not name the capability: %s", rgpot_last_error());
  CHECK(block.has_stress == 0, "has_stress set on a refused call");
  for (i = 0; i < 9; ++i) {
    CHECK(block.stress[i] == 0.0, "stress[%d] is not zero after a refusal", i);
  }

  /* A caller that declares a smaller block is refused before any write. */
  memset(&out, 0, sizeof out);
  block = rgpot_stress_out_create();
  block.struct_size = 8;
  block.stress[0] = 7.0;
  status = rgpot_potential_calculate_stress(capable, &built, &out, &block);
  CHECK(status == RGPOT_INVALID_PARAMETER, "undersized block status %d",
        (int)status);
  CHECK(block.stress[0] == 7.0 && block.has_stress == 0,
        "an undersized block was written");
  CHECK(stress_calls == 1, "the stress callback ran for an undersized block");

  rgpot_potential_free(plain);
  rgpot_potential_free(capable);
  rgpot_force_input_free(&built);

  if (failures != 0) {
    fprintf(stderr, "%d check(s) failed\n", failures);
    return 1;
  }
  printf("stress C ABI: all checks passed\n");
  return 0;
}
