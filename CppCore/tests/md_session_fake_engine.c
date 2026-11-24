/* MIT License
 * Copyright 2023--present rgpot developers
 *
 * Fake MD session plugin for testing rgpot_md_session_ops capacity
 * negotiation between older/newer hosts and older/newer engines.
 */

#include "rgpot/ExternalPot/external_potentials.h"
#include "rgpot/engine_c_abi.h"

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct FakeCtx {
  size_t seen_capacity;
  size_t reported_filled;
  double box[9];
} FakeCtx;

typedef struct FakeNewerOps {
  rgpot_md_session_ops base;
  void (*future_cb)(void *);
  uint64_t future_tag;
} FakeNewerOps;

static void set_err(char *err, size_t err_len, const char *msg) {
  if (err != NULL && err_len > 0) {
    snprintf(err, err_len, "%s", msg);
  }
}

static int fake_set_positions(void *ctx, const double *positions, size_t n3,
                              char *err, size_t err_len) {
  (void)ctx;
  (void)positions;
  (void)n3;
  (void)err;
  (void)err_len;
  return 0;
}

static int fake_set_velocities(void *ctx, const double *velocities, size_t n3,
                               char *err, size_t err_len) {
  (void)ctx;
  (void)velocities;
  (void)n3;
  (void)err;
  (void)err_len;
  return 0;
}

static int fake_step_langevin(void *ctx, double kT, double friction,
                              double time_step, const double *plane_point,
                              const double *plane_normal,
                              const double *wall_points,
                              const double *wall_normals, size_t n_walls,
                              size_t n3, size_t steps, uint64_t seed,
                              uint64_t *wall_hits, char *err, size_t err_len) {
  (void)ctx;
  (void)kT;
  (void)friction;
  (void)time_step;
  (void)plane_point;
  (void)plane_normal;
  (void)wall_points;
  (void)wall_normals;
  (void)n_walls;
  (void)n3;
  (void)steps;
  (void)seed;
  (void)err;
  (void)err_len;
  if (wall_hits != NULL) {
    *wall_hits = 0;
  }
  return 0;
}

static int fake_step_nve(void *ctx, double time_step, size_t steps, char *err,
                         size_t err_len) {
  (void)ctx;
  (void)time_step;
  (void)steps;
  (void)err;
  (void)err_len;
  return 0;
}

static int fake_get_state(void *ctx, double *positions, double *velocities,
                          double *forces, size_t n3, double *potential_energy,
                          double *kinetic_energy, char *err, size_t err_len) {
  (void)ctx;
  (void)err;
  (void)err_len;
  if (positions != NULL) {
    memset(positions, 0, n3 * sizeof(double));
  }
  if (velocities != NULL) {
    memset(velocities, 0, n3 * sizeof(double));
  }
  if (forces != NULL) {
    memset(forces, 0, n3 * sizeof(double));
  }
  if (potential_energy != NULL) {
    *potential_energy = 0.0;
  }
  if (kinetic_energy != NULL) {
    *kinetic_energy = 0.0;
  }
  return 0;
}

static int fake_get_masses(void *ctx, double *masses, size_t n_atoms, char *err,
                           size_t err_len) {
  const FakeCtx *state = (const FakeCtx *)ctx;
  if (masses == NULL || n_atoms < 2) {
    set_err(err, err_len, "expected two mass slots");
    return 1;
  }
  masses[0] = (double)state->seen_capacity;
  masses[1] = (double)state->reported_filled;
  return 0;
}

static void fake_destroy(void *ctx) { free(ctx); }

static int fake_set_periodic_box(void *ctx, const double *box9, char *err,
                                 size_t err_len) {
  FakeCtx *state = (FakeCtx *)ctx;
  (void)err;
  (void)err_len;
  memcpy(state->box, box9, sizeof(state->box));
  return 0;
}

static int fake_get_periodic_box(void *ctx, double *box9, char *err,
                                 size_t err_len) {
  const FakeCtx *state = (const FakeCtx *)ctx;
  (void)err;
  (void)err_len;
  memcpy(box9, state->box, sizeof(state->box));
  return 0;
}

static int fake_initialize(void *ctx, char *err, size_t err_len) {
  (void)ctx;
  (void)err;
  (void)err_len;
  return 0;
}

static int fake_run_langevin(void *ctx,
                             const rgpot_md_langevin_request *request,
                             rgpot_md_langevin_result *result, char *err,
                             size_t err_len) {
  (void)ctx;
  (void)request;
  (void)err;
  (void)err_len;
  if (result != NULL) {
    result->wall_hits = 0;
  }
  return 0;
}

static void fake_future_cb(void *ctx) { (void)ctx; }

static size_t min_size(size_t a, size_t b) { return a < b ? a : b; }

int rgpot_md_session_engine_create(const void *spec, size_t spec_len,
                                   const char *options,
                                   rgpot_md_session_ops *ops, void **ctx,
                                   size_t *n_atoms, char *err,
                                   size_t err_len) {
  (void)spec;
  (void)spec_len;
  if (ops == NULL || ctx == NULL || n_atoms == NULL) {
    set_err(err, err_len, "null output pointers");
    return 1;
  }
  const size_t capacity = ops->struct_size;
  if (capacity < RGPOT_MD_OPS_END(destroy)) {
    set_err(err, err_len, "ops struct_size capacity is too small");
    return 1;
  }

  FakeCtx *state = (FakeCtx *)calloc(1, sizeof(FakeCtx));
  if (state == NULL) {
    set_err(err, err_len, "out of memory");
    return 1;
  }
  state->seen_capacity = capacity;

  const char *mode = (options != NULL && options[0] != '\0') ? options : "current";
  size_t engine_size = sizeof(rgpot_md_session_ops);
  if (strcmp(mode, "older") == 0) {
    engine_size = RGPOT_MD_OPS_END(destroy);
  } else if (strcmp(mode, "newer") == 0) {
    engine_size = sizeof(FakeNewerOps);
  }

  FakeNewerOps full;
  memset(&full, 0, sizeof(full));
  size_t filled = min_size(capacity, engine_size);
  full.base.struct_size = filled;
  full.base.set_positions = fake_set_positions;
  full.base.set_velocities = fake_set_velocities;
  full.base.step_langevin = fake_step_langevin;
  full.base.step_nve = fake_step_nve;
  full.base.get_state = fake_get_state;
  full.base.get_masses = fake_get_masses;
  full.base.destroy = fake_destroy;

  if (filled >= RGPOT_MD_OPS_END(set_periodic_box)) {
    full.base.set_periodic_box = fake_set_periodic_box;
  }
  if (filled >= RGPOT_MD_OPS_END(get_periodic_box)) {
    full.base.get_periodic_box = fake_get_periodic_box;
  }
  if (filled >= RGPOT_MD_OPS_END(initialize)) {
    full.base.initialize = fake_initialize;
  }
  if (filled >= RGPOT_MD_OPS_END(run_langevin)) {
    full.base.run_langevin = fake_run_langevin;
  }
  if (filled >= offsetof(FakeNewerOps, future_cb) + sizeof(full.future_cb)) {
    full.future_cb = fake_future_cb;
  }
  if (filled >= offsetof(FakeNewerOps, future_tag) + sizeof(full.future_tag)) {
    full.future_tag = UINT64_C(0x123456789abcdef0);
  }

  memcpy(ops, &full, filled);
  if (strcmp(mode, "oversized") == 0) {
    /* Deliberately report a larger size than the host buffer to test clamping;
     * do not write past capacity bytes. */
    ops->struct_size = sizeof(rgpot_md_session_ops) + 64;
  }
  state->reported_filled = ops->struct_size;
  *ctx = state;
  *n_atoms = 2;
  return 0;
}

int rgpot_engine_abi_version(void) { return RGPOT_ENGINE_ABI_VERSION; }

RgpotEngineAbiStamp rgpot_engine_abi_stamp(void) {
  RgpotEngineAbiStamp stamp;
  stamp.abi_major = RGPOT_ENGINE_ABI_VERSION;
  stamp.abi_minor = RGPOT_ENGINE_ABI_MINOR;
  stamp.layout_revision = RGPOT_ENGINE_ABI_LAYOUT_REVISION;
  return stamp;
}

#ifdef FIXTURE_UNLOAD_POLICY
int rgpot_engine_unload_policy(void) { return FIXTURE_UNLOAD_POLICY; }
#endif
