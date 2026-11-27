/* MIT License
 * Copyright 2023--present rgpot developers
 *
 * A free-particle MD session plugin for tests/md_session.rs. Two atoms of
 * unit mass drift at constant velocity; neighbour queries report the one
 * pair. The options string selects a defect in what
 * rgpot_md_session_engine_create returns:
 *   "short_size"    struct_size ends before destroy
 *   "null_callback" get_state is null
 *   "zero_atoms"    the session reports no atoms
 *   "refuse"        creation fails with a message
 * md_fixture_destroyed counts the contexts released through destroy. */
#include "rgpot/ExternalPot/external_potentials.h"

#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#define FIXTURE_ATOMS 2

struct fixture_ctx {
  double x[3 * FIXTURE_ATOMS];
  double v[3 * FIXTURE_ATOMS];
};

static int destroyed = 0;

int md_fixture_destroyed(void) { return destroyed; }

static void write_err(char *err, size_t err_len, const char *message) {
  if (err == NULL || err_len == 0) {
    return;
  }
  strncpy(err, message, err_len - 1);
  err[err_len - 1] = '\0';
}

static int check_n3(size_t n3, char *err, size_t err_len) {
  if (n3 != 3 * FIXTURE_ATOMS) {
    write_err(err, err_len, "fixture expects six coordinates");
    return 1;
  }
  return 0;
}

static int set_positions(void *ctx, const double *positions, size_t n3,
                         char *err, size_t err_len) {
  if (check_n3(n3, err, err_len) != 0) {
    return 1;
  }
  memcpy(((struct fixture_ctx *)ctx)->x, positions, n3 * sizeof(double));
  return 0;
}

static int set_velocities(void *ctx, const double *velocities, size_t n3,
                          char *err, size_t err_len) {
  if (check_n3(n3, err, err_len) != 0) {
    return 1;
  }
  memcpy(((struct fixture_ctx *)ctx)->v, velocities, n3 * sizeof(double));
  return 0;
}

static int step_langevin(void *ctx, double kT, double friction,
                         double time_step, const double *plane_point,
                         const double *plane_normal, const double *wall_points,
                         const double *wall_normals, size_t n_walls, size_t n3,
                         size_t steps, uint64_t seed, uint64_t *wall_hits,
                         char *err, size_t err_len) {
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
  (void)wall_hits;
  write_err(err, err_len, "fixture has no Langevin dynamics");
  return 1;
}

static int step_nve(void *ctx, double time_step, size_t steps, char *err,
                    size_t err_len) {
  (void)err;
  (void)err_len;
  struct fixture_ctx *state = ctx;
  for (size_t k = 0; k < 3 * FIXTURE_ATOMS; ++k) {
    state->x[k] += time_step * (double)steps * state->v[k];
  }
  return 0;
}

static int get_state(void *ctx, double *positions, double *velocities,
                     double *forces, size_t n3, double *potential_energy,
                     double *kinetic_energy, char *err, size_t err_len) {
  if (check_n3(n3, err, err_len) != 0) {
    return 1;
  }
  const struct fixture_ctx *state = ctx;
  double kinetic = 0.0;
  for (size_t k = 0; k < n3; ++k) {
    if (positions != NULL) {
      positions[k] = state->x[k];
    }
    if (velocities != NULL) {
      velocities[k] = state->v[k];
    }
    if (forces != NULL) {
      forces[k] = 0.0;
    }
    kinetic += 0.5 * state->v[k] * state->v[k];
  }
  if (potential_energy != NULL) {
    *potential_energy = 0.0;
  }
  if (kinetic_energy != NULL) {
    *kinetic_energy = kinetic;
  }
  return 0;
}

static int get_masses(void *ctx, double *masses, size_t n_atoms, char *err,
                      size_t err_len) {
  (void)ctx;
  if (n_atoms != FIXTURE_ATOMS) {
    write_err(err, err_len, "fixture expects two atoms");
    return 1;
  }
  for (size_t i = 0; i < n_atoms; ++i) {
    masses[i] = 1.0;
  }
  return 0;
}

static void destroy(void *ctx) {
  free(ctx);
  ++destroyed;
}

static int get_neighbors(void *ctx, const rgpot_nlist_request *request,
                         rgpot_nlist_soa *out, size_t *count, size_t *needed,
                         char *err, size_t err_len) {
  (void)err;
  (void)err_len;
  const struct fixture_ctx *state = ctx;
  *count = 0;
  *needed = 1;
  if (out == NULL || out->cap == 0) {
    return RGPOT_MD_NLIST_OK;
  }
  out->first[0] = 0;
  out->second[0] = 1;
  if (request->want_distances && out->distance != NULL) {
    double sum = 0.0;
    for (int axis = 0; axis < 3; ++axis) {
      const double d = state->x[3 + axis] - state->x[axis];
      sum += d * d;
    }
    out->distance[0] = sqrt(sum);
  }
  *count = 1;
  return RGPOT_MD_NLIST_OK;
}

int rgpot_md_session_engine_create(const void *spec, size_t spec_len,
                                   const char *options,
                                   rgpot_md_session_ops *ops, void **ctx,
                                   size_t *n_atoms, char *err,
                                   size_t err_len) {
  (void)spec;
  (void)spec_len;
  const char *mode = options != NULL ? options : "";
  if (strcmp(mode, "refuse") == 0) {
    write_err(err, err_len, "fixture refused the session");
    return 1;
  }
  struct fixture_ctx *state = calloc(1, sizeof(*state));
  if (state == NULL) {
    write_err(err, err_len, "fixture is out of memory");
    return 1;
  }
  ops->struct_size = sizeof(*ops);
  ops->set_positions = set_positions;
  ops->set_velocities = set_velocities;
  ops->step_langevin = step_langevin;
  ops->step_nve = step_nve;
  ops->get_state = get_state;
  ops->get_masses = get_masses;
  ops->destroy = destroy;
  ops->get_neighbors = get_neighbors;
  ops->get_source_indices = NULL;
  if (strcmp(mode, "short_size") == 0) {
    ops->struct_size = offsetof(rgpot_md_session_ops, destroy);
  } else if (strcmp(mode, "null_callback") == 0) {
    ops->get_state = NULL;
  }
  *n_atoms = strcmp(mode, "zero_atoms") == 0 ? 0 : FIXTURE_ATOMS;
  *ctx = state;
  return 0;
}
