/*
 * Engine-generic molecular dynamics session C ABI.
 *
 * SPDX-License-Identifier: MIT
 * Copyright 2023--present rgpot developers
 *
 * An engine library (nblib today, OpenMM next) exports the calls below under
 * its own symbol prefix, <prefix>_md_<call>: rgpot_nblib_md_step_nve,
 * rgpot_openmm_md_step_nve and so on. The shape is the one of the abi.*
 * probes (<prefix>_md_abi_version, _available, _features, _last_error), so a
 * host dlopens the library, checks the version, reads the feature bits and
 * resolves the calls it needs by name; a call an engine does not implement
 * is simply absent. Nothing in this header is specific to one engine:
 * RGPOT_MD_DECLARE_API(prefix, api) declares the whole surface for an
 * engine, and a host that loads libraries at run time resolves the same
 * names with the prefix as data.
 *
 * What a session is. It owns one system (the engine decides what the creation
 * string means: a topology text for nblib, a serialized system for OpenMM),
 * the positions, the velocities, a periodic box and an integrator. The caller
 * sets the box, positions and velocities, steps, and reads the state back.
 *
 * Units at this boundary are Angstrom, picosecond, eV and atomic mass unit,
 * so velocities are Angstrom/ps and forces eV/Angstrom, whatever the engine
 * uses inside. A coordinate vector has dimension = 3 * atoms, atom-major.
 * A box is nine doubles, lattice vectors as rows. Positions are kept as
 * given and never folded into the box, so a trajectory is continuous across
 * the periodic faces.
 *
 * Velocities are v(t) at the time of the positions. An engine whose
 * integrator keeps them half a step away (leap-frog) converts at this
 * boundary, so a plain step is the velocity Verlet map and set_velocities
 * with -v reverses a trajectory.
 *
 * Every call that can fail returns 0, or nonzero with the reason in
 * <prefix>_md_last_error for the calling thread. A session is used by one
 * thread at a time; distinct sessions may run concurrently.
 *
 * Constrained, thermostatted dynamics. step_projected runs Langevin dynamics
 * restricted to a hyperplane n . (x - point) = 0 of the Angstrom positions,
 * with optional planar hard walls n_w . x <= offset_w:
 *   - positions are projected onto the plane and velocities onto its tangent
 *     space in the mass-weighted metric, so the dynamics is that of a
 *     holonomic constraint;
 *   - the forces are projected the same way (the constraint force removes
 *     the normal acceleration);
 *   - once per step the velocities relax toward the Maxwell distribution at
 *     kt_ev with friction friction_per_ps (an Ornstein-Uhlenbeck update with
 *     noise drawn per coordinate, projected onto the tangent space), so
 *     friction 0 is plain constrained NVE;
 *   - a step that ends outside a wall is undone: positions return to those
 *     before it and the velocity component along the wall's outward normal
 *     (taken in the tangent space of the plane) is reflected in the
 *     mass-weighted metric. The reflection conserves the kinetic energy.
 * Every normal is a gradient with respect to the Angstrom positions and need
 * not be a unit vector.
 */

#ifndef RGPOT_MD_SESSION_C_ABI_H
#define RGPOT_MD_SESSION_C_ABI_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* The major changes when a call an old host makes stops working. The minor
 * changes when a call or a trailing struct field is added. */
#define RGPOT_MD_ABI_VERSION 1
#define RGPOT_MD_ABI_MINOR 0

/* Bits of <prefix>_md_features. A host may also probe the symbol itself. */
/* step_nve, state, set_positions and set_velocities. Every engine has them. */
#define RGPOT_MD_FEATURE_NVE 1u
/* step_projected with a plane, a thermostat and walls. */
#define RGPOT_MD_FEATURE_PROJECTED 2u
/* set_hook: a per-step callback that may edit positions, velocities and
 * forces. */
#define RGPOT_MD_FEATURE_HOOK 4u
/* set_boundary: a hard-boundary callback. */
#define RGPOT_MD_FEATURE_BOUNDARY 8u
/* state reports the virial. */
#define RGPOT_MD_FEATURE_VIRIAL 16u

typedef struct rgpot_md_session rgpot_md_session;

typedef struct rgpot_md_config {
  uint32_t bytes; /* sizeof(rgpot_md_config) as the caller compiled it */
  double time_step_ps;
} rgpot_md_config;

typedef struct rgpot_md_info {
  uint32_t bytes; /* sizeof(rgpot_md_info) as the caller compiled it */
  double potential_ev;
  /* Kinetic energy of the velocities state returns. */
  double kinetic_ev;
  double time_ps;
  uint64_t steps;
  uint64_t force_evaluations;
  /* Xi = -1/2 sum r (x) F in eV, row-major, of the last force evaluation;
   * zero when the engine has no RGPOT_MD_FEATURE_VIRIAL. */
  double virial_ev[9];
  double box[9]; /* Angstrom */
} rgpot_md_info;

/* Langevin thermostat of step_projected. */
typedef struct rgpot_md_thermostat {
  uint32_t bytes;
  double kt_ev;
  double friction_per_ps;
} rgpot_md_thermostat;

/* The hyperplane n . (x - point) = 0. point and normal each hold
 * dimension doubles. */
typedef struct rgpot_md_plane {
  uint32_t bytes;
  size_t dimension;
  const double *point;
  const double *normal;
} rgpot_md_plane;

/* The allowed side of a planar hard wall: normal . x <= offset, normal of
 * dimension doubles, offset in Angstrom times the length of the normal. */
typedef struct rgpot_md_wall {
  const double *normal;
  double offset;
} rgpot_md_wall;

enum {
  /* After the position update and before the forces of the new positions.
   * positions and velocities may be edited, forces is read-only and holds the
   * forces of the step just taken. velocities are the ones half a step
   * before the new positions. */
  RGPOT_MD_PHASE_POSITIONS = 1,
  /* After the forces of the new positions. velocities are v(t). positions are
   * read-only; velocities and forces may be edited, and the edited forces
   * drive the next step. */
  RGPOT_MD_PHASE_FORCES = 2
};

/* masses holds one amu per coordinate. A nonzero return stops the step and
 * becomes the error of the stepping call. */
typedef int (*rgpot_md_step_hook)(void *user, int phase, uint64_t step,
                                  size_t dimension, double *positions,
                                  double *velocities, double *forces,
                                  const double *masses);

/* Hard boundary. Called with the positions a step ends on. Returns 0 when
 * they are inside the allowed region. Otherwise the step is undone, the
 * positions return to those before it, and the velocities are reflected in
 * the mass-weighted metric about the face whose outward normal the callback
 * wrote. A callback that leaves outward_normal all zero has the velocities
 * reversed instead. */
typedef int (*rgpot_md_boundary_fn)(void *user, size_t dimension,
                                    const double *positions,
                                    double *outward_normal);

/* Declares the surface for the engine whose symbols start with prefix, each
 * with the export attribute api. */
#define RGPOT_MD_DECLARE_API(prefix, api)                                      \
  /* Probes. */                                                                \
  api int prefix##_md_abi_version(void);                                       \
  api int prefix##_md_abi_minor(void);                                         \
  /* 1 when the engine behind the library is present; a stub build returns 0 \
   * and every constructor fails with the reason. */                           \
  api int prefix##_md_available(void);                                         \
  api unsigned prefix##_md_features(void);                                     \
  /* A short name: "nblib", "openmm". */                                       \
  api const char *prefix##_md_engine(void);                                    \
  /* The integrator a plain step uses: "leap-frog", "velocity-verlet". */      \
  api const char *prefix##_md_integrator(void);                                \
  api const char *prefix##_md_last_error(void);                                \
  /* Lifetime. spec is engine-defined text. Returns NULL and writes err on     \
   * failure. */                                                               \
  api rgpot_md_session *prefix##_md_create(                                    \
      const char *spec, const rgpot_md_config *config, char *err,              \
      size_t err_len);                                                         \
  api void prefix##_md_destroy(rgpot_md_session *session);                     \
  api size_t prefix##_md_atom_count(const rgpot_md_session *session);          \
  /* One mass in amu per coordinate, the layout of a hook. */                  \
  api int prefix##_md_masses(const rgpot_md_session *session, double *masses,  \
                             size_t dimension);                                \
  /* State. */                                                                 \
  api int prefix##_md_set_box(rgpot_md_session *session, const double *box9);  \
  api int prefix##_md_set_positions(rgpot_md_session *session,                 \
                                    const double *positions,                   \
                                    size_t dimension);                         \
  api int prefix##_md_set_velocities(rgpot_md_session *session,                \
                                     const double *velocities,                 \
                                     size_t dimension);                        \
  api int prefix##_md_set_time_step(rgpot_md_session *session,                 \
                                    double time_step_ps);                      \
  /* Seeds the generator of the thermostat noise. */                           \
  api int prefix##_md_set_seed(rgpot_md_session *session, uint64_t seed);      \
  /* Evaluates the forces of the current state so state can be read before any \
   * step; the first step does it otherwise. */                                \
  api int prefix##_md_initialize(rgpot_md_session *session);                   \
  /* Any of positions, velocities and forces may be NULL, and so may info.    \
   * forces is the force of the potential at positions, in eV/Angstrom, never  \
   * the projected or hook-edited force a step integrates. */                  \
  api int prefix##_md_state(rgpot_md_session *session, double *positions,      \
                            double *velocities, double *forces,                \
                            size_t dimension, rgpot_md_info *info);            \
  /* Stepping. */                                                              \
  api int prefix##_md_step_nve(rgpot_md_session *session, uint64_t nsteps);    \
  /* thermostat, plane and walls may each be NULL (nwalls 0). With all three   \
   * NULL it is step_nve. */                                                   \
  api int prefix##_md_step_projected(                                          \
      rgpot_md_session *session, uint64_t nsteps,                              \
      const rgpot_md_thermostat *thermostat, const rgpot_md_plane *plane,      \
      const rgpot_md_wall *walls, size_t nwalls);                              \
  /* Callbacks. NULL clears them. */                                           \
  api int prefix##_md_set_hook(rgpot_md_session *session,                      \
                               rgpot_md_step_hook hook, void *user);           \
  api int prefix##_md_set_boundary(rgpot_md_session *session,                  \
                                   rgpot_md_boundary_fn boundary, void *user);

/* rgpot_md_config with the time step at 0.001 ps and bytes set. */
static inline void rgpot_md_config_init(rgpot_md_config *config) {
  config->bytes = (uint32_t)sizeof(rgpot_md_config);
  config->time_step_ps = 0.001;
}

#ifdef __cplusplus
}
#endif

#endif
