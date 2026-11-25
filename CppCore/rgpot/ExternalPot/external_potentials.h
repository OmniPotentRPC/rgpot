/*
 * Public constructors for the LAMMPS, GROMACS and OpenMM potentials.
 *
 * SPDX-License-Identifier: MIT
 * Copyright 2023--present rgpot developers
 *
 * Positions are Angstrom, energies are eV, forces are eV/Angstrom.
 * A box is nine doubles, lattice vectors as rows. With an engine path,
 * GROMACS and OpenMM convert u0, cutoff and psi to kJ/mol and nanometres
 * once, in the constructor, and the geometry, energy and forces on every
 * call.
 */

#ifndef RGPOT_EXTERNAL_POTENTIALS_H
#define RGPOT_EXTERNAL_POTENTIALS_H

#include <stddef.h>
#include <stdint.h>
#include "rgpot/nlist/rgpot_nlist.h"

#include "rgpot/rgpot_api.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct rgpot_external_potential rgpot_external_potential;

/* kJ/mol in one eV, and nanometres in one Angstrom. */
RGPOT_API double rgpot_md_kj_per_ev(void);
RGPOT_API double rgpot_md_nm_per_angstrom(void);

/*
 * Shifted 12-6 Lennard-Jones. u0 is eV, cutoff and psi are Angstrom.
 * engine_path NULL or "" searches RGPOT_LAMMPS_ENGINE, then
 * LAMMPS_ENGINE when they are set and not empty, then
 * liblammps_engine.so and liblj_engine.so. The plugin stays the backend: the constructor
 * dlopens it and calls rgpot_engine_force. Returns NULL and writes
 * err on failure.
 *
 * The first successful open of an engine file is kept and marked
 * RTLD_NODELETE, so the library stays mapped after
 * rgpot_external_potential_free and a later constructor of that file reuses
 * the open. An open that does not become the resident engine is closed.
 * A file replaced at a path that was already loaded is refused.
 */
RGPOT_API rgpot_external_potential *rgpot_lammps_potential(const char *engine_path,
                                                          double u0, double cutoff,
                                                          double psi, char *err,
                                                          size_t err_len);

/*
 * Lammpc-backed LAMMPS potential configured by a flat PotentialConfig
 * message containing the lammps arm. engine_path NULL or "" tries
 * LAMMPC_LIBRARY and RGPOT_LAMMPC_ENGINE when either is set and liblammpc.so
 * only when neither is. Any other engine_path is the only library tried, and
 * err names it when it fails.
 */
rgpot_external_potential *rgpot_lammpc_potential(const char *engine_path,
                                                const void *config_capnp,
                                                size_t config_capnp_size,
                                                char *err, size_t err_len);

/*
 * The same potential. The constructor stores u0 in kJ/mol and the
 * lengths in nanometres. engine_path NULL evaluates that potential
 * in process. A path is a library that exports rgpot_engine_create,
 * rgpot_engine_destroy and rgpot_md_native_force in those units.
 */
RGPOT_API rgpot_external_potential *rgpot_gromacs_potential(const char *engine_path,
                                                           double u0, double cutoff,
                                                           double psi, char *err,
                                                           size_t err_len);

RGPOT_API rgpot_external_potential *rgpot_openmm_potential(const char *engine_path,
                                                          double u0, double cutoff,
                                                          double psi, char *err,
                                                          size_t err_len);

RGPOT_API void rgpot_external_potential_free(rgpot_external_potential *pot);

/* Returns 0, or 1 and writes err. forces holds 3 * n doubles. */
RGPOT_API int rgpot_external_potential_energy_forces(
    rgpot_external_potential *pot, const double *xyz, const int *atomic_numbers,
    size_t n, const double *box9, double *forces, double *energy, char *err,
    size_t err_len);

/*
 * Generic in-process MD session. All quantities cross the C ABI in rgpot
 * units: positions and plane/wall coordinates in Angstrom, velocities in
 * Angstrom/ps, forces in eV/Angstrom, energies and kT in eV, time_step in ps,
 * friction in 1/ps, masses in amu, and the periodic box in Angstrom. Flat
 * arrays hold 3 * n_atoms doubles in row-major (x, y, z) order; the box is
 * nine doubles, the three lattice vectors as rows.
 *
 * Any backend (OpenMM, GROMACS nblib, LAMMPS) implements rgpot_md_session_ops
 * or exports rgpot_md_session_engine_create from a plugin shared library.
 * The callbacks up to and including destroy are required. Those after destroy
 * are optional: a backend that predates one has a smaller struct_size, and a
 * backend that does not offer one leaves it null. The session reports
 * either as unsupported with a message rather than crashing. A backend keeps
 * the order of the members, so a newer member only ever follows an older one.
 *
 * Optional neighbour exchange uses rgpot_nlist_request and rgpot_nlist_soa and
 * returns RGPOT_MD_NLIST_OK, RGPOT_MD_NLIST_ERROR, or
 * RGPOT_MD_NLIST_UNSUPPORTED.
 *
 * get_neighbors: a null out, or a buffer whose cap is 0, sets *needed to the
 * full pair count, sets *count to 0 when count is not null, and returns
 * RGPOT_MD_NLIST_OK. A buffer shorter than that list returns
 * RGPOT_MD_NLIST_ERROR after writing the pairs that fit. *count is the number
 * written, *needed is the full count the caller must provide, and the error
 * text names that required count. The OpenMM session and the in-tree mock
 * follow this rule.
 */
typedef struct rgpot_md_session rgpot_md_session;

/* How a drift that ends outside a wall is handled by run_langevin. */
enum {
  /* Mirror the position and reflect the velocity in the mass-weighted metric. */
  RGPOT_MD_WALL_REFLECT = 0,
  /* Undo the drift and reflect the velocity. */
  RGPOT_MD_WALL_REJECT = 1
};

/*
 * One call of Langevin dynamics, optionally on a hyperplane and between
 * planar walls. Positions and velocities are those of the session and are
 * advanced in place.
 *
 * The plane n . (x - p) = 0 is given by plane_point and plane_normal, each n3
 * doubles; both null means unconstrained dynamics. A wall w bounds
 * n_w . (x - p_w) <= 0 with p_w = wall_points[w * n3 ...] in Angstrom and a
 * unit-free n_w = wall_normals[w * n3 ...]; the positions must start inside.
 * kT is in eV, friction in 1/ps and time_step in ps. seed names the session's
 * random stream, the same on every engine. A call with the seed of the
 * previous call continues that stream, so a run split in chunks equals one
 * long run; a new seed starts a new stream. Setting positions or velocities
 * ends the stream, so the same seed after a new state replays the same noise.
 * A distance constraint of the engine's own topology holds along with the
 * plane.
 */
typedef struct rgpot_md_langevin_request {
  size_t struct_size; /* sizeof(rgpot_md_langevin_request) */
  double kT;
  double friction;
  double time_step;
  size_t steps;
  uint64_t seed;
  size_t n3; /* 3 * n_atoms */
  const double *plane_point;
  const double *plane_normal;
  size_t n_walls;
  const double *wall_points;
  const double *wall_normals;
  int wall_mode; /* RGPOT_MD_WALL_REFLECT or RGPOT_MD_WALL_REJECT */
} rgpot_md_langevin_request;

/*
 * What run_langevin reports. constraint_force is null or a buffer of
 * constraint_force_len doubles, at least the number of steps when the plane is
 * set. Entry k is the Lagrange multiplier lambda of step k in eV/Angstrom: the
 * constraint force on the system is lambda n for the unit normal n, and
 * lambda = -(n . M^-1 F) / (n . M^-1 n) averaged over the two force kicks of
 * the step. Its mean over the constrained ensemble is dF/dc for the plane
 * offset c = n . p. Nothing is written without a plane.
 */
typedef struct rgpot_md_langevin_result {
  size_t struct_size; /* sizeof(rgpot_md_langevin_result) */
  uint64_t wall_hits;
  double *constraint_force;
  size_t constraint_force_len;
} rgpot_md_langevin_result;

typedef struct rgpot_md_session_ops {
  size_t struct_size;
  int (*set_positions)(void *ctx, const double *positions, size_t n3,
                       char *err, size_t err_len);
  int (*set_velocities)(void *ctx, const double *velocities, size_t n3,
                        char *err, size_t err_len);
  int (*step_langevin)(void *ctx, double kT, double friction, double time_step,
                       const double *plane_point, const double *plane_normal,
                       const double *wall_points, const double *wall_normals,
                       size_t n_walls, size_t n3, size_t steps, uint64_t seed,
                       uint64_t *wall_hits, char *err, size_t err_len);
  int (*step_nve)(void *ctx, double time_step, size_t steps, char *err,
                  size_t err_len);
  /* Any output pointer may be null; a null output is not written. */
  int (*get_state)(void *ctx, double *positions, double *velocities,
                   double *forces, size_t n3, double *potential_energy,
                   double *kinetic_energy, char *err, size_t err_len);
  int (*get_masses)(void *ctx, double *masses, size_t n_atoms, char *err,
                    size_t err_len);
  void (*destroy)(void *ctx);
  /* Optional from here on. */
  int (*set_periodic_box)(void *ctx, const double *box9, char *err,
                          size_t err_len);
  int (*get_periodic_box)(void *ctx, double *box9, char *err, size_t err_len);
  /* Evaluate the first forces once box and positions are set. */
  int (*initialize)(void *ctx, char *err, size_t err_len);
  int (*run_langevin)(void *ctx, const rgpot_md_langevin_request *request,
                      rgpot_md_langevin_result *result, char *err,
                      size_t err_len);
  /* Optional neighbour-list exchange. A null callback means unsupported. */
  int (*get_neighbors)(void *ctx, const rgpot_nlist_request *request,
                       rgpot_nlist_soa *out, size_t *count, size_t *needed,
                       char *err, size_t err_len);
  int (*set_neighbors)(void *ctx, const rgpot_nlist_request *request,
                       const rgpot_nlist_soa *pairs, size_t count, char *err,
                       size_t err_len);
} rgpot_md_session_ops;

enum {
  RGPOT_MD_NLIST_OK = 0,
  RGPOT_MD_NLIST_ERROR = 1,
  RGPOT_MD_NLIST_UNSUPPORTED = 2
};

/*
 * True when ops carries the optional member and a callback for it:
 * RGPOT_MD_OPS_HAS(ops, run_langevin).
 */
#define RGPOT_MD_OPS_END(member) (offsetof(rgpot_md_session_ops, member) + sizeof(((rgpot_md_session_ops *)0)->member))
#define RGPOT_MD_OPS_HAS(ops, member) ((ops)->struct_size >= RGPOT_MD_OPS_END(member) && (ops)->member != NULL)

typedef int (*rgpot_md_session_engine_create_fn)(
    const void *spec, size_t spec_len, const char *options,
    rgpot_md_session_ops *ops, void **ctx, size_t *n_atoms, char *err,
    size_t err_len);

RGPOT_API rgpot_md_session *
rgpot_md_session_from_ops(void *ctx, const rgpot_md_session_ops *ops,
                          size_t n_atoms, char *err, size_t err_len);
RGPOT_API rgpot_md_session *
rgpot_md_session_create(const char *engine, const char *engine_path,
                        const void *spec, size_t spec_len, const char *options,
                        char *err, size_t err_len);
RGPOT_API size_t rgpot_md_session_n_atoms(const rgpot_md_session *session);
RGPOT_API int rgpot_md_session_get_masses(rgpot_md_session *session,
                                          double *masses, size_t n_atoms,
                                          char *err, size_t err_len);
RGPOT_API int rgpot_md_session_set_periodic_box(rgpot_md_session *session,
                                                const double *box9, char *err,
                                                size_t err_len);
RGPOT_API int rgpot_md_session_get_periodic_box(rgpot_md_session *session,
                                                double *box9, char *err,
                                                size_t err_len);
RGPOT_API int rgpot_md_session_set_positions(rgpot_md_session *session,
                                             const double *positions, size_t n3,
                                             char *err, size_t err_len);
RGPOT_API int rgpot_md_session_set_velocities(rgpot_md_session *session,
                                              const double *velocities,
                                              size_t n3, char *err,
                                              size_t err_len);
RGPOT_API int rgpot_md_session_initialize(rgpot_md_session *session, char *err,
                                          size_t err_len);
RGPOT_API int rgpot_md_session_step_langevin(
    rgpot_md_session *session, double kT, double friction, double time_step,
    const double *plane_point, const double *plane_normal,
    const double *wall_points, const double *wall_normals, size_t n_walls,
    size_t n3, size_t steps, uint64_t seed, uint64_t *wall_hits, char *err,
    size_t err_len);
RGPOT_API int rgpot_md_session_run_langevin(
    rgpot_md_session *session, const rgpot_md_langevin_request *request,
    rgpot_md_langevin_result *result, char *err, size_t err_len);
RGPOT_API int rgpot_md_session_step_nve(rgpot_md_session *session,
                                        double time_step, size_t steps,
                                        char *err, size_t err_len);
/*
 * Draw velocities from the Maxwell-Boltzmann distribution at kT (eV) and set
 * them. With plane_normal (n3 doubles) the draw is conditioned on
 * n . v = 0, the constrained momentum ensemble of run_langevin on that plane;
 * null draws all 3 * n_atoms components. A particle of zero mass is at rest.
 * seed makes the draw reproducible and is the same stream on every engine.
 */
RGPOT_API int rgpot_md_session_reseed_velocities(rgpot_md_session *session,
                                                 double kT, uint64_t seed,
                                                 const double *plane_normal,
                                                 size_t n3, char *err,
                                                 size_t err_len);
RGPOT_API int rgpot_md_session_get_state(rgpot_md_session *session,
                                         double *positions, double *velocities,
                                         double *forces, size_t n3,
                                         double *potential_energy,
                                         double *kinetic_energy, char *err,
                                         size_t err_len);
RGPOT_API int rgpot_md_session_get_positions(rgpot_md_session *session,
                                             double *positions, size_t n3,
                                             char *err, size_t err_len);
RGPOT_API int rgpot_md_session_get_velocities(rgpot_md_session *session,
                                              double *velocities, size_t n3,
                                              char *err, size_t err_len);
/* Forces and the potential energy at the current positions. */
RGPOT_API int rgpot_md_session_get_forces(rgpot_md_session *session,
                                          double *forces, size_t n3,
                                          double *potential_energy, char *err,
                                          size_t err_len);
/* Potential and kinetic energy, either of which may be null. */
RGPOT_API int rgpot_md_session_get_energy(rgpot_md_session *session,
                                          double *potential_energy,
                                          double *kinetic_energy, char *err,
                                          size_t err_len);
/* See the neighbour-exchange note above for the short-buffer contract. */
RGPOT_API int rgpot_md_session_get_neighbors(rgpot_md_session *session,
                                             const rgpot_nlist_request *request,
                                             rgpot_nlist_soa *out,
                                             size_t *count, size_t *needed,
                                             char *err, size_t err_len);
RGPOT_API int rgpot_md_session_set_neighbors(rgpot_md_session *session,
                                             const rgpot_nlist_request *request,
                                             const rgpot_nlist_soa *pairs,
                                             size_t count, char *err,
                                             size_t err_len);
RGPOT_API void rgpot_md_session_destroy(rgpot_md_session *session);

#ifdef __cplusplus
}
#endif

#endif
