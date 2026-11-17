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

#ifdef __cplusplus
extern "C" {
#endif

typedef struct rgpot_external_potential rgpot_external_potential;

/* kJ/mol in one eV, and nanometres in one Angstrom. */
double rgpot_md_kj_per_ev(void);
double rgpot_md_nm_per_angstrom(void);

/*
 * Shifted 12-6 Lennard-Jones. u0 is eV, cutoff and psi are Angstrom.
 * engine_path NULL or "" searches RGPOT_LAMMPS_ENGINE, then
 * LAMMPS_ENGINE when they are set and not empty, then
 * liblammps_engine.so and liblj_engine.so. The plugin stays the backend: the constructor
 * dlopens it and calls rgpot_engine_force. Returns NULL and writes
 * err on failure.
 */
rgpot_external_potential *rgpot_lammps_potential(const char *engine_path,
                                                double u0, double cutoff,
                                                double psi, char *err,
                                                size_t err_len);

/*
 * The same potential. The constructor stores u0 in kJ/mol and the
 * lengths in nanometres. engine_path NULL evaluates that potential
 * in process. A path is a library that exports rgpot_engine_create,
 * rgpot_engine_destroy and rgpot_md_native_force in those units.
 */
rgpot_external_potential *rgpot_gromacs_potential(const char *engine_path,
                                                 double u0, double cutoff,
                                                 double psi, char *err,
                                                 size_t err_len);

rgpot_external_potential *rgpot_openmm_potential(const char *engine_path,
                                                double u0, double cutoff,
                                                double psi, char *err,
                                                size_t err_len);

void rgpot_external_potential_free(rgpot_external_potential *pot);

/* Returns 0, or 1 and writes err. forces holds 3 * n doubles. */
int rgpot_external_potential_energy_forces(
    rgpot_external_potential *pot, const double *xyz, const int *atomic_numbers,
    size_t n, const double *box9, double *forces, double *energy, char *err,
    size_t err_len);

/*
 * Generic in-process MD session. All quantities cross the C ABI in rgpot
 * units: positions and plane/wall coordinates in Angstrom, velocities in
 * Angstrom/ps, forces in eV/Angstrom, energies and kT in eV, time_step in ps,
 * friction in 1/ps, and masses in amu. Flat arrays hold 3 * n_atoms doubles
 * in row-major (x, y, z) order.
 * Optional neighbour exchange uses rgpot_nlist_request/rgpot_nlist_soa and
 * returns RGPOT_MD_NLIST_OK, RGPOT_MD_NLIST_ERROR, or
 * RGPOT_MD_NLIST_UNSUPPORTED.
 *
 * Any backend (OpenMM, GROMACS nblib, LAMMPS) implements rgpot_md_session_ops
 * or exports rgpot_md_session_engine_create from a plugin shared library.
 */
typedef struct rgpot_md_session rgpot_md_session;

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
  int (*get_state)(void *ctx, double *positions, double *velocities,
                   double *forces, size_t n3, double *potential_energy,
                   double *kinetic_energy, char *err, size_t err_len);
  int (*get_masses)(void *ctx, double *masses, size_t n_atoms, char *err,
                    size_t err_len);
  void (*destroy)(void *ctx);
  /* Optional neighbour-list exchange. Null callback means unsupported. */
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

typedef int (*rgpot_md_session_engine_create_fn)(
    const void *spec, size_t spec_len, const char *options,
    rgpot_md_session_ops *ops, void **ctx, size_t *n_atoms, char *err,
    size_t err_len);

rgpot_md_session *rgpot_md_session_from_ops(void *ctx,
                                            const rgpot_md_session_ops *ops,
                                            size_t n_atoms, char *err,
                                            size_t err_len);
rgpot_md_session *rgpot_md_session_create(const char *engine,
                                          const char *engine_path,
                                          const void *spec, size_t spec_len,
                                          const char *options, char *err,
                                          size_t err_len);
size_t rgpot_md_session_n_atoms(const rgpot_md_session *session);
int rgpot_md_session_get_masses(rgpot_md_session *session, double *masses,
                                size_t n_atoms, char *err, size_t err_len);
int rgpot_md_session_set_positions(rgpot_md_session *session,
                                   const double *positions, size_t n3,
                                   char *err, size_t err_len);
int rgpot_md_session_set_velocities(rgpot_md_session *session,
                                    const double *velocities, size_t n3,
                                    char *err, size_t err_len);
int rgpot_md_session_step_langevin(
    rgpot_md_session *session, double kT, double friction, double time_step,
    const double *plane_point, const double *plane_normal,
    const double *wall_points, const double *wall_normals, size_t n_walls,
    size_t n3, size_t steps, uint64_t seed, uint64_t *wall_hits, char *err,
    size_t err_len);
int rgpot_md_session_step_nve(rgpot_md_session *session, double time_step,
                              size_t steps, char *err, size_t err_len);
int rgpot_md_session_get_state(rgpot_md_session *session, double *positions,
                               double *velocities, double *forces, size_t n3,
                               double *potential_energy, double *kinetic_energy,
                               char *err, size_t err_len);
int rgpot_md_session_get_neighbors(rgpot_md_session *session,
                                   const rgpot_nlist_request *request,
                                   rgpot_nlist_soa *out, size_t *count,
                                   size_t *needed, char *err, size_t err_len);
int rgpot_md_session_set_neighbors(rgpot_md_session *session,
                                   const rgpot_nlist_request *request,
                                   const rgpot_nlist_soa *pairs, size_t count,
                                   char *err, size_t err_len);
void rgpot_md_session_destroy(rgpot_md_session *session);

#ifdef __cplusplus
}
#endif

#endif
