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
 *
 * The first successful open of an engine file is kept and marked
 * RTLD_NODELETE, so the library stays mapped after
 * rgpot_external_potential_free and a later constructor of that file reuses
 * the open. An open that does not become the resident engine is closed.
 * A file replaced at a path that was already loaded is refused.
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
 * Generic in-process MD session. Coordinates are OpenMM native units:
 * nanometres, kJ/mol, and ps. Positions and velocities are flat xyz arrays.
 * The engine owns the serialized System and applies projected BAOAB or NVE.
 */
typedef struct rgpot_md_session rgpot_md_session;
rgpot_md_session *rgpot_md_session_create(
    const char *system_xml, const char *platform, double time_step,
    double friction, double kT, int seed, char *err, size_t err_len);
int rgpot_md_session_set_positions(rgpot_md_session *, const double *, size_t,
                                   char *, size_t);
int rgpot_md_session_set_velocities(rgpot_md_session *, const double *, size_t,
                                    char *, size_t);
int rgpot_md_session_step_langevin(
    rgpot_md_session *, const double *plane_point, const double *plane_normal,
    const double *wall_points, const double *wall_normals, size_t n_walls,
    size_t steps, uint64_t *wall_hits, char *, size_t);
int rgpot_md_session_step_nve(rgpot_md_session *, size_t steps, char *,
                               size_t);
int rgpot_md_session_get_state(rgpot_md_session *, double *positions,
                               double *velocities, double *forces, size_t n3,
                               double *potential_energy, double *kinetic_energy,
                               char *, size_t);
void rgpot_md_session_destroy(rgpot_md_session *);

#ifdef __cplusplus
}
#endif

#endif
