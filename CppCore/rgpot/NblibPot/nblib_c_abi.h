/*
 * C ABI of librgpot_nblib: GROMACS nblib in process.
 *
 * SPDX-License-Identifier: MIT
 * Copyright 2023--present rgpot developers
 *
 * The library evaluates the nonbonded and listed forces of a nblib topology
 * (the potential below) and steps it with nblib's leap-frog integrator
 * through the engine-generic MD session ABI of
 * rgpot/MDSession/md_session_c_abi.h, exported here with the prefix
 * rgpot_nblib (rgpot_nblib_md_create, rgpot_nblib_md_step_nve, ...). The
 * spec string of rgpot_nblib_md_create is the topology text. GROMACS is
 * linked into the library only; a host dlopens it and probes the symbols, so
 * a library built before a call existed simply lacks the symbol.
 *
 * Units at this boundary are Angstrom, picosecond, eV, atomic mass unit.
 * The topology text (docs/orgmode/howto/engine-nblib.org) is the one place
 * that speaks the GROMACS units of a force field: nm, kJ/mol, amu and
 * elementary charge. A box is nine doubles, lattice vectors as rows; nblib
 * takes rectangular cells only, so the six off-diagonal entries must be
 * exactly zero.
 *
 * Every call that can fail returns 0 or nonzero. The reason is written to
 * err when the call takes one, and rgpot_nblib_last_error returns it for the
 * calling thread.
 *
 * Thread safety: one handle is used by one thread at a time. Distinct
 * handles may be used concurrently.
 */

#ifndef RGPOT_NBLIB_C_ABI_H
#define RGPOT_NBLIB_C_ABI_H

#include "rgpot/MDSession/md_session_c_abi.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32)
#define RGPOT_NBLIB_API
#else
#define RGPOT_NBLIB_API __attribute__((visibility("default")))
#endif

/* The major changes when a call an old host makes stops working. The minor
 * changes when a symbol or a trailing struct field is added. */
#define RGPOT_NBLIB_ABI_VERSION 1
#define RGPOT_NBLIB_ABI_MINOR 0

/* Bits of rgpot_nblib_features. A host may also probe the symbol itself. */
#define RGPOT_NBLIB_FEATURE_POTENTIAL 1u
#define RGPOT_NBLIB_FEATURE_VIRIAL 2u
/* The rgpot_nblib_md_* session surface is present. */
#define RGPOT_NBLIB_FEATURE_SESSION 4u

RGPOT_NBLIB_API int rgpot_nblib_abi_version(void);
RGPOT_NBLIB_API int rgpot_nblib_abi_minor(void);
/* 1 when the library was built against nblib. */
RGPOT_NBLIB_API int rgpot_nblib_available(void);
RGPOT_NBLIB_API unsigned rgpot_nblib_features(void);
/* Bits of the floating point type nblib was compiled with: 64 or 32. */
RGPOT_NBLIB_API int rgpot_nblib_real_bits(void);
/* GROMACS release nblib comes from, for example "2025.5". */
RGPOT_NBLIB_API const char *rgpot_nblib_gromacs_version(void);
RGPOT_NBLIB_API const char *rgpot_nblib_last_error(void);

/* ---------------------------------------------------------------------- */
/* Potential: energy, forces and virial of a configuration.                */
/* ---------------------------------------------------------------------- */

typedef struct rgpot_nblib_potential rgpot_nblib_potential;

/* Parses the topology text and builds the nblib topology and force
 * calculators. Returns NULL and writes err on failure. */
RGPOT_NBLIB_API rgpot_nblib_potential *
rgpot_nblib_potential_create(const char *topology, char *err, size_t err_len);

RGPOT_NBLIB_API void rgpot_nblib_potential_free(rgpot_nblib_potential *pot);

RGPOT_NBLIB_API size_t
rgpot_nblib_potential_atom_count(const rgpot_nblib_potential *pot);

/* Cutoff of the topology in Angstrom. */
RGPOT_NBLIB_API double
rgpot_nblib_potential_cutoff(const rgpot_nblib_potential *pot);

/* Writes one mass in amu per atom. Returns 0, or 1 when n differs from the
 * atom count. */
RGPOT_NBLIB_API int rgpot_nblib_potential_masses(const rgpot_nblib_potential *pot,
                                                 double *masses, size_t n);

/* xyz holds 3 * n coordinates and forces 3 * n doubles. energy is eV.
 * virial9 may be NULL; otherwise it receives the GROMACS virial
 * Xi = -1/2 sum r (x) F in eV, row-major, which is V/2 times the rgpot
 * stress. The positions need not lie in the box. Returns 0, or 1 and
 * writes err. */
RGPOT_NBLIB_API int rgpot_nblib_potential_energy_forces(
    rgpot_nblib_potential *pot, const double *xyz, size_t n, const double *box9,
    double *forces, double *energy, double *virial9, char *err, size_t err_len);

/* ---------------------------------------------------------------------- */
/* MD session: the engine-generic surface under the prefix rgpot_nblib.     */
/* ---------------------------------------------------------------------- */

RGPOT_MD_DECLARE_API(rgpot_nblib, RGPOT_NBLIB_API)

#ifdef __cplusplus
}
#endif

#endif
