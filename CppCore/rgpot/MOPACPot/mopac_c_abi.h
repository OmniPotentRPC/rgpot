/**
 * @file mopacc.h
 * @brief C ABI between an rgpot MOPACPot frontend and libmopacc.so.
 *
 * Split OpenMOPAC engine, same role as nwchemc / cpmdc. Every
 * mopac_system model and the potential-facing OpenMOPAC entry points
 * (SCF, relax, vibe; conventional or MOZYME) go through this ABI.
 * The omitted-params default Hamiltonian is AM1 (model 4).
 *
 * OpenMOPAC reports heat of formation in kcal/mol and Cartesian
 * derivatives in kcal/mol/Angstrom. This ABI reports Hartree and
 * Hartree/Bohr, same units as nwchemc.h. Dipole is Debye. Stress is
 * GPa Voigt. Lattice vectors are Angstrom.
 *
 * Geometry stays on the argument list. Method knobs are a packed
 * MopacCParams blob, or NULL for vacuum defaults. Cell is sticky
 * via mopacc_set_cell. There is no Cap'n Proto arm.
 *
 * Not thread-safe: OpenMOPAC's API is process-global.
 */
#pragma once

#include <stddef.h>

#define RGPOT_MOPACC_C_ABI_VERSION 1
#define MOPACC_ABI_VERSION 1 /* matches shared-library soversion 1 */

#ifdef __cplusplus
extern "C" {
#endif

#if defined(RGPOT_MOPACC_BUILD) && (defined(_WIN32) || defined(_WIN64))
#define RGPOT_MOPACC_API __declspec(dllexport)
#else
#define RGPOT_MOPACC_API
#endif

/** OpenMOPAC mopac_system.model. Omitted params default to AM1. */
#define MOPACC_MODEL_PM7 0
#define MOPACC_MODEL_PM6_D3H4 1
#define MOPACC_MODEL_PM6_ORG 2
#define MOPACC_MODEL_PM6 3
#define MOPACC_MODEL_AM1 4
#define MOPACC_MODEL_RM1 5

#define MOPACC_SOLVER_MOPAC 0
#define MOPACC_SOLVER_MOZYME 1

/**
 * Sticky method options. Omit the buffer for charge 0, closed shell,
 * vacuum, conventional solver, AM1.
 */
typedef struct MopacCParams {
  int charge;       /**< net charge */
  int spin;         /**< OpenMOPAC spin excitations */
  int model;        /**< 0 PM7 .. 5 RM1; omitted blob defaults to AM1 */
  double tolerance; /**< GNORM/RELSCF scale; 1.0 is the OpenMOPAC default */
  int max_time;     /**< seconds; 0 uses 3600 */
  double epsilon;   /**< COSMO dielectric; 1.0 is vacuum. Must be 1 if cell. */
  int solver;       /**< 0 conventional, 1 MOZYME */
} MopacCParams;

typedef struct MopacCResult {
  int ok;               /**< 1 success, 0 failure */
  double energy_h;      /**< heat of formation converted to Hartree */
  double dipole_debye[3];
  double stress_gpa[6]; /**< Voigt xx,yy,zz,yz,xz,xy; 0 if unavailable */
  char message[512];
} MopacCResult;

typedef struct MopacCSession MopacCSession;

RGPOT_MOPACC_API int mopacc_set_params(const void *params,
                                       size_t params_size_bytes);

RGPOT_MOPACC_API MopacCResult mopacc_energy_gradient(
    int n_atoms, const double *positions_ang, const int *atomic_numbers,
    const void *params, size_t params_size_bytes, double *grad_h_bohr);

RGPOT_MOPACC_API MopacCResult mopacc_energy(
    int n_atoms, const double *positions_ang, const int *atomic_numbers,
    const void *params, size_t params_size_bytes);

RGPOT_MOPACC_API MopacCResult mopacc_energy_forces(
    int n_atoms, const double *positions_ang, const int *atomic_numbers,
    const void *params, size_t params_size_bytes, double *forces_h_bohr);

RGPOT_MOPACC_API MopacCSession *mopacc_session_create(const void *params,
                                                      size_t params_size_bytes);

RGPOT_MOPACC_API void mopacc_session_destroy(MopacCSession *session);

RGPOT_MOPACC_API int mopacc_session_set_params(MopacCSession *session,
                                               const void *params,
                                               size_t params_size_bytes);

RGPOT_MOPACC_API MopacCResult mopacc_session_energy_gradient(
    MopacCSession *session, int n_atoms, const double *positions_ang,
    const int *atomic_numbers, double *grad_h_bohr);

RGPOT_MOPACC_API int mopacc_set_cell(const double *lattice_ang, int nlattice,
                                     int nlattice_move, double pressure_gpa);

RGPOT_MOPACC_API int mopacc_session_set_cell(MopacCSession *session,
                                             const double *lattice_ang,
                                             int nlattice, int nlattice_move,
                                             double pressure_gpa);

RGPOT_MOPACC_API MopacCResult mopacc_relax(
    int n_atoms, const double *positions_ang, const int *atomic_numbers,
    const void *params, size_t params_size_bytes, double *coord_update_ang,
    double *grad_h_bohr);

RGPOT_MOPACC_API MopacCResult mopacc_vibe(
    int n_atoms, const double *positions_ang, const int *atomic_numbers,
    const void *params, size_t params_size_bytes, double *freq_cm,
    double *disp);

RGPOT_MOPACC_API MopacCResult mopacc_session_relax(
    MopacCSession *session, int n_atoms, const double *positions_ang,
    const int *atomic_numbers, double *coord_update_ang, double *grad_h_bohr);

RGPOT_MOPACC_API MopacCResult mopacc_session_vibe(
    MopacCSession *session, int n_atoms, const double *positions_ang,
    const int *atomic_numbers, double *freq_cm, double *disp);

RGPOT_MOPACC_API int mopacc_last_charges(int n_atoms, double *charges);
RGPOT_MOPACC_API int mopacc_session_charges(const MopacCSession *session,
                                            int n_atoms, double *charges);

RGPOT_MOPACC_API const char *mopacc_version(void);
RGPOT_MOPACC_API int mopacc_c_abi_version(void);
RGPOT_MOPACC_API int mopacc_abi_version(void);
RGPOT_MOPACC_API int mopacc_available(void);
RGPOT_MOPACC_API const char *mopacc_last_error(void);
RGPOT_MOPACC_API void mopacc_finalize(void);

#ifdef __cplusplus
}
#endif
