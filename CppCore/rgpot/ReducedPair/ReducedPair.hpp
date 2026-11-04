#pragma once
// MIT License
// Copyright 2023--present rgpot developers

/**
 * @brief Reduced cluster pair kernels used by anneal.
 *
 * Lennard-Jones is the unshifted Wales and Doye form
 * `4 epsilon [(sigma/r)^12 - (sigma/r)^6]`, with no cutoff when
 * `cutoff <= 0`. Morse is the Doye and Wales form
 * `epsilon exp(rho (1 - r/re)) (exp(rho (1 - r/re)) - 2)`, again
 * unshifted. The unit Lennard-Jones branch (epsilon = 1, sigma = 1,
 * cutoff <= 0) uses the same expression tree as anneal's
 * PairKind::LennardJones so a search sees the same rounding.
 *
 * Gradients are derivatives of the energy, not physical forces.
 */

// clang-format off
#include <cstdint>
// clang-format on

#if defined(_WIN32)
#define RGPOT_PAIR_API __declspec(dllexport)
#else
#define RGPOT_PAIR_API __attribute__((visibility("default")))
#endif

extern "C" {

/// Pair energy, `(dV/dr) / r`, and `d2V/dr2`. `r2 <= 0` writes zeros.
RGPOT_PAIR_API int rgpot_lj_pair(double epsilon, double sigma, double r2,
                                 double *energy, double *dvdr_over_r,
                                 double *d2vdr2);

/// Pair energy, `(dV/dr) / r`, and `d2V/dr2` for `npairs` squared distances.
/// One call covers a whole pair list. Returns 0 on success.
RGPOT_PAIR_API int rgpot_lj_curvature_pairs(double epsilon, double sigma,
                                            int32_t npairs, const double *r2,
                                            double *energy,
                                            double *dvdr_over_r,
                                            double *d2vdr2);

/// Cluster energy and gradient. `cutoff <= 0` keeps every pair.
/// Returns 0 on success, 1 on a bad argument.
RGPOT_PAIR_API int rgpot_lj_gradient(double epsilon, double sigma,
                                     double cutoff, int32_t natoms,
                                     const double *positions, double *energy,
                                     double *gradient);

/// Pair energy of each atom, added to both ends of every pair.
/// The total energy is half the sum. Returns 0 on success.
RGPOT_PAIR_API int rgpot_lj_atom_energies(double epsilon, double sigma,
                                          int32_t natoms,
                                          const double *positions,
                                          double *atom_energy);

/// Gradient rows of `moved` only. Other rows stay zero.
RGPOT_PAIR_API int rgpot_lj_partial_gradient(double epsilon, double sigma,
                                             int32_t natoms,
                                             const double *positions,
                                             int32_t n_moved,
                                             const int32_t *moved,
                                             double *gradient);

/// Arithmetic-mean diameters. `sigma` has one entry per atom.
/// `cutoff <= 0` keeps every pair. Energy is `4 * sum (s^12 - s^6)`.
RGPOT_PAIR_API int rgpot_lj_arithmetic_gradient(const double *sigma,
                                                double cutoff, int32_t natoms,
                                                const double *positions,
                                                double *energy,
                                                double *gradient);

/// Morse pair. `re` is the pair minimum. `r2 <= 0` writes zeros.
RGPOT_PAIR_API int rgpot_morse_pair(double epsilon, double rho, double re,
                                    double r2, double *energy,
                                    double *dvdr_over_r, double *d2vdr2);

/// Morse cluster. `cutoff <= 0` keeps every pair.
RGPOT_PAIR_API int rgpot_morse_gradient(double epsilon, double rho,
                                        double re, double cutoff,
                                        int32_t natoms,
                                        const double *positions,
                                        double *energy, double *gradient);

/// Species A is the first `n_a` atoms, species B the rest.
/// `eps` and `sig` are ordered AA, AB, BB.
RGPOT_PAIR_API int rgpot_binary_lj_gradient(int32_t n_a, const double *eps,
                                            const double *sig, double cutoff,
                                            int32_t natoms,
                                            const double *positions,
                                            double *energy, double *gradient);

/// Oxygen Lennard-Jones plus Coulomb on rigid TIP4P sites.
/// `sites` and `gradient` are molecule-major O, H, H, M, three
/// coordinates each. `use_epsilon == 0` uses `c6` and `c12`.
/// `use_epsilon != 0` uses `epsilon` and `sigma`.
struct rgpot_tip4p_sites_config {
  double c6;
  double c12;
  double epsilon;
  double sigma;
  int32_t use_epsilon;
  double coulomb;
  double q_h;
  double q_m;
};

RGPOT_PAIR_API int rgpot_tip4p_sites(const rgpot_tip4p_sites_config *cfg,
                                     int32_t n_molecules, const double *sites,
                                     double *energy, double *gradient);

} // extern "C"
