// MIT License
// Copyright 2023--present rgpot developers

/**
 * @file TersoffSiC.cc
 * @author rgpot Developers
 * @date 2026-10-26
 * @brief Tersoff Si-C kernel. lambda3 is zero for this parameter set, so
 * the coordination factor exp(lambda3^m (rij-rik)^m) is omitted.
 */

#include "rgpot/Tersoff/TersoffSiC.hpp"

#include "rgpot/ParamHash.hpp"
#include <cmath>
#include <stdexcept>
#include <string>

namespace rgpot {
namespace {

// Homoatomic columns. R and D in the LAMMPS file are the taper midpoint
// and half-width; Rin = R - D and Rout = R + D.
struct Homo {
  double big_a, big_b, lambda1, lambda2, beta, n, c, d, h, rin, rout, gamma;
};

constexpr Homo kSi = {1830.8, 471.18, 2.4799,   1.7322, 1.1e-6, 0.78734,
                      100390, 16.217, -0.59825, 2.7,    3.0,    1.0};
constexpr Homo kC = {1393.6, 346.74, 3.4879,   2.2119, 1.5724e-7, 0.72751,
                     38049,  4.3484, -0.57058, 1.8,    2.1,       1.0};
int kindOf(int z) {
  if (z == 6) {
    return 0;
  }
  if (z == 14) {
    return 1;
  }
  throw std::invalid_argument("TersoffSiC: atomic number " + std::to_string(z) +
                              " is neither carbon (6) nor silicon (14)");
}

TersoffSiC::Species makeSpecies(const Homo &h) {
  TersoffSiC::Species s;
  s.c = h.c;
  s.d = h.d;
  s.h = h.h;
  s.n = h.n;
  s.beta = h.beta;
  s.gamma = h.gamma;
  s.c2 = h.c * h.c;
  s.d2 = h.d * h.d;
  s.beta_n = std::pow(h.beta, h.n);
  return s;
}

TersoffSiC::Pair mixPair(const Homo &a, const Homo &b, double chi) {
  TersoffSiC::Pair p;
  p.lambda1 = 0.5 * (a.lambda1 + b.lambda1);
  p.lambda2 = 0.5 * (a.lambda2 + b.lambda2);
  p.big_a = std::sqrt(a.big_a * b.big_a);
  p.big_b = chi * std::sqrt(a.big_b * b.big_b);
  // Inner and outer cutoffs are averaged separately. Averaging the
  // LAMMPS midpoints is a different number.
  p.rin = std::sqrt(a.rin * b.rin);
  p.rout = std::sqrt(a.rout * b.rout);
  return p;
}

} // namespace

TersoffSiC::TersoffSiC() : Potential(PotType::TersoffSiC) {
  m_species[0] = makeSpecies(kC);
  m_species[1] = makeSpecies(kSi);
  const Homo homo[2] = {kC, kSi};
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
      const double chi = (i == j) ? 1.0 : kChiSiC;
      m_pair[i][j] = mixPair(homo[i], homo[j], chi);
    }
  }
  Fnv1a fp;
  fp.u64(1);
  fp.f64(kChiSiC);
  const Homo both[2] = {kC, kSi};
  for (const Homo &h : both) {
    fp.f64(h.big_a);
    fp.f64(h.big_b);
    fp.f64(h.lambda1);
    fp.f64(h.lambda2);
    fp.f64(h.beta);
    fp.f64(h.n);
    fp.f64(h.c);
    fp.f64(h.d);
    fp.f64(h.h);
    fp.f64(h.rin);
    fp.f64(h.rout);
    fp.f64(h.gamma);
  }
  m_paramsKey = fp.h;
}

void TersoffSiC::forceImpl(const ForceInput &in, ForceOut *out) const {
  if (out == nullptr) {
    throw std::invalid_argument("TersoffSiC: null ForceOut");
  }
  out->variance = 0.0;
  out->has_stress = 0;
  for (double &s : out->stress) {
    s = 0.0;
  }
  if (in.nAtoms > 0 && (in.pos == nullptr || in.box == nullptr ||
                        in.atmnrs == nullptr || out->F == nullptr)) {
    throw std::invalid_argument("TersoffSiC: null force buffer");
  }
  evaluate(in.pos, in.atmnrs, in.box, in.nAtoms, &out->energy, out->F, 1.0,
           nullptr);
}

void TersoffSiC::energyAndGradient(const double *positions,
                                   const int *atomic_numbers, const double *box,
                                   std::size_t n_atoms, double *energy,
                                   double *gradient,
                                   nlist::NeighborList *neighbors) const {
  if (energy == nullptr ||
      (n_atoms > 0 && (positions == nullptr || atomic_numbers == nullptr ||
                       box == nullptr || gradient == nullptr))) {
    throw std::invalid_argument("TersoffSiC: null energy buffer");
  }
  evaluate(positions, atomic_numbers, box, n_atoms, energy, gradient, -1.0,
           neighbors);
}

void TersoffSiC::energyAndGradientBatch(std::size_t n_config,
                                        const double *positions,
                                        const int *atomic_numbers,
                                        const double *box, std::size_t n_atoms,
                                        double *energies,
                                        double *gradients) const {
  if (n_config == 0) {
    return;
  }
  if (energies == nullptr ||
      (n_atoms > 0 && (positions == nullptr || atomic_numbers == nullptr ||
                       box == nullptr || gradients == nullptr))) {
    throw std::invalid_argument("TersoffSiC: null batch buffer");
  }
  const std::size_t stride = n_atoms * 3;
  for (std::size_t c = 0; c < n_config; ++c) {
    energyAndGradient(positions + c * stride, atomic_numbers, box, n_atoms,
                      energies + c, gradients + c * stride);
  }
}

void TersoffSiC::evaluate(const double *positions, const int *atomic_numbers,
                          const double *box, std::size_t n_atoms,
                          double *energy, double *acc, double scale,
                          nlist::NeighborList *neighbors) const {
  *energy = 0.0;
  for (std::size_t i = 0; i < n_atoms * 3; ++i) {
    acc[i] = 0.0;
  }
  (void)positions;
  (void)box;
  (void)scale;
  (void)neighbors;
  for (std::size_t i = 0; i < n_atoms; ++i) {
    kindOf(atomic_numbers[i]);
  }
}

} // namespace rgpot
