/*
** This file is part of eOn.
**
** SPDX-License-Identifier: BSD-3-Clause
**
** Copyright (c) 2010--present, eOn Development Team
** All rights reserved.
**
** Repo:
** https://github.com/TheochemUI/eOn
*/
#pragma once

#include <array>
#include <cstdint>
#include <vector>

#ifndef EAM_STANDALONE
#include "eon/Potential.h"
#endif

/// Fraction of the widest element cutoff added to the neighbour search.
/// A later call reuses the list while every atom has moved by at most half
/// of that distance. One fraction for every geometry.
inline constexpr double kEamSkinOverCutoff = 1.0 / 5.0;

/// One OpenMP thread matches the direct pair loop bit for bit. Several
/// threads each write the forces of their own atoms and add partial
/// energies in thread-id order, so a fixed thread count repeats bit for
/// bit. Energy and every force component stay within this relative
/// tolerance of the one-thread value. A component whose one-thread
/// magnitude is below 1 is compared on an absolute scale of the same size.
inline constexpr double kEamThreadedTolerance = 1e-9;

/// Embedded-atom aluminium, palladium and gold.
///
/// The neighbour search is a linked-cell list with a Verlet skin. The force
/// loop visits the same pairs, in the same atom order, as the direct double
/// loop, and the arithmetic of each pair is unchanged. `OMP_NUM_THREADS`
/// above 1 splits atoms across threads. The variable is read on each call.
/// When it is unset the call stays on one thread.
class EAM
#ifndef EAM_STANDALONE
    : public eonc::Potential
#endif
{
public:
#ifdef EAM_STANDALONE
  EAM() = default;
  ~EAM() = default;
#else
  explicit EAM(const eonc::Parameters &params)
      : eonc::Potential(eonc::PotType::EAM_AL, params) {}

  ~EAM() override = default;
#endif

  void cleanMemory();
  void force(long N, const double *R, const int *atomicNrs, double *F,
             double *U, double *variance, const double *fullbox)
#ifndef EAM_STANDALONE
      override
#endif
      ;

  /// Neighbour-list builds since this object was created.
  [[nodiscard]] std::uint64_t listBuilds() const noexcept { return builds_; }

  /// Stored neighbour indices. Linear in the atom count for a short-ranged
  /// density; the direct loop kept a slot for every atom pair.
  [[nodiscard]] std::size_t neighborSlots() const noexcept {
    return nbr_.size();
  }

  /// Threads used by the most recent force call. 1 when OpenMP is off or
  /// `OMP_NUM_THREADS` is unset.
  [[nodiscard]] int threadsUsed() const noexcept { return threads_used_; }

#ifndef EAM_STANDALONE
  /// The list and the scratch live on the instance.
  [[nodiscard]] bool isSharedInstanceThreadSafe() const noexcept override {
    return false;
  }
#endif

private:
  struct element_parameters {
    const int Z;
    const double Dm;
    const double alphaM;
    const double Rm;
    const double beta1;
    const double beta2;
    const double r_cut;
    const double func_coeff[9];
  };
  static const element_parameters el_params[];

  /// One neighbour's density derivative, in atom-index order.
  struct Delta {
    long atom;
    double x;
    double y;
    double z;
  };

  std::vector<double> wrapped_;
  std::vector<double> ref_;
  std::vector<int> species_;
  std::vector<long> head_;
  std::vector<long> next_;
  std::vector<long> off_;
  std::vector<long> nbr_;
  std::vector<long> rev_;
  std::vector<long> gather_;
  std::vector<unsigned> stamp_;
  std::vector<Delta> deltas_;
  std::vector<double> rho_;
  std::vector<double> dF_;
  std::vector<double> fscale_;
  std::vector<double> thread_energy_;

  double box_len_[3]{};
  double box_ref_[3]{};
  double skin_ = 0;
  long n_atoms_ = 0;
  long n_axis_[3]{};
  std::uint64_t builds_ = 0;
  int threads_used_ = 1;
  bool built_ = false;

  void prepare(long N, const double *R, const double *box);
  [[nodiscard]] bool reusable(long N, const double *R, const int *atomicNrs,
                              const double *box) const;
  void buildList(long N, const int *atomicNrs);
  void evaluate(long N, const int *atomicNrs, double *F, double *U);
  void accumulate(long i, const int *atomicNrs, double *F, double &U,
                  std::vector<Delta> &deltas) const;
  void geometry(long i, long j, double &dx, double &dy, double &dz,
                double &r2) const;
  [[nodiscard]] int threadsRequested() const;

  [[nodiscard]] element_parameters
  get_element_parameters(int atomic_number) const;
  [[nodiscard]] static double embedding_function(const double *func_coeff,
                                                 double rho);
  [[nodiscard]] static double embedding_force(const double *func_coeff,
                                              double rho);
};
