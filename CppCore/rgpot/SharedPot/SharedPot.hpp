#pragma once
// MIT License
// Copyright 2023--present rgpot developers

/**
 * @brief A potential loaded from a shared library at runtime.
 *
 * The dlopen route: any engine adapter that exports the single C symbol
 * below becomes an rgpot backend without recompiling rgpot. NWChem, CP2K
 * and their kin attach as small adapter libraries built against their own
 * toolchains, and every rgpot consumer, the RPC server included, gains the
 * backend for the price of a path.
 *
 * The exported contract, C ABI:
 * @code
 * int rgpot_shared_force(size_t nAtoms, const double *pos,
 *                        const int *atmnrs, const double *box,
 *                        double *F, double *energy, double *variance);
 * @endcode
 * Positions in Angstrom, energies in eV, forces in eV/Angstrom, return 0 on
 * success. The adapter owns any engine state behind the symbol.
 */

#include <string>

#include "rgpot/Potential.hpp"

namespace rgpot {

class SharedPot : public Potential<SharedPot> {
public:
  explicit SharedPot(const std::string &library_path);
  ~SharedPot();
  SharedPot(const SharedPot &) = delete;
  SharedPot &operator=(const SharedPot &) = delete;

  void forceImpl(const ForceInput &in, ForceOut *out) const override;

private:
  void *m_handle{nullptr};
  int (*m_force)(size_t, const double *, const int *, const double *,
                 double *, double *, double *){nullptr};
};

} // namespace rgpot
