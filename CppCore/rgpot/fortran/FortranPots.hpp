#pragma once
// MIT License
// Copyright 2023--present rgpot developers

/**
 * @brief C++ faces for the Fortran 2018 potential kernels.
 *
 * Each kernel exposes one `bind(c)` entry taking flat buffers and
 * returning a status. Those entries stay inside librgpot: `forceImpl` is
 * defined out of line, and the archive holding the Fortran objects is
 * linked with `--exclude-libs`, so no Fortran symbol reaches the dynamic
 * table and consumers reach the kernels only through these classes.
 *
 * Each instance owns a Fortran workspace (rgpot_workspace.f90): its own
 * neighbour table, which vesin keeps between calls and rebuilds only when
 * an atom leaves the Verlet skin, and its own error message. The kernels
 * carry no other state, so instances evaluate on separate threads at once
 * (`Reentrancy::PerInstance`), and multi-image callers keep one instance
 * per image (`perImageInstances`) so each table follows one geometry
 * instead of every image rebuilding a shared one.
 */

#include "rgpot/ForceStructs.hpp"
#include "rgpot/Potential.hpp"
#include "rgpot/pot_caps.hpp"
#include "rgpot/pot_types.hpp"

namespace rgpot {
namespace fortranpots {

/// Owner of one Fortran workspace handle. Copies get a workspace of their
/// own, so a cloned potential never shares a neighbour table.
class FortranWorkspace {
public:
  FortranWorkspace();
  ~FortranWorkspace();
  FortranWorkspace(const FortranWorkspace &);
  FortranWorkspace &operator=(const FortranWorkspace &);
  FortranWorkspace(FortranWorkspace &&) = delete;
  FortranWorkspace &operator=(FortranWorkspace &&) = delete;
  [[nodiscard]] void *get() const noexcept { return m_handle; }

private:
  void *m_handle;
};

#define RGPOT_FORTRAN_POT_CLASS(ClassName, PotTypeValue)                       \
  class ClassName : public Potential<ClassName> {                              \
  public:                                                                      \
    ClassName() : Potential(PotType::PotTypeValue) {}                          \
    void forceImpl(const ForceInput &in, ForceOut *out) const override;        \
    [[nodiscard]] PotCaps caps() const noexcept override {                     \
      return {.reentrancy = Reentrancy::PerInstance,                           \
              .perImageInstances = true};                                      \
    }                                                                          \
                                                                               \
  private:                                                                     \
    FortranWorkspace m_ws;                                                     \
  }

RGPOT_FORTRAN_POT_CLASS(SWPot, SWSi);
RGPOT_FORTRAN_POT_CLASS(EDIPPot, EDIP);
RGPOT_FORTRAN_POT_CLASS(LenoskyPot, LenoskySi);
RGPOT_FORTRAN_POT_CLASS(TersoffPot, TersoffSi);
RGPOT_FORTRAN_POT_CLASS(EAMAlPot, EAMAl);
RGPOT_FORTRAN_POT_CLASS(FeHePot, FeHe);
RGPOT_FORTRAN_POT_CLASS(CuH2Pot, CuH2);
RGPOT_FORTRAN_POT_CLASS(WaterHPot, WaterH);

#undef RGPOT_FORTRAN_POT_CLASS

} // namespace fortranpots
} // namespace rgpot
