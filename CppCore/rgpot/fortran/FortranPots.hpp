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
 * neighbour table. The builder keeps that list between calls and rebuilds
 * it only when an atom leaves the Verlet skin. The kernels also keep an
 * error message. The kernels
 * carry no other state, so instances evaluate on separate threads at once
 * (`Reentrancy::PerInstance`), and multi-image callers keep one instance
 * per image (`perImageInstances`) so each table follows one geometry
 * instead of every image rebuilding a shared one.
 */

#include "rgpot/ForceStructs.hpp"
#include "rgpot/Potential.hpp"
#include "rgpot/Tersoff/TersoffSiC.hpp"
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

RGPOT_FORTRAN_POT_CLASS(EDIPPot, EDIP);
RGPOT_FORTRAN_POT_CLASS(LenoskyPot, LenoskySi);
RGPOT_FORTRAN_POT_CLASS(EAMAlPot, EAMAl);
RGPOT_FORTRAN_POT_CLASS(FeHePot, FeHe);
RGPOT_FORTRAN_POT_CLASS(WaterHPot, WaterH);

#undef RGPOT_FORTRAN_POT_CLASS

/// Copper-hydrogen EAM (PotType::CuH2), with the analytic Hessian-vector
/// product of rgpot_cuh2.f90.
class CuH2Pot : public Potential<CuH2Pot> {
public:
  CuH2Pot() : Potential(PotType::CuH2) {}
  void forceImpl(const ForceInput &in, ForceOut *out) const override;
  /// @f$H v@f$ from the pair, density and embedding second derivatives.
  /// @a v and @a hv hold @c 3*in.nAtoms doubles.
  void hessianVectorImpl(const ForceInput &in, const double *v,
                         double *hv) const override;
  [[nodiscard]] PotCaps caps() const noexcept override {
    return {.reentrancy = Reentrancy::PerInstance,
            .perImageInstances = true,
            .analyticHvp = true};
  }

private:
  FortranWorkspace m_ws;
};

/// Stillinger-Weber silicon (PotType::SWSi). Epsilon is eOn's
/// 2.16826 * 1.06767385 = 2.3149945 eV, not the published 2.16723920 eV
/// of PotType::SWSi1985. Their ratio is 1.068176741337. The factor
/// 1.06767385 is untraced; see rgpot_sw.f90.
class SWPot : public Potential<SWPot> {
public:
  SWPot() : Potential(PotType::SWSi) {}
  /// Forces, energy and the Cauchy stress sigma = (1/V) dE/dε in
  /// eV/Angstrom^3, the convention of stress.hpp.
  void forceImpl(const ForceInput &in, ForceOut *out) const override;
  /// Batched forces, energy and optional stress, with the same opt-in flag.
  void forceBatchImpl(const ForceBatch &batch) const override;
  [[nodiscard]] PotCaps caps() const noexcept override {
    return {.reentrancy = Reentrancy::PerInstance,
            .perImageInstances = true,
            .stress = true};
  }

private:
  FortranWorkspace m_ws;
};

/// Tersoff silicon (PotType::TersoffSi). All-silicon calls, and a null
/// species pointer, use the Fortran gather. Any other atomic number
/// uses the 1989 silicon-carbon kernel. There is no second PotType.
/// When ForceInput::compute_stress is set, periodic all-silicon calls
/// compute the Cauchy stress sigma = (1/V) dE/dε in eV/Angstrom^3 on the
/// gather path, and mixed-species calls compute it in TersoffSiC.
/// Force-only calls skip the virial and leave has_stress at 0.
/// A stress request with a null species pointer throws
/// std::invalid_argument, which the C ABI stress path reports as
/// RGPOT_INVALID_PARAMETER.
class TersoffPot : public Potential<TersoffPot> {
public:
  TersoffPot() : Potential(PotType::TersoffSi) {}
  void forceImpl(const ForceInput &in, ForceOut *out) const override;
  [[nodiscard]] PotCaps caps() const noexcept override {
    return {.reentrancy = Reentrancy::PerInstance,
            .perImageInstances = true,
            .stress = true};
  }

  /// Held list for the calls that reach the 1989 kernel, as
  /// TersoffSiC::setHeldSkin. The gather keeps its own neighbour table.
  void setHeldSkin(double skin) { m_sic.setHeldSkin(skin); }

private:
  FortranWorkspace m_ws;
  TersoffSiC m_sic;
};

} // namespace fortranpots
} // namespace rgpot
