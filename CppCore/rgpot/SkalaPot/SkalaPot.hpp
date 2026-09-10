#pragma once
// MIT License — Skala XC via NWChemPot / libnwchemc

#include "rgpot/NWChemPot/NWChemPot.hpp"
#include "rgpot/ParamHash.hpp"
#include "rgpot/Potential.hpp"
#include "rgpot/SkalaPot/SkalaConfig.hpp"
#include "rgpot/rpc/Potentials.capnp.h"

#include <cstdint>
#include <memory>

namespace rgpot {

/**
 * @brief DFT oracle: NWChem SCF, Skala exchange-correlation.
 *
 * Builds an NWChemParams message (theory=dft, dft.xc=skala-1.1 by
 * default) and evaluates through NWChemPot. No Python helper.
 */
class SkalaPot : public Potential<SkalaPot> {
public:
  SkalaPot();
  explicit SkalaPot(const SkalaConfig &config);
  ~SkalaPot() override = default;

  SkalaPot(const SkalaPot &) = delete;
  SkalaPot &operator=(const SkalaPot &) = delete;

  void forceImpl(const ForceInput &in, ForceOut *out) const override;

  [[nodiscard]] PotCaps caps() const noexcept override {
    return {.reentrancy = Reentrancy::ProcessSerial, .periodic = false};
  }

  [[nodiscard]] const SkalaConfig &config() const noexcept { return m_config; }

  void setChargeMultiplicity(int charge, int multiplicity);

  [[nodiscard]] bool available() const { return m_nwchem.available(); }

  void getParams(::NWChemParams::Builder out) const { m_nwchem.getParams(out); }

  [[nodiscard]] uint64_t paramsKey() const noexcept override {
    return m_paramsKey;
  }

private:
  static constexpr uint64_t kKernelVersion = 1;

  SkalaConfig m_config;
  uint64_t m_paramsKey{0};
  NWChemPot m_nwchem;

  void recomputeParamsKey();
  void applyToEngine();
};

} // namespace rgpot
