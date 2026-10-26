#pragma once
// MIT License
// Copyright 2023--present rgpot developers
//
// The TIP4P, SPC/E and TIP4P-Pt kernels are the eOn water tree
// (client/potentials/Water and client/potentials/Water_Pt), BSD-3-Clause,
// copyright the eOn Development Team. The eOn wrappers call them with a
// cutoff of 8.5 angstrom and a switching width of 1.0 angstrom.

#include <cstdint>
#include <memory>

#include "rgpot/ParamHash.hpp"
#include "rgpot/Potential.hpp"

namespace rgpot {

/// Cutoff and switching width shared by the flexible water kernels.
/// The defaults are the values the eOn wrappers pass.
struct WaterConfig {
  double cutoff = 8.5;
  double switching_width = 1.0;
};

/// Flexible TIP4P. Atoms are ordered H, H, ..., O, O, one molecule per
/// three atoms.
class TIP4PPot : public Potential<TIP4PPot> {
public:
  TIP4PPot();
  explicit TIP4PPot(const WaterConfig &config);
  ~TIP4PPot() override;

  TIP4PPot(const TIP4PPot &) = delete;
  TIP4PPot &operator=(const TIP4PPot &) = delete;

  [[nodiscard]] const WaterConfig &config() const noexcept { return m_config; }

  [[nodiscard]] PotCaps caps() const noexcept override {
    return {.reentrancy = Reentrancy::PerInstance, .perImageInstances = true};
  }

  [[nodiscard]] uint64_t paramsKey() const noexcept override {
    return m_paramsKey;
  }

  void forceImpl(const ForceInput &in, ForceOut *out) const override;

private:
  struct Impl;
  WaterConfig m_config;
  uint64_t m_paramsKey = 0;
  std::unique_ptr<Impl> m_impl;
};

/// Flexible SPC/E with the CCL intramolecular term. Same atom order as TIP4P.
class SPCEPot : public Potential<SPCEPot> {
public:
  SPCEPot();
  explicit SPCEPot(const WaterConfig &config);
  ~SPCEPot() override;

  SPCEPot(const SPCEPot &) = delete;
  SPCEPot &operator=(const SPCEPot &) = delete;

  [[nodiscard]] const WaterConfig &config() const noexcept { return m_config; }

  [[nodiscard]] PotCaps caps() const noexcept override {
    return {.reentrancy = Reentrancy::PerInstance, .perImageInstances = true};
  }

  [[nodiscard]] uint64_t paramsKey() const noexcept override {
    return m_paramsKey;
  }

  void forceImpl(const ForceInput &in, ForceOut *out) const override;

private:
  struct Impl;
  WaterConfig m_config;
  uint64_t m_paramsKey = 0;
  std::unique_ptr<Impl> m_impl;
};

/// TIP4P water on platinum (Zhu-Philpott). Hydrogens come in pairs at the
/// front, then the oxygens, then the platinum atoms.
class TIP4PPtPot : public Potential<TIP4PPtPot> {
public:
  TIP4PPtPot();
  explicit TIP4PPtPot(const WaterConfig &config);
  ~TIP4PPtPot() override;

  TIP4PPtPot(const TIP4PPtPot &) = delete;
  TIP4PPtPot &operator=(const TIP4PPtPot &) = delete;

  [[nodiscard]] const WaterConfig &config() const noexcept { return m_config; }

  [[nodiscard]] PotCaps caps() const noexcept override {
    return {.reentrancy = Reentrancy::PerInstance, .perImageInstances = true};
  }

  [[nodiscard]] uint64_t paramsKey() const noexcept override {
    return m_paramsKey;
  }

  void forceImpl(const ForceInput &in, ForceOut *out) const override;

private:
  struct Impl;
  WaterConfig m_config;
  uint64_t m_paramsKey = 0;
  std::unique_ptr<Impl> m_impl;
};

} // namespace rgpot
