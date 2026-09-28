#pragma once
// MIT License
// Copyright 2023--present rgpot developers

/**
 * @brief CPMD potential under rgpot `PotentialConfig` user parameters.
 */

#include "rgpot/Potential.hpp"
#include "rgpot/rpc/Potentials.capnp.h"

#include <string>

namespace rgpot {

class CPMDPot : public Potential<CPMDPot> {
public:
  CPMDPot();
  explicit CPMDPot(const ::CPMDParams::Reader &params);
  ~CPMDPot() override;

  CPMDPot(const CPMDPot &) = delete;
  CPMDPot &operator=(const CPMDPot &) = delete;

  void forceImpl(const ForceInput &in, ForceOut *out) const override;

  /// The dlopen'd engine keeps global session state: serialize
  /// process-wide.
  [[nodiscard]] PotCaps caps() const noexcept override {
    return {.reentrancy = Reentrancy::ProcessSerial};
  }


  bool setParams(const ::CPMDParams::Reader &params);

  void getParams(::CPMDParams::Builder out) const;

  bool setPotentialConfig(const ::PotentialConfig::Reader &cfg,
                          std::string *message_out = nullptr);

  bool available() const;
  static bool probe_available();
  static bool abi_available();

  // Collective on MPI_COMM_WORLD. Splits into calculators of
  // ranks_per_calc ranks via cpmdc_bind_calculator. One NEB image
  // is one calculator. A second band calls this on its own world.
  // Every rank must call it before the first force. Returns the
  // group index, or -1 when the engine has no bind symbol.
  static int bindCalculators(int ranks_per_calc);

private:
  struct Impl;
  Impl *impl_;
};

} // namespace rgpot
