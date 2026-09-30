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

  /// When a calculator group is bound and calculatorWorldSize() is
  /// greater than 1, every rank enters this call. The call fills
  /// ForceOut on the rank that calls it. A failure is printed on every
  /// rank in the call, then MPI_Abort runs on MPI_COMM_WORLD, and then
  /// the exception leaves the call.
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

  /// When this build has MPI and the engine exports
  /// cpmdc_adopt_calculator_comm, every rank calls this before the
  /// first force. One MPI_Comm_split builds the calculator
  /// communicator and that symbol receives it. One NEB image is one
  /// calculator. A second band calls this on its own world. Returns
  /// the group index. Returns -1, before any MPI call, when this
  /// build has MPI and the engine has no adopt symbol.
  static int bindCalculators(int ranks_per_calc);

private:
  struct Impl;
  Impl *impl_;

  void forceImplOrThrow(const ForceInput &in, ForceOut *out) const;
};

} // namespace rgpot
