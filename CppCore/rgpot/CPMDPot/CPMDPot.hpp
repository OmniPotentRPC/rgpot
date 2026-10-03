#pragma once
// MIT License
// Copyright 2023--present rgpot developers

/**
 * @brief CPMD potential under rgpot `PotentialConfig` user parameters.
 */

#include "rgpot/Potential.hpp"
#include "rgpot/rpc/Potentials.capnp.h"

#include <cstdint>
#include <string>

namespace rgpot {

class CPMDPot : public Potential<CPMDPot> {
public:
  CPMDPot();
  explicit CPMDPot(const ::CPMDParams::Reader &params);
  ~CPMDPot() override;

  CPMDPot(const CPMDPot &) = delete;
  CPMDPot &operator=(const CPMDPot &) = delete;

  /// Every rank of the calling calculator enters this call together; other
  /// calculators need not (a host may give them no system in a batch).
  /// The call fills ForceOut on the rank that calls it. A failure is
  /// printed on every rank of the calculator, then MPI_Abort runs on
  /// MPI_COMM_WORLD, and then the exception leaves the call.
  void forceImpl(const ForceInput &in, ForceOut *out) const override;

  /// The dlopen'd engine keeps global session state: serialize
  /// process-wide. Every rank of a calculator enters forceImpl together:
  /// the engine's SCF and the error exchange are collectives on the
  /// calculator's communicator.
  [[nodiscard]] PotCaps caps() const noexcept override {
    return {.reentrancy = Reentrancy::ProcessSerial, .groupCollective = true};
  }

  /// FNV-1a over the serialized CPMDParams bytes plus kKernelVersion, so
  /// two CPMDPot instances with different params never share result-cache
  /// entries. Recomputed by the constructors and setParams.
  [[nodiscard]] uint64_t paramsKey() const noexcept override;

  /// Bump whenever the engine-facing numerics change.
  static constexpr uint64_t kKernelVersion = 1;

  bool setParams(const ::CPMDParams::Reader &params);

  /// Names the calculation the following forces belong to: an image of a
  /// band, a bead of a ring polymer. The engine keeps the converged
  /// orbitals per key, so a calculator that evaluates several of them in
  /// turn starts each SCF from that key's own previous orbitals instead of
  /// from whichever calculation it ran last. Every rank of a calculator
  /// names the same key before the same force. An engine without
  /// cpmdc_session_select_orbitals ignores the key.
  void selectOrbitals(int64_t key);

  /// True when the loaded engine keeps orbitals per key.
  [[nodiscard]] bool keepsOrbitalsPerKey() const;

  void getParams(::CPMDParams::Builder out) const;

  bool setPotentialConfig(const ::PotentialConfig::Reader &cfg,
                          std::string *message_out = nullptr);

  bool available() const;
  static bool probe_available();
  static bool abi_available();

  /// Collective on MPI_COMM_WORLD. Splits into calculators of
  /// ranks_per_calc ranks via cpmdc_bind_calculator. One NEB image
  /// is one calculator. A second band calls this on its own world.
  /// Every rank must call it before the first force. Returns the
  /// group index, or -1 when the engine has no bind symbol.
  ///
  /// Ordering: the engine is loaded here and stays loaded for the rest
  /// of the process, whether or not a CPMDPot exists yet. The split
  /// communicator lives inside the engine and the calculator hooks run
  /// once, so an engine unloaded between this call and the first
  /// CPMDPot would lose the split with no way to redo it. Construct
  /// CPMDPot instances after this call; they share the loaded engine.
  static int bindCalculators(int ranks_per_calc);

private:
  struct Impl;
  Impl *impl_;

  void forceImplOrThrow(const ForceInput &in, ForceOut *out) const;
};

} // namespace rgpot
