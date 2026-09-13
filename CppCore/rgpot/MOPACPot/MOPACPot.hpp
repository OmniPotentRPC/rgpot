#pragma once
// MIT License
// Copyright 2023--present rgpot developers

/**
 * @brief MOPAC AM1 potential: dlopen `libmopacc.so`, no Cap'n Proto.
 *
 * User path: `MOPACConfig` only. The frontend packs that into
 * `MopacCParams` and passes it through C ABI symbols resolved by dlopen.
 */

#include "rgpot/Potential.hpp"

#include <cstdint>
#include <string>

namespace rgpot {

/** Method knobs for the mopacc engine. `engine_path` is host-only. */
struct MOPACConfig {
  int charge = 0;
  int spin = 0;  //!< OpenMOPAC spin excitations, not multiplicity
  int model = 4; //!< 4 = AM1
  double tolerance = 1.0;
  int max_time = 3600;
  std::string engine_path;
};

class MOPACPot : public Potential<MOPACPot> {
public:
  MOPACPot();
  explicit MOPACPot(const MOPACConfig &cfg);
  ~MOPACPot() override;

  MOPACPot(const MOPACPot &) = delete;
  MOPACPot &operator=(const MOPACPot &) = delete;

  void forceImpl(const ForceInput &in, ForceOut *out) const override;

  /// The dlopen'd engine keeps global session state: serialize
  /// process-wide.
  [[nodiscard]] PotCaps caps() const noexcept override {
    return {.reentrancy = Reentrancy::ProcessSerial, .periodic = false};
  }

  bool setParams(const MOPACConfig &cfg);
  [[nodiscard]] MOPACConfig getParams() const;

  bool available() const;
  static bool probe_available();
  static bool abi_available();

  [[nodiscard]] uint64_t paramsKey() const noexcept override;

private:
  struct Impl;
  Impl *impl_;
};

} // namespace rgpot
