#pragma once
// MIT License
// Copyright 2023--present rgpot developers

/**
 * @brief ORCA through the split liborcac shim (OmniPotentRPC/orcac), C ABI 1.
 *
 * The frontend dlopens liborcac on the pattern of MOPACPot, checks that it
 * speaks ABI 1 of the vendored orcac.h, lowers ORCACConfig into an
 * OrcaCConfig and keeps one orcac session, which owns one run directory
 * and one set of stored orbitals. liborcac writes the ORCA input, runs ORCA
 * and reads the `.engrad`; this class converts the Hartree energy to eV and
 * the Hartree/Bohr gradient to eV/Angstrom forces. ORCA itself is neither
 * linked nor loaded here.
 */

#include <string>

#include "rgpot/Potential.hpp"

namespace rgpot {

/// The fields of an orcac session, one for one with OrcaCConfig.
struct ORCACConfig {
  std::string method;     ///< One `!` token, e.g. "PBE0"; required.
  std::string basis;      ///< One `!` token; empty for a composite method.
  std::string keywords;   ///< Further `!` words on one line; may be empty.
  std::string blocks;     ///< Verbatim `%` blocks; may span lines.
  std::string executable; ///< orca to run; empty lets orcac search.
  std::string scratch_dir; ///< Parent of the run directory; empty for TMPDIR.
  int charge = 0;
  int multiplicity = 1;   ///< 2S+1, at least 1.
  int nprocs = 1;         ///< `%pal nprocs`; 1 omits the block.
  int maxcore_mb = 0;     ///< `%maxcore` MB per process; 0 omits it.
  double timeout_s = 3600.0; ///< Seconds per ORCA run; 0 waits forever.
  bool restart = true;    ///< Start from the last orbitals (MORead).
  /// The library to open. Empty searches ORCAC_LIBRARY, RGPOT_ORCAC_ENGINE,
  /// liborcac.so.1, liborcac.so and the other default names.
  std::string engine_path;
};

class ORCACPot : public Potential<ORCACPot> {
public:
  explicit ORCACPot(ORCACConfig config);
  ~ORCACPot() override;

  ORCACPot(const ORCACPot &) = delete;
  ORCACPot &operator=(const ORCACPot &) = delete;

  void forceImpl(const ForceInput &in, ForceOut *out) const override;

  [[nodiscard]] PotCaps caps() const noexcept override {
    return {.reentrancy = Reentrancy::PerInstance, .periodic = false};
  }

  /// The library path and every ORCACConfig field that shapes a run.
  [[nodiscard]] uint64_t paramsKey() const noexcept override;

  /// True when liborcac loaded, speaks ABI 1 and the session was created.
  [[nodiscard]] bool available() const noexcept;

  /// Why available() is false; empty when it is true.
  [[nodiscard]] const std::string &loadError() const noexcept;

  /// The candidate liborcac was opened from; empty when none opened.
  [[nodiscard]] const std::string &libraryPath() const noexcept;

  /// orcac_version() of the loaded library; empty when none loaded.
  [[nodiscard]] std::string engineVersion() const;

  [[nodiscard]] const ORCACConfig &config() const noexcept;

private:
  struct Impl;
  Impl *impl_;
};

} // namespace rgpot
