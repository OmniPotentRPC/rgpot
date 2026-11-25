#pragma once
// MIT License
// Copyright 2023--present rgpot developers

/**
 * @brief ORCA through the split liborcac shim (OmniPotentRPC/orcac).
 *
 * The frontend dlopens liborcac on the pattern of MOPACPot and keeps one
 * orcac session, which owns one working directory and one set of stored
 * orbitals. liborcac writes the ORCA input, runs ORCA and reads the
 * `.engrad`; this class converts Hartree and Hartree/Bohr to eV and
 * eV/Angstrom. ORCA itself is neither linked nor loaded here.
 */

#include <string>
#include <vector>

#include "rgpot/Potential.hpp"

namespace rgpot {

/// What an ORCACPot session is created from.
struct ORCACConfig {
  /// An unpacked flat Cap'n Proto message of the rgconf schema.
  std::vector<unsigned char> message;
  /// True when @c message is a PotentialConfig with the orca arm (liborcac
  /// then lowers its common overlay); false when it is ORCAParams.
  bool potential_config = false;
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

  /// The library path and the configuration bytes.
  [[nodiscard]] uint64_t paramsKey() const noexcept override;

  /// True when liborcac loaded and the session was created.
  [[nodiscard]] bool available() const noexcept;

  /// Why available() is false; empty when it is true.
  [[nodiscard]] const std::string &loadError() const noexcept;

  /// The candidate liborcac was opened from; empty when none opened.
  [[nodiscard]] const std::string &libraryPath() const noexcept;

  /// The session's working directory; empty without a session.
  [[nodiscard]] std::string runDirectory() const;

private:
  struct Impl;
  Impl *impl_;
};

} // namespace rgpot
