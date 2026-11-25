#pragma once
// MIT License
// Copyright 2023--present rgpot developers

/**
 * @file ORCAPot.hpp
 * @brief ORCA energies and gradients through the runtime-loaded liborcac.
 *
 * Pure consumer of the split orcac library: the frontend dlopens liborcac,
 * checks it against the vendored orcac.h, and runs every force call through
 * one orcac session. rgpot links no ORCA and no orcac code.
 *
 * liborcac is found from ORCAConfig::library, else ORCAC_LIBRARY, else
 * RGPOT_ORCAC_ENGINE, else the platform library names. A path that is given
 * is the only one tried. A library whose ABI version is not 1, or whose
 * reported OrcaCConfig or OrcaCResult size is below the ABI 1 minimum, is
 * refused before any session or evaluation. Positions go to orcac in
 * Angstrom; the Eh energy and the Eh/bohr gradient come back and are
 * converted to eV and eV/Angstrom forces with the constants of units.hpp.
 *
 * The box rule and the pinning of the orca executable are those of
 * ORCALegacyPot (ORCAGuards.hpp). Each call starts from ORCA's own guess
 * unless ORCAConfig::guess_policy is MoRead; with MoRead the result cache is
 * not used (see setCache).
 */

#include "rgpot/ORCAPot/ORCAError.hpp"
#include "rgpot/Potential.hpp"

#include <cstdint>
#include <string>

namespace rgpot {

/**
 * @brief Parameters for ORCAPot, mirrored by ORCAParams on the wire.
 */
struct ORCAConfig {
  static constexpr double kDefaultTimeout = 3600.0;

  /// How a run starts its SCF.
  enum class GuessPolicy : int {
    MoRead = 0, ///< From the session's last orbitals (ORCAC_GUESS_CHAIN).
    Fresh = 1,  ///< Always from ORCA's own guess (ORCAC_GUESS_FRESH).
  };

  std::string method;   ///< Simple-input method keyword; required.
  std::string basis;    ///< Basis keyword; empty for a composite method.
  std::string keywords; ///< Further simple-input keywords, one line.
  std::string blocks;   ///< Verbatim % blocks; may span lines.
  int charge = 0;
  int multiplicity = 1; ///< 2S+1, at least 1.
  int nprocs = 1;       ///< At least 1.
  int maxcore_mb = 0;   ///< Memory per process in MB; 0 keeps ORCA's default.
  std::string executable; ///< orca to run; empty lets orcac search.
  std::string library;    ///< liborcac to load; empty searches.
  /// Fresh by default, so a result does not depend on the calls before it.
  /// With MoRead, ORCAPot does not use a result cache it is given.
  GuessPolicy guess_policy = GuessPolicy::Fresh;
  double timeout_s = kDefaultTimeout; ///< Seconds per run; 0 waits forever.
  std::string scratch_dir; ///< Run directory parent; empty for the system one.
  /// Must stay false: orcac ABI 1 always removes the run directory.
  bool keep_files = false;
  bool fresh_retry = true; ///< Rerun once from a fresh guess on failure.
  /// Deprecated: kept from the 3.8.0 ORCAConfig; set guess_policy instead.
  /// True makes the instance use GuessPolicy::MoRead whatever guess_policy
  /// says. Not marked [[deprecated]], since the implicit copy and move of
  /// ORCAConfig would then warn in every program that copies one.
  bool restart = false;
};

/**
 * @class ORCAPot
 * @brief ORCA through liborcac, one orcac session per instance.
 *
 * The constructor never throws for a missing or mismatched liborcac or a
 * missing orca executable: it records the reason, available() is false and
 * the first force throws ORCAError with it. A config that cannot run
 * throws std::invalid_argument.
 */
class ORCAPot : public Potential<ORCAPot> {
public:
  explicit ORCAPot(const ORCAConfig &config);
  ~ORCAPot() override;

  ORCAPot(const ORCAPot &) = delete;
  ORCAPot &operator=(const ORCAPot &) = delete;

  /// One ORCA energy and gradient. ORCA has no periodic boundary
  /// conditions, so a box is only padding: absent, all zero, or a cell in
  /// which every pair of atoms is its own nearest periodic image, checked
  /// along every nonzero lattice vector. Positions are passed unchanged; a
  /// box with a closer periodic image, a singular box, a cell too skewed to
  /// check (including one whose image bound is not finite or out of range),
  /// a non-finite box entry and, with a box, a non-finite position throw
  /// std::invalid_argument. The call throws ORCAError without running ORCA
  /// once the executable has changed (see executableIdentity()), and
  /// discards the result when it changed during the run.
  void forceImpl(const ForceInput &in, ForceOut *out) const override;

  [[nodiscard]] PotCaps caps() const noexcept override {
    return {.reentrancy = Reentrancy::PerInstance,
            .perImageInstances = true,
            .periodic = false,
            .stress = false};
  }

  /// Low 64 bits of the SHA-256 of the method, basis, keywords and blocks,
  /// the resolved executable and the size, modification time, change time,
  /// device, inode and content hash of that file, ORCAC_ABI_VERSION, the
  /// charge, the multiplicity, the guess policy, and whether the executable
  /// has changed since. Once it has, the key matches no entry an unchanged
  /// instance writes.
  [[nodiscard]] uint64_t paramsKey() const noexcept override;

#ifdef RGPOT_HAS_CACHE
  /// With GuessPolicy::MoRead the result cache is not used: a call it
  /// served would run no ORCA and leave orcac's stored guess unchanged, so
  /// the next result would depend on which calls happened to be cached.
  /// With Fresh the cache is used as given.
  void setCache(rgpot::cache::PotentialCache *c) override {
    Potential<ORCAPot>::setCache(
        config().guess_policy == ORCAConfig::GuessPolicy::MoRead ? nullptr
                                                                 : c);
  }
#endif

  /// True when liborcac loaded, passed the handshake, found an executable
  /// and opened a session.
  [[nodiscard]] bool available() const noexcept;

  /// Why available() is false; empty when it is true.
  [[nodiscard]] const std::string &loadError() const noexcept;

  /// Path liborcac was opened from; empty when none opened.
  [[nodiscard]] const std::string &libraryPath() const noexcept;

  /// orcac_version() of the loaded library; empty when none loaded.
  [[nodiscard]] std::string engineVersion() const;

  /// The absolute orca path orcac_probe_executable resolved, with symlinks
  /// followed: the file orcac runs. Every call resolves the executable
  /// again and throws ORCAError, without running anything, once it names
  /// another file or the size, modification time, change time, device,
  /// inode or content hash of this file changed.
  [[nodiscard]] const std::string &executableIdentity() const noexcept;

  [[nodiscard]] const ORCAConfig &config() const noexcept;

  /// Charge and multiplicity of later runs, through orcac_configure.
  /// Throws std::invalid_argument for a multiplicity below 1 and
  /// ORCAError when orcac refuses; the previous values then stay.
  void setChargeMultiplicity(int charge, int multiplicity);

  /// Stops the force call running on this instance (orcac_cancel); it then
  /// throws ORCAError with ORCAC_E_CANCELLED. Safe from any thread of the
  /// process. A call that starts after cancel() is not affected.
  void cancel() noexcept;

  /// True when liborcac loads from the default search and passes the
  /// handshake. Cached after the first call.
  [[nodiscard]] static bool probeAvailable();

  /// probeAvailable(), and orcac_available() reports an orca executable on
  /// orcac's own search. Cached after the first call.
  [[nodiscard]] static bool abiAvailable();

private:
  struct Impl;
  Impl *impl_;
};

} // namespace rgpot
