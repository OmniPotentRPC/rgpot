#pragma once
// MIT License
// Copyright 2023--present rgpot developers

/**
 * @file OrcaPot.hpp
 * @brief Molecular ORCA potential. One `orca` process per force call.
 *
 * Positions are Angstrom. ORCA's `.engrad` file reports the energy in
 * hartree and the Cartesian gradient in hartree/bohr. Forces returned
 * here are eV/Angstrom, the negative of that gradient. The box is
 * ignored: an EnGrad job is a molecule, not a periodic cell.
 *
 * After a successful call the `.gbw` is kept. Later calls pass
 * `! MORead` and `%moinp` unless `OrcaConfig::reuse_gbw` is false. A
 * failed MORead attempt is retried once from a fresh guess.
 */

#include "rgpot/Potential.hpp"

#include <cstddef>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

namespace rgpot {

/**
 * @brief How one ORCA process is started.
 *
 * `executable` empty means `RGPOT_ORCA`, then `ORCA`, then `orca` on
 * `PATH`. `nprocs` and `maxcore_mb` of 0 omit `%pal` and `%maxcore`.
 * `keywords` is extra simple-input tokens, or a raw block when it
 * contains a newline or starts with `!` or `%`.
 */
struct OrcaConfig {
  int charge = 0;
  int multiplicity = 1;
  std::string method = "HF";
  std::string basis = "STO-3G";
  std::string keywords;
  std::string executable;
  bool reuse_gbw = true;
  int nprocs = 0;
  int maxcore_mb = 0;
};

/// Cartesian geometry in Angstrom, row-major, length `3 * n`.
struct OrcaAtoms {
  std::size_t n = 0;
  const double *positions = nullptr;
  const int *atomic_numbers = nullptr;
};

/// Parsed `.engrad` contents, still in ORCA's units.
struct OrcaEngrad {
  int n_atoms = 0;
  double energy_eh = 0.0;
  std::vector<double> gradient_eh_bohr;
};

/**
 * @brief Text of an EnGrad input.
 *
 * `moread` adds `! MORead` and `%moinp "<moinp>"`. `moinp` is the file
 * name ORCA opens, relative to the working directory of the process.
 */
std::string renderOrcaInput(const OrcaConfig &cfg, const OrcaAtoms &atoms,
                            bool moread, std::string_view moinp);

/// Throws std::runtime_error when a required section is missing or short.
OrcaEngrad parseOrcaEngrad(std::string_view text);

/**
 * @brief ORCA energy and forces through a subprocess.
 * @ingroup rgpot_potentials
 */
class OrcaPot : public Potential<OrcaPot> {
public:
  OrcaPot();
  explicit OrcaPot(const OrcaConfig &config);
  ~OrcaPot() override;

  OrcaPot(const OrcaPot &) = delete;
  OrcaPot &operator=(const OrcaPot &) = delete;

  void forceImpl(const ForceInput &in, ForceOut *out) const override;

  /// One scratch directory and one `.gbw`. A saddle image needs its own
  /// instance. The calculation is molecular.
  [[nodiscard]] PotCaps caps() const noexcept override {
    return {.reentrancy = Reentrancy::PerInstance,
            .perImageInstances = true,
            .periodic = false};
  }

  [[nodiscard]] uint64_t paramsKey() const noexcept override {
    return m_paramsKey;
  }

  [[nodiscard]] const OrcaConfig &config() const noexcept { return m_cfg; }

  /// Drop the stored `.gbw`. The next call uses ORCA's own guess.
  void reset();

  /// Replace the charge and multiplicity and drop the stored `.gbw`.
  void setChargeSpin(int charge, int multiplicity);

  /// True when the last successful call started from the previous `.gbw`.
  [[nodiscard]] bool reusedWavefunction() const;

  /**
   * @brief Absolute path of the binary `configured` names.
   *
   * An empty string uses `RGPOT_ORCA`, then `ORCA`, then `orca` on
   * `PATH`. Throws std::runtime_error when nothing executable is found.
   */
  static std::string resolveExecutable(const std::string &configured);

private:
  void validate(const OrcaConfig &cfg) const;
  uint64_t hashConfig(const OrcaConfig &cfg) const;

  OrcaConfig m_cfg;
  uint64_t m_paramsKey = 0;
  std::string m_scratch;
  mutable bool m_have_gbw = false;
  mutable bool m_last_reused = false;
  mutable std::mutex m_mu;
};

} // namespace rgpot
