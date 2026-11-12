#pragma once
// MIT License
// Copyright 2023--present rgpot developers

/**
 * @file ORCAPot.hpp
 * @brief Potential that evaluates energy and forces by running ORCA.
 *
 * Each call writes an ORCA input with `! EnGrad`, runs the ORCA
 * executable as a subprocess, and reads the `.engrad` file. ORCA reports
 * the energy in Hartree and the Cartesian gradient in Hartree/Bohr.
 * rgpot returns the energy in eV and the forces in eV/Angstrom. The force
 * is the negative of the gradient.
 *
 * The executable is the config path when that is set, otherwise the
 * `ORCA_EXE` environment variable, otherwise an `orca` binary on `PATH`.
 * A missing binary throws when a geometry is evaluated.
 *
 * One instance keeps a private work directory. The first evaluation is a
 * cold guess. Later evaluations copy the previous `.gbw` to a different
 * name and pass it with `! MORead` and `%moinp`. ORCA refuses to read and
 * write the same `.gbw`. A missing `.gbw`, or one failed MORead run, falls
 * back to a single cold guess.
 *
 * The calculation is molecular. The cell is not used and there is no
 * neighbour list. Saddle-search drivers should give each image its own
 * instance: the saved wavefunction belongs to one geometry.
 */

#include "rgpot/Potential.hpp"

#include <atomic>
#include <mutex>
#include <string>

namespace rgpot {

/**
 * @brief Method, basis, charge, multiplicity, and the ORCA executable.
 *
 * `keywords` adds simple-input tokens on the `!` line (`TightSCF`,
 * `D3BJ`). Text after the first newline is written as extra input lines
 * between that line and the coordinate block, which is where an ORCA
 * `%` block belongs. The engine always adds `EnGrad`, and `MORead` when
 * it reuses a wavefunction.
 */
struct ORCAConfig {
  std::string method = "HF";
  std::string basis = "def2-SVP";
  int charge = 0;
  int multiplicity = 1;
  std::string keywords;
  /// Empty: `ORCA_EXE`, then `orca` on `PATH`.
  std::string executable;
};

/**
 * @brief ORCA energy and gradient for one molecule.
 * @ingroup rgpot_potentials
 */
class ORCAPot : public Potential<ORCAPot> {
public:
  using Config = ORCAConfig;

  explicit ORCAPot(const Config &config);
  ~ORCAPot() override;

  ORCAPot(const ORCAPot &) = delete;
  ORCAPot &operator=(const ORCAPot &) = delete;

  void forceImpl(const ForceInput &in, ForceOut *out) const override;

  /// One work directory and one wavefunction. Not periodic.
  [[nodiscard]] PotCaps caps() const noexcept override {
    return {.reentrancy = Reentrancy::PerInstance,
            .perImageInstances = true,
            .periodic = false};
  }

  [[nodiscard]] uint64_t paramsKey() const noexcept override {
    return params_key_.load(std::memory_order_relaxed);
  }

  [[nodiscard]] const Config &config() const noexcept { return config_; }

  /// Replace method options. A change of method, basis, charge,
  /// multiplicity, or keywords drops the saved wavefunction.
  void applyParams(const Config &config);

  /// Charge and spin multiplicity for later evaluations. A change drops
  /// the saved wavefunction. `multiplicity` is 2S+1.
  void setChargeSpin(int charge, int multiplicity);

  /// Absolute or relative path that `forceImpl` would execute.
  /// Throws std::runtime_error when no executable can be found.
  [[nodiscard]] static std::string locate(const std::string &executable);

  /// False when @c locate would throw.
  [[nodiscard]] static bool available(const std::string &executable = {});

private:
  void rehash();
  void dropWavefunction() const;

  Config config_;
  std::string work_;
  mutable std::mutex mu_;
  std::atomic<uint64_t> params_key_{0};
};

} // namespace rgpot
