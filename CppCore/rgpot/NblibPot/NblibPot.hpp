#pragma once
// MIT License
// Copyright 2023--present rgpot developers

/**
 * @brief A GROMACS nblib topology as a regular rgpot potential.
 *
 * The constructor dlopens @c librgpot_nblib (built with -Dwith_nblib) and
 * keeps one nblib force calculator in it. GROMACS is linked into that
 * library only, so a program that links librgpot does not link GROMACS. The
 * call surface is Angstrom and eV. The potential fills the energy, the
 * forces and the stress, the stress being 2 Xi / V for the GROMACS virial Xi.
 * It needs a rectangular box and a topology with the atom count of the
 * call; the atomic numbers are not read.
 */

#include "rgpot/Potential.hpp"

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace rgpot {

struct NblibConfig {
  /// Topology text (docs/orgmode/howto/engine-nblib.org). Read from
  /// topology_file when empty.
  std::string topology;
  std::string topology_file;
  /// librgpot_nblib. Empty searches RGPOT_NBLIB_ENGINE, then librgpot_nblib.so.
  std::string engine_path;
};

class NblibPot : public Potential<NblibPot> {
public:
  explicit NblibPot(const NblibConfig &config);
  ~NblibPot() override;
  NblibPot(const NblibPot &) = delete;
  NblibPot &operator=(const NblibPot &) = delete;
  NblibPot(NblibPot &&) noexcept;
  NblibPot &operator=(NblibPot &&) noexcept;

  void forceImpl(const ForceInput &in, ForceOut *out) const override;
  [[nodiscard]] PotCaps caps() const noexcept override;

  [[nodiscard]] const NblibConfig &config() const noexcept { return config_; }
  [[nodiscard]] std::size_t atomCount() const noexcept;
  /// Cutoff of the topology, Angstrom.
  [[nodiscard]] double cutoff() const noexcept;
  /// amu, one per atom.
  [[nodiscard]] std::vector<double> masses() const;
  /// GROMACS virial -1/2 sum r (x) F of the last call, eV, row-major.
  [[nodiscard]] std::vector<double> lastVirial() const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  NblibConfig config_;
};

std::unique_ptr<NblibPot> make_nblib_potential(const NblibConfig &config);

} // namespace rgpot
