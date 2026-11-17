#pragma once
// MIT License
// Copyright 2023--present rgpot developers

/**
 * @brief Energy, forces and virial of a topology through nblib.
 *
 * Internal to librgpot_nblib: this header includes nblib and GROMACS headers
 * and is not installed. Lengths are nm, energies kJ/mol, forces kJ/mol/nm and
 * masses amu, the units nblib takes.
 */

#include "NblibTopology.hpp"

#include <array>
#include <cstddef>
#include <memory>
#include <vector>

#include "gromacs/utility/arrayref.h"
#include "nblib/nblib.h"

namespace rgpot::nblibpot {

/// nblib's floating point type, the one GROMACS was compiled with.
using Real = ::real;

struct Evaluation {
  double energy = 0.0;               //!< kJ/mol
  std::vector<double> forces;        //!< kJ/mol/nm, 3 per atom
  std::array<double, 9> virial{};    //!< -1/2 sum r (x) F, kJ/mol
};

class Engine {
public:
  explicit Engine(TopologyDesc description);
  ~Engine();
  Engine(const Engine &) = delete;
  Engine &operator=(const Engine &) = delete;

  [[nodiscard]] const TopologyDesc &description() const { return description_; }
  [[nodiscard]] const nblib::Topology &topology() const { return *topology_; }
  [[nodiscard]] std::size_t atoms() const { return atoms_; }
  /// amu, one per atom.
  [[nodiscard]] const std::vector<double> &masses() const { return masses_; }

  /// Positions in nm, 3 per atom, in any periodic image. The box is the
  /// three edge lengths in nm of a rectangular cell. Throws
  /// std::invalid_argument for a cell that is shorter than twice the cutoff.
  void evaluate(const double *positions, const std::array<double, 3> &box,
                bool wantVirial, Evaluation &out);

  /// The nblib box of three edge lengths.
  [[nodiscard]] static nblib::Box makeBox(const std::array<double, 3> &edges);

private:
  void rebuildListed(const std::array<double, 3> &box);

  TopologyDesc description_;
  std::size_t atoms_ = 0;
  std::vector<double> masses_;
  std::unique_ptr<nblib::Topology> topology_;
  std::unique_ptr<nblib::GmxNBForceCalculatorCpu> nonbonded_;
  // nblib's calculator adds the energies of every call to the ones it holds.
  // The engine returns the difference and starts a new calculator after
  // kEnergyEpoch calls so the running sums stay small.
  std::array<double, 5> accumulated_{};
  std::size_t epochCalls_ = 0;
  std::unique_ptr<nblib::ListedForceCalculator> listed_;
  // nblib 2025.5 returns the improper dihedral forces with the wrong sign
  // and the right energy, so impropers run in a calculator of their own
  // and their forces are subtracted.
  std::unique_ptr<nblib::ListedForceCalculator> impropers_;
  std::vector<nblib::Vec3> improperForces_;
  std::array<double, 3> listedBox_{};
  std::vector<std::array<std::size_t, 2>> instances_;
  std::vector<nblib::Vec3> wrapped_;
  std::vector<nblib::Vec3> nonbondedForces_;
  std::vector<nblib::Vec3> listedForces_;
};

} // namespace rgpot::nblibpot
