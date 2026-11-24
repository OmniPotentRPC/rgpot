#pragma once
// MIT License
// Copyright 2026 Rohit Goswami

/**
 * @file TprEngine.hpp
 * @brief Forces of a GROMACS run input file, evaluated in process.
 *
 * nblib's own topology covers rectangular cells, cutoff electrostatics and
 * flexible molecules. A run input file (TPR) can hold more: triclinic cells,
 * PME, modified Lennard-Jones, position restraints, rigid water and virtual
 * sites. This engine opens the TPR and evaluates the same nbnxm, PME and
 * listed kernels mdrun uses.
 *
 * Virtual sites have no equation of motion. The engine builds them from the
 * atoms that carry mass and spreads their forces back, so callers see the
 * massive atoms only, in TPR order.
 *
 * A TPR that uses a feature the engine does not evaluate is refused.
 */

#include "rgpot/MDSession/ProjectedLangevin.hpp"

#include <array>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace rgpot::nblibpot {

class TprEngine {
public:
  TprEngine(const std::string &path, int threads);
  ~TprEngine();
  TprEngine(const TprEngine &) = delete;
  TprEngine &operator=(const TprEngine &) = delete;

  /// Massive atoms.
  [[nodiscard]] std::size_t atoms() const;
  [[nodiscard]] const std::vector<double> &masses() const;
  /// The distances SETTLE and LINCS hold in mdrun, nm, massive-atom indices.
  [[nodiscard]] const std::vector<md::DistanceConstraint> &constraints() const;
  /// Positions (nm), velocities (nm/ps) and box (nm, rows) of the TPR. Atoms
  /// joined by constraints are in one periodic image.
  [[nodiscard]] const std::vector<double> &initialPositions() const;
  [[nodiscard]] const std::vector<double> &initialVelocities() const;
  [[nodiscard]] std::array<double, 9> box() const;
  /// A lower-triangular GROMACS box, nm.
  void setBox(const std::array<double, 9> &box);

  /// Potential energy (kJ/mol); forces (kJ/mol/nm) go to the second argument.
  /// Positions may be in any periodic image.
  double evaluate(const double *positions, double *forces);

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace rgpot::nblibpot
