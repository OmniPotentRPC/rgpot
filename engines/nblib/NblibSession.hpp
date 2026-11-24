#pragma once
// MIT License
// Copyright 2023--present rgpot developers

/**
 * @brief Molecular dynamics of a topology on top of nblib forces.
 *
 * Internal to librgpot_nblib. The units are nm, ps, kJ/mol and amu. The
 * session holds positions, full-step velocities and the forces at those
 * positions. Langevin dynamics, the hyperplane, the walls, the constraint
 * force and the random stream are the shared driver of
 * rgpot/MDSession/ProjectedLangevin.hpp, the one OpenMM runs, with nblib
 * supplying the forces. Unconstrained NVE steps use nblib's leap-frog
 * integrator on the half-step velocities and give back full-step ones.
 */

#include "NblibEngine.hpp"
#include "rgpot/MDSession/ProjectedLangevin.hpp"

#include <array>
#include <cstdint>
#include <memory>
#include <vector>

namespace rgpot::nblibpot {

struct SessionInfo {
  double potential = 0.0; //!< kJ/mol
  double kinetic = 0.0;   //!< kJ/mol
  double time = 0.0;      //!< ps
  std::uint64_t steps = 0;
  std::uint64_t evaluations = 0;
  std::array<double, 9> virial{}; //!< kJ/mol
  std::array<double, 9> box{};    //!< nm
};

class Session {
public:
  explicit Session(TopologyDesc description);
  ~Session();
  Session(const Session &) = delete;
  Session &operator=(const Session &) = delete;

  [[nodiscard]] std::size_t atoms() const { return engine_.atoms(); }
  [[nodiscard]] const std::vector<double> &masses() const {
    return engine_.masses();
  }

  /// Three edge lengths from a rectangular box, nm. Throws for any other cell.
  void setBox(const double *box9);
  /// The box as nine values, rows are the lattice vectors, nm.
  [[nodiscard]] std::array<double, 9> box() const;
  void setPositions(const double *positions, std::size_t dimension);
  void setVelocities(const double *velocities, std::size_t dimension);

  /// Evaluate the forces at the positions set.
  void initialize();
  /// Unconstrained velocity-Verlet steps of length timeStep.
  void step(std::uint64_t count, double timeStep);
  /// Langevin dynamics with an optional hyperplane and walls; the settings
  /// carry the seed that names the random stream.
  md::LangevinResult langevin(const md::LangevinSettings &settings,
                              const md::PlaneSpec *plane,
                              const std::vector<md::WallSpec> &walls);
  void state(double *positions, double *velocities, double *forces,
             std::size_t dimension, SessionInfo *info);

private:
  void requireReady() const;
  void evaluate();
  void ensureForces();

  Engine engine_;
  std::array<double, 3> box_{};
  bool haveBox_ = false;
  bool havePositions_ = false;
  std::vector<double> x_;     //!< nm, continuous across the cell faces
  std::vector<double> v_;     //!< nm/ps, at the time of x_
  std::vector<double> force_; //!< kJ/mol/nm at x_ when forcesValid_
  std::vector<double> mass3_; //!< amu per coordinate
  bool forcesValid_ = false;
  double potential_ = 0.0; //!< kJ/mol
  std::array<double, 9> virial_{};
  std::uint64_t steps_ = 0;
  double time_ = 0.0;
  std::uint64_t evaluations_ = 0;
  md::ProjectedLangevin driver_;
  md::SeededStream stream_;

  std::unique_ptr<nblib::SimulationState> simulation_;
  std::unique_ptr<nblib::LeapFrog> integrator_;
  Evaluation scratch_;
};

} // namespace rgpot::nblibpot
