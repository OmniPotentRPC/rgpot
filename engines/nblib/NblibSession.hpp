#pragma once
// MIT License
// Copyright 2023--present rgpot developers

/**
 * @brief Molecular dynamics of a topology with nblib's leap-frog integrator.
 *
 * Internal to librgpot_nblib. The public units match the canonical session
 * ABI: nm, ps, kJ/mol and amu. The velocities a caller sees are the ones at the time of the
 * positions, v(t). nblib's leap-frog keeps the velocity half a step away, so
 * the session stores v(t) - dt/2 F(t)/m and a step is the velocity Verlet
 * map.
 */

#include "NblibEngine.hpp"
#include "rgpot/MDSession/ProjectedDynamics.hpp"

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
  Session(TopologyDesc description, double timeStepPs);
  ~Session();
  Session(const Session &) = delete;
  Session &operator=(const Session &) = delete;

  [[nodiscard]] std::size_t atoms() const { return engine_.atoms(); }
  [[nodiscard]] const std::vector<double> &masses() const {
    return engine_.masses();
  }

  void setTimeStep(double timeStepPs);
  void setBox(const double *box9);
  void setPositions(const double *positions, std::size_t dimension);
  void setVelocities(const double *velocities, std::size_t dimension);
  void setSeed(std::uint64_t seed);

  void initialize();
  void step(std::uint64_t count);
  /// Langevin dynamics on a full-dimensional hyperplane with planar walls.
  std::uint64_t stepProjected(
      std::uint64_t count, double kT, double friction,
      const double *planePoint, const double *planeNormal,
      const double *wallPoints, const double *wallNormals,
      std::size_t wallCount);
  void state(double *positions, double *velocities, double *forces,
             std::size_t dimension, SessionInfo *info);

private:
  void requireReady() const;
  void evaluate();
  void ensureForces();
  void toFull();
  void toHalf();
  void projectState(bool forces, bool thermalize);
  void loop(std::uint64_t count);
  [[nodiscard]] bool applyBoundary(const std::vector<double> &previous,
                                   const std::vector<double> &previousHalf);
  void exchange(bool toState);

  Engine engine_;
  double dt_;
  std::array<double, 3> box_{};
  bool haveBox_ = false;
  bool havePositions_ = false;
  std::vector<double> x_;        //!< nm, continuous across the cell faces
  std::vector<double> velocity_; //!< nm/ps, full step or half step
  std::vector<double> force_;    //!< kJ/mol/nm, the forces that drive a step
  std::vector<double> rawForce_; //!< kJ/mol/nm, the forces of the engine
  std::vector<double> mass3_;    //!< amu per coordinate
  bool halfStep_ = false;
  bool forcesValid_ = false;
  double potential_ = 0.0;       //!< kJ/mol
  std::array<double, 9> virial_{};
  std::uint64_t steps_ = 0;
  std::uint64_t evaluations_ = 0;
  md::ProjectedDynamics dynamics_;
  bool projected_ = false;
  double kT_ = 0.0;       //!< kJ/mol
  double friction_ = 0.0; //!< 1/ps
  std::uint64_t reflections_ = 0;

  std::unique_ptr<nblib::SimulationState> simulation_;
  std::unique_ptr<nblib::LeapFrog> integrator_;
  Evaluation scratch_;
};

} // namespace rgpot::nblibpot
