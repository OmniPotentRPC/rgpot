#pragma once

// MIT License
// Copyright 2026 Rohit Goswami

#include <memory>
#include <string>
#include <vector>

#include <openmm/Vec3.h>

namespace OpenMM {
class Context;
class CustomIntegrator;
class Integrator;
class Platform;
class System;
} // namespace OpenMM

namespace rgpot {

/**
 * @brief Controls for an in-process OpenMM dynamics session.
 *
 * Coordinates, energies, masses, and time use OpenMM's native units. The
 * session owns the OpenMM System and Context; no OpenMM object crosses the
 * C ABI boundary.
 */
struct OpenMMSessionConfig {
  std::string platform = "Reference";
  double time_step = 0.001;
  double friction = 1.0;
  double temperature = 0.02585;
  int random_seed = 1;
};

/** A state returned by an OpenMM session. */
struct OpenMMState {
  std::vector<OpenMM::Vec3> positions;
  std::vector<OpenMM::Vec3> velocities;
  std::vector<OpenMM::Vec3> forces;
  double potential_energy = 0.0;
  double kinetic_energy = 0.0;
};

/**
 * @brief In-process OpenMM CustomIntegrator and Verlet session.
 *
 * Langevin steps use BAOAB splitting. A plane is supplied in Cartesian
 * coordinates and is enforced by projected force/velocity and affine
 * position restoration. Planar walls reflect the tangent velocity and
 * report their count through the return value.
 */
class OpenMMSession {
public:
  OpenMMSession(std::unique_ptr<OpenMM::System> system,
                const OpenMMSessionConfig &config);
  ~OpenMMSession();
  OpenMMSession(const OpenMMSession &) = delete;
  OpenMMSession &operator=(const OpenMMSession &) = delete;
  OpenMMSession(OpenMMSession &&) noexcept;
  OpenMMSession &operator=(OpenMMSession &&) noexcept;

  void setPositions(const std::vector<OpenMM::Vec3> &positions);
  void setVelocities(const std::vector<OpenMM::Vec3> &velocities);
  /** Advance unconstrained NVE Verlet dynamics. */
  void stepNVE(int steps);
  /**
   * Advance projected BAOAB dynamics. Returns the number of wall
   * reflections, including repeated hits in this call.
   */
  int stepLangevin(const OpenMM::Vec3 &plane_point,
                   const OpenMM::Vec3 &plane_normal,
                   const std::vector<OpenMM::Vec3> &wall_points,
                   const std::vector<OpenMM::Vec3> &wall_normals,
                   int steps);
  /** Read positions, velocities, and energies from the Context. */
  [[nodiscard]] OpenMMState state() const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace rgpot
