#pragma once

// MIT License
// Copyright 2023--present rgpot developers

#include "rgpot/ExternalPot/external_potentials.h"
#include "rgpot/ExternalPot/md_units.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <openmm/Vec3.h>

namespace OpenMM {
class Context;
class Integrator;
class System;
} // namespace OpenMM

namespace rgpot {

/**
 * @brief Controls for an in-process OpenMM dynamics session.
 *
 * Coordinates, energies, masses, and time use OpenMM's native units
 * (nanometres, kJ/mol, amu, and picoseconds). The session owns the OpenMM
 * System and Context; no OpenMM object crosses the C ABI boundary.
 */
struct OpenMMSessionConfig {
  std::string platform = "Reference";
  //! Time step in picoseconds (1 fs).
  double time_step = 0.001;
  //! Langevin friction in inverse picoseconds.
  double friction = 1.0;
  //! Thermal energy kT in kJ/mol (~300 K).
  double temperature = 0.02585 * md::kKjPerEv;
  //! Deterministic 64-bit PRNG seed for Langevin noise.
  uint64_t random_seed = 1;
};

/** What happens to a drift that ends outside a wall. */
enum class OpenMMWallMode { Reflect, Reject };

/** What a Langevin run reports, in OpenMM native units. */
struct OpenMMLangevinResult {
  uint64_t wall_hits = 0;
  /**
   * One Lagrange multiplier per step when requested and a plane is set:
   * lambda = -(n . M^-1 F) / (n . M^-1 n) in kJ/mol/nm for the unit normal n
   * over the 3N coordinates, the mean of the two force kicks of the step.
   */
  std::vector<double> constraint_force;
};

/** A state returned by an OpenMM session in OpenMM native units. */
struct OpenMMState {
  std::vector<OpenMM::Vec3> positions;
  std::vector<OpenMM::Vec3> velocities;
  std::vector<OpenMM::Vec3> forces;
  double potential_energy = 0.0;
  double kinetic_energy = 0.0;
};

/**
 * @brief In-process OpenMM projected-BAOAB and Verlet NVE session.
 *
 * Langevin steps run the shared md::ProjectedLangevin driver on the OpenMM
 * Context: BAOAB splitting on a full 3N-dimensional hyperplane in Cartesian
 * coordinates using the inverse-mass metric (equivalent to Euclidean
 * projection in mass-weighted coordinates). The first force of a call is
 * evaluated on the hyperplane, not at the positions the caller left.
 * Optional planar hard walls reflect positions and velocities specularly on
 * both half-drifts, as often as the drift needs, and return the total number
 * of reflections. The distance constraints of the System hold in every
 * Langevin step together with the hyperplane; a System with virtual sites is
 * refused.
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

  [[nodiscard]] size_t numParticles() const noexcept;
  [[nodiscard]] std::vector<double> masses() const;

  void setPositions(const std::vector<OpenMM::Vec3> &positions);
  void setVelocities(const std::vector<OpenMM::Vec3> &velocities);

  /** Advance unconstrained NVE Verlet dynamics using the configured step size. */
  void stepNVE(int steps);
  /** Advance unconstrained NVE Verlet dynamics with an explicit time step (ps). */
  void stepNVE(double time_step, size_t steps);

  /**
   * Advance projected BAOAB dynamics with explicit controls and 3N plane/wall
   * geometry in OpenMM native units. Returns the number of wall reflections.
   *
   * seed names the random stream. A call with the seed of the previous call
   * continues that stream, so a run split in chunks equals one long run; a new
   * seed starts a new stream. Setting positions or velocities ends the stream,
   * so the same seed after a new state replays.
   */
  uint64_t stepLangevin(
      double kT, double friction, double time_step,
      const std::vector<OpenMM::Vec3> &plane_point,
      const std::vector<OpenMM::Vec3> &plane_normal,
      const std::vector<std::vector<OpenMM::Vec3>> &wall_points,
      const std::vector<std::vector<OpenMM::Vec3>> &wall_normals,
      size_t steps, uint64_t seed);

  /**
   * Run projected BAOAB dynamics like stepLangevin, with the wall mode and
   * the per-step constraint force. An empty plane_point and plane_normal run
   * without the hyperplane; the distance constraints of the System still
   * hold. The constraint force is recorded only when record_constraint_force
   * is set and a plane is given.
   */
  OpenMMLangevinResult runLangevin(
      double kT, double friction, double time_step,
      const std::vector<OpenMM::Vec3> &plane_point,
      const std::vector<OpenMM::Vec3> &plane_normal,
      const std::vector<std::vector<OpenMM::Vec3>> &wall_points,
      const std::vector<std::vector<OpenMM::Vec3>> &wall_normals,
      size_t steps, uint64_t seed, OpenMMWallMode wall_mode,
      bool record_constraint_force);

  /** Read positions, velocities, forces, and energies from the Context. */
  [[nodiscard]] OpenMMState state() const;

  /** Read the periodic box vectors from the Context. */
  void boxVectors(OpenMM::Vec3 &a, OpenMM::Vec3 &b, OpenMM::Vec3 &c) const;
  /**
   * True when a force of the System uses periodic boundary conditions.
   * A false result still has box vectors; they are a placeholder and are
   * not a periodic cell.
   */
  [[nodiscard]] bool usesPeriodicBoundaryConditions() const;
  /** Read the positions and box vectors from the Context without forces. */
  [[nodiscard]] std::vector<OpenMM::Vec3>
  geometry(OpenMM::Vec3 &a, OpenMM::Vec3 &b, OpenMM::Vec3 &c) const;
  /** Set the box vectors (a, b, c as rows, nanometres) on the Context. */
  void setPeriodicBox(const std::array<double, 9> &box);
  /** Evaluate forces at the current positions, so a failure shows early. */
  void initialize();

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

/**
 * Populate an rgpot_md_session_ops vtable backed by an OpenMMSession
 * deserialized from system_xml on the requested platform. All C ABI
 * inputs/outputs use rgpot units (Angstrom, eV, ps, amu).
 */
int openmm_md_session_init(const char *system_xml, const char *platform,
                           rgpot_md_session_ops *ops, void **ctx,
                           size_t *n_atoms, char *err, size_t err_len);

} // namespace rgpot
