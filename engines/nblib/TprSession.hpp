#pragma once
// MIT License
// Copyright 2026 Rohit Goswami

#include "NblibSession.hpp"
#include "TprEngine.hpp"

namespace rgpot::nblibpot {

/**
 * @brief An MD session on a GROMACS run input file.
 *
 * Opens with the positions, velocities and box of the TPR. Every run keeps
 * the constraints of the TPR; the thermostat, time step and integrator come
 * from the call and not from the TPR.
 */
class TprSession final : public SessionBase {
public:
  TprSession(const std::string &path, int threads);

  [[nodiscard]] std::size_t atoms() const override { return engine_.atoms(); }
  [[nodiscard]] const std::vector<double> &masses() const override {
    return engine_.masses();
  }
  void setBox(const double *box9) override;
  [[nodiscard]] std::array<double, 9> box() const override {
    return engine_.box();
  }
  /// Atoms joined by a constraint must be in one periodic image.
  void setPositions(const double *positions, std::size_t dimension) override;
  void setVelocities(const double *velocities, std::size_t dimension) override;
  void initialize() override { ensureForces(); }
  void step(std::uint64_t count, double timeStep) override;
  md::LangevinResult langevin(const md::LangevinSettings &settings,
                              const md::PlaneSpec *plane,
                              const std::vector<md::WallSpec> &walls) override;
  void state(double *positions, double *velocities, double *forces,
             std::size_t dimension, SessionInfo *info) override;
  /// Terms of the forces in state(), kJ/mol.
  [[nodiscard]] TprEngine::Terms terms();

private:
  void ensureForces();
  md::LangevinResult run(const md::LangevinSettings &settings,
                         const md::PlaneSpec *plane,
                         const std::vector<md::WallSpec> &walls,
                         md::SplitMix64 &rng);

  TprEngine engine_;
  std::vector<double> mass3_;
  md::ProjectedLangevin driver_;
  md::SeededStream stream_;
  std::vector<double> x_;
  std::vector<double> v_;
  std::vector<double> force_;
  double potential_ = 0.0;
  bool forcesValid_ = false;
  std::uint64_t steps_ = 0;
  std::uint64_t evaluations_ = 0;
  double time_ = 0.0;
};

} // namespace rgpot::nblibpot
