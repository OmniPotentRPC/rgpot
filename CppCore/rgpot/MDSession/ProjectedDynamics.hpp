#pragma once
// MIT License
// Copyright 2023--present rgpot developers

/**
 * @brief Plane-projected Langevin dynamics, independent of the MD engine.
 *
 * The part of rgpot_md_session_step_langevin that does not
 * depend on how an engine integrates: projecting positions, velocities and
 * forces onto the tangent space of a hyperplane n . (x - point) = 0 in the
 * mass-weighted metric, the Ornstein-Uhlenbeck velocity update, and the test
 * against planar hard walls. An engine calls these from the points of its
 * step where its callbacks run, so nblib and OpenMM share one definition of
 * the constrained dynamics.
 *
 * Arrays have one entry per coordinate, atom-major. The class does not know
 * the units; the caller passes kT in mass times velocity squared of its
 * arrays.
 */

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

namespace rgpot::md {

class ProjectedDynamics {
public:
  /// masses: one value per coordinate.
  explicit ProjectedDynamics(std::vector<double> masses)
      : mass_(std::move(masses)) {}

  void seed(std::uint64_t value) { rng_.seed(value); }

  void setPlane(const double *point, const double *normal) {
    point_.assign(point, point + mass_.size());
    normal_.assign(normal, normal + mass_.size());
    double c = 0.0;
    for (std::size_t k = 0; k < mass_.size(); ++k) {
      c += normal_[k] * normal_[k] / mass_[k];
    }
    if (!(c > 0.0) || !std::isfinite(c)) {
      throw std::invalid_argument("the plane normal must be finite and not "
                                  "zero");
    }
    weighted_ = c;
  }

  void clearPlane() {
    point_.clear();
    normal_.clear();
    weighted_ = 0.0;
  }

  void addWall(const double *normal, double offset) {
    Wall wall;
    wall.normal.assign(normal, normal + mass_.size());
    wall.offset = offset;
    double norm2 = 0.0;
    for (const double value : wall.normal) {
      norm2 += value * value;
    }
    if (!(norm2 > 0.0) || !std::isfinite(norm2) || !std::isfinite(offset)) {
      throw std::invalid_argument("a wall needs a finite, nonzero normal");
    }
    walls_.push_back(std::move(wall));
  }

  void clearWalls() { walls_.clear(); }

  [[nodiscard]] bool hasPlane() const { return !normal_.empty(); }
  [[nodiscard]] std::size_t wallCount() const { return walls_.size(); }

  /// x -= M^-1 n g / (n M^-1 n) with g = n . (x - point).
  void projectPosition(double *x) const {
    if (!hasPlane()) {
      return;
    }
    double g = 0.0;
    for (std::size_t k = 0; k < mass_.size(); ++k) {
      g += normal_[k] * (x[k] - point_[k]);
    }
    const double scale = g / weighted_;
    for (std::size_t k = 0; k < mass_.size(); ++k) {
      x[k] -= scale * normal_[k] / mass_[k];
    }
  }

  /// v -= M^-1 n (n . v) / (n M^-1 n).
  void projectVelocity(double *v) const {
    if (!hasPlane()) {
      return;
    }
    double along = 0.0;
    for (std::size_t k = 0; k < mass_.size(); ++k) {
      along += normal_[k] * v[k];
    }
    const double scale = along / weighted_;
    for (std::size_t k = 0; k < mass_.size(); ++k) {
      v[k] -= scale * normal_[k] / mass_[k];
    }
  }

  /// f -= n (n . M^-1 f) / (n M^-1 n): the constraint force removes the
  /// acceleration along the normal. Any covector projects the same way.
  void projectForce(double *f) const { projectGradient(f); }

  /// One Ornstein-Uhlenbeck update, then the projection of the result.
  /// friction * dt = 0 leaves v alone and draws nothing.
  void thermalize(double *v, double kT, double friction, double dt) {
    if (!(kT >= 0.0) || !(friction >= 0.0) || !(dt > 0.0)) {
      throw std::invalid_argument("kT and friction must not be negative");
    }
    const double decay = std::exp(-friction * dt);
    if (decay != 1.0) {
      const double spread = 1.0 - decay * decay;
      for (std::size_t k = 0; k < mass_.size(); ++k) {
        v[k] = decay * v[k] +
               std::sqrt(spread * kT / mass_[k]) * normal_draw_(rng_);
      }
    }
    projectVelocity(v);
  }

  /// True when x is outside a wall. outward then holds the gradient of the
  /// first wall violated, with the component along the plane normal
  /// removed in the mass-weighted metric, so a reflection about it keeps the
  /// velocity in the plane.
  bool violation(const double *x, double *outward) const {
    for (const Wall &wall : walls_) {
      double value = 0.0;
      for (std::size_t k = 0; k < mass_.size(); ++k) {
        value += wall.normal[k] * x[k];
      }
      if (value > wall.offset) {
        std::copy(wall.normal.begin(), wall.normal.end(), outward);
        projectGradient(outward);
        return true;
      }
    }
    return false;
  }

  [[nodiscard]] bool inside(const double *x) const {
    std::vector<double> scratch(mass_.size());
    return !violation(x, scratch.data());
  }

private:
  struct Wall {
    std::vector<double> normal;
    double offset = 0.0;
  };

  void projectGradient(double *g) const {
    if (!hasPlane()) {
      return;
    }
    double along = 0.0;
    for (std::size_t k = 0; k < mass_.size(); ++k) {
      along += normal_[k] * g[k] / mass_[k];
    }
    const double scale = along / weighted_;
    for (std::size_t k = 0; k < mass_.size(); ++k) {
      g[k] -= scale * normal_[k];
    }
  }

  std::vector<double> mass_;
  std::vector<double> point_;
  std::vector<double> normal_;
  double weighted_ = 0.0;
  std::vector<Wall> walls_;
  std::mt19937_64 rng_{0};
  std::normal_distribution<double> normal_draw_{0.0, 1.0};
};

} // namespace rgpot::md
