// MIT License
// Copyright 2026 Rohit Goswami

#include "TprSession.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace rgpot::nblibpot {

namespace {

std::vector<double> perCoordinate(const std::vector<double> &masses) {
  std::vector<double> out;
  out.reserve(3 * masses.size());
  for (const double mass : masses) {
    out.insert(out.end(), {mass, mass, mass});
  }
  return out;
}

void requireFinite(const double *values, std::size_t count, std::size_t expect,
                   const char *what) {
  if (count != expect) {
    throw std::invalid_argument(std::string(what) +
                                " need three values per atom");
  }
  for (std::size_t k = 0; k < count; ++k) {
    if (!std::isfinite(values[k])) {
      throw std::invalid_argument(std::string(what) + " must be finite");
    }
  }
}

} // namespace

TprSession::TprSession(const std::string &path, int threads)
    : engine_(path, threads), mass3_(perCoordinate(engine_.masses())),
      driver_(mass3_, engine_.constraints()), x_(engine_.initialPositions()),
      v_(engine_.initialVelocities()), force_(x_.size(), 0.0) {}

void TprSession::setBox(const double *box9) {
  std::array<double, 9> box{};
  std::copy_n(box9, 9, box.begin());
  engine_.setBox(box);
}

void TprSession::setPositions(const double *positions, std::size_t dimension) {
  requireFinite(positions, dimension, x_.size(), "positions");
  // A pair split over the cell would be pulled together by the constraint
  // solver, which knows no periodicity.
  for (const md::DistanceConstraint &c : engine_.constraints()) {
    double r2 = 0.0;
    for (std::size_t d = 0; d < 3; ++d) {
      const double delta =
          positions[3 * c.first + d] - positions[3 * c.second + d];
      r2 += delta * delta;
    }
    if (std::abs(std::sqrt(r2) - c.length) > 0.5 * c.length) {
      throw std::invalid_argument(
          "a constrained pair is far from its length: atoms " +
          std::to_string(c.first) + " and " + std::to_string(c.second) +
          " may be in different periodic images");
    }
  }
  x_.assign(positions, positions + dimension);
  forcesValid_ = false;
  stream_.reset();
}

void TprSession::setVelocities(const double *velocities,
                               std::size_t dimension) {
  requireFinite(velocities, dimension, v_.size(), "velocities");
  v_.assign(velocities, velocities + dimension);
  stream_.reset();
}

void TprSession::ensureForces() {
  if (!forcesValid_) {
    potential_ = engine_.evaluate(x_.data(), force_.data());
    ++evaluations_;
    forcesValid_ = true;
  }
}

TprEngine::Terms TprSession::terms() {
  ensureForces();
  return engine_.terms();
}

md::LangevinResult TprSession::run(const md::LangevinSettings &settings,
                                   const md::PlaneSpec *plane,
                                   const std::vector<md::WallSpec> &walls,
                                   md::SplitMix64 &rng) {
  std::uint64_t evaluations = 0;
  const md::ForceFunction forcesAt = [&](const double *positions,
                                         double *forces) {
    // A chunked run starts where the last one ended, with forces in hand.
    if (forcesValid_ && std::equal(x_.begin(), x_.end(), positions)) {
      std::copy(force_.begin(), force_.end(), forces);
      return potential_;
    }
    ++evaluations;
    return engine_.evaluate(positions, forces);
  };
  // A failed run leaves the session at its last good state.
  std::vector<double> x = x_;
  std::vector<double> v = v_;
  md::LangevinResult result = driver_.run(settings, plane, walls, x.data(),
                                          v.data(), forcesAt, rng);
  if (settings.steps > 0) {
    x_ = std::move(x);
    v_ = std::move(v);
    force_ = result.forces;
    potential_ = result.potential;
    evaluations_ += evaluations;
    steps_ += settings.steps;
    time_ += static_cast<double>(settings.steps) * settings.timeStep;
    forcesValid_ = true;
  }
  return result;
}

void TprSession::step(std::uint64_t count, double timeStep) {
  // Without friction the driver is velocity Verlet with RATTLE and draws no
  // noise, so kT has no effect beyond having to be positive.
  md::LangevinSettings settings;
  settings.kT = 1.0;
  settings.friction = 0.0;
  settings.timeStep = timeStep;
  settings.steps = count;
  md::SplitMix64 unused(0);
  run(settings, nullptr, {}, unused);
}

md::LangevinResult
TprSession::langevin(const md::LangevinSettings &settings,
                     const md::PlaneSpec *plane,
                     const std::vector<md::WallSpec> &walls) {
  return run(settings, plane, walls, stream_.forSeed(settings.seed));
}

void TprSession::state(double *positions, double *velocities, double *forces,
                       std::size_t dimension, SessionInfo *info) {
  if (dimension != x_.size()) {
    throw std::invalid_argument("state buffers need three values per atom");
  }
  ensureForces();
  if (positions != nullptr) {
    std::copy(x_.begin(), x_.end(), positions);
  }
  if (velocities != nullptr) {
    std::copy(v_.begin(), v_.end(), velocities);
  }
  if (forces != nullptr) {
    std::copy(force_.begin(), force_.end(), forces);
  }
  if (info != nullptr) {
    double kinetic = 0.0;
    for (std::size_t k = 0; k < v_.size(); ++k) {
      kinetic += 0.5 * mass3_[k] * v_[k] * v_[k];
    }
    *info = {};
    info->potential = potential_;
    info->kinetic = kinetic;
    info->time = time_;
    info->steps = steps_;
    info->evaluations = evaluations_;
    info->box = engine_.box();
  }
}

} // namespace rgpot::nblibpot
