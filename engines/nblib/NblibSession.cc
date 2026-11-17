// MIT License
// Copyright 2023--present rgpot developers

#include "NblibSession.hpp"

#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

namespace rgpot::nblibpot {

namespace {

std::vector<double> massesPerCoordinate(const std::vector<double> &masses) {
  std::vector<double> out;
  out.reserve(3 * masses.size());
  for (const double mass : masses) {
    out.insert(out.end(), {mass, mass, mass});
  }
  return out;
}

void requireFinitePositive(double value, const char *what) {
  if (!(value > 0.0) || !std::isfinite(value)) {
    throw std::invalid_argument(std::string(what) +
                                " must be positive and finite");
  }
}

} // namespace

Session::Session(TopologyDesc description, double timeStepPs)
    : engine_(std::move(description)), dt_(timeStepPs),
      dynamics_(massesPerCoordinate(engine_.masses())) {
  requireFinitePositive(dt_, "the time step");
  const std::size_t dim = 3 * engine_.atoms();
  x_.assign(dim, 0.0);
  velocity_.assign(dim, 0.0);
  force_.assign(dim, 0.0);
  rawForce_.assign(dim, 0.0);
  mass3_.reserve(dim);
  for (const double mass : engine_.masses()) {
    mass3_.insert(mass3_.end(), {mass, mass, mass});
  }
}

Session::~Session() = default;

void Session::setTimeStep(double timeStepPs) {
  requireFinitePositive(timeStepPs, "the time step");
  if (forcesValid_) {
    toFull();
  }
  dt_ = timeStepPs;
  if (forcesValid_) {
    toHalf();
  }
}

void Session::toFull() {
  if (!halfStep_) {
    return;
  }
  for (std::size_t k = 0; k < velocity_.size(); ++k) {
    velocity_[k] += 0.5 * dt_ * force_[k] / mass3_[k];
  }
  halfStep_ = false;
}

void Session::toHalf() {
  if (halfStep_) {
    return;
  }
  for (std::size_t k = 0; k < velocity_.size(); ++k) {
    velocity_[k] -= 0.5 * dt_ * force_[k] / mass3_[k];
  }
  halfStep_ = true;
}

void Session::setBox(const double *box9) {
  for (const std::size_t k : {1u, 2u, 3u, 5u, 6u, 7u}) {
    if (box9[k] != 0.0) {
      throw std::invalid_argument(
          "nblib takes rectangular cells: the off-diagonal box entries must "
          "be zero");
    }
  }
  const std::array<double, 3> edges{box9[0], box9[4], box9[8]};
  for (const double edge : edges) {
    requireFinitePositive(edge, "a box edge");
  }
  if (forcesValid_) {
    toFull();
  }
  forcesValid_ = false;
  box_ = edges;
  haveBox_ = true;
  simulation_.reset();
  integrator_.reset();
}

void Session::setPositions(const double *positions, std::size_t dimension) {
  if (dimension != x_.size()) {
    throw std::invalid_argument("positions need three values per atom");
  }
  if (forcesValid_) {
    toFull();
  }
  forcesValid_ = false;
  for (std::size_t k = 0; k < dimension; ++k) {
    if (!std::isfinite(positions[k])) {
      throw std::invalid_argument("positions must be finite");
    }
    x_[k] = positions[k];
  }
  havePositions_ = true;
}

void Session::setVelocities(const double *velocities, std::size_t dimension) {
  if (dimension != velocity_.size()) {
    throw std::invalid_argument("velocities need three values per atom");
  }
  for (std::size_t k = 0; k < dimension; ++k) {
    if (!std::isfinite(velocities[k])) {
      throw std::invalid_argument("velocities must be finite");
    }
    velocity_[k] = velocities[k];
  }
  halfStep_ = false;
  if (forcesValid_) {
    toHalf();
  }
}

void Session::setSeed(std::uint64_t seed) { dynamics_.seed(seed); }

void Session::requireReady() const {
  if (!haveBox_ || !havePositions_) {
    throw std::logic_error("set the box and the positions first");
  }
}

void Session::evaluate() {
  engine_.evaluate(x_.data(), box_, true, scratch_);
  force_ = scratch_.forces;
  rawForce_ = force_;
  potential_ = scratch_.energy;
  virial_ = scratch_.virial;
  ++evaluations_;
}

void Session::projectState(bool forces, bool thermalize) {
  if (forces) {
    toFull();
  }
  if (projected_) {
    if (!forces) {
      dynamics_.projectPosition(x_.data());
      dynamics_.projectVelocity(velocity_.data());
    } else {
      if (thermalize && friction_ > 0.0) {
        dynamics_.thermalize(velocity_.data(), kT_, friction_, dt_);
      }
      dynamics_.projectVelocity(velocity_.data());
      dynamics_.projectForce(force_.data());
    }
  }
  if (forces) {
    toHalf();
  }
}

void Session::ensureForces() {
  requireReady();
  if (forcesValid_) {
    return;
  }
  if (simulation_ == nullptr) {
    const std::size_t atoms = engine_.atoms();
    const std::vector<nblib::Vec3> zeros(atoms, nblib::Vec3(0, 0, 0));
    const nblib::Box box = Engine::makeBox(box_);
    simulation_ = std::make_unique<nblib::SimulationState>(
        zeros, zeros, zeros, box, engine_.topology());
    integrator_ = std::make_unique<nblib::LeapFrog>(simulation_->topology(), box);
  }
  halfStep_ = false;
  evaluate();
  forcesValid_ = true;
  projectState(true, false);
}

void Session::initialize() { ensureForces(); }

void Session::exchange(bool toState) {
  auto &x = simulation_->coordinates();
  auto &v = simulation_->velocities();
  auto &f = simulation_->forces();
  for (std::size_t i = 0; i < engine_.atoms(); ++i) {
    for (std::size_t d = 0; d < 3; ++d) {
      const std::size_t k = 3 * i + d;
      if (toState) {
        x[i][d] = static_cast<Real>(x_[k]);
        v[i][d] = static_cast<Real>(velocity_[k]);
        f[i][d] = static_cast<Real>(force_[k]);
      } else {
        x_[k] = x[i][d];
        velocity_[k] = v[i][d];
      }
    }
  }
}

bool Session::applyBoundary(const std::vector<double> &previous,
                            const std::vector<double> &previousHalf) {
  const std::size_t dim = x_.size();
  std::vector<double> normal(dim, 0.0);
  const bool outside =
      projected_ && dynamics_.violation(x_.data(), normal.data());
  if (!outside) {
    return false;
  }
  x_ = previous;
  velocity_ = previousHalf;
  halfStep_ = true;
  toFull();
  // Reflection in the mass-weighted metric: v -= 2 (g . v) M^-1 g / (g M^-1 g)
  // when v points outward. The scale of g drops out.
  double weighted = 0.0;
  double outward = 0.0;
  for (std::size_t k = 0; k < dim; ++k) {
    weighted += normal[k] * normal[k] / mass3_[k];
    outward += normal[k] * velocity_[k];
  }
  if (weighted > 0.0) {
    if (outward > 0.0) {
      const double scale = 2.0 * outward / weighted;
      for (std::size_t k = 0; k < dim; ++k) {
        velocity_[k] -= scale * normal[k] / mass3_[k];
      }
    }
  } else {
    for (double &component : velocity_) {
      component = -component;
    }
  }
  toHalf();
  ++reflections_;
  return true;
}

void Session::loop(std::uint64_t count) {
  for (std::uint64_t s = 0; s < count; ++s) {
    const std::vector<double> previous = x_;
    const std::vector<double> previousHalf = velocity_;
    exchange(true);
    integrator_->integrate(static_cast<Real>(dt_), simulation_->coordinates(),
                           simulation_->velocities(), simulation_->forces());
    exchange(false);
    ++steps_;
    if (projected_) {
      projectState(false, true);
    }
    if (projected_ && applyBoundary(previous, previousHalf)) {
      continue;
    }
    evaluate();
    projectState(true, true);
  }
}

void Session::step(std::uint64_t count) {
  ensureForces();
  try {
    loop(count);
  } catch (...) {
    forcesValid_ = false;
    halfStep_ = false;
    throw;
  }
}

std::uint64_t Session::stepProjected(
    std::uint64_t count, double kT, double friction,
    const double *planePoint, const double *planeNormal,
    const double *wallPoints, const double *wallNormals,
    std::size_t wallCount) {
  requireReady();
  const std::size_t dim = x_.size();
  if (planePoint == nullptr || planeNormal == nullptr ||
      (wallCount > 0 && (wallPoints == nullptr || wallNormals == nullptr))) {
    throw std::invalid_argument("projected dynamics inputs are incomplete");
  }
  if (!(kT >= 0.0) || !(friction >= 0.0) || !std::isfinite(kT) ||
      !std::isfinite(friction)) {
    throw std::invalid_argument(
        "the thermostat needs finite, non-negative kT and friction");
  }
  dynamics_.clearPlane();
  dynamics_.clearWalls();
  kT_ = kT;
  friction_ = friction;
  dynamics_.setPlane(planePoint, planeNormal);
  for (std::size_t w = 0; w < wallCount; ++w) {
    double offset = 0.0;
    for (std::size_t k = 0; k < dim; ++k) {
      offset += wallNormals[w * dim + k] * wallPoints[w * dim + k];
    }
    dynamics_.addWall(wallNormals + w * dim, offset);
  }
  const std::uint64_t beforeReflections = reflections_;
  projected_ = true;
  try {
    if (forcesValid_) {
      toFull();
    }
    const std::vector<double> before = x_;
    dynamics_.projectPosition(x_.data());
    dynamics_.projectVelocity(velocity_.data());
    if (x_ != before) {
      forcesValid_ = false;
    } else if (forcesValid_) {
      force_ = rawForce_;
      dynamics_.projectForce(force_.data());
      toHalf();
    }
    if (dynamics_.wallCount() > 0 && !dynamics_.inside(x_.data())) {
      throw std::invalid_argument("the positions start outside a wall");
    }
    ensureForces();
    loop(count);
  } catch (...) {
    projected_ = false;
    forcesValid_ = false;
    halfStep_ = false;
    throw;
  }
  projected_ = false;
  toFull();
  force_ = rawForce_;
  toHalf();
  return reflections_ - beforeReflections;
}

void Session::state(double *positions, double *velocities, double *forces,
                    std::size_t dimension, SessionInfo *info) {
  if (dimension != x_.size()) {
    throw std::invalid_argument("state buffers need three values per atom");
  }
  ensureForces();
  std::vector<double> full(velocity_);
  if (halfStep_) {
    for (std::size_t k = 0; k < full.size(); ++k) {
      full[k] += 0.5 * dt_ * force_[k] / mass3_[k];
    }
  }
  for (std::size_t k = 0; k < dimension; ++k) {
    if (positions != nullptr) {
      positions[k] = x_[k];
    }
    if (velocities != nullptr) {
      velocities[k] = full[k];
    }
    if (forces != nullptr) {
      forces[k] = rawForce_[k];
    }
  }
  if (info != nullptr) {
    double kinetic = 0.0;
    for (std::size_t k = 0; k < full.size(); ++k) {
      kinetic += 0.5 * mass3_[k] * full[k] * full[k];
    }
    info->potential = potential_;
    info->kinetic = kinetic;
    info->time = static_cast<double>(steps_) * dt_;
    info->steps = steps_;
    info->evaluations = evaluations_;
    for (std::size_t k = 0; k < 9; ++k) {
      info->virial[k] = virial_[k];
    }
    info->box = {box_[0], 0.0, 0.0, 0.0, box_[1],
                 0.0,     0.0, 0.0, box_[2]};
  }
}

} // namespace rgpot::nblibpot
