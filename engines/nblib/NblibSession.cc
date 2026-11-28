// MIT License
// Copyright 2023--present rgpot developers

#include "NblibSession.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
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

Session::Session(TopologyDesc description)
    : engine_(std::move(description)),
      mass3_(massesPerCoordinate(engine_.masses())), driver_(mass3_) {
  const std::size_t dim = 3 * engine_.atoms();
  x_.assign(dim, 0.0);
  v_.assign(dim, 0.0);
  force_.assign(dim, 0.0);
  sourceIndices_.resize(engine_.atoms());
  std::iota(sourceIndices_.begin(), sourceIndices_.end(), 0);
}

Session::~Session() = default;

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
  forcesValid_ = false;
  box_ = edges;
  haveBox_ = true;
  simulation_.reset();
  integrator_.reset();
}

std::array<double, 9> Session::box() const {
  if (!haveBox_) {
    throw std::logic_error("the box has not been set");
  }
  return {box_[0], 0.0, 0.0, 0.0, box_[1], 0.0, 0.0, 0.0, box_[2]};
}

void Session::setPositions(const double *positions, std::size_t dimension) {
  if (dimension != x_.size()) {
    throw std::invalid_argument("positions need three values per atom");
  }
  for (std::size_t k = 0; k < dimension; ++k) {
    if (!std::isfinite(positions[k])) {
      throw std::invalid_argument("positions must be finite");
    }
  }
  for (std::size_t k = 0; k < dimension; ++k) {
    x_[k] = positions[k];
  }
  forcesValid_ = false;
  havePositions_ = true;
  stream_.reset();
}

void Session::setVelocities(const double *velocities, std::size_t dimension) {
  if (dimension != v_.size()) {
    throw std::invalid_argument("velocities need three values per atom");
  }
  for (std::size_t k = 0; k < dimension; ++k) {
    if (!std::isfinite(velocities[k])) {
      throw std::invalid_argument("velocities must be finite");
    }
  }
  for (std::size_t k = 0; k < dimension; ++k) {
    v_[k] = velocities[k];
  }
  stream_.reset();
}

void Session::requireReady() const {
  if (!haveBox_ || !havePositions_) {
    throw std::logic_error("set the box and the positions first");
  }
}

void Session::evaluate() {
  engine_.evaluate(x_.data(), box_, true, scratch_);
  force_ = scratch_.forces;
  potential_ = scratch_.energy;
  virial_ = scratch_.virial;
  ++evaluations_;
}

void Session::ensureForces() {
  requireReady();
  if (forcesValid_) {
    return;
  }
  evaluate();
  forcesValid_ = true;
}

void Session::initialize() { ensureForces(); }

void Session::step(std::uint64_t count, double timeStep) {
  requireFinitePositive(timeStep, "the time step");
  ensureForces();
  if (simulation_ == nullptr) {
    const std::size_t atoms = engine_.atoms();
    const std::vector<nblib::Vec3> zeros(atoms, nblib::Vec3(0, 0, 0));
    const nblib::Box box = Engine::makeBox(box_);
    simulation_ = std::make_unique<nblib::SimulationState>(
        zeros, zeros, zeros, box, engine_.topology());
    integrator_ =
        std::make_unique<nblib::LeapFrog>(simulation_->topology(), box);
  }
  // nblib's leap-frog keeps the velocity half a step behind the positions:
  // v(t - dt/2) = v(t) - dt/2 F(t)/m. A step from there is velocity Verlet.
  std::vector<double> half(v_);
  for (std::size_t k = 0; k < half.size(); ++k) {
    half[k] -= 0.5 * timeStep * force_[k] / mass3_[k];
  }
  const auto exchange = [&](bool toState) {
    auto &x = simulation_->coordinates();
    auto &v = simulation_->velocities();
    auto &f = simulation_->forces();
    for (std::size_t i = 0; i < engine_.atoms(); ++i) {
      for (std::size_t d = 0; d < 3; ++d) {
        const std::size_t k = 3 * i + d;
        if (toState) {
          x[i][d] = static_cast<Real>(x_[k]);
          v[i][d] = static_cast<Real>(half[k]);
          f[i][d] = static_cast<Real>(force_[k]);
        } else {
          x_[k] = x[i][d];
          half[k] = v[i][d];
        }
      }
    }
  };
  try {
    for (std::uint64_t s = 0; s < count; ++s) {
      exchange(true);
      integrator_->integrate(static_cast<Real>(timeStep),
                             simulation_->coordinates(),
                             simulation_->velocities(),
                             simulation_->forces());
      exchange(false);
      ++steps_;
      time_ += timeStep;
      evaluate();
    }
  } catch (...) {
    forcesValid_ = false;
    throw;
  }
  for (std::size_t k = 0; k < v_.size(); ++k) {
    v_[k] = half[k] + 0.5 * timeStep * force_[k] / mass3_[k];
  }
}

md::LangevinResult Session::langevin(const md::LangevinSettings &settings,
                                     const md::PlaneSpec *plane,
                                     const std::vector<md::WallSpec> &walls) {
  requireReady();
  std::array<double, 9> virial = virial_;
  std::uint64_t evaluations = 0;
  const md::ForceFunction forcesAt = [&](const double *positions,
                                         double *forces) {
    engine_.evaluate(positions, box_, true, scratch_);
    std::copy(scratch_.forces.begin(), scratch_.forces.end(), forces);
    virial = scratch_.virial;
    ++evaluations;
    return scratch_.energy;
  };
  std::vector<double> x = x_;
  std::vector<double> v = v_;
  md::LangevinResult result =
      driver_.run(settings, plane, walls, x.data(), v.data(), forcesAt,
                  stream_.forSeed(settings.seed));
  if (settings.steps > 0) {
    x_ = std::move(x);
    v_ = std::move(v);
    force_ = result.forces;
    potential_ = result.potential;
    virial_ = virial;
    evaluations_ += evaluations;
    steps_ += settings.steps;
    time_ += static_cast<double>(settings.steps) * settings.timeStep;
    forcesValid_ = true;
  }
  return result;
}

void Session::state(double *positions, double *velocities, double *forces,
                    std::size_t dimension, SessionInfo *info) {
  if (dimension != x_.size()) {
    throw std::invalid_argument("state buffers need three values per atom");
  }
  ensureForces();
  for (std::size_t k = 0; k < dimension; ++k) {
    if (positions != nullptr) {
      positions[k] = x_[k];
    }
    if (velocities != nullptr) {
      velocities[k] = v_[k];
    }
    if (forces != nullptr) {
      forces[k] = force_[k];
    }
  }
  if (info != nullptr) {
    double kinetic = 0.0;
    for (std::size_t k = 0; k < v_.size(); ++k) {
      kinetic += 0.5 * mass3_[k] * v_[k] * v_[k];
    }
    info->potential = potential_;
    info->kinetic = kinetic;
    info->time = time_;
    info->steps = steps_;
    info->evaluations = evaluations_;
    info->virial = virial_;
    info->box = {box_[0], 0.0, 0.0, 0.0, box_[1], 0.0, 0.0, 0.0, box_[2]};
  }
}

} // namespace rgpot::nblibpot
