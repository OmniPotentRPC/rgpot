// MIT License
// Copyright 2023--present rgpot developers

#include "NblibSession.hpp"

#include "rgpot/ExternalPot/md_units.hpp"

#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

namespace rgpot::nblibpot {

namespace {

constexpr double kAngstrom = md::kAngstromPerNm;
constexpr double kNm = md::kNmPerAngstrom;
constexpr double kEvAPerKjNm = md::kEvAPerKjNm;
constexpr double kKjNmPerEvA = 1.0 / md::kEvAPerKjNm;

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
  const std::array<double, 3> edges{box9[0] * kNm, box9[4] * kNm,
                                    box9[8] * kNm};
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
    x_[k] = positions[k] * kNm;
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
    velocity_[k] = velocities[k] * kNm;
  }
  halfStep_ = false;
  if (forcesValid_) {
    toHalf();
  }
}

void Session::setSeed(std::uint64_t seed) { dynamics_.seed(seed); }

void Session::setHook(rgpot_md_step_hook hook, void *user) {
  hook_ = hook;
  hookUser_ = user;
}

void Session::setBoundary(rgpot_md_boundary_fn boundary, void *user) {
  boundary_ = boundary;
  boundaryUser_ = user;
}

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

void Session::runHook(int phase, bool thermalize) {
  if (phase == RGPOT_MD_PHASE_FORCES) {
    toFull();
  }
  if (hook_ != nullptr) {
    const std::size_t dim = x_.size();
    std::vector<double> x(dim);
    std::vector<double> v(dim);
    std::vector<double> f(dim);
    for (std::size_t k = 0; k < dim; ++k) {
      x[k] = x_[k] * kAngstrom;
      v[k] = velocity_[k] * kAngstrom;
      f[k] = force_[k] * kEvAPerKjNm;
    }
    const int code = hook_(hookUser_, phase, steps_, dim, x.data(), v.data(),
                           f.data(), mass3_.data());
    if (code != 0) {
      throw std::runtime_error("the step hook returned " +
                               std::to_string(code));
    }
    for (std::size_t k = 0; k < dim; ++k) {
      if (!std::isfinite(v[k]) || !std::isfinite(f[k]) ||
          !std::isfinite(x[k])) {
        throw std::runtime_error("the step hook left a non-finite value");
      }
      velocity_[k] = v[k] * kNm;
      if (phase == RGPOT_MD_PHASE_FORCES) {
        force_[k] = f[k] * kKjNmPerEvA;
      } else {
        x_[k] = x[k] * kNm;
      }
    }
  }
  if (projected_) {
    if (phase == RGPOT_MD_PHASE_POSITIONS) {
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
  if (phase == RGPOT_MD_PHASE_FORCES) {
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
  runHook(RGPOT_MD_PHASE_FORCES, false);
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
  bool outside = false;
  if (boundary_ != nullptr) {
    std::vector<double> positions(dim);
    for (std::size_t k = 0; k < dim; ++k) {
      positions[k] = x_[k] * kAngstrom;
    }
    outside = boundary_(boundaryUser_, dim, positions.data(), normal.data()) != 0;
  }
  if (!outside && projected_ && dynamics_.wallCount() > 0) {
    outside = dynamics_.violation(x_.data(), normal.data());
  }
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
    if (hook_ != nullptr || projected_) {
      runHook(RGPOT_MD_PHASE_POSITIONS, true);
    }
    if ((boundary_ != nullptr || projected_) &&
        applyBoundary(previous, previousHalf)) {
      continue;
    }
    evaluate();
    runHook(RGPOT_MD_PHASE_FORCES, true);
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

void Session::stepProjected(std::uint64_t count,
                            const rgpot_md_thermostat *thermostat,
                            const rgpot_md_plane *plane,
                            const rgpot_md_wall *walls, std::size_t wallCount) {
  requireReady();
  const std::size_t dim = x_.size();
  if (thermostat == nullptr && plane == nullptr && wallCount == 0) {
    step(count);
    return;
  }
  if (wallCount > 0 && walls == nullptr) {
    throw std::invalid_argument("walls is NULL");
  }
  dynamics_.clearPlane();
  dynamics_.clearWalls();
  kT_ = 0.0;
  friction_ = 0.0;
  if (thermostat != nullptr) {
    if (thermostat->bytes < sizeof(rgpot_md_thermostat)) {
      throw std::invalid_argument("rgpot_md_thermostat is too short");
    }
    if (!(thermostat->kt_ev >= 0.0) || !(thermostat->friction_per_ps >= 0.0) ||
        !std::isfinite(thermostat->kt_ev) ||
        !std::isfinite(thermostat->friction_per_ps)) {
      throw std::invalid_argument(
          "the thermostat needs finite, non-negative kT and friction");
    }
    kT_ = thermostat->kt_ev * md::kKjPerEv;
    friction_ = thermostat->friction_per_ps;
  }
  // The caller's normals are gradients in Angstrom; the session holds nm.
  std::vector<double> scaled(dim);
  if (plane != nullptr) {
    if (plane->bytes < sizeof(rgpot_md_plane)) {
      throw std::invalid_argument("rgpot_md_plane is too short");
    }
    if (plane->dimension != dim || plane->point == nullptr ||
        plane->normal == nullptr) {
      throw std::invalid_argument("the plane needs a point and a normal of "
                                  "three values per atom");
    }
    std::vector<double> point(dim);
    for (std::size_t k = 0; k < dim; ++k) {
      point[k] = plane->point[k] * kNm;
      scaled[k] = plane->normal[k] * kAngstrom;
    }
    dynamics_.setPlane(point.data(), scaled.data());
  }
  for (std::size_t w = 0; w < wallCount; ++w) {
    if (walls[w].normal == nullptr) {
      throw std::invalid_argument("a wall has no normal");
    }
    for (std::size_t k = 0; k < dim; ++k) {
      scaled[k] = walls[w].normal[k] * kAngstrom;
    }
    dynamics_.addWall(scaled.data(), walls[w].offset);
  }
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
      positions[k] = x_[k] * kAngstrom;
    }
    if (velocities != nullptr) {
      velocities[k] = full[k] * kAngstrom;
    }
    if (forces != nullptr) {
      forces[k] = rawForce_[k] * kEvAPerKjNm;
    }
  }
  if (info != nullptr) {
    double kinetic = 0.0;
    for (std::size_t k = 0; k < full.size(); ++k) {
      kinetic += 0.5 * mass3_[k] * full[k] * full[k];
    }
    info->potential = potential_ * md::kEvPerKj;
    info->kinetic = kinetic * md::kEvPerKj;
    info->time = static_cast<double>(steps_) * dt_;
    info->steps = steps_;
    info->evaluations = evaluations_;
    for (std::size_t k = 0; k < 9; ++k) {
      info->virial[k] = virial_[k] * md::kEvPerKj;
    }
    info->box = {box_[0] * kAngstrom, 0.0, 0.0, 0.0, box_[1] * kAngstrom,
                 0.0,                 0.0, 0.0, box_[2] * kAngstrom};
  }
}

} // namespace rgpot::nblibpot
