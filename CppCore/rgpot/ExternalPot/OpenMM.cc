// MIT License
// Copyright 2026 Rohit Goswami

#include "rgpot/ExternalPot/OpenMM.hpp"

#include <openmm/Context.h>
#include <openmm/CustomIntegrator.h>
#include <openmm/Platform.h>
#include <openmm/State.h>
#include <openmm/System.h>
#include <openmm/VerletIntegrator.h>
#include <openmm/serialization/XmlSerializer.h>

#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <sstream>
#include <utility>

namespace rgpot {
namespace {

double dot(const OpenMM::Vec3 &a, const OpenMM::Vec3 &b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

OpenMM::Vec3 subtract(const OpenMM::Vec3 &a, const OpenMM::Vec3 &b,
                      double scale) {
  return OpenMM::Vec3(a[0] - scale * b[0], a[1] - scale * b[1],
                      a[2] - scale * b[2]);
}

void require_positive(double value, const char *name) {
  if (!std::isfinite(value) || value <= 0.0) {
    throw std::invalid_argument(std::string(name) +
                                " must be finite and positive");
  }
}

} // namespace

struct OpenMMSession::Impl {
  std::unique_ptr<OpenMM::System> system;
  std::unique_ptr<OpenMM::Integrator> integrator;
  OpenMM::CustomIntegrator *custom = nullptr;
  std::unique_ptr<OpenMM::Context> context;
  int particles = 0;
  double friction = 0.0;
};

OpenMMSession::OpenMMSession(std::unique_ptr<OpenMM::System> system,
                             const OpenMMSessionConfig &config)
    : impl_(std::make_unique<Impl>()) {
  if (system == nullptr) {
    throw std::invalid_argument("OpenMM session needs a System");
  }
  require_positive(config.time_step, "OpenMM time_step");
  if (!std::isfinite(config.friction) || config.friction < 0.0) {
    throw std::invalid_argument(
        "OpenMM friction must be finite and non-negative");
  }
  require_positive(config.temperature, "OpenMM temperature");
  if (config.platform.empty()) {
    throw std::invalid_argument("OpenMM platform is empty");
  }
  impl_->particles = system->getNumParticles();
  impl_->friction = config.friction;
  if (impl_->particles < 1) {
    throw std::invalid_argument("OpenMM System has no particles");
  }
  impl_->system = std::move(system);

  auto custom = std::make_unique<OpenMM::CustomIntegrator>(config.time_step);
  custom->addGlobalVariable("normal_force", 0.0);
  custom->addGlobalVariable("kT", config.temperature);
  custom->addGlobalVariable("friction", config.friction);
  custom->addGlobalVariable("noise_scale", 1.0);
  custom->addPerDofVariable("normal", 0.0);
  custom->addComputePerDof("v", "v+0.5*dt*(f-normal*normal_force)/m");
  custom->addComputePerDof("x", "x+0.5*dt*v");
  custom->addComputePerDof(
      "v",
      "v*exp(-friction*dt)+sqrt(noise_scale*kT*(1-exp(-2*friction*dt))/m)*gaussian");
  custom->addComputePerDof("x", "x+0.5*dt*v");
  custom->addComputePerDof("v", "v+0.5*dt*(f-normal*normal_force)/m");
  custom->setRandomNumberSeed(config.random_seed);
  impl_->custom = custom.get();
  impl_->integrator = std::move(custom);

  OpenMM::Platform &platform =
      OpenMM::Platform::getPlatformByName(config.platform);
  impl_->context =
      std::make_unique<OpenMM::Context>(*impl_->system, *impl_->integrator,
                                         platform);
}

OpenMMSession::~OpenMMSession() = default;
OpenMMSession::OpenMMSession(OpenMMSession &&) noexcept = default;
OpenMMSession &OpenMMSession::operator=(OpenMMSession &&) noexcept = default;

void OpenMMSession::setPositions(
    const std::vector<OpenMM::Vec3> &positions) {
  if (static_cast<int>(positions.size()) != impl_->particles) {
    throw std::invalid_argument("OpenMM position count differs from System");
  }
  impl_->context->setPositions(positions);
}

void OpenMMSession::setVelocities(
    const std::vector<OpenMM::Vec3> &velocities) {
  if (static_cast<int>(velocities.size()) != impl_->particles) {
    throw std::invalid_argument("OpenMM velocity count differs from System");
  }
  impl_->context->setVelocities(velocities);
}

void OpenMMSession::stepNVE(int steps) {
  if (steps < 0) {
    throw std::invalid_argument("OpenMM NVE steps must be non-negative");
  }
  impl_->custom->setGlobalVariableByName("noise_scale", 0.0);
  impl_->custom->setGlobalVariableByName("friction", 0.0);
  impl_->integrator->step(steps);
  impl_->custom->setGlobalVariableByName("friction", impl_->friction);
  impl_->custom->setGlobalVariableByName("noise_scale", 1.0);
}

int OpenMMSession::stepLangevin(
    const OpenMM::Vec3 &plane_point, const OpenMM::Vec3 &plane_normal,
    const std::vector<OpenMM::Vec3> &wall_points,
    const std::vector<OpenMM::Vec3> &wall_normals, int steps) {
  if (steps < 0 || wall_points.size() != wall_normals.size()) {
    throw std::invalid_argument("invalid OpenMM Langevin step or wall count");
  }
  const double normal_norm = std::sqrt(dot(plane_normal, plane_normal));
  require_positive(normal_norm, "OpenMM plane normal");
  const OpenMM::Vec3 n(plane_normal[0] / normal_norm,
                       plane_normal[1] / normal_norm,
                       plane_normal[2] / normal_norm);
  std::vector<OpenMM::Vec3> normals(
      static_cast<size_t>(impl_->particles), n);
  impl_->custom->setPerDofVariableByName("normal", normals);
  int reflections = 0;
  for (int step = 0; step < steps; ++step) {
    const OpenMM::State before =
        impl_->context->getState(OpenMM::State::Forces);
    double normal_force = 0.0;
    for (const auto &force : before.getForces()) {
      normal_force += dot(force, n);
    }
    impl_->custom->setGlobalVariableByName("normal_force", normal_force);
    impl_->custom->step(1);
    OpenMM::State after = impl_->context->getState(
        OpenMM::State::Positions | OpenMM::State::Velocities);
    auto positions = after.getPositions();
    auto velocities = after.getVelocities();
    for (int i = 0; i < impl_->particles; ++i) {
      const double distance = dot(positions[static_cast<size_t>(i)] -
                                      plane_point,
                                  n);
      positions[static_cast<size_t>(i)] =
          subtract(positions[static_cast<size_t>(i)], n, distance);
      velocities[static_cast<size_t>(i)] =
          subtract(velocities[static_cast<size_t>(i)], n,
                   dot(velocities[static_cast<size_t>(i)], n));
      for (size_t wall = 0; wall < wall_points.size(); ++wall) {
        const OpenMM::Vec3 &wall_normal = wall_normals[wall];
        const double penetration =
            dot(positions[static_cast<size_t>(i)] - wall_points[wall],
                wall_normal);
        if (penetration > 0.0) {
          positions[static_cast<size_t>(i)] =
              subtract(positions[static_cast<size_t>(i)], wall_normal,
                       penetration);
          velocities[static_cast<size_t>(i)] =
              subtract(velocities[static_cast<size_t>(i)], wall_normal,
                       2.0 * dot(velocities[static_cast<size_t>(i)],
                                  wall_normal));
          ++reflections;
        }
      }
    }
    impl_->context->setPositions(positions);
    impl_->context->setVelocities(velocities);
  }
  return reflections;
}

OpenMMState OpenMMSession::state() const {
  const OpenMM::State state = impl_->context->getState(
      OpenMM::State::Positions | OpenMM::State::Velocities |
      OpenMM::State::Forces | OpenMM::State::Energy);
  return {state.getPositions(), state.getVelocities(),
          state.getForces(),
          state.getPotentialEnergy(), state.getKineticEnergy()};
}

} // namespace rgpot

struct rgpot_md_session {
  std::unique_ptr<rgpot::OpenMMSession> session;
  size_t n3 = 0;
};

namespace {
void md_error(char *err, size_t len, const std::exception &error) {
  if (err != nullptr && len > 0) {
    std::snprintf(err, len, "%s", error.what());
  }
}

std::vector<OpenMM::Vec3> md_vectors(const double *values, size_t n3) {
  if (values == nullptr || n3 == 0 || n3 % 3 != 0) {
    throw std::invalid_argument("MD vector buffer must contain xyz triples");
  }
  std::vector<OpenMM::Vec3> out(n3 / 3);
  for (size_t i = 0; i < out.size(); ++i) {
    out[i] = OpenMM::Vec3(values[3 * i], values[3 * i + 1],
                          values[3 * i + 2]);
  }
  return out;
}

void copy_md_vectors(const std::vector<OpenMM::Vec3> &values, double *output,
                     size_t n3) {
  if (output == nullptr || n3 != 3 * values.size()) {
    throw std::invalid_argument("MD output buffer has the wrong length");
  }
  for (size_t i = 0; i < values.size(); ++i) {
    output[3 * i] = values[i][0];
    output[3 * i + 1] = values[i][1];
    output[3 * i + 2] = values[i][2];
  }
}
} // namespace

extern "C" {

rgpot_md_session *rgpot_md_session_create(
    const char *system_xml, const char *platform, double time_step,
    double friction, double kT, int seed, char *err, size_t err_len) {
  try {
    if (system_xml == nullptr || platform == nullptr) {
      throw std::invalid_argument("MD session needs XML and platform");
    }
    std::istringstream xml(system_xml);
    auto system = std::unique_ptr<OpenMM::System>(
        OpenMM::XmlSerializer::deserialize<OpenMM::System>(xml));
    rgpot::OpenMMSessionConfig config;
    config.platform = platform;
    config.time_step = time_step;
    config.friction = friction;
    config.temperature = kT;
    config.random_seed = seed;
    auto *out = new rgpot_md_session;
    out->session =
        std::make_unique<rgpot::OpenMMSession>(std::move(system), config);
    return out;
  } catch (const std::exception &error) {
    md_error(err, err_len, error);
    return nullptr;
  }
}

int rgpot_md_session_set_positions(rgpot_md_session *session,
                                   const double *values, size_t n3, char *err,
                                   size_t err_len) {
  try {
    if (session == nullptr) {
      throw std::invalid_argument("null MD session");
    }
    session->session->setPositions(md_vectors(values, n3));
    session->n3 = n3;
    return 0;
  } catch (const std::exception &error) {
    md_error(err, err_len, error);
    return 1;
  }
}

int rgpot_md_session_set_velocities(rgpot_md_session *session,
                                    const double *values, size_t n3, char *err,
                                    size_t err_len) {
  try {
    if (session == nullptr) {
      throw std::invalid_argument("null MD session");
    }
    session->session->setVelocities(md_vectors(values, n3));
    return 0;
  } catch (const std::exception &error) {
    md_error(err, err_len, error);
    return 1;
  }
}

int rgpot_md_session_step_langevin(
    rgpot_md_session *session, const double *plane_point,
    const double *plane_normal, const double *wall_points,
    const double *wall_normals, size_t n_walls, size_t steps,
    uint64_t *wall_hits, char *err, size_t err_len) {
  try {
    if (session == nullptr || plane_point == nullptr || plane_normal == nullptr ||
        (n_walls > 0 && (wall_points == nullptr || wall_normals == nullptr))) {
      throw std::invalid_argument("incomplete MD Langevin inputs");
    }
    std::vector<OpenMM::Vec3> points(n_walls);
    std::vector<OpenMM::Vec3> normals(n_walls);
    for (size_t i = 0; i < n_walls; ++i) {
      points[i] = OpenMM::Vec3(wall_points[3 * i], wall_points[3 * i + 1],
                               wall_points[3 * i + 2]);
      normals[i] = OpenMM::Vec3(wall_normals[3 * i], wall_normals[3 * i + 1],
                                wall_normals[3 * i + 2]);
    }
    const int hits = session->session->stepLangevin(
        OpenMM::Vec3(plane_point[0], plane_point[1], plane_point[2]),
        OpenMM::Vec3(plane_normal[0], plane_normal[1], plane_normal[2]),
        points, normals, static_cast<int>(steps));
    if (wall_hits != nullptr) {
      *wall_hits = static_cast<uint64_t>(hits);
    }
    return 0;
  } catch (const std::exception &error) {
    md_error(err, err_len, error);
    return 1;
  }
}

int rgpot_md_session_step_nve(rgpot_md_session *session, size_t steps,
                              char *err, size_t err_len) {
  try {
    if (session == nullptr) {
      throw std::invalid_argument("null MD session");
    }
    session->session->stepNVE(static_cast<int>(steps));
    return 0;
  } catch (const std::exception &error) {
    md_error(err, err_len, error);
    return 1;
  }
}

int rgpot_md_session_get_state(rgpot_md_session *session, double *positions,
                               double *velocities, double *forces, size_t n3,
                               double *potential_energy, double *kinetic_energy,
                               char *err, size_t err_len) {
  try {
    if (session == nullptr || potential_energy == nullptr ||
        kinetic_energy == nullptr) {
      throw std::invalid_argument("incomplete MD state outputs");
    }
    const auto state = session->session->state();
    copy_md_vectors(state.positions, positions, n3);
    copy_md_vectors(state.velocities, velocities, n3);
    copy_md_vectors(state.forces, forces, n3);
    *potential_energy = state.potential_energy;
    *kinetic_energy = state.kinetic_energy;
    return 0;
  } catch (const std::exception &error) {
    md_error(err, err_len, error);
    return 1;
  }
}

void rgpot_md_session_destroy(rgpot_md_session *session) { delete session; }

} // extern "C"
