// MIT License
// Copyright 2026 Rohit Goswami

#include "rgpot/ExternalPot/OpenMM.hpp"

#include "rgpot/MDSession/ProjectedLangevin.hpp"

#include <openmm/Context.h>
#include <openmm/Platform.h>
#include <openmm/State.h>
#include <openmm/System.h>
#include <openmm/VerletIntegrator.h>
#include <openmm/serialization/XmlSerializer.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace rgpot {
namespace {

void ensure_plugins_loaded() {
  static std::once_flag once;
  std::call_once(once, []() {
    if (const char *dir = std::getenv("OPENMM_PLUGIN_DIR")) {
      if (dir[0] != '\0') {
        OpenMM::Platform::loadPluginsFromDirectory(dir);
      }
    }
    OpenMM::Platform::loadPluginsFromDirectory(
        OpenMM::Platform::getDefaultPluginsDirectory());
  });
}

void require_positive(double value, const char *name) {
  if (!std::isfinite(value) || value <= 0.0) {
    throw std::invalid_argument(std::string(name) +
                                " must be finite and positive");
  }
}

void require_non_negative(double value, const char *name) {
  if (!std::isfinite(value) || value < 0.0) {
    throw std::invalid_argument(std::string(name) +
                                " must be finite and non-negative");
  }
}

void require_finite_vec3(const OpenMM::Vec3 &v, const char *name) {
  if (!std::isfinite(v[0]) || !std::isfinite(v[1]) || !std::isfinite(v[2])) {
    throw std::invalid_argument(std::string(name) + " must be finite");
  }
}

std::vector<double> flatten(const std::vector<OpenMM::Vec3> &values) {
  std::vector<double> out(3 * values.size());
  for (size_t i = 0; i < values.size(); ++i) {
    out[3 * i] = values[i][0];
    out[3 * i + 1] = values[i][1];
    out[3 * i + 2] = values[i][2];
  }
  return out;
}

void unflatten(const double *values, std::vector<OpenMM::Vec3> &out) {
  for (size_t i = 0; i < out.size(); ++i) {
    out[i] = OpenMM::Vec3(values[3 * i], values[3 * i + 1], values[3 * i + 2]);
  }
}

} // namespace

struct OpenMMSession::Impl {
  std::unique_ptr<OpenMM::System> system;
  std::unique_ptr<OpenMM::VerletIntegrator> integrator;
  std::unique_ptr<OpenMM::Context> context;
  std::vector<double> masses;
  std::unique_ptr<md::ProjectedLangevin> driver;
  md::SeededStream stream;
  std::vector<OpenMM::Vec3> scratch;
  size_t particles = 0;
  double default_time_step = 0.0;
};

OpenMMSession::OpenMMSession(std::unique_ptr<OpenMM::System> system,
                             const OpenMMSessionConfig &config)
    : impl_(std::make_unique<Impl>()) {
  if (system == nullptr) {
    throw std::invalid_argument("OpenMM session needs a System");
  }
  require_positive(config.time_step, "OpenMM time_step");
  require_non_negative(config.friction, "OpenMM friction");
  require_positive(config.temperature, "OpenMM temperature");
  if (config.platform.empty()) {
    throw std::invalid_argument("OpenMM platform is empty");
  }
  const int count = system->getNumParticles();
  if (count < 1) {
    throw std::invalid_argument("OpenMM System has no particles");
  }
  impl_->particles = static_cast<size_t>(count);
  impl_->default_time_step = config.time_step;
  impl_->masses.resize(impl_->particles);
  bool has_massive = false;
  for (size_t i = 0; i < impl_->particles; ++i) {
    const double mass = system->getParticleMass(static_cast<int>(i));
    require_non_negative(mass, "OpenMM particle mass");
    if (mass > 0.0) {
      has_massive = true;
    }
    impl_->masses[i] = mass;
  }
  if (!has_massive) {
    throw std::invalid_argument("OpenMM System has no massive particles");
  }
  for (int i = 0; i < count; ++i) {
    if (system->isVirtualSite(i)) {
      throw std::invalid_argument(
          "OpenMM session does not support virtual sites");
    }
  }
  std::vector<md::DistanceConstraint> distances;
  for (int c = 0; c < system->getNumConstraints(); ++c) {
    int first = 0;
    int second = 0;
    double length = 0.0;
    system->getConstraintParameters(c, first, second, length);
    distances.push_back({static_cast<size_t>(first),
                         static_cast<size_t>(second), length});
  }
  std::vector<double> coordinate_masses;
  coordinate_masses.reserve(3 * impl_->particles);
  for (const double mass : impl_->masses) {
    coordinate_masses.insert(coordinate_masses.end(), {mass, mass, mass});
  }
  impl_->driver =
      std::make_unique<md::ProjectedLangevin>(std::move(coordinate_masses),
                                              std::move(distances));
  impl_->scratch.resize(impl_->particles);
  impl_->system = std::move(system);
  ensure_plugins_loaded();
  impl_->integrator = std::make_unique<OpenMM::VerletIntegrator>(config.time_step);
  OpenMM::Platform &platform =
      OpenMM::Platform::getPlatformByName(config.platform);
  impl_->context = std::make_unique<OpenMM::Context>(
      *impl_->system, *impl_->integrator, platform);
}

OpenMMSession::~OpenMMSession() = default;
OpenMMSession::OpenMMSession(OpenMMSession &&) noexcept = default;
OpenMMSession &OpenMMSession::operator=(OpenMMSession &&) noexcept = default;

size_t OpenMMSession::numParticles() const noexcept {
  return impl_->particles;
}

std::vector<double> OpenMMSession::masses() const { return impl_->masses; }

void OpenMMSession::setPositions(const std::vector<OpenMM::Vec3> &positions) {
  if (positions.size() != impl_->particles) {
    throw std::invalid_argument("OpenMM position count differs from System");
  }
  for (const auto &p : positions) {
    require_finite_vec3(p, "OpenMM position");
  }
  impl_->context->setPositions(positions);
  impl_->stream.reset();
}

void OpenMMSession::setVelocities(const std::vector<OpenMM::Vec3> &velocities) {
  if (velocities.size() != impl_->particles) {
    throw std::invalid_argument("OpenMM velocity count differs from System");
  }
  for (const auto &v : velocities) {
    require_finite_vec3(v, "OpenMM velocity");
  }
  impl_->context->setVelocities(velocities);
  impl_->stream.reset();
}

void OpenMMSession::stepNVE(int steps) {
  if (steps < 0) {
    throw std::invalid_argument("OpenMM NVE steps must be non-negative");
  }
  stepNVE(impl_->default_time_step, static_cast<size_t>(steps));
}

void OpenMMSession::stepNVE(double time_step, size_t steps) {
  require_positive(time_step, "OpenMM NVE time_step");
  impl_->integrator->setStepSize(time_step);
  constexpr size_t kMaxChunk =
      static_cast<size_t>(std::numeric_limits<int>::max());
  size_t remaining = steps;
  while (remaining > 0) {
    const int chunk = static_cast<int>(std::min(remaining, kMaxChunk));
    impl_->integrator->step(chunk);
    remaining -= static_cast<size_t>(chunk);
  }
}

uint64_t OpenMMSession::stepLangevin(
    double kT, double friction, double time_step,
    const std::vector<OpenMM::Vec3> &plane_point,
    const std::vector<OpenMM::Vec3> &plane_normal,
    const std::vector<std::vector<OpenMM::Vec3>> &wall_points,
    const std::vector<std::vector<OpenMM::Vec3>> &wall_normals, size_t steps,
    uint64_t seed) {
  if (plane_point.size() != impl_->particles ||
      plane_normal.size() != impl_->particles) {
    throw std::invalid_argument(
        "OpenMM hyperplane vector count differs from System");
  }
  return runLangevin(kT, friction, time_step, plane_point, plane_normal,
                     wall_points, wall_normals, steps, seed,
                     OpenMMWallMode::Reflect, false)
      .wall_hits;
}

OpenMMLangevinResult OpenMMSession::runLangevin(
    double kT, double friction, double time_step,
    const std::vector<OpenMM::Vec3> &plane_point,
    const std::vector<OpenMM::Vec3> &plane_normal,
    const std::vector<std::vector<OpenMM::Vec3>> &wall_points,
    const std::vector<std::vector<OpenMM::Vec3>> &wall_normals, size_t steps,
    uint64_t seed, OpenMMWallMode wall_mode, bool record_constraint_force) {
  require_positive(kT, "OpenMM Langevin kT");
  require_non_negative(friction, "OpenMM Langevin friction");
  require_positive(time_step, "OpenMM Langevin time_step");
  const bool has_plane = !plane_point.empty() || !plane_normal.empty();
  if (has_plane && (plane_point.size() != impl_->particles ||
                    plane_normal.size() != impl_->particles)) {
    throw std::invalid_argument(
        "OpenMM hyperplane vector count differs from System");
  }
  if (wall_points.size() != wall_normals.size()) {
    throw std::invalid_argument("OpenMM wall point and normal count mismatch");
  }
  md::PlaneSpec plane;
  if (has_plane) {
    for (size_t i = 0; i < impl_->particles; ++i) {
      require_finite_vec3(plane_point[i], "OpenMM plane_point");
      require_finite_vec3(plane_normal[i], "OpenMM plane_normal");
    }
    plane.point = flatten(plane_point);
    plane.normal = flatten(plane_normal);
  }
  std::vector<md::WallSpec> walls(wall_points.size());
  for (size_t w = 0; w < wall_points.size(); ++w) {
    if (wall_points[w].size() != impl_->particles ||
        wall_normals[w].size() != impl_->particles) {
      throw std::invalid_argument(
          "OpenMM wall vector count differs from System");
    }
    for (size_t i = 0; i < impl_->particles; ++i) {
      require_finite_vec3(wall_points[w][i], "OpenMM wall_point");
      require_finite_vec3(wall_normals[w][i], "OpenMM wall_normal");
    }
    walls[w].point = flatten(wall_points[w]);
    walls[w].normal = flatten(wall_normals[w]);
  }
  OpenMMLangevinResult out;
  if (steps == 0) {
    return out;
  }

  const OpenMM::State initial = impl_->context->getState(
      OpenMM::State::Positions | OpenMM::State::Velocities);
  std::vector<double> x = flatten(initial.getPositions());
  std::vector<double> v = flatten(initial.getVelocities());

  md::LangevinSettings settings;
  settings.kT = kT;
  settings.friction = friction;
  settings.timeStep = time_step;
  settings.steps = steps;
  settings.wallMode = wall_mode == OpenMMWallMode::Reject
                          ? md::WallMode::Reject
                          : md::WallMode::Reflect;
  settings.recordConstraintForce = record_constraint_force;
  const md::ForceFunction force_at = [this](const double *positions,
                                            double *forces) {
    unflatten(positions, impl_->scratch);
    impl_->context->setPositions(impl_->scratch);
    const OpenMM::State at = impl_->context->getState(OpenMM::State::Forces |
                                                      OpenMM::State::Energy);
    const std::vector<OpenMM::Vec3> &f = at.getForces();
    for (size_t i = 0; i < f.size(); ++i) {
      forces[3 * i] = f[i][0];
      forces[3 * i + 1] = f[i][1];
      forces[3 * i + 2] = f[i][2];
    }
    return at.getPotentialEnergy();
  };
  md::LangevinResult result = impl_->driver->run(
      settings, has_plane ? &plane : nullptr, walls, x.data(), v.data(),
      force_at, impl_->stream.forSeed(seed));

  std::vector<OpenMM::Vec3> state(impl_->particles);
  unflatten(x.data(), state);
  impl_->context->setPositions(state);
  unflatten(v.data(), state);
  impl_->context->setVelocities(state);
  out.wall_hits = result.wallHits;
  out.constraint_force = std::move(result.constraintForce);
  return out;
}

void OpenMMSession::boxVectors(OpenMM::Vec3 &a, OpenMM::Vec3 &b,
                               OpenMM::Vec3 &c) const {
  const OpenMM::State state =
      impl_->context->getState(OpenMM::State::Positions);
  state.getPeriodicBoxVectors(a, b, c);
}

void OpenMMSession::setPeriodicBox(const std::array<double, 9> &box) {
  for (const double value : box) {
    if (!std::isfinite(value)) {
      throw std::invalid_argument("OpenMM periodic box must be finite");
    }
  }
  impl_->context->setPeriodicBoxVectors(
      OpenMM::Vec3(box[0], box[1], box[2]),
      OpenMM::Vec3(box[3], box[4], box[5]),
      OpenMM::Vec3(box[6], box[7], box[8]));
}

void OpenMMSession::initialize() {
  impl_->context->getState(OpenMM::State::Forces | OpenMM::State::Energy);
}

OpenMMState OpenMMSession::state() const {
  const OpenMM::State state = impl_->context->getState(
      OpenMM::State::Positions | OpenMM::State::Velocities |
      OpenMM::State::Forces | OpenMM::State::Energy);
  return {state.getPositions(), state.getVelocities(), state.getForces(),
          state.getPotentialEnergy(), state.getKineticEnergy()};
}

namespace {

void write_c_err(char *err, size_t len, const char *msg) {
  if (err != nullptr && len > 0) {
    std::snprintf(err, len, "%s", msg != nullptr ? msg : "unknown error");
  }
}

std::vector<OpenMM::Vec3> to_openmm_vec3(const double *values, size_t n3,
                                         double scale) {
  if (values == nullptr || n3 == 0 || n3 % 3 != 0) {
    throw std::invalid_argument("MD vector buffer must contain xyz triples");
  }
  std::vector<OpenMM::Vec3> out(n3 / 3);
  for (size_t i = 0; i < out.size(); ++i) {
    out[i] = OpenMM::Vec3(values[3 * i] * scale, values[3 * i + 1] * scale,
                          values[3 * i + 2] * scale);
  }
  return out;
}

void copy_from_openmm_vec3(const std::vector<OpenMM::Vec3> &values,
                           double *output, size_t n3, double scale) {
  if (output == nullptr) {
    return;
  }
  if (n3 != 3 * values.size()) {
    throw std::invalid_argument("MD output buffer has the wrong length");
  }
  for (size_t i = 0; i < values.size(); ++i) {
    output[3 * i] = values[i][0] * scale;
    output[3 * i + 1] = values[i][1] * scale;
    output[3 * i + 2] = values[i][2] * scale;
  }
}

int c_set_positions(void *ctx, const double *positions, size_t n3, char *err,
                    size_t err_len) {
  try {
    auto *session = static_cast<OpenMMSession *>(ctx);
    session->setPositions(to_openmm_vec3(positions, n3, md::kNmPerAngstrom));
    return 0;
  } catch (const std::exception &ex) {
    write_c_err(err, err_len, ex.what());
    return 1;
  } catch (...) {
    write_c_err(err, err_len, "unknown C++ exception");
    return 1;
  }
}

int c_set_velocities(void *ctx, const double *velocities, size_t n3, char *err,
                     size_t err_len) {
  try {
    auto *session = static_cast<OpenMMSession *>(ctx);
    session->setVelocities(to_openmm_vec3(velocities, n3, md::kNmPerAngstrom));
    return 0;
  } catch (const std::exception &ex) {
    write_c_err(err, err_len, ex.what());
    return 1;
  } catch (...) {
    write_c_err(err, err_len, "unknown C++ exception");
    return 1;
  }
}

int c_step_langevin(void *ctx, double kT, double friction, double time_step,
                    const double *plane_point, const double *plane_normal,
                    const double *wall_points, const double *wall_normals,
                    size_t n_walls, size_t n3, size_t steps, uint64_t seed,
                    uint64_t *wall_hits, char *err, size_t err_len) {
  try {
    auto *session = static_cast<OpenMMSession *>(ctx);
    if (n3 != 3 * session->numParticles()) {
      throw std::invalid_argument("MD Langevin vector length differs from 3*N");
    }
    const auto p_point = to_openmm_vec3(plane_point, n3, md::kNmPerAngstrom);
    const auto p_normal = to_openmm_vec3(plane_normal, n3, 1.0);
    std::vector<std::vector<OpenMM::Vec3>> w_points(n_walls);
    std::vector<std::vector<OpenMM::Vec3>> w_normals(n_walls);
    for (size_t w = 0; w < n_walls; ++w) {
      w_points[w] =
          to_openmm_vec3(wall_points + w * n3, n3, md::kNmPerAngstrom);
      w_normals[w] = to_openmm_vec3(wall_normals + w * n3, n3, 1.0);
    }
    const uint64_t hits = session->stepLangevin(
        kT * md::kKjPerEv, friction, time_step, p_point, p_normal, w_points,
        w_normals, steps, seed);
    if (wall_hits != nullptr) {
      *wall_hits = hits;
    }
    return 0;
  } catch (const std::exception &ex) {
    write_c_err(err, err_len, ex.what());
    return 1;
  } catch (...) {
    write_c_err(err, err_len, "unknown C++ exception");
    return 1;
  }
}

int c_set_periodic_box(void *ctx, const double *box9, char *err,
                       size_t err_len) {
  try {
    if (box9 == nullptr) {
      throw std::invalid_argument("MD periodic box is null");
    }
    std::array<double, 9> box{};
    for (size_t index = 0; index < box.size(); ++index) {
      box[index] = box9[index] * md::kNmPerAngstrom;
    }
    static_cast<OpenMMSession *>(ctx)->setPeriodicBox(box);
    return 0;
  } catch (const std::exception &ex) {
    write_c_err(err, err_len, ex.what());
    return 1;
  } catch (...) {
    write_c_err(err, err_len, "unknown C++ exception");
    return 1;
  }
}

int c_get_periodic_box(void *ctx, double *box9, char *err, size_t err_len) {
  try {
    if (box9 == nullptr) {
      throw std::invalid_argument("MD periodic box output is null");
    }
    OpenMM::Vec3 a;
    OpenMM::Vec3 b;
    OpenMM::Vec3 c;
    static_cast<OpenMMSession *>(ctx)->boxVectors(a, b, c);
    const OpenMM::Vec3 rows[3] = {a, b, c};
    for (size_t row = 0; row < 3; ++row) {
      for (size_t axis = 0; axis < 3; ++axis) {
        box9[3 * row + axis] = rows[row][static_cast<int>(axis)] *
                               md::kAngstromPerNm;
      }
    }
    return 0;
  } catch (const std::exception &ex) {
    write_c_err(err, err_len, ex.what());
    return 1;
  } catch (...) {
    write_c_err(err, err_len, "unknown C++ exception");
    return 1;
  }
}

int c_initialize(void *ctx, char *err, size_t err_len) {
  try {
    static_cast<OpenMMSession *>(ctx)->initialize();
    return 0;
  } catch (const std::exception &ex) {
    write_c_err(err, err_len, ex.what());
    return 1;
  } catch (...) {
    write_c_err(err, err_len, "unknown C++ exception");
    return 1;
  }
}

int c_run_langevin(void *ctx, const rgpot_md_langevin_request *request,
                   rgpot_md_langevin_result *result, char *err,
                   size_t err_len) {
  try {
    auto *session = static_cast<OpenMMSession *>(ctx);
    const size_t n3 = request->n3;
    if (n3 != 3 * session->numParticles()) {
      throw std::invalid_argument("MD Langevin vector length differs from 3*N");
    }
    std::vector<OpenMM::Vec3> p_point;
    std::vector<OpenMM::Vec3> p_normal;
    if (request->plane_point != nullptr) {
      p_point = to_openmm_vec3(request->plane_point, n3, md::kNmPerAngstrom);
      p_normal = to_openmm_vec3(request->plane_normal, n3, 1.0);
    }
    std::vector<std::vector<OpenMM::Vec3>> w_points(request->n_walls);
    std::vector<std::vector<OpenMM::Vec3>> w_normals(request->n_walls);
    for (size_t w = 0; w < request->n_walls; ++w) {
      w_points[w] = to_openmm_vec3(request->wall_points + w * n3, n3,
                                   md::kNmPerAngstrom);
      w_normals[w] = to_openmm_vec3(request->wall_normals + w * n3, n3, 1.0);
    }
    const bool record =
        request->plane_point != nullptr && result->constraint_force != nullptr;
    const OpenMMLangevinResult run = session->runLangevin(
        request->kT * md::kKjPerEv, request->friction, request->time_step,
        p_point, p_normal, w_points, w_normals, request->steps, request->seed,
        request->wall_mode == RGPOT_MD_WALL_REJECT ? OpenMMWallMode::Reject
                                                   : OpenMMWallMode::Reflect,
        record);
    result->wall_hits = run.wall_hits;
    if (record) {
      for (size_t k = 0; k < run.constraint_force.size(); ++k) {
        result->constraint_force[k] = run.constraint_force[k] * md::kEvAPerKjNm;
      }
    }
    return 0;
  } catch (const std::exception &ex) {
    write_c_err(err, err_len, ex.what());
    return 1;
  } catch (...) {
    write_c_err(err, err_len, "unknown C++ exception");
    return 1;
  }
}

int c_step_nve(void *ctx, double time_step, size_t steps, char *err,
               size_t err_len) {
  try {
    auto *session = static_cast<OpenMMSession *>(ctx);
    session->stepNVE(time_step, steps);
    return 0;
  } catch (const std::exception &ex) {
    write_c_err(err, err_len, ex.what());
    return 1;
  } catch (...) {
    write_c_err(err, err_len, "unknown C++ exception");
    return 1;
  }
}

int c_get_state(void *ctx, double *positions, double *velocities,
                double *forces, size_t n3, double *potential_energy,
                double *kinetic_energy, char *err, size_t err_len) {
  try {
    auto *session = static_cast<OpenMMSession *>(ctx);
    const OpenMMState st = session->state();
    copy_from_openmm_vec3(st.positions, positions, n3, md::kAngstromPerNm);
    copy_from_openmm_vec3(st.velocities, velocities, n3, md::kAngstromPerNm);
    copy_from_openmm_vec3(st.forces, forces, n3, md::kEvAPerKjNm);
    if (potential_energy != nullptr) {
      *potential_energy = st.potential_energy * md::kEvPerKj;
    }
    if (kinetic_energy != nullptr) {
      *kinetic_energy = st.kinetic_energy * md::kEvPerKj;
    }
    return 0;
  } catch (const std::exception &ex) {
    write_c_err(err, err_len, ex.what());
    return 1;
  } catch (...) {
    write_c_err(err, err_len, "unknown C++ exception");
    return 1;
  }
}

int c_get_masses(void *ctx, double *masses, size_t n_atoms, char *err,
                 size_t err_len) {
  try {
    auto *session = static_cast<OpenMMSession *>(ctx);
    const auto m = session->masses();
    if (masses == nullptr || n_atoms != m.size()) {
      throw std::invalid_argument("MD mass output buffer has wrong length");
    }
    for (size_t i = 0; i < n_atoms; ++i) {
      masses[i] = m[i];
    }
    return 0;
  } catch (const std::exception &ex) {
    write_c_err(err, err_len, ex.what());
    return 1;
  } catch (...) {
    write_c_err(err, err_len, "unknown C++ exception");
    return 1;
  }
}

void c_destroy(void *ctx) { delete static_cast<OpenMMSession *>(ctx); }

} // namespace

int openmm_md_session_init(const char *system_xml, const char *platform,
                           rgpot_md_session_ops *ops, void **ctx,
                           size_t *n_atoms, char *err, size_t err_len) {
  try {
    if (system_xml == nullptr || ops == nullptr || ctx == nullptr ||
        n_atoms == nullptr) {
      throw std::invalid_argument(
          "OpenMM MD session needs XML and output pointers");
    }
    std::istringstream xml(system_xml);
    auto system = std::unique_ptr<OpenMM::System>(
        OpenMM::XmlSerializer::deserialize<OpenMM::System>(xml));
    OpenMMSessionConfig config;
    if (platform != nullptr && platform[0] != '\0') {
      config.platform = platform;
    }
    auto session = std::make_unique<OpenMMSession>(std::move(system), config);
    *n_atoms = session->numParticles();
    ops->struct_size = sizeof(rgpot_md_session_ops);
    ops->set_positions = c_set_positions;
    ops->set_velocities = c_set_velocities;
    ops->step_langevin = c_step_langevin;
    ops->step_nve = c_step_nve;
    ops->get_state = c_get_state;
    ops->get_masses = c_get_masses;
    ops->destroy = c_destroy;
    ops->set_periodic_box = c_set_periodic_box;
    ops->get_periodic_box = c_get_periodic_box;
    ops->initialize = c_initialize;
    ops->run_langevin = c_run_langevin;
    *ctx = session.release();
    return 0;
  } catch (const std::exception &ex) {
    write_c_err(err, err_len, ex.what());
    return 1;
  } catch (...) {
    write_c_err(err, err_len, "unknown C++ exception");
    return 1;
  }
}

} // namespace rgpot
