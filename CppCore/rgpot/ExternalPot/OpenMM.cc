// MIT License
// Copyright 2026 Rohit Goswami

#include "rgpot/ExternalPot/OpenMM.hpp"

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

constexpr size_t kMaxReflectionPasses = 8;

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

class SplitMix64 {
public:
  explicit SplitMix64(uint64_t seed) noexcept : state_(seed) {}

  uint64_t next_u64() noexcept {
    state_ += 0x9e3779b97f4a7c15ULL;
    uint64_t z = state_;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
  }

  double next_uniform() noexcept {
    const uint64_t bits = next_u64() >> 11;
    return (static_cast<double>(bits) + 0.5) * 0x1.0p-53;
  }

  double next_normal() noexcept {
    if (has_spare_) {
      has_spare_ = false;
      return spare_;
    }
    const double u1 = next_uniform();
    const double u2 = next_uniform();
    const double r = std::sqrt(-2.0 * std::log(u1));
    const double theta = 2.0 * kPi * u2;
    spare_ = r * std::sin(theta);
    has_spare_ = true;
    return r * std::cos(theta);
  }

private:
  static constexpr double kPi = 3.14159265358979323846;
  uint64_t state_;
  bool has_spare_ = false;
  double spare_ = 0.0;
};

struct PreparedWall {
  std::vector<OpenMM::Vec3> normal;
  std::vector<OpenMM::Vec3> point;
  std::vector<OpenMM::Vec3> tangent_m_inv;
  double scale = 0.0;
};

void project_velocity(std::vector<OpenMM::Vec3> &velocities,
                      const std::vector<OpenMM::Vec3> &plane_normal,
                      const std::vector<OpenMM::Vec3> &plane_w) {
  double nv = 0.0;
  for (size_t i = 0; i < velocities.size(); ++i) {
    nv += plane_normal[i].dot(velocities[i]);
  }
  for (size_t i = 0; i < velocities.size(); ++i) {
    velocities[i] -= plane_w[i] * nv;
  }
}

void restore_position(std::vector<OpenMM::Vec3> &positions,
                      const std::vector<OpenMM::Vec3> &plane_point,
                      const std::vector<OpenMM::Vec3> &plane_normal,
                      const std::vector<OpenMM::Vec3> &plane_w) {
  double nx = 0.0;
  for (size_t i = 0; i < positions.size(); ++i) {
    nx += plane_normal[i].dot(positions[i] - plane_point[i]);
  }
  for (size_t i = 0; i < positions.size(); ++i) {
    positions[i] -= plane_w[i] * nx;
  }
}

uint64_t reflect_walls(std::vector<OpenMM::Vec3> &positions,
                       std::vector<OpenMM::Vec3> &velocities,
                       const std::vector<PreparedWall> &walls,
                       const std::vector<double> &masses,
                       const std::vector<OpenMM::Vec3> &plane_point,
                       const std::vector<OpenMM::Vec3> &plane_normal,
                       const std::vector<OpenMM::Vec3> &plane_w) {
  if (walls.empty()) {
    return 0;
  }
  uint64_t hits = 0;
  for (size_t pass = 0; pass < kMaxReflectionPasses; ++pass) {
    bool any_hit = false;
    for (const PreparedWall &wall : walls) {
      if (wall.scale <= 0.0) {
        continue;
      }
      double raw = 0.0;
      for (size_t i = 0; i < positions.size(); ++i) {
        raw += wall.normal[i].dot(positions[i] - wall.point[i]);
      }
      const double excess = raw / wall.scale;
      if (excess > 0.0) {
        double v_dot = 0.0;
        for (size_t i = 0; i < velocities.size(); ++i) {
          if (masses[i] > 0.0) {
            v_dot += (wall.tangent_m_inv[i] * masses[i]).dot(velocities[i]);
          }
        }
        for (size_t i = 0; i < positions.size(); ++i) {
          positions[i] -= wall.tangent_m_inv[i] * (2.0 * excess);
          velocities[i] -= wall.tangent_m_inv[i] * (2.0 * v_dot);
        }
        restore_position(positions, plane_point, plane_normal, plane_w);
        project_velocity(velocities, plane_normal, plane_w);
        any_hit = true;
        ++hits;
      }
    }
    if (!any_hit) {
      break;
    }
  }
  return hits;
}

} // namespace

struct OpenMMSession::Impl {
  std::unique_ptr<OpenMM::System> system;
  std::unique_ptr<OpenMM::VerletIntegrator> integrator;
  std::unique_ptr<OpenMM::Context> context;
  std::vector<double> masses;
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
}

void OpenMMSession::setVelocities(const std::vector<OpenMM::Vec3> &velocities) {
  if (velocities.size() != impl_->particles) {
    throw std::invalid_argument("OpenMM velocity count differs from System");
  }
  for (const auto &v : velocities) {
    require_finite_vec3(v, "OpenMM velocity");
  }
  impl_->context->setVelocities(velocities);
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
  require_positive(kT, "OpenMM Langevin kT");
  require_non_negative(friction, "OpenMM Langevin friction");
  require_positive(time_step, "OpenMM Langevin time_step");
  if (plane_point.size() != impl_->particles ||
      plane_normal.size() != impl_->particles) {
    throw std::invalid_argument("OpenMM hyperplane vector count differs from System");
  }
  if (wall_points.size() != wall_normals.size()) {
    throw std::invalid_argument("OpenMM wall point and normal count mismatch");
  }

  double norm_sq = 0.0;
  for (size_t i = 0; i < impl_->particles; ++i) {
    require_finite_vec3(plane_point[i], "OpenMM plane_point");
    require_finite_vec3(plane_normal[i], "OpenMM plane_normal");
    norm_sq += plane_normal[i].dot(plane_normal[i]);
  }
  const double normal_norm = std::sqrt(norm_sq);
  require_positive(normal_norm, "OpenMM plane_normal");
  std::vector<OpenMM::Vec3> n(impl_->particles);
  double alpha = 0.0;
  for (size_t i = 0; i < impl_->particles; ++i) {
    n[i] = plane_normal[i] * (1.0 / normal_norm);
    if (impl_->masses[i] > 0.0) {
      alpha += n[i].dot(n[i]) / impl_->masses[i];
    }
  }
  require_positive(alpha, "OpenMM hyperplane inverse-mass norm");
  std::vector<OpenMM::Vec3> plane_w(impl_->particles, OpenMM::Vec3(0.0, 0.0, 0.0));
  for (size_t i = 0; i < impl_->particles; ++i) {
    if (impl_->masses[i] > 0.0) {
      plane_w[i] = n[i] * (1.0 / (alpha * impl_->masses[i]));
    }
  }

  std::vector<PreparedWall> walls(wall_points.size());
  for (size_t w = 0; w < wall_points.size(); ++w) {
    if (wall_points[w].size() != impl_->particles ||
        wall_normals[w].size() != impl_->particles) {
      throw std::invalid_argument("OpenMM wall vector count differs from System");
    }
    double u_sq = 0.0;
    for (size_t i = 0; i < impl_->particles; ++i) {
      require_finite_vec3(wall_points[w][i], "OpenMM wall_point");
      require_finite_vec3(wall_normals[w][i], "OpenMM wall_normal");
      u_sq += wall_normals[w][i].dot(wall_normals[w][i]);
    }
    const double u_norm = std::sqrt(u_sq);
    require_positive(u_norm, "OpenMM wall_normal");
    walls[w].point = wall_points[w];
    walls[w].normal.resize(impl_->particles);
    double n_minv_u = 0.0;
    for (size_t i = 0; i < impl_->particles; ++i) {
      walls[w].normal[i] = wall_normals[w][i] * (1.0 / u_norm);
      if (impl_->masses[i] > 0.0) {
        n_minv_u += n[i].dot(walls[w].normal[i]) / impl_->masses[i];
      }
    }
    const double coeff = n_minv_u / alpha;
    std::vector<OpenMM::Vec3> u_tan(impl_->particles);
    double scale_sq = 0.0;
    for (size_t i = 0; i < impl_->particles; ++i) {
      u_tan[i] = walls[w].normal[i] - n[i] * coeff;
      if (impl_->masses[i] > 0.0) {
        scale_sq += u_tan[i].dot(u_tan[i]) / impl_->masses[i];
      }
    }
    const double scale = std::sqrt(scale_sq);
    walls[w].scale = scale;
    walls[w].tangent_m_inv.assign(impl_->particles, OpenMM::Vec3(0.0, 0.0, 0.0));
    if (scale > 0.0) {
      for (size_t i = 0; i < impl_->particles; ++i) {
        if (impl_->masses[i] > 0.0) {
          walls[w].tangent_m_inv[i] = u_tan[i] * (1.0 / (scale * impl_->masses[i]));
        }
      }
    }
  }

  if (steps == 0) {
    return 0;
  }

  const OpenMM::State initial = impl_->context->getState(
      OpenMM::State::Positions | OpenMM::State::Velocities |
      OpenMM::State::Forces);
  auto positions = initial.getPositions();
  auto velocities = initial.getVelocities();
  auto forces = initial.getForces();

  restore_position(positions, plane_point, n, plane_w);
  project_velocity(velocities, n, plane_w);

  const double half_dt = 0.5 * time_step;
  const double decay = std::exp(-friction * time_step);
  const double noise_amp =
      std::sqrt(kT * (1.0 - std::exp(-2.0 * friction * time_step)));
  SplitMix64 rng(seed);
  uint64_t reflections = 0;

  for (size_t step = 0; step < steps; ++step) {
    for (size_t i = 0; i < impl_->particles; ++i) {
      if (impl_->masses[i] > 0.0) {
        velocities[i] += forces[i] * (half_dt / impl_->masses[i]);
      }
    }
    project_velocity(velocities, n, plane_w);

    for (size_t i = 0; i < impl_->particles; ++i) {
      positions[i] += velocities[i] * half_dt;
    }
    restore_position(positions, plane_point, n, plane_w);
    reflections += reflect_walls(positions, velocities, walls, impl_->masses,
                                 plane_point, n, plane_w);

    for (size_t i = 0; i < impl_->particles; ++i) {
      if (impl_->masses[i] > 0.0) {
        const double sigma = noise_amp / std::sqrt(impl_->masses[i]);
        const OpenMM::Vec3 kick(sigma * rng.next_normal(),
                                sigma * rng.next_normal(),
                                sigma * rng.next_normal());
        velocities[i] = velocities[i] * decay + kick;
      }
    }
    project_velocity(velocities, n, plane_w);

    for (size_t i = 0; i < impl_->particles; ++i) {
      positions[i] += velocities[i] * half_dt;
    }
    restore_position(positions, plane_point, n, plane_w);
    reflections += reflect_walls(positions, velocities, walls, impl_->masses,
                                 plane_point, n, plane_w);

    impl_->context->setPositions(positions);
    const OpenMM::State post = impl_->context->getState(OpenMM::State::Forces);
    forces = post.getForces();
    for (size_t i = 0; i < impl_->particles; ++i) {
      if (impl_->masses[i] > 0.0) {
        velocities[i] += forces[i] * (half_dt / impl_->masses[i]);
      }
    }
    project_velocity(velocities, n, plane_w);
  }

  impl_->context->setPositions(positions);
  impl_->context->setVelocities(velocities);
  return reflections;
}

OpenMMState OpenMMSession::state() const {
  const OpenMM::State state = impl_->context->getState(
      OpenMM::State::Positions | OpenMM::State::Velocities |
      OpenMM::State::Forces | OpenMM::State::Energy);
  return {state.getPositions(), state.getVelocities(), state.getForces(),
          state.getPotentialEnergy(), state.getKineticEnergy()};
}

void OpenMMSession::boxVectors(OpenMM::Vec3 &a, OpenMM::Vec3 &b,
                               OpenMM::Vec3 &c) const {
  const OpenMM::State state = impl_->context->getState(OpenMM::State::Positions);
  state.getPeriodicBoxVectors(a, b, c);
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

int c_get_neighbors(void *ctx, const rgpot_nlist_request *request,
                    rgpot_nlist_soa *out, size_t *count, size_t *needed,
                    char *err, size_t err_len) {
  try {
    auto *session = static_cast<OpenMMSession *>(ctx);
    const OpenMMState st = session->state();
    std::vector<double> pos_ang(st.positions.size() * 3);
    for (size_t i = 0; i < st.positions.size(); ++i) {
      pos_ang[3 * i] = st.positions[i][0] * md::kAngstromPerNm;
      pos_ang[3 * i + 1] = st.positions[i][1] * md::kAngstromPerNm;
      pos_ang[3 * i + 2] = st.positions[i][2] * md::kAngstromPerNm;
    }
    OpenMM::Vec3 a, b, c;
    session->boxVectors(a, b, c);
    const std::array<double, 9> box_ang = {
        a[0] * md::kAngstromPerNm, a[1] * md::kAngstromPerNm,
        a[2] * md::kAngstromPerNm, b[0] * md::kAngstromPerNm,
        b[1] * md::kAngstromPerNm, b[2] * md::kAngstromPerNm,
        c[0] * md::kAngstromPerNm, c[1] * md::kAngstromPerNm,
        c[2] * md::kAngstromPerNm,
    };
    const int32_t status = rgpot_nlist_build(pos_ang.data(), st.positions.size(),
                                             box_ang.data(), request, out,
                                             count, needed);
    if (status != RGPOT_NLIST_OK) {
      write_c_err(err, err_len, rgpot_nlist_strerror(status));
      return RGPOT_MD_NLIST_ERROR;
    }
    return RGPOT_MD_NLIST_OK;
  } catch (const std::exception &ex) {
    write_c_err(err, err_len, ex.what());
    return RGPOT_MD_NLIST_ERROR;
  } catch (...) {
    write_c_err(err, err_len, "unknown C++ exception");
    return RGPOT_MD_NLIST_ERROR;
  }
}

int c_set_neighbors(void *ctx, const rgpot_nlist_request *request,
                    const rgpot_nlist_soa *pairs, size_t count, char *err,
                    size_t err_len) {
  (void)ctx;
  (void)request;
  (void)pairs;
  (void)count;
  write_c_err(err, err_len,
              "OpenMM MD session does not accept externally supplied neighbor lists");
  return RGPOT_MD_NLIST_UNSUPPORTED;
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
    ops->get_neighbors = c_get_neighbors;
    ops->set_neighbors = c_set_neighbors;
    ops->destroy = c_destroy;
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
