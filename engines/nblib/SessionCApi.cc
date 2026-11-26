// MIT License
// Copyright 2026 Rohit Goswami

#include "NblibSession.hpp"
#include "NblibTopology.hpp"
#include "rgpot/ExternalPot/external_potentials.h"
#include "rgpot/ExternalPot/md_units.hpp"
#include "rgpot/engine_c_abi.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

// The generic loader recognizes RGPOT_NBLIB_ENGINE for this plugin.
using rgpot::md::kAngstromPerNm;
using rgpot::md::kEvAPerKjNm;
using rgpot::md::kEvPerKj;
using rgpot::md::kKjPerEv;
using rgpot::md::kNmPerAngstrom;
namespace md = rgpot::md;
using rgpot::nblibpot::Session;

void error(char *buffer, size_t length, const char *message) {
  if (buffer != nullptr && length != 0) {
    std::snprintf(buffer, length, "%s", message);
  }
}

template <class Function>
int guarded(char *buffer, size_t length, Function &&function) {
  try {
    function();
    return 0;
  } catch (const std::exception &exception) {
    error(buffer, length, exception.what());
    return 1;
  } catch (...) {
    error(buffer, length, "unknown nblib session error");
    return 1;
  }
}

std::vector<double> scaled(const double *values, size_t count, double factor) {
  if (values == nullptr) {
    throw std::invalid_argument("MD input is null");
  }
  std::vector<double> result(count);
  for (size_t index = 0; index < count; ++index) {
    result[index] = values[index] * factor;
  }
  return result;
}

int set_periodic_box(void *context, const double *box9, char *buffer,
                     size_t length) {
  return guarded(buffer, length, [&] {
    const auto box = scaled(box9, 9, kNmPerAngstrom);
    static_cast<Session *>(context)->setBox(box.data());
  });
}

int get_periodic_box(void *context, double *box9, char *buffer,
                     size_t length) {
  return guarded(buffer, length, [&] {
    if (box9 == nullptr) {
      throw std::invalid_argument("MD periodic box output is null");
    }
    const auto box = static_cast<Session *>(context)->box();
    for (size_t index = 0; index < box.size(); ++index) {
      box9[index] = box[index] * kAngstromPerNm;
    }
  });
}

int set_positions(void *context, const double *positions, size_t n3,
                  char *buffer, size_t length) {
  return guarded(buffer, length, [&] {
    const auto value = scaled(positions, n3, kNmPerAngstrom);
    static_cast<Session *>(context)->setPositions(value.data(), value.size());
  });
}

int set_velocities(void *context, const double *velocities, size_t n3,
                   char *buffer, size_t length) {
  return guarded(buffer, length, [&] {
    const auto value = scaled(velocities, n3, kNmPerAngstrom);
    static_cast<Session *>(context)->setVelocities(value.data(), value.size());
  });
}

// One entry for the legacy step_langevin and run_langevin: nblib units in,
// the multiplier per step out.
md::LangevinResult langevin(Session *session, double k_t, double friction,
                            double time_step, const double *plane_point,
                            const double *plane_normal,
                            const double *wall_points,
                            const double *wall_normals, size_t n_walls,
                            size_t n3, size_t steps, uint64_t seed,
                            int wall_mode, bool record) {
  md::LangevinSettings settings;
  settings.kT = k_t * kKjPerEv;
  settings.friction = friction;
  settings.timeStep = time_step;
  settings.steps = steps;
  settings.seed = seed;
  settings.wallMode =
      wall_mode == RGPOT_MD_WALL_REJECT ? md::WallMode::Reject
                                        : md::WallMode::Reflect;
  settings.recordConstraintForce = record;
  md::PlaneSpec plane;
  const bool has_plane = plane_point != nullptr && plane_normal != nullptr;
  if (has_plane) {
    plane.point = scaled(plane_point, n3, kNmPerAngstrom);
    plane.normal = scaled(plane_normal, n3, 1.0);
  }
  std::vector<md::WallSpec> walls(n_walls);
  for (size_t wall = 0; wall < n_walls; ++wall) {
    walls[wall].point = scaled(wall_points + wall * n3, n3, kNmPerAngstrom);
    walls[wall].normal = scaled(wall_normals + wall * n3, n3, 1.0);
  }
  return session->langevin(settings, has_plane ? &plane : nullptr, walls);
}

int step_langevin(void *context, double k_t, double friction, double time_step,
                  const double *plane_point, const double *plane_normal,
                  const double *wall_points, const double *wall_normals,
                  size_t n_walls, size_t n3, size_t steps, uint64_t seed,
                  uint64_t *wall_hits, char *buffer, size_t length) {
  return guarded(buffer, length, [&] {
    const md::LangevinResult result = langevin(
        static_cast<Session *>(context), k_t, friction, time_step,
        plane_point, plane_normal, wall_points, wall_normals, n_walls, n3,
        steps, seed, RGPOT_MD_WALL_REFLECT, false);
    if (wall_hits != nullptr) {
      *wall_hits = result.wallHits;
    }
  });
}

int get_neighbors(void *context, const rgpot_nlist_request *request,
                 rgpot_nlist_soa *out, size_t *count, size_t *needed,
                 char *buffer, size_t length) {
  (void)context;
  (void)request;
  (void)out;
  (void)count;
  (void)needed;
  error(buffer, length, "nblib MD session does not expose neighbor queries");
  return RGPOT_MD_NLIST_UNSUPPORTED;
}

int set_neighbors(void *context, const rgpot_nlist_request *request,
                  const rgpot_nlist_soa *pairs, size_t count, char *buffer,
                  size_t length) {
  (void)context;
  (void)request;
  (void)pairs;
  (void)count;
  error(buffer, length, "nblib MD session does not accept supplied neighbors");
  return RGPOT_MD_NLIST_UNSUPPORTED;
}

int run_langevin(void *context, const rgpot_md_langevin_request *request,
                 rgpot_md_langevin_result *result, char *buffer,
                 size_t length) {
  return guarded(buffer, length, [&] {
    if (request == nullptr || result == nullptr) {
      throw std::invalid_argument(
          "nblib Langevin needs a request and a result");
    }
    if (request->struct_size < sizeof(rgpot_md_langevin_request) ||
        result->struct_size < sizeof(rgpot_md_langevin_result)) {
      throw std::invalid_argument(
          "nblib Langevin request or result struct_size is too small");
    }
    const bool record = request->plane_point != nullptr &&
                        result->constraint_force != nullptr;
    const md::LangevinResult run = langevin(
        static_cast<Session *>(context), request->kT, request->friction,
        request->time_step, request->plane_point, request->plane_normal,
        request->wall_points, request->wall_normals, request->n_walls,
        request->n3, request->steps, request->seed, request->wall_mode,
        record);
    result->wall_hits = run.wallHits;
    if (record) {
      if (result->constraint_force_len < run.constraintForce.size()) {
        throw std::invalid_argument(
            "nblib Langevin constraint_force buffer is shorter than steps");
      }
      for (size_t step = 0; step < run.constraintForce.size(); ++step) {
        result->constraint_force[step] =
            run.constraintForce[step] * kEvAPerKjNm;
      }
    }
  });
}

int step_nve(void *context, double time_step, size_t steps, char *buffer,
             size_t length) {
  return guarded(buffer, length, [&] {
    static_cast<Session *>(context)->step(steps, time_step);
  });
}

int get_state(void *context, double *positions, double *velocities,
              double *forces, size_t n3, double *potential, double *kinetic,
              char *buffer, size_t length) {
  return guarded(buffer, length, [&] {
    std::vector<double> x(n3);
    std::vector<double> v(n3);
    std::vector<double> f(n3);
    rgpot::nblibpot::SessionInfo info;
    static_cast<Session *>(context)->state(x.data(), v.data(), f.data(), n3,
                                           &info);
    for (size_t index = 0; index < n3; ++index) {
      if (positions != nullptr) positions[index] = x[index] * kAngstromPerNm;
      if (velocities != nullptr) velocities[index] = v[index] * kAngstromPerNm;
      if (forces != nullptr) forces[index] = f[index] * kEvAPerKjNm;
    }
    if (potential != nullptr) *potential = info.potential * kEvPerKj;
    if (kinetic != nullptr) *kinetic = info.kinetic * kEvPerKj;
  });
}

int get_masses(void *context, double *masses, size_t n_atoms, char *buffer,
               size_t length) {
  return guarded(buffer, length, [&] {
    const auto &values = static_cast<Session *>(context)->masses();
    if (masses == nullptr || values.size() != n_atoms) {
      throw std::invalid_argument("MD mass output has wrong length");
    }
    std::memcpy(masses, values.data(), values.size() * sizeof(double));
  });
}

int initialize(void *context, char *buffer, size_t length) {
  return guarded(buffer, length,
                 [&] { static_cast<Session *>(context)->initialize(); });
}

void destroy(void *context) { delete static_cast<Session *>(context); }

} // namespace

extern "C" int rgpot_md_session_engine_create(
    const void *spec, size_t spec_len, const char *options,
    rgpot_md_session_ops *ops, void **context, size_t *n_atoms, char *buffer,
    size_t length) {
  return guarded(buffer, length, [&] {
    if (spec == nullptr || spec_len == 0 || ops == nullptr ||
        context == nullptr || n_atoms == nullptr) {
      throw std::invalid_argument("nblib MD session needs topology and outputs");
    }
    (void)options;
    const size_t capacity = ops->struct_size;
    if (capacity < RGPOT_MD_OPS_END(destroy)) {
      throw std::invalid_argument(
          "nblib MD session ops struct_size capacity is too small");
    }
    const std::string topology(static_cast<const char *>(spec), spec_len);
    auto session = std::make_unique<Session>(
        rgpot::nblibpot::parseTopology(topology));
    const size_t filled = std::min(capacity, sizeof(rgpot_md_session_ops));
    rgpot_md_session_ops full{};
    full.struct_size = filled;
    full.set_positions = set_positions;
    full.set_velocities = set_velocities;
    full.step_langevin = step_langevin;
    full.step_nve = step_nve;
    full.get_state = get_state;
    full.get_masses = get_masses;
    full.destroy = destroy;
    if (filled >= RGPOT_MD_OPS_END(set_periodic_box)) {
      full.set_periodic_box = set_periodic_box;
    }
    if (filled >= RGPOT_MD_OPS_END(get_periodic_box)) {
      full.get_periodic_box = get_periodic_box;
    }
    if (filled >= RGPOT_MD_OPS_END(initialize)) {
      full.initialize = initialize;
    }
    if (filled >= RGPOT_MD_OPS_END(run_langevin)) {
      full.run_langevin = run_langevin;
    }
    if (filled >= RGPOT_MD_OPS_END(get_neighbors)) {
      full.get_neighbors = get_neighbors;
    }
    if (filled >= RGPOT_MD_OPS_END(set_neighbors)) {
      full.set_neighbors = set_neighbors;
    }
    std::memcpy(ops, &full, filled);
    *n_atoms = session->atoms();
    *context = session.release();
  });
}

extern "C" int rgpot_engine_abi_version(void) {
  return RGPOT_ENGINE_ABI_VERSION;
}

// GROMACS keeps process state, so the host must not dlclose this library.
extern "C" int rgpot_engine_unload_policy(void) {
  return RGPOT_ENGINE_UNLOAD_RESIDENT;
}
