// MIT License
// Copyright 2026 Rohit Goswami

#include "NblibSession.hpp"
#include "NblibTopology.hpp"
#include "rgpot/ExternalPot/external_potentials.h"
#include "rgpot/ExternalPot/md_units.hpp"

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
    if (box9 == nullptr) {
      throw std::invalid_argument("MD periodic box is null");
    }
    std::array<double, 9> box{};
    for (size_t index = 0; index < box.size(); ++index) {
      box[index] = box9[index] * kNmPerAngstrom;
    }
    static_cast<Session *>(context)->setBox(box.data());
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

int step_langevin(void *context, double k_t, double friction, double time_step,
                  const double *plane_point, const double *plane_normal,
                  const double *wall_points, const double *wall_normals,
                  size_t n_walls, size_t n3, size_t steps, uint64_t seed,
                  uint64_t *wall_hits, char *buffer, size_t length) {
  return guarded(buffer, length, [&] {
    Session *session = static_cast<Session *>(context);
    const auto point = scaled(plane_point, n3, kNmPerAngstrom);
    const auto normal = scaled(plane_normal, n3, 1.0);
    const std::vector<double> points =
        n_walls == 0 ? std::vector<double>{}
                     : scaled(wall_points, n_walls * n3, kNmPerAngstrom);
    const std::vector<double> normals =
        n_walls == 0 ? std::vector<double>{}
                     : scaled(wall_normals, n_walls * n3, 1.0);
    session->setSeed(seed);
    session->setTimeStep(time_step);
    const uint64_t hits = session->stepProjected(
        steps, k_t * kKjPerEv, friction, point.data(), normal.data(),
        n_walls == 0 ? nullptr : points.data(),
        n_walls == 0 ? nullptr : normals.data(), n_walls);
    if (wall_hits != nullptr) {
      *wall_hits = hits;
    }
  });
}

int step_nve(void *context, double time_step, size_t steps, char *buffer,
             size_t length) {
  return guarded(buffer, length, [&] {
    Session *session = static_cast<Session *>(context);
    session->setTimeStep(time_step);
    session->step(steps);
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
    const std::string topology(static_cast<const char *>(spec), spec_len);
    auto session = std::make_unique<Session>(
        rgpot::nblibpot::parseTopology(topology), 0.001);
    *n_atoms = session->atoms();
    *ops = {};
    ops->struct_size = sizeof(*ops);
    ops->set_periodic_box = set_periodic_box;
    ops->set_positions = set_positions;
    ops->set_velocities = set_velocities;
    ops->step_langevin = step_langevin;
    ops->step_nve = step_nve;
    ops->get_state = get_state;
    ops->get_masses = get_masses;
    ops->initialize = initialize;
    ops->destroy = destroy;
    *context = session.release();
  });
}
