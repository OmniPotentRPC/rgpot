// MIT License
// Copyright 2026 Rohit Goswami

#include "lammpc.h"
#include "lammpc_engine_spec.h"
#include "rgpot/ExternalPot/external_potentials.h"
#include "rgpot/ExternalPot/md_units.hpp"
#include "rgpot/abi/ProfileLoader.hpp"
#include "rgpot/rpc/Potentials.capnp.h"

#include <capnp/message.h>
#include <capnp/serialize.h>
#include <kj/array.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr double kAmuA2Ps2PerEv = rgpot::md::kKjPerEv * 100.0;
constexpr size_t kReflectionPasses = 16;

void write_error(char *buffer, size_t length, const char *message) {
  if (buffer != nullptr && length != 0) {
    std::snprintf(buffer, length, "%s",
                  message != nullptr ? message : "unknown lammpc error");
  }
}

template <typename Function>
int guarded(char *buffer, size_t length, Function &&function) {
  try {
    function();
    return 0;
  } catch (const std::exception &error) {
    write_error(buffer, length, error.what());
    return 1;
  } catch (...) {
    write_error(buffer, length, "unknown lammpc session error");
    return 1;
  }
}

std::vector<capnp::word> words(const unsigned char *data, size_t size,
                               const char *name) {
  if (data == nullptr || size == 0 || size % sizeof(capnp::word) != 0) {
    throw std::invalid_argument(std::string(name) +
                                " must be a nonempty flat Cap'n Proto message");
  }
  std::vector<capnp::word> output(size / sizeof(capnp::word));
  std::memcpy(output.data(), data, size);
  return output;
}

std::vector<unsigned char> flatten(capnp::MessageBuilder &message) {
  kj::Array<capnp::word> flat = capnp::messageToFlatArray(message);
  const auto bytes = flat.asBytes();
  return std::vector<unsigned char>(bytes.begin(), bytes.end());
}

class SplitMix64 {
public:
  explicit SplitMix64(uint64_t seed) : state_(seed) {}

  uint64_t next() {
    state_ += UINT64_C(0x9e3779b97f4a7c15);
    uint64_t value = state_;
    value = (value ^ (value >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27)) * UINT64_C(0x94d049bb133111eb);
    return value ^ (value >> 31);
  }

  double normal() {
    const double first = (static_cast<double>(next() >> 11) + 0.5) * 0x1.0p-53;
    const double second = (static_cast<double>(next() >> 11) + 0.5) * 0x1.0p-53;
    return std::sqrt(-2.0 * std::log(first)) *
           std::cos(6.283185307179586476925286766559 * second);
  }

private:
  uint64_t state_;
};

struct Wall {
  std::vector<double> point;
  std::vector<double> normal;
  std::vector<double> tangent_m_inv;
  double scale = 0.0;
};

struct EngineSession {
  using MdInitializeFn = int (*)(void *, const void *, size_t, const double *,
                                 const double *, size_t,
                                 const LammpcMdConfig *);
  using MdSetVelocitiesFn = int (*)(void *, const double *, size_t,
                                    unsigned long long);
  using MdStepFn = int (*)(void *, size_t);
  using MdStateFn = int (*)(void *, double *, double *, double *, double *,
                            size_t, LammpcMdObservation *);
  using MassesFn = int (*)(void *, double *, size_t);
  using NveSetPositionsFn = int (*)(void *, const double *, size_t,
                                    const double *, size_t);
  using NveSetVelocitiesFn = int (*)(void *, const double *, size_t);
  using NveStepFn = int (*)(void *, size_t, double);
  using NveStateFn = int (*)(void *, double *, double *, size_t,
                             LammpcNveObservation *);
  using SetBoundaryFn = int (*)(void *, LammpcHardBoundaryFn, void *);
  using ClearBoundaryFn = int (*)(void *);

  rgpot::abi::ProfileLoader loader;
  void *session = nullptr;
  std::vector<unsigned char> config;
  std::vector<int> atomic_numbers;
  std::vector<double> positions;
  std::vector<double> velocities;
  std::vector<double> forces;
  std::vector<double> masses;
  std::vector<double> box;
  std::vector<double> plane_normal;
  std::vector<Wall> walls;
  double potential = 0.0;
  double kinetic = 0.0;
  double constraint_force = 0.0;
  uint64_t wall_hits = 0;
  bool dirty = true;

  MdInitializeFn md_initialize = nullptr;
  MdSetVelocitiesFn md_set_velocities = nullptr;
  MdStepFn md_step = nullptr;
  MdStateFn md_state = nullptr;
  MassesFn session_masses = nullptr;
  NveSetPositionsFn nve_set_positions = nullptr;
  NveSetVelocitiesFn nve_set_velocities = nullptr;
  NveStepFn nve_step = nullptr;
  NveStateFn nve_state = nullptr;
  SetBoundaryFn set_boundary = nullptr;
  ClearBoundaryFn clear_boundary = nullptr;

  ~EngineSession() {
    if (session != nullptr && loader.session_destroy != nullptr) {
      loader.session_destroy(session);
    }
  }

  const char *lastError() const {
    return loader.last_error != nullptr ? loader.last_error()
                                        : "lammpc operation failed";
  }

  template <typename Function> Function required(const char *name) {
    Function result = loader.optional<Function>(name);
    if (result == nullptr) {
      throw std::runtime_error(std::string("liblammpc is missing ") +
                               "lammpc_" + name);
    }
    return result;
  }

  std::vector<unsigned char> forceInput() const {
    capnp::MallocMessageBuilder message;
    auto input = message.initRoot<::ForceInput>();
    auto pos = input.initPos(positions.size());
    for (size_t index = 0; index < positions.size(); ++index) {
      pos.set(index, positions[index]);
    }
    auto numbers = input.initAtmnrs(atomic_numbers.size());
    for (size_t index = 0; index < atomic_numbers.size(); ++index) {
      numbers.set(index, atomic_numbers[index]);
    }
    auto cell = input.initBox(box.size());
    for (size_t index = 0; index < box.size(); ++index) {
      cell.set(index, box[index]);
    }
    input.setLengthUnit("angstrom");
    input.setEnergyUnit("eV");
    return flatten(message);
  }

  void calculate() {
    const std::vector<unsigned char> input = forceInput();
    const size_t output_size = loader.potential_result_size_for_force_input(
        input.data(), input.size());
    if (output_size == 0) {
      throw std::runtime_error("lammpc refused PotentialResult sizing");
    }
    std::vector<unsigned char> output(output_size);
    size_t written = 0;
    const rgpot::abi::ProfileResult result =
        loader.session_calculate_result(session, input.data(), input.size(),
                                        output.data(), output.size(), &written);
    if (!result.ok) {
      throw std::runtime_error(std::string("lammpc evaluation failed: ") +
                               result.message);
    }
    output.resize(written);
    auto storage = words(output.data(), output.size(), "PotentialResult");
    capnp::FlatArrayMessageReader reader(
        kj::arrayPtr(storage.data(), storage.size()));
    const auto result_message = reader.getRoot<::PotentialResult>();
    const auto result_forces = result_message.getForces();
    if (result_forces.size() != forces.size()) {
      throw std::runtime_error("lammpc returned the wrong force count");
    }
    potential = result_message.getEnergy();
    for (size_t index = 0; index < forces.size(); ++index) {
      forces[index] = result_forces[index];
    }
    dirty = false;
  }

  void updateKinetic() {
    kinetic = 0.0;
    for (size_t atom = 0; atom < masses.size(); ++atom) {
      const size_t offset = 3 * atom;
      const double speed2 = velocities[offset] * velocities[offset] +
                            velocities[offset + 1] * velocities[offset + 1] +
                            velocities[offset + 2] * velocities[offset + 2];
      kinetic += 0.5 * masses[atom] * speed2 / kAmuA2Ps2PerEv;
    }
  }

  void prepareWalls(const double *plane, const double *points,
                    const double *normals, size_t count) {
    const size_t n3 = positions.size();
    plane_normal.assign(plane, plane + n3);
    double plane_norm2 = 0.0;
    for (double value : plane_normal) {
      plane_norm2 += value * value;
    }
    if (!(plane_norm2 > 0.0)) {
      throw std::invalid_argument("lammpc plane normal vanishes");
    }
    const double plane_norm = std::sqrt(plane_norm2);
    for (double &value : plane_normal) {
      value /= plane_norm;
    }
    std::vector<double> coordinate_masses(n3);
    for (size_t index = 0; index < n3; ++index) {
      coordinate_masses[index] = masses[index / 3];
    }
    double alpha = 0.0;
    for (size_t index = 0; index < n3; ++index) {
      alpha +=
          plane_normal[index] * plane_normal[index] / coordinate_masses[index];
    }
    walls.clear();
    walls.reserve(count);
    for (size_t wall_index = 0; wall_index < count; ++wall_index) {
      Wall wall;
      wall.point.assign(points + wall_index * n3,
                        points + (wall_index + 1) * n3);
      wall.normal.assign(normals + wall_index * n3,
                         normals + (wall_index + 1) * n3);
      double norm2 = 0.0;
      for (double value : wall.normal) {
        norm2 += value * value;
      }
      if (!(norm2 > 0.0)) {
        throw std::invalid_argument("lammpc wall normal vanishes");
      }
      const double norm = std::sqrt(norm2);
      for (double &value : wall.normal) {
        value /= norm;
      }
      double cross = 0.0;
      for (size_t index = 0; index < n3; ++index) {
        cross +=
            plane_normal[index] * wall.normal[index] / coordinate_masses[index];
      }
      const double coefficient = cross / alpha;
      std::vector<double> tangent(n3);
      double scale2 = 0.0;
      for (size_t index = 0; index < n3; ++index) {
        tangent[index] = wall.normal[index] - plane_normal[index] * coefficient;
        scale2 += tangent[index] * tangent[index] / coordinate_masses[index];
      }
      wall.scale = std::sqrt(scale2);
      wall.tangent_m_inv.assign(n3, 0.0);
      if (wall.scale > 0.0) {
        for (size_t index = 0; index < n3; ++index) {
          wall.tangent_m_inv[index] =
              tangent[index] / (wall.scale * coordinate_masses[index]);
        }
      }
      walls.push_back(std::move(wall));
    }
  }

  static int boundary(void *user, int atom_count, double *x, double *v,
                      const LammpcBoundaryBox *) {
    auto *self = static_cast<EngineSession *>(user);
    if (atom_count < 0 ||
        static_cast<size_t>(atom_count) != self->masses.size()) {
      return -1;
    }
    bool reflected = false;
    for (size_t pass = 0; pass < kReflectionPasses; ++pass) {
      bool pass_hit = false;
      for (const Wall &wall : self->walls) {
        if (!(wall.scale > 0.0)) {
          continue;
        }
        double raw = 0.0;
        for (size_t index = 0; index < self->positions.size(); ++index) {
          raw += wall.normal[index] * (x[index] - wall.point[index]);
        }
        const double excess = raw / wall.scale;
        if (excess <= 0.0) {
          continue;
        }
        double velocity_component = 0.0;
        for (size_t index = 0; index < self->positions.size(); ++index) {
          velocity_component +=
              wall.tangent_m_inv[index] * self->masses[index / 3] * v[index];
        }
        for (size_t index = 0; index < self->positions.size(); ++index) {
          x[index] -= 2.0 * excess * wall.tangent_m_inv[index];
          v[index] -= 2.0 * velocity_component * wall.tangent_m_inv[index];
        }
        ++self->wall_hits;
        reflected = true;
        pass_hit = true;
      }
      if (!pass_hit) {
        break;
      }
    }
    return reflected ? LAMMPC_BOUNDARY_REFLECT : LAMMPC_BOUNDARY_ACCEPT;
  }
};

void require_size(size_t got, size_t expected, const char *name) {
  if (got != expected) {
    throw std::invalid_argument(std::string(name) + " has the wrong length");
  }
}

int set_periodic_box(void *context, const double *box9, char *buffer,
                     size_t length) {
  return guarded(buffer, length, [&] {
    auto *session = static_cast<EngineSession *>(context);
    if (box9 == nullptr) {
      throw std::invalid_argument("lammpc periodic box is null");
    }
    session->box.assign(box9, box9 + 9);
    session->dirty = true;
  });
}

int set_positions(void *context, const double *positions, size_t n3,
                  char *buffer, size_t length) {
  return guarded(buffer, length, [&] {
    auto *session = static_cast<EngineSession *>(context);
    require_size(n3, session->positions.size(), "lammpc positions");
    session->positions.assign(positions, positions + n3);
    session->dirty = true;
  });
}

int set_velocities(void *context, const double *velocities, size_t n3,
                   char *buffer, size_t length) {
  return guarded(buffer, length, [&] {
    auto *session = static_cast<EngineSession *>(context);
    require_size(n3, session->velocities.size(), "lammpc velocities");
    session->velocities.assign(velocities, velocities + n3);
    session->updateKinetic();
  });
}

int reseed_velocities(void *context, double kT, uint64_t seed, char *buffer,
                      size_t length) {
  return guarded(buffer, length, [&] {
    auto *session = static_cast<EngineSession *>(context);
    SplitMix64 random(seed);
    for (size_t atom = 0; atom < session->masses.size(); ++atom) {
      const double sigma =
          std::sqrt(kT * kAmuA2Ps2PerEv / session->masses[atom]);
      for (size_t component = 0; component < 3; ++component) {
        session->velocities[3 * atom + component] = sigma * random.normal();
      }
    }
    session->updateKinetic();
  });
}

int step_langevin(void *context, double kT, double friction, double time_step,
                  const double *plane_point, const double *plane_normal,
                  const double *wall_points, const double *wall_normals,
                  size_t n_walls, size_t n3, size_t steps, uint64_t seed,
                  uint64_t *wall_hits, char *buffer, size_t length) {
  return guarded(buffer, length, [&] {
    auto *session = static_cast<EngineSession *>(context);
    require_size(n3, session->positions.size(), "lammpc Langevin vectors");
    const std::vector<unsigned char> input = session->forceInput();
    const LammpcMdConfig config = {time_step, kT, friction,
                                   static_cast<unsigned long long>(seed),
                                   LAMMPC_PLANE_CARTESIAN};
    if (session->md_initialize(session->session, input.data(), input.size(),
                               plane_point, plane_normal, n3, &config) != 0) {
      throw std::runtime_error(session->lastError());
    }
    if (session->md_set_velocities(session->session, session->velocities.data(),
                                   n3, seed) != 0) {
      throw std::runtime_error(session->lastError());
    }
    session->prepareWalls(plane_normal, wall_points, wall_normals, n_walls);
    session->wall_hits = 0;
    if (n_walls > 0 &&
        session->set_boundary(session->session, EngineSession::boundary,
                              session) != 0) {
      throw std::runtime_error(session->lastError());
    }
    struct BoundaryGuard {
      EngineSession *owner;
      ~BoundaryGuard() {
        if (owner->clear_boundary != nullptr) {
          (void)owner->clear_boundary(owner->session);
        }
      }
    } boundary_guard{session};
    if (session->md_step(session->session, steps) != 0) {
      throw std::runtime_error(session->lastError());
    }
    LammpcMdObservation observation{};
    std::vector<double> projected(n3);
    if (session->md_state(session->session, session->positions.data(),
                          session->velocities.data(), projected.data(), nullptr,
                          n3, &observation) != 0) {
      throw std::runtime_error(session->lastError());
    }
    session->constraint_force = observation.constraint_force;
    session->dirty = true;
    session->calculate();
    session->updateKinetic();
    if (wall_hits != nullptr) {
      *wall_hits = session->wall_hits;
    }
  });
}

int step_nve(void *context, double time_step, size_t steps, char *buffer,
             size_t length) {
  return guarded(buffer, length, [&] {
    auto *session = static_cast<EngineSession *>(context);
    if (session->dirty) {
      session->calculate();
    }
    const size_t n3 = session->positions.size();
    if (session->nve_set_positions(session->session, session->positions.data(),
                                   n3, session->box.data(),
                                   session->box.size()) != 0 ||
        session->nve_set_velocities(session->session,
                                    session->velocities.data(), n3) != 0 ||
        session->nve_step(session->session, steps, time_step) != 0) {
      throw std::runtime_error(session->lastError());
    }
    LammpcNveObservation observation{};
    if (session->nve_state(session->session, session->positions.data(),
                           session->velocities.data(), n3, &observation) != 0) {
      throw std::runtime_error(session->lastError());
    }
    session->potential = observation.potential_energy_ev;
    session->kinetic = observation.kinetic_energy_ev;
    session->constraint_force = 0.0;
    session->dirty = true;
    session->calculate();
  });
}

int get_state(void *context, double *positions, double *velocities,
              double *forces, size_t n3, double *potential, double *kinetic,
              char *buffer, size_t length) {
  return guarded(buffer, length, [&] {
    auto *session = static_cast<EngineSession *>(context);
    require_size(n3, session->positions.size(), "lammpc state");
    if (session->dirty) {
      session->calculate();
    }
    if (positions != nullptr) {
      std::copy(session->positions.begin(), session->positions.end(),
                positions);
    }
    if (velocities != nullptr) {
      std::copy(session->velocities.begin(), session->velocities.end(),
                velocities);
    }
    if (forces != nullptr) {
      std::copy(session->forces.begin(), session->forces.end(), forces);
    }
    if (potential != nullptr) {
      *potential = session->potential;
    }
    if (kinetic != nullptr) {
      *kinetic = session->kinetic;
    }
  });
}

int get_masses(void *context, double *masses, size_t atom_count, char *buffer,
               size_t length) {
  return guarded(buffer, length, [&] {
    auto *session = static_cast<EngineSession *>(context);
    require_size(atom_count, session->masses.size(), "lammpc masses");
    std::copy(session->masses.begin(), session->masses.end(), masses);
  });
}

int get_constraint_force(void *context, double *constraint_force, char *buffer,
                         size_t length) {
  return guarded(buffer, length, [&] {
    if (constraint_force == nullptr) {
      throw std::invalid_argument("lammpc constraint force output is null");
    }
    *constraint_force = static_cast<EngineSession *>(context)->constraint_force;
  });
}

int initialize(void *context, char *buffer, size_t length) {
  return guarded(buffer, length,
                 [&] { static_cast<EngineSession *>(context)->calculate(); });
}

void destroy(void *context) { delete static_cast<EngineSession *>(context); }

} // namespace

extern "C" int rgpot_md_session_engine_create(const void *spec, size_t spec_len,
                                              const char *options,
                                              rgpot_md_session_ops *ops,
                                              void **context,
                                              size_t *atom_count, char *buffer,
                                              size_t length) {
  return guarded(buffer, length, [&] {
    if (spec == nullptr || ops == nullptr || context == nullptr ||
        atom_count == nullptr ||
        spec_len < sizeof(rgpot_lammpc_session_spec_header)) {
      throw std::invalid_argument("lammpc MD session spec or output is null");
    }
    rgpot_lammpc_session_spec_header header{};
    std::memcpy(&header, spec, sizeof(header));
    if (header.magic != RGPOT_LAMMPC_SESSION_SPEC_MAGIC ||
        header.config_size > spec_len - sizeof(header) ||
        header.force_input_size >
            spec_len - sizeof(header) - header.config_size ||
        sizeof(header) + header.config_size + header.force_input_size !=
            spec_len) {
      throw std::invalid_argument("invalid lammpc MD session spec framing");
    }
    const auto *payload =
        static_cast<const unsigned char *>(spec) + sizeof(header);
    auto owner = std::make_unique<EngineSession>();
    owner->config.assign(payload, payload + header.config_size);
    const unsigned char *input_bytes = payload + header.config_size;
    auto input_storage =
        words(input_bytes, header.force_input_size, "ForceInput");
    capnp::FlatArrayMessageReader input_reader(
        kj::arrayPtr(input_storage.data(), input_storage.size()));
    const auto input = input_reader.getRoot<::ForceInput>();
    if (input.getLengthUnit() != "angstrom" || input.getEnergyUnit() != "eV") {
      throw std::invalid_argument(
          "lammpc session ForceInput must use angstrom and eV");
    }
    const auto input_positions = input.getPos();
    const auto numbers = input.getAtmnrs();
    const auto input_box = input.getBox();
    if (numbers.size() == 0 || input_positions.size() != 3 * numbers.size() ||
        input_box.size() != 9) {
      throw std::invalid_argument(
          "lammpc session ForceInput needs atoms, positions, and box");
    }
    owner->positions.resize(input_positions.size());
    for (size_t index = 0; index < owner->positions.size(); ++index) {
      owner->positions[index] = input_positions[index];
    }
    owner->velocities.assign(owner->positions.size(), 0.0);
    owner->forces.assign(owner->positions.size(), 0.0);
    owner->atomic_numbers.resize(numbers.size());
    for (size_t index = 0; index < owner->atomic_numbers.size(); ++index) {
      owner->atomic_numbers[index] = numbers[index];
    }
    owner->box.resize(input_box.size());
    for (size_t index = 0; index < owner->box.size(); ++index) {
      owner->box[index] = input_box[index];
    }
    owner->loader.load("lammpc", options != nullptr ? std::string(options)
                                                    : std::string{});
    if (owner->loader.available() != 1) {
      throw std::runtime_error("lammpc backend reports unavailable");
    }
    owner->session = owner->loader.session_create_from_config(
        owner->config.data(), owner->config.size());
    if (owner->session == nullptr) {
      throw std::runtime_error("lammpc_session_create_from_config failed");
    }
    owner->md_initialize =
        owner->required<EngineSession::MdInitializeFn>("session_md_initialize");
    owner->md_set_velocities =
        owner->required<EngineSession::MdSetVelocitiesFn>(
            "session_md_set_velocities");
    owner->md_step =
        owner->required<EngineSession::MdStepFn>("session_md_step");
    owner->md_state =
        owner->required<EngineSession::MdStateFn>("session_md_state");
    owner->session_masses =
        owner->required<EngineSession::MassesFn>("session_masses");
    owner->nve_set_positions =
        owner->required<EngineSession::NveSetPositionsFn>(
            "session_nve_set_positions");
    owner->nve_set_velocities =
        owner->required<EngineSession::NveSetVelocitiesFn>(
            "session_nve_set_velocities");
    owner->nve_step =
        owner->required<EngineSession::NveStepFn>("session_nve_step");
    owner->nve_state =
        owner->required<EngineSession::NveStateFn>("session_nve_state");
    owner->set_boundary = owner->required<EngineSession::SetBoundaryFn>(
        "session_set_hard_boundary");
    owner->clear_boundary = owner->required<EngineSession::ClearBoundaryFn>(
        "session_clear_hard_boundary");
    owner->calculate();
    owner->masses.resize(numbers.size());
    if (owner->session_masses(owner->session, owner->masses.data(),
                              owner->masses.size()) != 0) {
      throw std::runtime_error(owner->lastError());
    }
    owner->updateKinetic();

    *ops = {};
    ops->struct_size = sizeof(*ops);
    ops->set_periodic_box = set_periodic_box;
    ops->set_positions = set_positions;
    ops->set_velocities = set_velocities;
    ops->reseed_velocities = reseed_velocities;
    ops->step_langevin = step_langevin;
    ops->step_nve = step_nve;
    ops->get_state = get_state;
    ops->get_masses = get_masses;
    ops->get_constraint_force = get_constraint_force;
    ops->initialize = initialize;
    ops->destroy = destroy;
    *atom_count = owner->masses.size();
    *context = owner.release();
  });
}
