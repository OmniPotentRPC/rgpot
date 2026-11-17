// MIT License
// Copyright 2023--present rgpot developers

#include "rgpot/NblibPot/nblib_c_abi.h"

#include "rgpot/ExternalPot/md_units.hpp"
#include "NblibEngine.hpp"
#include "NblibSession.hpp"

#include "gromacs/version.h"

#include <cstdio>
#include <cstring>
#include <exception>
#include <memory>
#include <string>
#include <utility>

struct rgpot_nblib_potential {
  explicit rgpot_nblib_potential(rgpot::nblibpot::TopologyDesc description)
      : engine(std::move(description)) {}
  rgpot::nblibpot::Engine engine;
  rgpot::nblibpot::Evaluation scratch;
  std::vector<double> positions;
};

struct rgpot_md_session {
  rgpot_md_session(rgpot::nblibpot::TopologyDesc description, double dt)
      : session(std::move(description), dt) {}
  rgpot::nblibpot::Session session;
};

namespace {

namespace md = rgpot::md;

thread_local std::string g_error;

void setError(const std::string &message, char *err, std::size_t errLen) {
  g_error = message;
  if (err != nullptr && errLen > 0) {
    std::snprintf(err, errLen, "%s", message.c_str());
  }
}

template <typename Body>
int guarded(char *err, std::size_t errLen, Body &&body) {
  try {
    body();
    return 0;
  } catch (const std::exception &e) {
    setError(e.what(), err, errLen);
  } catch (...) {
    setError("unknown failure in librgpot_nblib", err, errLen);
  }
  return 1;
}

std::array<double, 3> edgesFromBox(const double *box9) {
  for (const std::size_t k : {1u, 2u, 3u, 5u, 6u, 7u}) {
    if (box9[k] != 0.0) {
      throw std::invalid_argument(
          "nblib takes rectangular cells: the off-diagonal box entries must "
          "be zero");
    }
  }
  return {box9[0] * md::kNmPerAngstrom, box9[4] * md::kNmPerAngstrom,
          box9[8] * md::kNmPerAngstrom};
}

} // namespace

extern "C" {

int rgpot_nblib_abi_version(void) { return RGPOT_NBLIB_ABI_VERSION; }
int rgpot_nblib_abi_minor(void) { return RGPOT_NBLIB_ABI_MINOR; }
int rgpot_nblib_available(void) { return 1; }

unsigned rgpot_nblib_features(void) {
  return RGPOT_NBLIB_FEATURE_POTENTIAL | RGPOT_NBLIB_FEATURE_VIRIAL |
         RGPOT_NBLIB_FEATURE_SESSION;
}

int rgpot_nblib_real_bits(void) {
  return static_cast<int>(8 * sizeof(rgpot::nblibpot::Real));
}

const char *rgpot_nblib_gromacs_version(void) {
  static const std::string version = std::to_string(GMX_VERSION / 10000) +
                                     "." + std::to_string(GMX_VERSION % 10000);
  return version.c_str();
}

const char *rgpot_nblib_last_error(void) { return g_error.c_str(); }

rgpot_nblib_potential *rgpot_nblib_potential_create(const char *topology,
                                                    char *err,
                                                    std::size_t errLen) {
  rgpot_nblib_potential *made = nullptr;
  guarded(err, errLen, [&] {
    if (topology == nullptr) {
      throw std::invalid_argument("the topology text is NULL");
    }
    made = new rgpot_nblib_potential(rgpot::nblibpot::parseTopology(topology));
  });
  return made;
}

void rgpot_nblib_potential_free(rgpot_nblib_potential *pot) { delete pot; }

std::size_t rgpot_nblib_potential_atom_count(const rgpot_nblib_potential *pot) {
  return pot == nullptr ? 0 : pot->engine.atoms();
}

double rgpot_nblib_potential_cutoff(const rgpot_nblib_potential *pot) {
  return pot->engine.description().cutoff * md::kAngstromPerNm;
}

int rgpot_nblib_potential_masses(const rgpot_nblib_potential *pot,
                                 double *masses, std::size_t n) {
  if (n != pot->engine.atoms()) {
    return 1;
  }
  for (std::size_t i = 0; i < n; ++i) {
    masses[i] = pot->engine.masses()[i];
  }
  return 0;
}

int rgpot_nblib_potential_energy_forces(rgpot_nblib_potential *pot,
                                        const double *xyz, std::size_t n,
                                        const double *box9, double *forces,
                                        double *energy, double *virial9,
                                        char *err, std::size_t errLen) {
  return guarded(err, errLen, [&] {
    if (n != pot->engine.atoms()) {
      throw std::invalid_argument("the atom count differs from the topology");
    }
    pot->positions.resize(3 * n);
    for (std::size_t k = 0; k < 3 * n; ++k) {
      pot->positions[k] = xyz[k] * md::kNmPerAngstrom;
    }
    pot->engine.evaluate(pot->positions.data(), edgesFromBox(box9),
                         virial9 != nullptr, pot->scratch);
    for (std::size_t k = 0; k < 3 * n; ++k) {
      forces[k] = pot->scratch.forces[k] * md::kEvAPerKjNm;
    }
    *energy = pot->scratch.energy * md::kEvPerKj;
    if (virial9 != nullptr) {
      for (std::size_t k = 0; k < 9; ++k) {
        virial9[k] = pot->scratch.virial[k] * md::kEvPerKj;
      }
    }
  });
}

int rgpot_nblib_md_abi_version(void) { return RGPOT_MD_ABI_VERSION; }
int rgpot_nblib_md_abi_minor(void) { return RGPOT_MD_ABI_MINOR; }
int rgpot_nblib_md_available(void) { return 1; }

unsigned rgpot_nblib_md_features(void) {
  return RGPOT_MD_FEATURE_NVE | RGPOT_MD_FEATURE_PROJECTED |
         RGPOT_MD_FEATURE_HOOK | RGPOT_MD_FEATURE_BOUNDARY |
         RGPOT_MD_FEATURE_VIRIAL;
}

const char *rgpot_nblib_md_engine(void) { return "nblib"; }
const char *rgpot_nblib_md_integrator(void) { return "leap-frog"; }
const char *rgpot_nblib_md_last_error(void) { return g_error.c_str(); }

rgpot_md_session *rgpot_nblib_md_create(const char *spec,
                                        const rgpot_md_config *config,
                                        char *err, std::size_t errLen) {
  rgpot_md_session *made = nullptr;
  guarded(err, errLen, [&] {
    if (spec == nullptr) {
      throw std::invalid_argument("the topology text is NULL");
    }
    rgpot_md_config use;
    rgpot_md_config_init(&use);
    if (config != nullptr) {
      if (config->bytes < sizeof(rgpot_md_config)) {
        throw std::invalid_argument("rgpot_md_config is too short");
      }
      use = *config;
    }
    made = new rgpot_md_session(rgpot::nblibpot::parseTopology(spec),
                                use.time_step_ps);
  });
  return made;
}

void rgpot_nblib_md_destroy(rgpot_md_session *session) { delete session; }

std::size_t rgpot_nblib_md_atom_count(const rgpot_md_session *session) {
  return session == nullptr ? 0 : session->session.atoms();
}

int rgpot_nblib_md_masses(const rgpot_md_session *session, double *masses,
                          std::size_t dimension) {
  return guarded(nullptr, 0, [&] {
    if (dimension != 3 * session->session.atoms()) {
      throw std::invalid_argument("masses need three values per atom");
    }
    for (std::size_t k = 0; k < dimension; ++k) {
      masses[k] = session->session.masses()[k / 3];
    }
  });
}

int rgpot_nblib_md_set_box(rgpot_md_session *session, const double *box9) {
  return guarded(nullptr, 0, [&] { session->session.setBox(box9); });
}

int rgpot_nblib_md_set_positions(rgpot_md_session *session,
                                 const double *positions,
                                 std::size_t dimension) {
  return guarded(nullptr, 0,
                 [&] { session->session.setPositions(positions, dimension); });
}

int rgpot_nblib_md_set_velocities(rgpot_md_session *session,
                                  const double *velocities,
                                  std::size_t dimension) {
  return guarded(nullptr, 0, [&] {
    session->session.setVelocities(velocities, dimension);
  });
}

int rgpot_nblib_md_set_time_step(rgpot_md_session *session,
                                 double timeStepPs) {
  return guarded(nullptr, 0,
                 [&] { session->session.setTimeStep(timeStepPs); });
}

int rgpot_nblib_md_set_seed(rgpot_md_session *session, std::uint64_t seed) {
  return guarded(nullptr, 0, [&] { session->session.setSeed(seed); });
}

int rgpot_nblib_md_set_hook(rgpot_md_session *session, rgpot_md_step_hook hook,
                            void *user) {
  return guarded(nullptr, 0, [&] { session->session.setHook(hook, user); });
}

int rgpot_nblib_md_set_boundary(rgpot_md_session *session,
                                rgpot_md_boundary_fn boundary, void *user) {
  return guarded(nullptr, 0,
                 [&] { session->session.setBoundary(boundary, user); });
}

int rgpot_nblib_md_initialize(rgpot_md_session *session) {
  return guarded(nullptr, 0, [&] { session->session.initialize(); });
}

int rgpot_nblib_md_step_nve(rgpot_md_session *session, std::uint64_t nsteps) {
  return guarded(nullptr, 0, [&] { session->session.step(nsteps); });
}

int rgpot_nblib_md_step_projected(rgpot_md_session *session,
                                  std::uint64_t nsteps,
                                  const rgpot_md_thermostat *thermostat,
                                  const rgpot_md_plane *plane,
                                  const rgpot_md_wall *walls,
                                  std::size_t nwalls) {
  return guarded(nullptr, 0, [&] {
    session->session.stepProjected(nsteps, thermostat, plane, walls, nwalls);
  });
}

int rgpot_nblib_md_state(rgpot_md_session *session, double *positions,
                         double *velocities, double *forces,
                         std::size_t dimension, rgpot_md_info *info) {
  return guarded(nullptr, 0, [&] {
    if (info != nullptr && info->bytes < sizeof(rgpot_md_info)) {
      throw std::invalid_argument("rgpot_md_info is too short");
    }
    rgpot::nblibpot::SessionInfo state;
    session->session.state(positions, velocities, forces, dimension, &state);
    if (info != nullptr) {
      info->potential_ev = state.potential;
      info->kinetic_ev = state.kinetic;
      info->time_ps = state.time;
      info->steps = state.steps;
      info->force_evaluations = state.evaluations;
      for (std::size_t k = 0; k < 9; ++k) {
        info->virial_ev[k] = state.virial[k];
        info->box[k] = state.box[k];
      }
    }
  });
}

} // extern "C"
