// MIT License
// Copyright 2026 Rohit Goswami

// The generic MD session ABI around the OpenMM engine: the periodic box in
// Angstrom, the constraint force of run_langevin in eV/Angstrom, optional
// callbacks of a short vtable, and the errors of invalid requests.

#include "rgpot/ExternalPot/external_potentials.h"
#include "rgpot/ExternalPot/md_units.hpp"

#include <openmm/CustomExternalForce.h>
#include <openmm/CustomNonbondedForce.h>
#include <openmm/System.h>
#include <openmm/serialization/XmlSerializer.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, const std::string &message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

bool near(double a, double b, double tolerance) {
  return std::abs(a - b) <= tolerance * std::max(1.0, std::abs(b));
}

std::string xml_of(OpenMM::System &system) {
  std::ostringstream xml;
  OpenMM::XmlSerializer::serialize<OpenMM::System>(&system, "System", xml);
  return xml.str();
}

struct Owned {
  rgpot_md_session *session;
  ~Owned() { rgpot_md_session_destroy(session); }
};

Owned open(const char *library, const std::string &xml) {
  char error[512] = {};
  rgpot_md_session *session = rgpot_md_session_create(
      "openmm", library, xml.data(), xml.size(), "Reference", error,
      sizeof(error));
  require(session != nullptr, std::string("create: ") + error);
  return {session};
}

// A pair of particles that interact only through the periodic image.
void check_box_in_angstrom(const char *library) {
  OpenMM::System system;
  system.addParticle(10.0);
  system.addParticle(10.0);
  system.setDefaultPeriodicBoxVectors(OpenMM::Vec3(30.0, 0, 0),
                                      OpenMM::Vec3(0, 30.0, 0),
                                      OpenMM::Vec3(0, 0, 30.0));
  auto *pair = new OpenMM::CustomNonbondedForce("100*(1-r/rc)^2");
  pair->addGlobalParameter("rc", 0.8);
  pair->addParticle({});
  pair->addParticle({});
  pair->setNonbondedMethod(OpenMM::CustomNonbondedForce::CutoffPeriodic);
  pair->setCutoffDistance(0.8);
  system.addForce(pair);
  Owned owned = open(library, xml_of(system));

  char error[512] = {};
  const double box[9] = {30.0, 0, 0, 0, 30.0, 0, 0, 0, 30.0};
  require(rgpot_md_session_set_periodic_box(owned.session, box, error,
                                            sizeof(error)) == 0,
          std::string("set_periodic_box: ") + error);
  double back[9] = {};
  require(rgpot_md_session_get_periodic_box(owned.session, back, error,
                                            sizeof(error)) == 0,
          std::string("get_periodic_box: ") + error);
  for (int k = 0; k < 9; ++k) {
    require(near(back[k], box[k], 1.0e-12), "box round trip in Angstrom");
  }

  // 1.0 and 29.0 Angstrom are 2.0 Angstrom apart through the boundary.
  const double positions[6] = {1.0, 5.0, 5.0, 29.0, 5.0, 5.0};
  require(rgpot_md_session_set_positions(owned.session, positions, 6, error,
                                         sizeof(error)) == 0,
          error);
  require(rgpot_md_session_initialize(owned.session, error, sizeof(error)) == 0,
          error);
  double potential = 0.0;
  double forces[6] = {};
  require(rgpot_md_session_get_forces(owned.session, forces, 6, &potential,
                                      error, sizeof(error)) == 0,
          error);
  const double expected_kj = 100.0 * std::pow(1.0 - 0.2 / 0.8, 2);
  require(near(potential, expected_kj * rgpot::md::kEvPerKj, 1.0e-9),
          "the image pair energy is missing: the box is not in Angstrom");
  require(forces[0] != 0.0 && near(forces[0], -forces[3], 1.0e-9),
          "the image pair has no force");
  double kinetic = 0.0;
  require(rgpot_md_session_get_energy(owned.session, nullptr, &kinetic, error,
                                      sizeof(error)) == 0,
          error);
  require(std::isfinite(kinetic) && kinetic >= 0.0, "get_energy kinetic");
}

// A free particle in a harmonic well: lambda = -F_n for a unit axis normal.
void check_constraint_force(const char *library) {
  OpenMM::System system;
  system.addParticle(12.0);
  auto *well = new OpenMM::CustomExternalForce("0.5*k*(x*x+y*y+z*z)");
  well->addGlobalParameter("k", 500.0);
  well->addParticle(0);
  system.addForce(well);
  Owned owned = open(library, xml_of(system));
  char error[512] = {};

  const double start[3] = {0.0, 0.7, 0.0};
  const double rest[3] = {0.0, 0.0, 0.0};
  require(rgpot_md_session_set_positions(owned.session, start, 3, error,
                                         sizeof(error)) == 0,
          error);
  require(rgpot_md_session_set_velocities(owned.session, rest, 3, error,
                                          sizeof(error)) == 0,
          error);
  const double normal[3] = {0.0, 1.0, 0.0};
  std::vector<double> lambda(4, -1.0);
  rgpot_md_langevin_request request{};
  request.struct_size = sizeof(request);
  request.kT = 0.02585;
  request.friction = 0.0;
  request.time_step = 1.0e-5;
  request.steps = 3;
  request.seed = 1;
  request.n3 = 3;
  request.plane_point = start;
  request.plane_normal = normal;
  rgpot_md_langevin_result result{};
  result.struct_size = sizeof(result);
  result.constraint_force = lambda.data();
  result.constraint_force_len = lambda.size();
  require(rgpot_md_session_run_langevin(owned.session, &request, &result, error,
                                        sizeof(error)) == 0,
          error);
  // F_y = -k y with k = 500 kJ/mol/nm^2 and y = 0.07 nm, and lambda = -F_y.
  const double expected = 500.0 * 0.07 * rgpot::md::kEvAPerKjNm;
  for (int step = 0; step < 3; ++step) {
    require(near(lambda[static_cast<std::size_t>(step)], expected, 1.0e-9),
            "constraint force per step");
  }
  require(lambda[3] == -1.0, "constraint force wrote past the steps");

  // A buffer shorter than the steps is refused, and so is a short struct.
  result.constraint_force_len = 2;
  require(rgpot_md_session_run_langevin(owned.session, &request, &result, error,
                                        sizeof(error)) != 0,
          "a short constraint buffer was accepted");
  result.constraint_force_len = lambda.size();
  request.struct_size = sizeof(request) - 1;
  require(rgpot_md_session_run_langevin(owned.session, &request, &result, error,
                                        sizeof(error)) != 0,
          "a short request was accepted");
  request.struct_size = sizeof(request);
  request.n3 = 6;
  require(rgpot_md_session_run_langevin(owned.session, &request, &result, error,
                                        sizeof(error)) != 0,
          "a mismatched dimension was accepted");
  request.n3 = 3;

  // The same engine without a plane reports no constraint force.
  request.plane_point = nullptr;
  request.plane_normal = nullptr;
  result.constraint_force = nullptr;
  result.constraint_force_len = 0;
  require(rgpot_md_session_run_langevin(owned.session, &request, &result, error,
                                        sizeof(error)) == 0,
          error);

  // Reseeding wants a normal of the right length.
  require(rgpot_md_session_reseed_velocities(owned.session, 0.02585, 3, normal,
                                             2, error, sizeof(error)) != 0,
          "a short normal was accepted by reseed_velocities");
}

// A vtable that stops at destroy has no optional callback: the session says
// so and does not read past the end.
int unsupported_set(void *, const double *, size_t, char *, size_t) { return 0; }
int unsupported_step(void *, double, double, double, const double *,
                     const double *, const double *, const double *, size_t,
                     size_t, size_t, uint64_t, uint64_t *, char *, size_t) {
  return 0;
}
int unsupported_nve(void *, double, size_t, char *, size_t) { return 0; }
size_t g_state_n3 = 0;
int unsupported_state(void *, double *, double *, double *, size_t n3, double *,
                      double *kinetic, char *err, size_t err_len) {
  g_state_n3 = n3;
  if (kinetic != nullptr) {
    *kinetic = 1.5;
  }
  if (n3 != 3 && err != nullptr && err_len > 0) {
    std::snprintf(err, err_len, "state buffers need three values per atom");
    return 1;
  }
  return 0;
}
int unsupported_masses(void *, double *masses, size_t n, char *, size_t) {
  for (size_t i = 0; i < n; ++i) {
    masses[i] = 1.0;
  }
  return 0;
}
void unsupported_destroy(void *) {}

void check_short_vtable() {
  rgpot_md_session_ops ops;
  std::memset(&ops, 0xff, sizeof(ops));
  ops.struct_size = offsetof(rgpot_md_session_ops, destroy) + sizeof(ops.destroy);
  ops.set_positions = unsupported_set;
  ops.set_velocities = unsupported_set;
  ops.step_langevin = unsupported_step;
  ops.step_nve = unsupported_nve;
  ops.get_state = unsupported_state;
  ops.get_masses = unsupported_masses;
  ops.destroy = unsupported_destroy;
  int token = 0;
  char error[512] = {};
  rgpot_md_session *session = rgpot_md_session_from_ops(&token, &ops, 1, error,
                                                       sizeof(error));
  require(session != nullptr, std::string("from_ops: ") + error);
  const double box[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  require(rgpot_md_session_set_periodic_box(session, box, error,
                                            sizeof(error)) != 0 &&
              std::strstr(error, "set_periodic_box") != nullptr,
          "set_periodic_box should be reported as unsupported");
  rgpot_md_langevin_request request{};
  request.struct_size = sizeof(request);
  request.kT = 1.0;
  request.time_step = 0.1;
  request.n3 = 3;
  rgpot_md_langevin_result result{};
  result.struct_size = sizeof(result);
  require(rgpot_md_session_run_langevin(session, &request, &result, error,
                                        sizeof(error)) != 0 &&
              std::strstr(error, "run_langevin") != nullptr,
          "run_langevin should be reported as unsupported");
  require(rgpot_md_session_initialize(session, error, sizeof(error)) == 0,
          "initialize needs no callback");
  // An engine that checks the length of its buffers is told 3N even when
  // only the energies are asked for.
  double kinetic = 0.0;
  require(rgpot_md_session_get_energy(session, nullptr, &kinetic, error,
                                      sizeof(error)) == 0,
          std::string("get_energy: ") + error);
  require(g_state_n3 == 3 && kinetic == 1.5,
          "get_energy did not pass 3N to get_state");
  rgpot_md_session_destroy(session);

  ops.struct_size = offsetof(rgpot_md_session_ops, destroy);
  require(rgpot_md_session_from_ops(&token, &ops, 1, error, sizeof(error)) ==
              nullptr,
          "a vtable without destroy was accepted");
}

} // namespace

int main(int argc, char **argv) {
  try {
    require(argc == 2, "usage: test_openmm_abi <openmm-engine>");
    check_box_in_angstrom(argv[1]);
    check_constraint_force(argv[1]);
    check_short_vtable();
    std::cout << "OpenMM session ABI checks passed\n";
    return EXIT_SUCCESS;
  } catch (const std::exception &failure) {
    std::cerr << failure.what() << '\n';
    return EXIT_FAILURE;
  }
}
