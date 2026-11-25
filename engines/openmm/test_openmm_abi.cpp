// MIT License
// Copyright 2023--present rgpot developers

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
#include <dlfcn.h>
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
  rgpot_nlist_request nlist{};
  size_t count = 7;
  size_t needed = 7;
  require(rgpot_md_session_get_neighbors(session, &nlist, nullptr, &count,
                                         &needed, error, sizeof(error)) ==
                  RGPOT_MD_NLIST_UNSUPPORTED &&
              std::strstr(error, "neighbor queries") != nullptr &&
              count == 0 && needed == 0,
          "get_neighbors should be reported as unsupported");
  rgpot_nlist_soa pairs{};
  pairs.bytes = sizeof(pairs);
  require(rgpot_md_session_set_neighbors(session, &nlist, &pairs, 0, error,
                                         sizeof(error)) ==
                  RGPOT_MD_NLIST_UNSUPPORTED &&
              std::strstr(error, "supplied neighbors") != nullptr,
          "set_neighbors should be reported as unsupported");
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

void check_ops_capacity(const char *library) {
  OpenMM::System system;
  system.addParticle(12.0);
  const std::string xml = xml_of(system);

  void *handle = dlopen(library, RTLD_NOW | RTLD_LOCAL);
  require(handle != nullptr, "dlopen failed in check_ops_capacity");
  struct DlCloser {
    void *h;
    ~DlCloser() {
      if (h != nullptr) {
        dlclose(h);
      }
    }
  } closer{handle};

  auto create_fn = reinterpret_cast<rgpot_md_session_engine_create_fn>(
      dlsym(handle, "rgpot_md_session_engine_create"));
  require(create_fn != nullptr, "missing rgpot_md_session_engine_create");

  char error[512] = {};

  // Capacity smaller than required callbacks (including 0) is refused without
  // writing past struct_size or leaking a context.
  for (const size_t tiny_cap :
       {size_t{0}, RGPOT_MD_OPS_END(destroy) - 1}) {
    std::vector<unsigned char> buf(sizeof(rgpot_md_session_ops) + 64, 0xA5);
    auto *ops = reinterpret_cast<rgpot_md_session_ops *>(buf.data());
    std::memcpy(&ops->struct_size, &tiny_cap, sizeof(tiny_cap));
    void *ctx = nullptr;
    size_t n_atoms = 0;
    require(create_fn(xml.data(), xml.size(), "Reference", ops, &ctx, &n_atoms,
                      error, sizeof(error)) != 0,
            "too-small capacity was accepted");
    require(ctx == nullptr, "refused tiny capacity returned non-null ctx");
    for (size_t i = sizeof(ops->struct_size); i < buf.size(); ++i) {
      require(buf[i] == 0xA5, "refused call wrote into ops buffer");
    }
  }

  // Older/smaller host capacity ending at destroy: guard bytes stay intact.
  {
    const size_t small_cap = RGPOT_MD_OPS_END(destroy);
    std::vector<unsigned char> buf(sizeof(rgpot_md_session_ops) + 64, 0xA5);
    std::memset(buf.data(), 0, small_cap);
    auto *ops = reinterpret_cast<rgpot_md_session_ops *>(buf.data());
    ops->struct_size = small_cap;
    void *ctx = nullptr;
    size_t n_atoms = 0;
    require(create_fn(xml.data(), xml.size(), "Reference", ops, &ctx, &n_atoms,
                      error, sizeof(error)) == 0,
            std::string("small capacity failed: ") + error);
    require(ctx != nullptr && n_atoms == 1, "small capacity bad ctx/n_atoms");
    require(ops->struct_size == small_cap,
            "engine did not report filled size == small_cap");
    for (size_t i = small_cap; i < buf.size(); ++i) {
      require(buf[i] == 0xA5, "engine overran truncated ops struct");
    }
    rgpot_md_session *session =
        rgpot_md_session_from_ops(ctx, ops, n_atoms, error, sizeof(error));
    require(session != nullptr, std::string("from_ops small_cap: ") + error);
    Owned owned{session};
    double mass = 0.0;
    require(rgpot_md_session_get_masses(owned.session, &mass, 1, error,
                                        sizeof(error)) == 0 &&
                near(mass, 12.0, 1.0e-12),
            "small_cap get_masses");
    const double box[9] = {10, 0, 0, 0, 10, 0, 0, 0, 10};
    require(rgpot_md_session_set_periodic_box(owned.session, box, error,
                                              sizeof(error)) != 0,
            "small_cap unexpectedly supported set_periodic_box");
  }

  // Unaligned mid-optional capacity: partial optional slot stays zero and
  // guard bytes at and after capacity stay intact.
  {
    const size_t mid_cap = RGPOT_MD_OPS_END(set_periodic_box) + 1;
    std::vector<unsigned char> buf(sizeof(rgpot_md_session_ops) + 64, 0xA5);
    std::memset(buf.data(), 0, mid_cap);
    auto *ops = reinterpret_cast<rgpot_md_session_ops *>(buf.data());
    ops->struct_size = mid_cap;
    void *ctx = nullptr;
    size_t n_atoms = 0;
    require(create_fn(xml.data(), xml.size(), "Reference", ops, &ctx, &n_atoms,
                      error, sizeof(error)) == 0,
            std::string("mid capacity failed: ") + error);
    require(ops->struct_size == mid_cap,
            "engine did not report filled size == mid_cap");
    for (size_t i = RGPOT_MD_OPS_END(set_periodic_box); i < mid_cap; ++i) {
      require(buf[i] == 0, "partial optional slot byte was non-zero");
    }
    for (size_t i = mid_cap; i < buf.size(); ++i) {
      require(buf[i] == 0xA5, "engine overran mid-capacity ops struct");
    }
    rgpot_md_session *session =
        rgpot_md_session_from_ops(ctx, ops, n_atoms, error, sizeof(error));
    require(session != nullptr, std::string("from_ops mid_cap: ") + error);
    Owned owned{session};
    const double box[9] = {10, 0, 0, 0, 10, 0, 0, 0, 10};
    require(rgpot_md_session_set_periodic_box(owned.session, box, error,
                                              sizeof(error)) == 0,
            "mid_cap set_periodic_box failed");
    double back[9] = {};
    require(rgpot_md_session_get_periodic_box(owned.session, back, error,
                                              sizeof(error)) != 0,
            "mid_cap unexpectedly supported get_periodic_box");
  }

  // Larger host capacity: engine fills only sizeof(rgpot_md_session_ops),
  // reports that filled size, and leaves trailing host bytes untouched.
  {
    struct LargerOps {
      rgpot_md_session_ops base;
      void (*future_cb)(void *);
      unsigned char tail_guard[32];
    };
    LargerOps larger{};
    std::memset(&larger, 0xA5, sizeof(larger));
    std::memset(&larger.base, 0, sizeof(larger.base));
    larger.base.struct_size = sizeof(larger);
    void *ctx = nullptr;
    size_t n_atoms = 0;
    require(create_fn(xml.data(), xml.size(), "Reference", &larger.base, &ctx,
                      &n_atoms, error, sizeof(error)) == 0,
            std::string("larger capacity failed: ") + error);
    require(larger.base.struct_size == sizeof(rgpot_md_session_ops),
            "engine did not report min(capacity, sizeof(ops))");
    require(RGPOT_MD_OPS_HAS(&larger.base, run_langevin),
            "larger host missing run_langevin");
    const auto *tail = reinterpret_cast<const unsigned char *>(&larger) +
                       sizeof(rgpot_md_session_ops);
    for (size_t i = 0; i < sizeof(larger) - sizeof(rgpot_md_session_ops); ++i) {
      require(tail[i] == 0xA5, "engine wrote past sizeof(rgpot_md_session_ops)");
    }
    rgpot_md_session *session = rgpot_md_session_from_ops(
        ctx, &larger.base, n_atoms, error, sizeof(error));
    require(session != nullptr, std::string("from_ops larger: ") + error);
    Owned owned{session};
    double back[9] = {};
    require(rgpot_md_session_get_periodic_box(owned.session, back, error,
                                              sizeof(error)) == 0,
            "larger host get_periodic_box failed");
  }
}

} // namespace

int main(int argc, char **argv) {
  try {
    require(argc == 2, "usage: test_openmm_abi <openmm-engine>");
    check_box_in_angstrom(argv[1]);
    check_constraint_force(argv[1]);
    check_short_vtable();
    check_ops_capacity(argv[1]);
    std::cout << "OpenMM session ABI checks passed\n";
    return EXIT_SUCCESS;
  } catch (const std::exception &failure) {
    std::cerr << failure.what() << '\n';
    return EXIT_FAILURE;
  }
}
