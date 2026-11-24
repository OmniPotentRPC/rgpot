// MIT License
// Copyright 2023--present rgpot developers

#include "rgpot/ExternalPot/OpenMM.hpp"
#include "rgpot/ExternalPot/external_potentials.h"
#include "rgpot/nlist/rgpot_nlist.h"

#include <openmm/HarmonicBondForce.h>
#include <openmm/System.h>
#include <openmm/serialization/XmlSerializer.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <map>
#include <limits>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace {

constexpr double kAcceptanceSigmas = 4.0;
constexpr std::size_t kTemperatureBatches = 16;

std::uint64_t test_seed(std::uint64_t fallback) {
  const char *value = std::getenv("RGPOT_TEST_SEED");
  if (value == nullptr) {
    return fallback;
  }
  char *end = nullptr;
  const auto parsed = std::strtoull(value, &end, 10);
  if (end == value || *end != '\0') {
    throw std::runtime_error("RGPOT_TEST_SEED is not an integer");
  }
  return parsed;
}

std::unique_ptr<OpenMM::System> make_harmonic_system(double m0, double m1) {
  auto out = std::make_unique<OpenMM::System>();
  out->addParticle(m0);
  out->addParticle(m1);
  auto *bond = new OpenMM::HarmonicBondForce();
  bond->addBond(0, 1, 0.35, 1000.0);
  out->addForce(bond);
  return out;
}

std::unique_ptr<OpenMM::System> make_free_system(double m0, double m1) {
  auto out = std::make_unique<OpenMM::System>();
  out->addParticle(m0);
  out->addParticle(m1);
  return out;
}

void near(double got, double expected, double scale, const char *what) {
  const double tolerance = 4096.0 *
                           std::sqrt(std::numeric_limits<double>::epsilon()) *
                           std::max(1.0, scale);
  if (std::abs(got - expected) > tolerance) {
    throw std::runtime_error(std::string(what) +
                             " differs: " + std::to_string(got) + " vs " +
                             std::to_string(expected));
  }
}

void test_cpp_session() {
  rgpot::OpenMMSessionConfig config;
  config.platform = "Reference";
  config.time_step = 0.0005;
  config.friction = 1.0;
  config.temperature = 0.02585 * rgpot::md::kKjPerEv;
  config.random_seed = static_cast<int>(test_seed(19));

  rgpot::OpenMMSession session(make_harmonic_system(39.9, 39.9), config);
  const std::vector<OpenMM::Vec3> positions{OpenMM::Vec3(0.0, 0.0, 0.0),
                                            OpenMM::Vec3(0.4, 0.0, 0.0)};
  session.setPositions(positions);
  session.setVelocities(
      {OpenMM::Vec3(0.0, 0.0, 0.0), OpenMM::Vec3(0.0, 0.0, 0.0)});
  const auto initial = session.state();
  near(initial.potential_energy, 1.25, 1.25, "harmonic energy");
  near(initial.forces[1][0], -50.0, 50.0, "harmonic force");

  // 3N hyperplane across both particles with unequal masses (4.0 and 39.9 amu).
  rgpot::OpenMMSession sample_session(make_free_system(4.0, 39.9), config);
  const std::vector<OpenMM::Vec3> plane_point{OpenMM::Vec3(0.1, 0.0, 0.0),
                                              OpenMM::Vec3(0.1, 0.0, 0.0)};
  const std::vector<OpenMM::Vec3> plane_normal{OpenMM::Vec3(1.0, 0.0, 0.0),
                                               OpenMM::Vec3(-1.0, 0.0, 0.0)};
  const std::vector<std::vector<OpenMM::Vec3>> wall_points{
      {OpenMM::Vec3(0.0, 0.0, 0.0), OpenMM::Vec3(0.0, 0.0, 0.0)}};
  const std::vector<std::vector<OpenMM::Vec3>> wall_normals{
      {OpenMM::Vec3(0.0, 0.0, -1.0), OpenMM::Vec3(0.0, 0.0, 0.0)}};

  auto wall_start = plane_point;
  wall_start[0][2] = 0.001;
  sample_session.setPositions(wall_start);
  sample_session.setVelocities(
      {OpenMM::Vec3(0.0, 0.0, -10.0), OpenMM::Vec3(0.0, 0.0, 0.0)});
  const std::uint64_t wall_hits = sample_session.stepLangevin(
      config.temperature, 0.0, config.time_step, plane_point, plane_normal,
      wall_points, wall_normals, 1, 1);
  if (wall_hits != 1) {
    throw std::runtime_error("OpenMM hard wall did not reflect");
  }

  sample_session.setPositions(plane_point);
  sample_session.setVelocities(
      {OpenMM::Vec3(0.0, 0.0, 0.0), OpenMM::Vec3(0.0, 0.0, 0.0)});
  sample_session.stepLangevin(config.temperature, config.friction,
                              config.time_step, plane_point, plane_normal,
                              wall_points, wall_normals, 2000, test_seed(19));

  const double tolerance =
      4096.0 * std::sqrt(std::numeric_limits<double>::epsilon());
  std::vector<double> temperatures;
  for (int sample = 0; sample < 2048; ++sample) {
    sample_session.stepLangevin(config.temperature, config.friction,
                                config.time_step, plane_point, plane_normal,
                                wall_points, wall_normals, 20,
                                test_seed(20) + static_cast<uint64_t>(sample));
    const auto sampled = sample_session.state();
    // 5 free degrees of freedom: 2*3 - 1 hyperplane constraint.
    temperatures.push_back(2.0 * sampled.kinetic_energy / 5.0);
    if (sampled.positions[0][2] < -tolerance) {
      throw std::runtime_error("OpenMM wall was crossed");
    }
    const double plane_dist =
        (sampled.positions[0][0] - sampled.positions[1][0]) / std::sqrt(2.0);
    if (std::abs(plane_dist) > tolerance) {
      throw std::runtime_error("OpenMM 3N hyperplane was crossed");
    }
    const double normal_vel =
        (sampled.velocities[0][0] - sampled.velocities[1][0]) / std::sqrt(2.0);
    if (std::abs(normal_vel) > tolerance) {
      throw std::runtime_error(
          "OpenMM 3N hyperplane velocity was not projected");
    }
  }
  // Sixteen batches of 128 observations span 1.28 ps each. That is longer
  // than the kinetic-energy autocorrelation time set by the 1/ps thermostat,
  // unlike a standard error that treats all 2048 observations as independent.
  std::vector<double> batch_temperatures;
  batch_temperatures.reserve(kTemperatureBatches);
  const std::size_t batch_size = temperatures.size() / kTemperatureBatches;
  for (std::size_t batch = 0; batch < kTemperatureBatches; ++batch) {
    batch_temperatures.push_back(
        std::accumulate(temperatures.begin() + batch * batch_size,
                        temperatures.begin() + (batch + 1) * batch_size, 0.0) /
        static_cast<double>(batch_size));
  }
  const double mean_temp = std::accumulate(batch_temperatures.begin(),
                                           batch_temperatures.end(), 0.0) /
                           static_cast<double>(batch_temperatures.size());
  const double batch_variance =
      std::accumulate(batch_temperatures.begin(), batch_temperatures.end(), 0.0,
                      [mean_temp](double sum, double value) {
                        const double delta = value - mean_temp;
                        return sum + delta * delta;
                      }) /
      static_cast<double>(batch_temperatures.size() - 1);
  const double stderr_temp =
      std::sqrt(batch_variance / batch_temperatures.size());
  if (std::abs(mean_temp - config.temperature) >
      kAcceptanceSigmas * stderr_temp + tolerance) {
    throw std::runtime_error(
        "OpenMM kinetic temperature is outside error: mean=" +
        std::to_string(mean_temp) +
        " target=" + std::to_string(config.temperature) +
        " stderr=" + std::to_string(stderr_temp));
  }

  // Determinism check for identical u64 seeds (including 0) and divergence for
  // distinct seeds.
  rgpot::OpenMMSession det_a(make_free_system(4.0, 39.9), config);
  rgpot::OpenMMSession det_b(make_free_system(4.0, 39.9), config);
  det_a.setPositions(plane_point);
  det_b.setPositions(plane_point);
  det_a.setVelocities(
      {OpenMM::Vec3(0.0, 0.0, 0.0), OpenMM::Vec3(0.0, 0.0, 0.0)});
  det_b.setVelocities(
      {OpenMM::Vec3(0.0, 0.0, 0.0), OpenMM::Vec3(0.0, 0.0, 0.0)});
  det_a.stepLangevin(config.temperature, config.friction, config.time_step,
                     plane_point, plane_normal, {}, {}, 50, 0);
  det_b.stepLangevin(config.temperature, config.friction, config.time_step,
                     plane_point, plane_normal, {}, {}, 50, 0);
  const auto st_a = det_a.state();
  const auto st_b = det_b.state();
  for (size_t i = 0; i < 2; ++i) {
    for (int k = 0; k < 3; ++k) {
      near(st_a.positions[i][k], st_b.positions[i][k], 1.0,
           "deterministic pos");
      near(st_a.velocities[i][k], st_b.velocities[i][k], 1.0,
           "deterministic vel");
    }
  }
  det_b.setPositions(plane_point);
  det_b.setVelocities(
      {OpenMM::Vec3(0.0, 0.0, 0.0), OpenMM::Vec3(0.0, 0.0, 0.0)});
  det_b.stepLangevin(config.temperature, config.friction, config.time_step,
                     plane_point, plane_normal, {}, {}, 50, 77);
  const auto st_c = det_b.state();
  if (std::abs(st_a.positions[0][1] - st_c.positions[0][1]) <= tolerance) {
    throw std::runtime_error(
        "Different Langevin seeds produced identical motion");
  }

  // Verify stepNVE is completely unconstrained even after stepLangevin.
  session.stepLangevin(config.temperature, config.friction, config.time_step,
                       positions, plane_normal, {}, {}, 5, 1);
  session.setPositions(positions);
  session.setVelocities(
      {OpenMM::Vec3(0.02, 0.0, 0.0), OpenMM::Vec3(-0.02, 0.0, 0.0)});
  const auto nve_before = session.state();
  session.stepNVE(config.time_step, 200);
  const auto nve_after = session.state();
  const double total_before =
      nve_before.potential_energy + nve_before.kinetic_energy;
  const double total_after =
      nve_after.potential_energy + nve_after.kinetic_energy;
  const double nve_tol =
      std::max(1.0, std::abs(total_before)) * config.time_step * 10.0;
  if (std::abs(total_after - total_before) > nve_tol) {
    throw std::runtime_error("OpenMM NVE energy drift exceeded bound");
  }
  const double moved_x0 = std::abs(nve_after.positions[0][0] - positions[0][0]);
  if (moved_x0 <= tolerance) {
    throw std::runtime_error("stepNVE remained constrained after stepLangevin");
  }
}

void test_c_abi_session() {
  auto sys = make_harmonic_system(4.0, 39.9);
  std::ostringstream xml_stream;
  OpenMM::XmlSerializer::serialize<OpenMM::System>(sys.get(), "System",
                                                   xml_stream);
  const std::string xml = xml_stream.str();

  char err[512] = {};
  if (rgpot_md_session_create("nonexistent_engine", nullptr, xml.data(),
                              xml.size(), "Reference", err,
                              sizeof(err)) != nullptr ||
      err[0] == '\0') {
    throw std::runtime_error("expected failure for nonexistent engine");
  }

  rgpot_md_session *session = rgpot_md_session_create(
      "openmm", nullptr, xml.data(), xml.size(), "Reference", err, sizeof(err));
  if (session == nullptr) {
    throw std::runtime_error(std::string("rgpot_md_session_create failed: ") +
                             err);
  }
  std::unique_ptr<rgpot_md_session, decltype(&rgpot_md_session_destroy)> guard(
      session, rgpot_md_session_destroy);

  if (rgpot_md_session_n_atoms(session) != 2) {
    throw std::runtime_error("expected 2 atoms in MD session");
  }
  double masses[2] = {};
  if (rgpot_md_session_get_masses(session, masses, 2, err, sizeof(err)) != 0) {
    throw std::runtime_error(err);
  }
  near(masses[0], 4.0, 4.0, "mass[0]");
  near(masses[1], 39.9, 39.9, "mass[1]");

  // Positions in Angstrom: 4.0 A = 0.4 nm (delta = 0.05 nm -> PE = 1.25
  // kJ/mol).
  const double pos_ang[6] = {0.0, 0.0, 0.0, 4.0, 0.0, 0.0};
  const double vel_ang[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  if (rgpot_md_session_set_positions(session, pos_ang, 6, err, sizeof(err)) !=
          0 ||
      rgpot_md_session_set_velocities(session, vel_ang, 6, err, sizeof(err)) !=
          0) {
    throw std::runtime_error(err);
  }

  double out_pos[6] = {};
  double out_forces[6] = {};
  double pe_ev = 0.0;
  double ke_ev = 0.0;
  if (rgpot_md_session_get_state(session, out_pos, nullptr, out_forces, 6,
                                 &pe_ev, &ke_ev, err, sizeof(err)) != 0) {
    throw std::runtime_error(err);
  }
  if (rgpot_md_session_get_state(session, nullptr, nullptr, nullptr, 5, &pe_ev,
                                 &ke_ev, err, sizeof(err)) == 0 ||
      err[0] == '\0') {
    throw std::runtime_error(
        "get_state accepted the wrong n3 with null coordinate buffers");
  }
  const double kj_per_ev = rgpot_md_kj_per_ev();
  const double nm_per_a = rgpot_md_nm_per_angstrom();
  near(out_pos[3], 4.0, 4.0, "C ABI pos[3] in Angstrom");
  near(pe_ev * kj_per_ev, 1.25, 1.25, "C ABI PE in eV");
  near(out_forces[3] * kj_per_ev / nm_per_a, -50.0, 50.0,
       "C ABI force[3] in eV/Angstrom");
  near(ke_ev, 0.0, 1.0, "C ABI KE");

  // Error handling checks at the C boundary.
  const double bad_pos[6] = {
      0.0, 0.0, 0.0, std::numeric_limits<double>::quiet_NaN(), 0.0, 0.0};
  if (rgpot_md_session_set_positions(session, bad_pos, 6, err, sizeof(err)) ==
          0 ||
      err[0] == '\0') {
    throw std::runtime_error("expected NaN position rejection");
  }
  if (rgpot_md_session_set_positions(session, pos_ang, 3, err, sizeof(err)) ==
          0 ||
      err[0] == '\0') {
    throw std::runtime_error("expected wrong length rejection");
  }
  const double plane_p[6] = {0.0, 0.0, 0.0, 4.0, 0.0, 0.0};
  const double plane_n[6] = {1.0, 0.0, 0.0, -1.0, 0.0, 0.0};
  const double wall_p[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  const double wall_n[6] = {0.0, 0.0, -1.0, 0.0, 0.0, 0.0};
  uint64_t hits = 0;
  if (rgpot_md_session_step_langevin(session, 0.02585, 1.0, 0.0005, plane_p,
                                     plane_n, wall_p, wall_n, 1, 6, 50, 42,
                                     &hits, err, sizeof(err)) != 0) {
    throw std::runtime_error(err);
  }
  if (rgpot_md_session_step_nve(session, 0.0005, 10, err, sizeof(err)) != 0) {
    throw std::runtime_error(err);
  }

}

rgpot_md_session *open_neighbor_session(std::unique_ptr<OpenMM::System> system,
                                        char *err, size_t err_len) {
  std::ostringstream xml_stream;
  OpenMM::XmlSerializer::serialize<OpenMM::System>(system.get(), "System",
                                                   xml_stream);
  const std::string xml = xml_stream.str();
  rgpot_md_session *session = rgpot_md_session_create(
      "openmm", nullptr, xml.data(), xml.size(), "Reference", err, err_len);
  if (session == nullptr) {
    throw std::runtime_error(std::string("rgpot_md_session_create failed: ") +
                             err);
  }
  return session;
}

constexpr double kNeighbourNoise = 1.0e-9;

using PairKey = std::tuple<int32_t, int32_t, int32_t, int32_t, int32_t>;

struct PairColumns {
  bool has_vector = false;
  double x = 0.0;
  double y = 0.0;
  double z = 0.0;
  bool has_distance = false;
  double distance = 0.0;
};

void expect_noise(double got, double expected, const char *what) {
  if (std::abs(got - expected) > kNeighbourNoise) {
    throw std::runtime_error(
        std::string(what) + " differs by more than 1e-9: " +
        std::to_string(got) + " vs " + std::to_string(expected));
  }
}

void collect_pairs(const rgpot_nlist_request &request,
                   const rgpot_nlist_soa &soa, size_t count,
                   std::map<PairKey, PairColumns> *out) {
  const bool shifts = request.want_shifts != 0;
  const bool vectors = request.want_vectors != 0 && soa.vector != nullptr;
  const bool distances = request.want_distances != 0 && soa.distance != nullptr;
  for (size_t pair = 0; pair < count; ++pair) {
    const PairKey key{soa.first[pair], soa.second[pair],
                      shifts ? soa.shift[3 * pair] : 0,
                      shifts ? soa.shift[3 * pair + 1] : 0,
                      shifts ? soa.shift[3 * pair + 2] : 0};
    PairColumns columns;
    if (vectors) {
      columns.has_vector = true;
      columns.x = soa.vector[3 * pair];
      columns.y = soa.vector[3 * pair + 1];
      columns.z = soa.vector[3 * pair + 2];
    }
    if (distances) {
      columns.has_distance = true;
      columns.distance = soa.distance[pair];
    }
    if (!out->emplace(key, columns).second) {
      throw std::runtime_error("neighbour list repeated a pair");
    }
  }
}

void expect_same_pairs(const std::map<PairKey, PairColumns> &session,
                       const std::map<PairKey, PairColumns> &builder,
                       const char *what) {
  if (session.size() != builder.size()) {
    throw std::runtime_error(std::string(what) +
                             " pair count differs from the shared builder");
  }
  for (const auto &item : session) {
    const auto found = builder.find(item.first);
    if (found == builder.end()) {
      throw std::runtime_error(std::string(what) +
                               " is missing a pair the session returned");
    }
    if (item.second.has_vector) {
      expect_noise(item.second.x, found->second.x, what);
      expect_noise(item.second.y, found->second.y, what);
      expect_noise(item.second.z, found->second.z, what);
    }
    if (item.second.has_distance) {
      expect_noise(item.second.distance, found->second.distance, what);
    }
  }
}

rgpot_nlist_soa allocate_soa(const rgpot_nlist_request &request, size_t cap,
                             std::vector<int32_t> *first,
                             std::vector<int32_t> *second,
                             std::vector<int32_t> *shift,
                             std::vector<double> *vector,
                             std::vector<double> *distance) {
  first->assign(cap, 0);
  second->assign(cap, 0);
  rgpot_nlist_soa soa{};
  soa.bytes = sizeof(soa);
  soa.first = first->data();
  soa.second = second->data();
  soa.cap = cap;
  if (request.want_shifts != 0) {
    shift->assign(3 * cap, 0);
    soa.shift = shift->data();
  }
  if (request.want_vectors != 0) {
    vector->assign(3 * cap, 0.0);
    soa.vector = vector->data();
  }
  if (request.want_distances != 0) {
    distance->assign(cap, 0.0);
    soa.distance = distance->data();
  }
  return soa;
}

// Session neighbours against rgpot_nlist_build. A non-periodic system
// clears the request's periodic flags before that call, so the reference
// does the same. A difference of 1e-9 is noise.
void expect_builder_parity(rgpot_md_session *session,
                           const rgpot_nlist_request &request, bool periodic,
                           const char *what) {
  char err[512] = {};
  const size_t n_atoms = rgpot_md_session_n_atoms(session);
  std::vector<double> positions(3 * n_atoms);
  if (rgpot_md_session_get_positions(session, positions.data(),
                                     positions.size(), err, sizeof(err)) != 0) {
    throw std::runtime_error(std::string(what) + ": " + err);
  }
  double box[9] = {};
  if (rgpot_md_session_get_periodic_box(session, box, err, sizeof(err)) != 0) {
    throw std::runtime_error(std::string(what) + ": " + err);
  }
  rgpot_nlist_request built = request;
  built.periodic[0] = periodic ? 1 : 0;
  built.periodic[1] = periodic ? 1 : 0;
  built.periodic[2] = periodic ? 1 : 0;

  size_t session_need = 0;
  const int session_rc = rgpot_md_session_get_neighbors(
      session, &request, nullptr, nullptr, &session_need, err, sizeof(err));
  if (session_rc != RGPOT_MD_NLIST_OK) {
    throw std::runtime_error(std::string(what) + " size probe failed: " + err);
  }
  size_t builder_need = 0;
  const int32_t builder_rc = rgpot_nlist_build(
      positions.data(), n_atoms, box, &built, nullptr, nullptr, &builder_need);
  if (builder_rc != RGPOT_NLIST_OK || builder_need != session_need) {
    throw std::runtime_error(std::string(what) +
                             " size differs from the shared builder");
  }
  if (session_need == 0) {
    return;
  }
  std::vector<int32_t> session_first, session_second, session_shift;
  std::vector<double> session_vector, session_distance;
  std::vector<int32_t> builder_first, builder_second, builder_shift;
  std::vector<double> builder_vector, builder_distance;
  rgpot_nlist_soa session_soa =
      allocate_soa(request, session_need, &session_first, &session_second,
                   &session_shift, &session_vector, &session_distance);
  rgpot_nlist_soa builder_soa =
      allocate_soa(built, builder_need, &builder_first, &builder_second,
                   &builder_shift, &builder_vector, &builder_distance);
  size_t session_got = 0;
  size_t session_full = 0;
  const int fill_rc = rgpot_md_session_get_neighbors(
      session, &request, &session_soa, &session_got, &session_full, err,
      sizeof(err));
  if (fill_rc != RGPOT_MD_NLIST_OK || session_got != session_need) {
    throw std::runtime_error(std::string(what) + " fill failed: " + err);
  }
  size_t builder_got = 0;
  const int32_t build_rc =
      rgpot_nlist_build(positions.data(), n_atoms, box, &built, &builder_soa,
                        &builder_got, &builder_need);
  if (build_rc != RGPOT_NLIST_OK || builder_got != session_got) {
    throw std::runtime_error(std::string(what) +
                             " build differs from the shared builder");
  }
  std::map<PairKey, PairColumns> session_pairs;
  std::map<PairKey, PairColumns> builder_pairs;
  collect_pairs(request, session_soa, session_got, &session_pairs);
  collect_pairs(built, builder_soa, builder_got, &builder_pairs);
  expect_same_pairs(session_pairs, builder_pairs, what);
}

std::unique_ptr<OpenMM::System> make_periodic_system() {
  auto out = std::make_unique<OpenMM::System>();
  out->addParticle(1.0);
  out->addParticle(1.0);
  auto *bond = new OpenMM::HarmonicBondForce();
  bond->setUsesPeriodicBoundaryConditions(true);
  bond->addBond(0, 1, 0.1, 0.0);
  out->addForce(bond);
  return out;
}

// Two free atoms. The system has no periodic force, so the session clears
// the request's periodic flags before rgpot_nlist_build. The hand-computed
// pairs check that geometry. The shared builder is called with those flags
// cleared, which is the geometry the session actually queries.
void test_nonperiodic_neighbors() {
  char err[512] = {};
  rgpot_md_session *session =
      open_neighbor_session(make_free_system(1.0, 1.0), err, sizeof(err));
  std::unique_ptr<rgpot_md_session, decltype(&rgpot_md_session_destroy)> guard(
      session, rgpot_md_session_destroy);

  const double close[6] = {0.0, 0.0, 0.0, 4.0, 0.0, 0.0};
  if (rgpot_md_session_set_positions(session, close, 6, err, sizeof(err)) !=
      0) {
    throw std::runtime_error(err);
  }
  rgpot_nlist_request nreq{};
  rgpot_nlist_request_init(&nreq);
  nreq.cutoff = 8.0;
  nreq.shell = 1; // full list: both directed pairs
  nreq.skin = 0.0;
  size_t need = 0;
  int nrc = rgpot_md_session_get_neighbors(session, &nreq, nullptr, nullptr,
                                           &need, err, sizeof(err));
  if (nrc != RGPOT_MD_NLIST_OK || need != 2) {
    throw std::runtime_error(
        std::string("4 A pair size probe expected 2, got ") +
        std::to_string(need) + " " + err);
  }
  std::vector<int32_t> first(need), second(need), shift(3 * need, 9);
  std::vector<double> vector(3 * need), distance(need);
  rgpot_nlist_soa soa{};
  soa.bytes = sizeof(soa);
  soa.first = first.data();
  soa.second = second.data();
  soa.shift = shift.data();
  soa.vector = vector.data();
  soa.distance = distance.data();
  soa.cap = need;
  size_t got = 0;
  nrc = rgpot_md_session_get_neighbors(session, &nreq, &soa, &got, &need, err,
                                       sizeof(err));
  if (nrc != RGPOT_MD_NLIST_OK || got != 2) {
    throw std::runtime_error(std::string("4 A pair fill failed: ") + err);
  }
  std::map<std::pair<int, int>, std::array<double, 4>> found;
  for (size_t p = 0; p < got; ++p) {
    if (shift[3 * p] != 0 || shift[3 * p + 1] != 0 || shift[3 * p + 2] != 0) {
      throw std::runtime_error("non-periodic pair carried an image shift");
    }
    found[{first[p], second[p]}] = {vector[3 * p], vector[3 * p + 1],
                                    vector[3 * p + 2], distance[p]};
  }
  const auto forward = found.find({0, 1});
  const auto backward = found.find({1, 0});
  if (found.size() != 2 || forward == found.end() || backward == found.end()) {
    throw std::runtime_error("4 A geometry was not the two directed pairs");
  }
  near(forward->second[0], 4.0, 4.0, "pair 0->1 x");
  near(forward->second[1], 0.0, 1.0, "pair 0->1 y");
  near(forward->second[2], 0.0, 1.0, "pair 0->1 z");
  near(forward->second[3], 4.0, 4.0, "pair 0->1 distance");
  near(backward->second[0], -4.0, 4.0, "pair 1->0 x");
  near(backward->second[1], 0.0, 1.0, "pair 1->0 y");
  near(backward->second[2], 0.0, 1.0, "pair 1->0 z");
  near(backward->second[3], 4.0, 4.0, "pair 1->0 distance");
  expect_builder_parity(session, nreq, false, "non-periodic 4 A pair");

  nreq.want_shifts = 0;
  nreq.want_vectors = 0;
  nreq.want_distances = 0;
  int32_t one_first = -1;
  int32_t one_second = -1;
  rgpot_nlist_soa short_soa{};
  short_soa.bytes = sizeof(short_soa);
  short_soa.first = &one_first;
  short_soa.second = &one_second;
  short_soa.cap = 1;
  size_t short_got = 0;
  size_t short_need = 0;
  nrc = rgpot_md_session_get_neighbors(session, &nreq, &short_soa, &short_got,
                                       &short_need, err, sizeof(err));
  if (nrc != RGPOT_MD_NLIST_ERROR || short_need != 2 ||
      std::strstr(err, "2 are required") == nullptr) {
    throw std::runtime_error(
        std::string("short neighbour buffer was not reported: ") + err);
  }

  // 15 A apart with an 8 A cutoff. The placeholder 20 A box would report a
  // phantom 5 A pair and shift (-1, 0, 0). A non-periodic system has no pair.
  const double far[6] = {0.0, 0.0, 0.0, 15.0, 0.0, 0.0};
  if (rgpot_md_session_set_positions(session, far, 6, err, sizeof(err)) != 0) {
    throw std::runtime_error(err);
  }
  need = 99;
  nrc = rgpot_md_session_get_neighbors(session, &nreq, nullptr, nullptr, &need,
                                       err, sizeof(err));
  if (nrc != RGPOT_MD_NLIST_OK || need != 0) {
    throw std::runtime_error(
        "non-periodic 15 A / 8 A query wrapped through the placeholder box");
  }
  nreq.want_shifts = 1;
  int32_t phantom_shift[3] = {7, 7, 7};
  double phantom_distance = -1.0;
  rgpot_nlist_soa far_soa{};
  far_soa.bytes = sizeof(far_soa);
  far_soa.first = &one_first;
  far_soa.second = &one_second;
  far_soa.shift = phantom_shift;
  far_soa.distance = &phantom_distance;
  far_soa.cap = 1;
  got = 99;
  nrc = rgpot_md_session_get_neighbors(session, &nreq, &far_soa, &got, &need,
                                       err, sizeof(err));
  if (nrc != RGPOT_MD_NLIST_OK || got != 0 || need != 0 ||
      phantom_shift[0] == -1) {
    throw std::runtime_error(
        "non-periodic 15 A / 8 A query returned a phantom image pair");
  }
  expect_builder_parity(session, nreq, false, "non-periodic 15 A pair");

  const int set_rc = rgpot_md_session_set_neighbors(
      session, &nreq, &far_soa, 0, err, sizeof(err));
  if (set_rc != RGPOT_MD_NLIST_UNSUPPORTED) {
    throw std::runtime_error(
        "expected OpenMM neighbor setter to be unsupported");
  }
}

// Atoms on opposite sides of a face. The minimum image crosses the
// boundary and the direct gap is outside the cutoff. The session pair
// set has to match rgpot_nlist_build in that box.
void test_periodic_boundary_neighbors() {
  char err[512] = {};
  rgpot_md_session *session =
      open_neighbor_session(make_periodic_system(), err, sizeof(err));
  std::unique_ptr<rgpot_md_session, decltype(&rgpot_md_session_destroy)> guard(
      session, rgpot_md_session_destroy);

  const double box[9] = {12.0, 0.0, 0.0, 0.0, 12.0, 0.0, 0.0, 0.0, 12.0};
  if (rgpot_md_session_set_periodic_box(session, box, err, sizeof(err)) != 0) {
    throw std::runtime_error(err);
  }
  const double pos[6] = {0.4, 3.0, 3.0, 11.6, 3.0, 3.0};
  if (rgpot_md_session_set_positions(session, pos, 6, err, sizeof(err)) != 0) {
    throw std::runtime_error(err);
  }
  rgpot_nlist_request nreq{};
  rgpot_nlist_request_init(&nreq);
  nreq.cutoff = 2.0;
  nreq.shell = 1;
  nreq.skin = 0.0;
  expect_builder_parity(session, nreq, true, "periodic boundary neighbours");

  size_t need = 0;
  int nrc = rgpot_md_session_get_neighbors(session, &nreq, nullptr, nullptr,
                                           &need, err, sizeof(err));
  if (nrc != RGPOT_MD_NLIST_OK || need != 2) {
    throw std::runtime_error(
        "periodic boundary pair was not the two directed images");
  }
  std::vector<int32_t> first(need), second(need), shift(3 * need, 0);
  std::vector<double> distance(need, 0.0);
  rgpot_nlist_soa soa{};
  soa.bytes = sizeof(soa);
  soa.first = first.data();
  soa.second = second.data();
  soa.shift = shift.data();
  soa.distance = distance.data();
  soa.cap = need;
  size_t got = 0;
  nrc = rgpot_md_session_get_neighbors(session, &nreq, &soa, &got, &need, err,
                                       sizeof(err));
  if (nrc != RGPOT_MD_NLIST_OK || got != 2) {
    throw std::runtime_error(std::string("periodic boundary fill failed: ") +
                             err);
  }
  bool crossed = false;
  for (size_t pair = 0; pair < got; ++pair) {
    if (shift[3 * pair] == 0 && shift[3 * pair + 1] == 0 &&
        shift[3 * pair + 2] == 0) {
      throw std::runtime_error("periodic boundary pair did not cross a face");
    }
    if (shift[3 * pair] != 0) {
      crossed = true;
    }
    near(distance[pair], 0.8, 0.8, "periodic image distance");
  }
  if (!crossed) {
    throw std::runtime_error("periodic image shift was not along the box");
  }
}

} // namespace

int main() {
  try {
    test_cpp_session();
    test_c_abi_session();
    test_nonperiodic_neighbors();
    test_periodic_boundary_neighbors();
    std::cout
        << "OpenMM session energy, plane, wall, temperature, determinism, "
           "C ABI, NVE, non-periodic, and periodic neighbour checks passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
