// MIT License
// Copyright 2023--present rgpot developers

#include "rgpot/ExternalPot/OpenMM.hpp"
#include "rgpot/ExternalPot/external_potentials.h"

#include <openmm/HarmonicBondForce.h>
#include <openmm/System.h>
#include <openmm/serialization/XmlSerializer.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
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

} // namespace

int main() {
  try {
    test_cpp_session();
    test_c_abi_session();
    std::cout
        << "OpenMM session energy, plane, wall, temperature, determinism, "
           "C ABI, and NVE checks passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
