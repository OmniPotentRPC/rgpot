// MIT License
// Copyright 2026 Rohit Goswami

#include "lammpc_engine_spec.h"
#include "rgpot/ExternalPot/external_potentials.h"
#include "rgpot/ExternalPot/md_units.hpp"
#include "rgpot/rpc/Potentials.capnp.h"

#include <capnp/message.h>
#include <capnp/serialize.h>
#include <kj/array.h>
#include <openmm/HarmonicBondForce.h>
#include <openmm/System.h>
#include <openmm/serialization/XmlSerializer.h>

#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr double kSpring = 1.0;
constexpr double kMass = 39.9;
constexpr double kKt = 0.025;
constexpr double kDistance = 2.0;

void require(bool condition, const std::string &message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

std::vector<unsigned char> flatten(capnp::MessageBuilder &message) {
  kj::Array<capnp::word> flat = capnp::messageToFlatArray(message);
  const auto bytes = flat.asBytes();
  return std::vector<unsigned char>(bytes.begin(), bytes.end());
}

std::vector<unsigned char> config() {
  capnp::MallocMessageBuilder message;
  auto root = message.initRoot<::PotentialConfig>();
  auto lammps = root.initLammps();
  lammps.setUnitsStyle("metal");
  lammps.setAtomStyle("atomic");
  lammps.setPairStyle("lepton 9.0");
  auto coefficients = lammps.initPairCoeffs(1);
  coefficients.set(0, "1 1 \"0.5*k*r^2; k=1.0\"");
  auto type_map = lammps.initTypeToAtomicNumber(1);
  type_map.set(0, 18);
  auto masses = lammps.initMasses(1);
  masses.set(0, kMass);
  lammps.setBoundary("p p p");
  return flatten(message);
}

std::vector<unsigned char> input() {
  capnp::MallocMessageBuilder message;
  auto root = message.initRoot<::ForceInput>();
  auto positions = root.initPos(6);
  const std::array<double, 6> xyz{0.0, 0.0, 0.0, kDistance, 0.0, 0.0};
  for (size_t index = 0; index < xyz.size(); ++index) {
    positions.set(index, xyz[index]);
  }
  auto numbers = root.initAtmnrs(2);
  numbers.set(0, 18);
  numbers.set(1, 18);
  auto box = root.initBox(9);
  const std::array<double, 9> cell{30.0, 0.0, 0.0, 0.0, 30.0,
                                   0.0,  0.0, 0.0, 30.0};
  for (size_t index = 0; index < cell.size(); ++index) {
    box.set(index, cell[index]);
  }
  root.setLengthUnit("angstrom");
  root.setEnergyUnit("eV");
  return flatten(message);
}

std::vector<unsigned char> spec() {
  const auto cfg = config();
  const auto geometry = input();
  rgpot_lammpc_session_spec_header header{RGPOT_LAMMPC_SESSION_SPEC_MAGIC,
                                          cfg.size(), geometry.size()};
  std::vector<unsigned char> result(sizeof(header) + cfg.size() +
                                    geometry.size());
  std::memcpy(result.data(), &header, sizeof(header));
  std::memcpy(result.data() + sizeof(header), cfg.data(), cfg.size());
  std::memcpy(result.data() + sizeof(header) + cfg.size(), geometry.data(),
              geometry.size());
  return result;
}

std::string openmmSystem() {
  auto system = std::make_unique<OpenMM::System>();
  system->addParticle(kMass);
  system->addParticle(kMass);
  auto *bond = new OpenMM::HarmonicBondForce();
  bond->addBond(0, 1, 0.0,
                kSpring * rgpot::md::kKjPerEv /
                    (rgpot::md::kNmPerAngstrom * rgpot::md::kNmPerAngstrom));
  system->addForce(bond);
  std::ostringstream output;
  OpenMM::XmlSerializer::serialize<OpenMM::System>(system.get(), "System",
                                                   output);
  return output.str();
}

using Session =
    std::unique_ptr<rgpot_md_session, decltype(&rgpot_md_session_destroy)>;

Session create(const char *engine, const char *engine_path, const void *bytes,
               size_t size, const char *options) {
  char error[1024] = {};
  rgpot_md_session *raw = rgpot_md_session_create(
      engine, engine_path, bytes, size, options, error, sizeof(error));
  require(raw != nullptr, std::string("session create failed: ") + error);
  return Session(raw, rgpot_md_session_destroy);
}

struct Sample {
  double potential;
  double kinetic;
  double constraint;
};

Sample state(rgpot_md_session *session) {
  char error[1024] = {};
  std::array<double, 6> positions{};
  std::array<double, 6> velocities{};
  std::array<double, 6> forces{};
  Sample sample{};
  require(rgpot_md_session_get_state(
              session, positions.data(), velocities.data(), forces.data(),
              positions.size(), &sample.potential, &sample.kinetic, error,
              sizeof(error)) == 0,
          error);
  require(rgpot_md_session_get_constraint_force(session, &sample.constraint,
                                                error, sizeof(error)) == 0,
          error);
  return sample;
}

void initialize(rgpot_md_session *session) {
  char error[1024] = {};
  const std::array<double, 9> box{30.0, 0.0, 0.0, 0.0, 30.0,
                                  0.0,  0.0, 0.0, 30.0};
  const std::array<double, 6> positions{0.0, 0.0, 0.0, kDistance, 0.0, 0.0};
  const std::array<double, 6> velocities{};
  require(rgpot_md_session_set_periodic_box(session, box.data(), error,
                                            sizeof(error)) == 0,
          error);
  require(rgpot_md_session_set_positions(session, positions.data(),
                                         positions.size(), error,
                                         sizeof(error)) == 0,
          error);
  require(rgpot_md_session_set_velocities(session, velocities.data(),
                                          velocities.size(), error,
                                          sizeof(error)) == 0,
          error);
  require(rgpot_md_session_initialize(session, error, sizeof(error)) == 0,
          error);
}

void advance(rgpot_md_session *session, size_t steps, uint64_t seed,
             uint64_t *hits = nullptr, const double *wall_point = nullptr,
             const double *wall_normal = nullptr) {
  char error[1024] = {};
  const std::array<double, 6> point{0.0, 0.0, 0.0, kDistance, 0.0, 0.0};
  const std::array<double, 6> normal{-1.0, 0.0, 0.0, 1.0, 0.0, 0.0};
  uint64_t local_hits = 0;
  require(rgpot_md_session_step_langevin(
              session, kKt, 10.0, 0.0005, point.data(), normal.data(),
              wall_point, wall_normal, wall_point != nullptr ? 1 : 0,
              point.size(), steps, seed, hits != nullptr ? hits : &local_hits,
              error, sizeof(error)) == 0,
          error);
}

double mean(const std::vector<double> &values) {
  return std::accumulate(values.begin(), values.end(), 0.0) / values.size();
}

double stderr(const std::vector<double> &values) {
  const double average = mean(values);
  double sum = 0.0;
  for (double value : values) {
    const double delta = value - average;
    sum += delta * delta;
  }
  return std::sqrt(sum / (values.size() - 1) / values.size());
}

struct Statistics {
  double potential;
  double potential_se;
  double kinetic;
  double kinetic_se;
  double constraint;
  double constraint_se;
};

Statistics sample(rgpot_md_session *session) {
  char error[1024] = {};
  require(rgpot_md_session_reseed_velocities(session, kKt, 701, error,
                                             sizeof(error)) == 0,
          error);
  advance(session, 5000, 991);
  std::vector<double> potential;
  std::vector<double> kinetic;
  std::vector<double> constraint;
  for (uint64_t block = 0; block < 400; ++block) {
    advance(session, 200, 1000 + block);
    const Sample value = state(session);
    potential.push_back(value.potential);
    kinetic.push_back(value.kinetic);
    constraint.push_back(value.constraint);
  }
  return {mean(potential), stderr(potential), mean(kinetic),
          stderr(kinetic), mean(constraint),  stderr(constraint)};
}

void within_se(double got, double expected, double standard_error,
               const char *what) {
  const double tolerance = 7.0 * standard_error + 2.0e-3;
  require(std::abs(got - expected) <= tolerance,
          std::string(what) + ": got " + std::to_string(got) + ", expected " +
              std::to_string(expected) + ", SE " +
              std::to_string(standard_error));
}

void run(const char *lammpc_engine, const char *lammpc_library,
         const char *openmm_engine) {
  const auto lammpc_spec = spec();
  const std::string xml = openmmSystem();
  auto lammps = create("lammpc", lammpc_engine, lammpc_spec.data(),
                       lammpc_spec.size(), lammpc_library);
  auto openmm =
      create("openmm", openmm_engine, xml.data(), xml.size(), "Reference");
  initialize(lammps.get());
  initialize(openmm.get());

  std::array<double, 2> masses{};
  char error[1024] = {};
  require(rgpot_md_session_get_masses(lammps.get(), masses.data(),
                                      masses.size(), error, sizeof(error)) == 0,
          error);
  require(std::abs(masses[0] - kMass) < 1.0e-12 &&
              std::abs(masses[1] - kMass) < 1.0e-12,
          "lammpc masses differ");
  const Sample initial = state(lammps.get());
  require(std::abs(initial.potential - 2.0) < 1.0e-10,
          "lammpc harmonic energy differs");

  auto replay_a = create("lammpc", lammpc_engine, lammpc_spec.data(),
                         lammpc_spec.size(), lammpc_library);
  auto replay_b = create("lammpc", lammpc_engine, lammpc_spec.data(),
                         lammpc_spec.size(), lammpc_library);
  initialize(replay_a.get());
  initialize(replay_b.get());
  require(rgpot_md_session_reseed_velocities(replay_a.get(), kKt, 42, error,
                                             sizeof(error)) == 0 &&
              rgpot_md_session_reseed_velocities(replay_b.get(), kKt, 42, error,
                                                 sizeof(error)) == 0,
          error);
  advance(replay_a.get(), 128, 77);
  advance(replay_b.get(), 128, 77);
  std::array<double, 6> a_pos{}, a_vel{}, b_pos{}, b_vel{};
  require(rgpot_md_session_get_state(replay_a.get(), a_pos.data(), a_vel.data(),
                                     nullptr, a_pos.size(), nullptr, nullptr,
                                     error, sizeof(error)) == 0 &&
              rgpot_md_session_get_state(
                  replay_b.get(), b_pos.data(), b_vel.data(), nullptr,
                  b_pos.size(), nullptr, nullptr, error, sizeof(error)) == 0,
          error);
  require(a_pos == b_pos && a_vel == b_vel,
          "lammpc seed did not replay byte-identically");

  const Statistics lammps_stats = sample(lammps.get());
  const Statistics openmm_stats = sample(openmm.get());
  const double analytic_potential = 0.5 * kSpring * kDistance * kDistance + kKt;
  const double analytic_kinetic = 2.5 * kKt;
  const double analytic_constraint = -std::sqrt(2.0) * kSpring * kDistance;
  within_se(lammps_stats.potential, analytic_potential,
            lammps_stats.potential_se, "lammpc harmonic equipartition");
  within_se(openmm_stats.potential, analytic_potential,
            openmm_stats.potential_se, "OpenMM harmonic equipartition");
  within_se(lammps_stats.kinetic, analytic_kinetic, lammps_stats.kinetic_se,
            "lammpc kinetic equipartition");
  within_se(openmm_stats.kinetic, analytic_kinetic, openmm_stats.kinetic_se,
            "OpenMM kinetic equipartition");
  within_se(lammps_stats.constraint, analytic_constraint,
            lammps_stats.constraint_se, "lammpc analytic dF/dc");
  within_se(openmm_stats.constraint, analytic_constraint,
            openmm_stats.constraint_se, "OpenMM analytic dF/dc");
  within_se(lammps_stats.potential, openmm_stats.potential,
            std::hypot(lammps_stats.potential_se, openmm_stats.potential_se),
            "engine potential means");
  within_se(lammps_stats.constraint, openmm_stats.constraint,
            std::hypot(lammps_stats.constraint_se, openmm_stats.constraint_se),
            "engine constrained mean forces");

  const std::array<double, 6> wall_point{};
  const std::array<double, 6> wall_normal{0.0, 0.0, -1.0, 0.0, 0.0, 0.0};
  const std::array<double, 6> wall_velocity{0.0, 0.0, -100.0, 0.0, 0.0, 0.0};
  require(rgpot_md_session_set_velocities(lammps.get(), wall_velocity.data(),
                                          wall_velocity.size(), error,
                                          sizeof(error)) == 0,
          error);
  uint64_t hits = 0;
  advance(lammps.get(), 100, 31, &hits, wall_point.data(), wall_normal.data());
  require(hits > 0, "lammpc planar wall did not reflect");
  std::array<double, 6> reflected_positions{};
  require(rgpot_md_session_get_state(lammps.get(), reflected_positions.data(),
                                     nullptr, nullptr,
                                     reflected_positions.size(), nullptr,
                                     nullptr, error, sizeof(error)) == 0,
          error);
  require(reflected_positions[2] >= -1.0e-12,
          "lammpc planar wall left a particle outside");

  require(rgpot_md_session_step_nve(lammps.get(), 0.0001, 10, error,
                                    sizeof(error)) == 0,
          error);
  std::cout << "LAMMPS/OpenMM harmonic equipartition, constrained mean force, "
               "seed replay, walls, state, and NVE checks passed\n";
}

} // namespace

int main(int argc, char **argv) {
  try {
    require(argc == 4,
            "usage: test_lammpc_session <engine> <liblammpc> <openmm-engine>");
    run(argv[1], argv[2], argv[3]);
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
