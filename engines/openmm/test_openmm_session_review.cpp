// MIT License
// Copyright 2026 Rohit Goswami

#include "rgpot/ExternalPot/OpenMM.hpp"

#include <openmm/CustomExternalForce.h>
#include <openmm/System.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr double kComparisonTolerance = 0.0000000001;
constexpr double kPlaneTolerance = 0.00000001;

void require(bool condition, const std::string &message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

void require_close(double actual, double expected, double tolerance,
                   const std::string &message) {
  if (std::abs(actual - expected) > tolerance) {
    throw std::runtime_error(message + ": got " + std::to_string(actual) +
                             ", expected " + std::to_string(expected));
  }
}

rgpot::OpenMMSessionConfig reference_config() {
  rgpot::OpenMMSessionConfig config;
  config.platform = "Reference";
  config.time_step = 0.0005;
  config.friction = 5.0;
  config.temperature = 2.5;
  config.random_seed = 17;
  return config;
}

std::vector<OpenMM::Vec3> zeros(std::size_t count) {
  return std::vector<OpenMM::Vec3>(count, OpenMM::Vec3(0.0, 0.0, 0.0));
}

struct BatchMean {
  double mean = 0.0;
  double error = 0.0;
};

// Mean and standard error of a correlated series by batch means.
BatchMean batch_mean(const std::vector<double> &series,
                     std::size_t batches = 40) {
  const std::size_t size = series.size() / batches;
  std::vector<double> means;
  for (std::size_t b = 0; b < batches; ++b) {
    means.push_back(std::accumulate(series.begin() + b * size,
                                    series.begin() + (b + 1) * size, 0.0) /
                    static_cast<double>(size));
  }
  const double mean = std::accumulate(means.begin(), means.end(), 0.0) /
                      static_cast<double>(means.size());
  double square = 0.0;
  for (const double value : means) {
    square += (value - mean) * (value - mean);
  }
  return {mean, std::sqrt(square / static_cast<double>(means.size() - 1) /
                          static_cast<double>(means.size()))};
}

void test_harmonic_equipartition_on_plane() {
  constexpr double kT = 2.5;
  constexpr double spring = 200.0;
  auto system = std::make_unique<OpenMM::System>();
  system->addParticle(4.0);
  system->addParticle(40.0);
  auto *force = new OpenMM::CustomExternalForce("0.5*k*(x*x+y*y+z*z)");
  force->addGlobalParameter("k", spring);
  force->addParticle(0);
  force->addParticle(1);
  system->addForce(force);

  auto config = reference_config();
  rgpot::OpenMMSession session(std::move(system), config);
  const std::vector<OpenMM::Vec3> point = zeros(2);
  const std::vector<OpenMM::Vec3> normal{OpenMM::Vec3(1.0, 0.0, 0.0),
                                         OpenMM::Vec3(1.0, 0.0, 0.0)};
  session.setPositions(point);
  session.setVelocities(zeros(2));
  session.stepLangevin(kT, config.friction, config.time_step, point, normal, {},
                       {}, 20000, 701);

  constexpr std::size_t sample_count = 60000;
  std::vector<double> potential_samples;
  potential_samples.reserve(sample_count);
  for (std::size_t sample = 0; sample < sample_count; ++sample) {
    session.stepLangevin(kT, config.friction, config.time_step, point, normal,
                         {}, {}, 20, 9001 + sample);
    const auto state = session.state();
    const double plane = state.positions[0][0] + state.positions[1][0];
    require(std::abs(plane) < kPlaneTolerance,
            "harmonic sample left the plane");
    potential_samples.push_back(state.potential_energy);
  }
  const BatchMean potential = batch_mean(potential_samples);
  const double expected = 2.5 * kT;
  require(potential.error < 0.3,
          "equipartition run is too short to resolve the mean: error " +
              std::to_string(potential.error));
  require(std::abs(potential.mean - expected) < 4.0 * potential.error,
          "constrained harmonic configurational equipartition failed: got " +
              std::to_string(potential.mean) + " +- " +
              std::to_string(potential.error) + ", expected " +
              std::to_string(expected));
}

void test_closed_form_constrained_mean_force() {
  constexpr double kT = 2.0;
  constexpr double coupling = 80.0;
  constexpr double x_spring = 20.0;
  constexpr double y_spring = 30.0;
  constexpr double z_spring = 100.0;
  constexpr double constrained_y = 0.15;

  auto system = std::make_unique<OpenMM::System>();
  system->addParticle(12.0);
  auto *force = new OpenMM::CustomExternalForce(
      "0.5*kc*(x-y)^2+0.5*kx*x^2+0.5*ky*y^2+0.5*kz*z^2");
  force->addGlobalParameter("kc", coupling);
  force->addGlobalParameter("kx", x_spring);
  force->addGlobalParameter("ky", y_spring);
  force->addGlobalParameter("kz", z_spring);
  force->addParticle(0);
  system->addForce(force);

  auto config = reference_config();
  rgpot::OpenMMSession session(std::move(system), config);
  const std::vector<OpenMM::Vec3> point{OpenMM::Vec3(0.0, constrained_y, 0.0)};
  const std::vector<OpenMM::Vec3> normal{OpenMM::Vec3(0.0, 1.0, 0.0)};
  session.setPositions(point);
  session.setVelocities(zeros(1));
  session.stepLangevin(kT, config.friction, config.time_step, point, normal, {},
                       {}, 20000, 1103);

  constexpr std::size_t sample_count = 6000;
  double sum_x = 0.0;
  double sum_lambda = 0.0;
  for (std::size_t sample = 0; sample < sample_count; ++sample) {
    session.stepLangevin(kT, config.friction, config.time_step, point, normal,
                         {}, {}, 200, 12001 + sample);
    const auto state = session.state();
    require_close(state.positions[0][1], constrained_y, kPlaneTolerance,
                  "mean-force model left the plane");
    sum_x += state.positions[0][0];
    sum_lambda += -state.forces[0][1];
  }
  const double mean_x = sum_x / static_cast<double>(sample_count);
  const double mean_lambda = sum_lambda / static_cast<double>(sample_count);
  const double expected_x = coupling * constrained_y / (coupling + x_spring);
  const double expected_lambda =
      (coupling * x_spring / (coupling + x_spring) + y_spring) * constrained_y;
  require(std::abs(mean_x - expected_x) < 0.01,
          "closed-form constrained coordinate mean failed: got " +
              std::to_string(mean_x) + ", expected " +
              std::to_string(expected_x));
  require(std::abs(mean_lambda - expected_lambda) < 0.8,
          "closed-form constrained mean force failed: got " +
              std::to_string(mean_lambda) + ", expected " +
              std::to_string(expected_lambda));
}

void test_wall_reflection_conserves_kinetic_energy() {
  auto system = std::make_unique<OpenMM::System>();
  system->addParticle(7.0);
  auto config = reference_config();
  rgpot::OpenMMSession session(std::move(system), config);
  const std::vector<OpenMM::Vec3> point = zeros(1);
  const std::vector<OpenMM::Vec3> plane_normal{OpenMM::Vec3(1.0, 0.0, 0.0)};
  const std::vector<std::vector<OpenMM::Vec3>> wall_points{zeros(1)};
  const std::vector<std::vector<OpenMM::Vec3>> wall_normals{
      {OpenMM::Vec3(0.0, 1.0, 0.0)}};
  session.setPositions({OpenMM::Vec3(0.0, -0.001, 0.0)});
  session.setVelocities({OpenMM::Vec3(0.0, 5.0, 2.0)});
  const double before = session.state().kinetic_energy;
  const auto hits = session.stepLangevin(2.5, 0.0, 0.001, point, plane_normal,
                                         wall_points, wall_normals, 1, 77);
  const auto state = session.state();
  require(hits == 1, "expected exactly one wall reflection");
  require(state.positions[0][1] <= 0.0,
          "wall reflection left particle outside");
  require(state.velocities[0][1] < 0.0,
          "wall reflection did not reverse velocity");
  require_close(state.kinetic_energy, before, kComparisonTolerance,
                "wall reflection changed kinetic energy");
}

void test_mass_metric_projection_and_seed_reproducibility() {
  auto make_system = []() {
    auto system = std::make_unique<OpenMM::System>();
    system->addParticle(1.0);
    system->addParticle(3.0);
    return system;
  };
  auto config = reference_config();
  const std::vector<OpenMM::Vec3> point = zeros(2);
  const std::vector<OpenMM::Vec3> normal{OpenMM::Vec3(1.0, 0.0, 0.0),
                                         OpenMM::Vec3(1.0, 0.0, 0.0)};

  rgpot::OpenMMSession projected(make_system(), config);
  projected.setPositions(point);
  projected.setVelocities(
      {OpenMM::Vec3(1.0, 0.0, 0.0), OpenMM::Vec3(0.0, 0.0, 0.0)});
  projected.stepLangevin(2.5, 0.0, config.time_step, point, normal, {}, {}, 1,
                         0);
  const auto projected_state = projected.state();
  require_close(projected_state.velocities[0][0], 0.25, kComparisonTolerance,
                "light-particle projected velocity");
  require_close(projected_state.velocities[1][0], -0.25, kComparisonTolerance,
                "heavy-particle projected velocity");

  rgpot::OpenMMSession first(make_system(), config);
  rgpot::OpenMMSession second(make_system(), config);
  first.setPositions(point);
  second.setPositions(point);
  first.setVelocities(zeros(2));
  second.setVelocities(zeros(2));
  first.stepLangevin(2.5, 3.0, config.time_step, point, normal, {}, {}, 1000,
                     0);
  second.stepLangevin(2.5, 3.0, config.time_step, point, normal, {}, {}, 1000,
                      0);
  const auto first_state = first.state();
  const auto second_state = second.state();
  for (std::size_t particle = 0; particle < 2; ++particle) {
    for (int coordinate = 0; coordinate < 3; ++coordinate) {
      require_close(first_state.positions[particle][coordinate],
                    second_state.positions[particle][coordinate],
                    kComparisonTolerance, "seeded position reproducibility");
      require_close(first_state.velocities[particle][coordinate],
                    second_state.velocities[particle][coordinate],
                    kComparisonTolerance, "seeded velocity reproducibility");
    }
  }
}

void test_seeded_stream_is_chunk_invariant() {
  auto make_system = []() {
    auto system = std::make_unique<OpenMM::System>();
    system->addParticle(2.0);
    return system;
  };
  auto config = reference_config();
  const std::vector<OpenMM::Vec3> point = zeros(1);
  const std::vector<OpenMM::Vec3> normal{OpenMM::Vec3(1.0, 0.0, 0.0)};
  rgpot::OpenMMSession monolithic(make_system(), config);
  rgpot::OpenMMSession chunked(make_system(), config);
  monolithic.setPositions(point);
  chunked.setPositions(point);
  monolithic.setVelocities(zeros(1));
  chunked.setVelocities(zeros(1));
  monolithic.stepLangevin(2.5, 3.0, config.time_step, point, normal, {}, {}, 2,
                          29);
  chunked.stepLangevin(2.5, 3.0, config.time_step, point, normal, {}, {}, 1,
                       29);
  chunked.stepLangevin(2.5, 3.0, config.time_step, point, normal, {}, {}, 1,
                       29);
  const auto monolithic_state = monolithic.state();
  const auto chunked_state = chunked.state();
  for (int coordinate = 0; coordinate < 3; ++coordinate) {
    require_close(chunked_state.positions[0][coordinate],
                  monolithic_state.positions[0][coordinate],
                  kComparisonTolerance, "seeded position chunk invariance");
    require_close(chunked_state.velocities[0][coordinate],
                  monolithic_state.velocities[0][coordinate],
                  kComparisonTolerance, "seeded velocity chunk invariance");
  }
}

void test_many_wall_crossings_remain_inside() {
  auto system = std::make_unique<OpenMM::System>();
  system->addParticle(1.0);
  auto config = reference_config();
  rgpot::OpenMMSession session(std::move(system), config);
  const std::vector<OpenMM::Vec3> point = zeros(1);
  const std::vector<OpenMM::Vec3> plane_normal{OpenMM::Vec3(1.0, 0.0, 0.0)};
  const std::vector<std::vector<OpenMM::Vec3>> wall_points{
      {OpenMM::Vec3(0.0, 1.0, 0.0)}, {OpenMM::Vec3(0.0, -1.0, 0.0)}};
  const std::vector<std::vector<OpenMM::Vec3>> wall_normals{
      {OpenMM::Vec3(0.0, 1.0, 0.0)}, {OpenMM::Vec3(0.0, -1.0, 0.0)}};
  session.setPositions(point);
  session.setVelocities({OpenMM::Vec3(0.0, 100.0, 0.0)});
  session.stepLangevin(2.5, 0.0, 1.0, point, plane_normal, wall_points,
                       wall_normals, 1, 9);
  const auto state = session.state();
  require(std::abs(state.positions[0][1]) <= 1.0 + kPlaneTolerance,
          "wall reflection pass limit returned an out-of-bounds state");
}

void test_off_plane_projection_refreshes_force() {
  auto system = std::make_unique<OpenMM::System>();
  system->addParticle(1.0);
  auto *force = new OpenMM::CustomExternalForce("k*x*y");
  force->addGlobalParameter("k", 100.0);
  force->addParticle(0);
  system->addForce(force);
  auto config = reference_config();
  rgpot::OpenMMSession session(std::move(system), config);
  const std::vector<OpenMM::Vec3> point = zeros(1);
  const std::vector<OpenMM::Vec3> normal{OpenMM::Vec3(1.0, 0.0, 0.0)};
  session.setPositions({OpenMM::Vec3(0.1, 0.0, 0.0)});
  session.setVelocities(zeros(1));
  session.stepLangevin(2.5, 0.0, 0.001, point, normal, {}, {}, 1, 3);
  const auto state = session.state();
  require_close(state.positions[0][1], 0.0, kComparisonTolerance,
                "stale off-plane force displaced tangent coordinate");
  require_close(state.velocities[0][1], 0.0, kComparisonTolerance,
                "stale off-plane force changed tangent velocity");
}

void test_openmm_distance_constraints_are_preserved() {
  auto system = std::make_unique<OpenMM::System>();
  system->addParticle(1.0);
  system->addParticle(1.0);
  system->addConstraint(0, 1, 0.2);
  auto config = reference_config();
  rgpot::OpenMMSession session(std::move(system), config);
  const std::vector<OpenMM::Vec3> point = zeros(2);
  const std::vector<OpenMM::Vec3> normal{OpenMM::Vec3(0.0, 0.0, 1.0),
                                         OpenMM::Vec3(0.0, 0.0, 1.0)};
  session.setPositions(
      {OpenMM::Vec3(-0.1, 0.0, 0.0), OpenMM::Vec3(0.1, 0.0, 0.0)});
  session.setVelocities(
      {OpenMM::Vec3(-1.0, 0.0, 0.0), OpenMM::Vec3(1.0, 0.0, 0.0)});
  session.stepLangevin(2.5, 0.0, 0.01, point, normal, {}, {}, 1, 5);
  const auto state = session.state();
  const OpenMM::Vec3 delta = state.positions[1] - state.positions[0];
  require_close(std::sqrt(delta.dot(delta)), 0.2, kComparisonTolerance,
                "serialized OpenMM distance constraint drifted");
}

} // namespace

int main(int argc, char **argv) {
  try {
    require(argc == 2, "one test case name is required");
    const std::string test = argv[1];
    if (test == "equipartition") {
      test_harmonic_equipartition_on_plane();
    } else if (test == "mean-force") {
      test_closed_form_constrained_mean_force();
    } else if (test == "wall-energy") {
      test_wall_reflection_conserves_kinetic_energy();
    } else if (test == "metric-seed") {
      test_mass_metric_projection_and_seed_reproducibility();
    } else if (test == "seed-chunking") {
      test_seeded_stream_is_chunk_invariant();
    } else if (test == "wall-multi-bounce") {
      test_many_wall_crossings_remain_inside();
    } else if (test == "off-plane-force") {
      test_off_plane_projection_refreshes_force();
    } else if (test == "openmm-constraints") {
      test_openmm_distance_constraints_are_preserved();
    } else {
      throw std::runtime_error("unknown test case");
    }
    std::cout << test << " passed\n";
    return EXIT_SUCCESS;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
