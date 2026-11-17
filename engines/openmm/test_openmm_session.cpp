// MIT License
// Copyright 2026 Rohit Goswami

#include "rgpot/ExternalPot/OpenMM.hpp"

#include <openmm/HarmonicBondForce.h>
#include <openmm/System.h>

#include <cmath>
#include <algorithm>
#include <iostream>
#include <memory>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

double dot(const OpenMM::Vec3 &value) {
  return value[0] * value[0] + value[1] * value[1] + value[2] * value[2];
}

std::unique_ptr<OpenMM::System> system() {
  auto out = std::make_unique<OpenMM::System>();
  out->addParticle(39.9);
  out->addParticle(39.9);
  auto *bond = new OpenMM::HarmonicBondForce();
  bond->addBond(0, 1, 0.35, 1000.0);
  out->addForce(bond);
  return out;
}

std::unique_ptr<OpenMM::System> free_system() {
  auto out = std::make_unique<OpenMM::System>();
  out->addParticle(39.9);
  out->addParticle(39.9);
  return out;
}

void near(double got, double expected, double scale, const char *what) {
  const double tolerance = 4096.0 * std::sqrt(std::numeric_limits<double>::epsilon()) *
                           std::max(1.0, scale);
  if (std::abs(got - expected) > tolerance) {
    throw std::runtime_error(std::string(what) + " differs: " +
                             std::to_string(got) + " vs " +
                             std::to_string(expected));
  }
}

} // namespace

int main() {
  try {
    rgpot::OpenMMSessionConfig config;
    config.platform = "Reference";
    config.time_step = 0.0005;
    config.friction = 1.0;
    config.temperature = 0.02585;
    config.random_seed = 19;
    rgpot::OpenMMSession session(system(), config);
    const std::vector<OpenMM::Vec3> positions{
        OpenMM::Vec3(0.0, 0.0, 0.0), OpenMM::Vec3(0.4, 0.0, 0.0)};
    session.setPositions(positions);
    session.setVelocities(
        {OpenMM::Vec3(0.0, 0.0, 0.0), OpenMM::Vec3(0.0, 0.0, 0.0)});
    const auto initial = session.state();
    near(initial.potential_energy, 1.25, 1.25, "harmonic energy");
    near(initial.forces[1][0], -50.0, 50.0, "harmonic force");
    rgpot::OpenMMSession sample_session(free_system(), config);
    sample_session.setPositions(
        {OpenMM::Vec3(0.0, 0.0, 0.0), OpenMM::Vec3(0.0, 0.0, 0.0)});
    sample_session.setVelocities(
        {OpenMM::Vec3(0.0, 0.0, 0.0), OpenMM::Vec3(0.0, 0.0, 0.0)});
    sample_session.stepLangevin(
        OpenMM::Vec3(0.0, 0.0, 0.0), OpenMM::Vec3(1.0, 0.0, 0.0),
        {OpenMM::Vec3(0.0, 0.0, 0.0)},
        {OpenMM::Vec3(0.0, 0.0, -1.0)}, 2'000);
    const double tolerance =
        4096.0 * std::sqrt(std::numeric_limits<double>::epsilon());
    std::vector<double> temperatures;
    int reflections = 0;
    for (int sample = 0; sample < 256; ++sample) {
      reflections += sample_session.stepLangevin(
          OpenMM::Vec3(0.0, 0.0, 0.0), OpenMM::Vec3(1.0, 0.0, 0.0),
          {OpenMM::Vec3(0.0, 0.0, 0.0)},
          {OpenMM::Vec3(0.0, 0.0, -1.0)}, 20);
      const auto sampled = sample_session.state();
      temperatures.push_back(
          (39.9 * dot(sampled.velocities[0]) +
           39.9 * dot(sampled.velocities[1])) /
          4.0);
      for (const auto &position : sampled.positions) {
        if (position[2] < -tolerance) {
          throw std::runtime_error("OpenMM wall was crossed");
        }
        if (std::abs(position[0]) > tolerance) {
          throw std::runtime_error("OpenMM plane was crossed");
        }
      }
    }
    if (reflections == 0) {
      throw std::runtime_error("OpenMM hard wall did not reflect");
    }
    const double temperature =
        std::accumulate(temperatures.begin(), temperatures.end(), 0.0) /
        static_cast<double>(temperatures.size());
    const double variance =
        std::accumulate(temperatures.begin(), temperatures.end(), 0.0,
                        [temperature](double sum, double value) {
                          const double delta = value - temperature;
                          return sum + delta * delta;
                        }) /
        static_cast<double>(temperatures.size() - 1);
    if (std::abs(temperature - config.temperature) >
        6.0 * std::sqrt(variance / temperatures.size()) + tolerance) {
      throw std::runtime_error(
          "OpenMM kinetic temperature is outside error: mean=" +
          std::to_string(temperature) +
          " stderr=" + std::to_string(std::sqrt(variance / temperatures.size())));
    }
    session.setPositions(positions);
    session.setVelocities(
        {OpenMM::Vec3(0.01, 0.0, 0.0), OpenMM::Vec3(-0.01, 0.0, 0.0)});
    const auto nve_before = session.state();
    session.stepNVE(200);
    const auto nve_after = session.state();
    const double nve_scale =
        std::max(1.0, std::abs(nve_before.potential_energy +
                                nve_before.kinetic_energy));
    if (std::abs((nve_after.potential_energy + nve_after.kinetic_energy) -
                 (nve_before.potential_energy + nve_before.kinetic_energy)) >
        nve_scale / 100.0) {
      throw std::runtime_error("OpenMM NVE energy drift exceeded one percent");
    }
    std::cout << "OpenMM session energy, plane, wall, temperature, and NVE "
                 "checks passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
