// MIT License
// Copyright 2026 Rohit Goswami

#include "rgpot/ExternalPot/external_potentials.h"
#include "lj_eval.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr double kKjPerEv = 96.48533212331002;
constexpr double kComparisonTolerance =
    450000.0 * std::numeric_limits<double>::epsilon();
constexpr double kNblibForceRelativeTolerance = 0.08;

void require(bool condition, const char *message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

void require_near(double actual, double expected, double scale,
                  const char *message) {
  require(std::abs(actual - expected) <= scale, message);
}

std::string argon_topology() {
  constexpr double epsilon_ev = 0.01032;
  constexpr double sigma_a = 3.4;
  const double epsilon_kj = epsilon_ev * kKjPerEv;
  const double sigma_nm = sigma_a * 0.1;
  const double c6 = 4.0 * epsilon_kj * std::pow(sigma_nm, 6);
  const double c12 = 4.0 * epsilon_kj * std::pow(sigma_nm, 12);
  return "nblib 1\ncutoff 0.8\ncoulomb none\natomtype Ar 39.948 " +
         std::to_string(c6) + " " + std::to_string(c12) +
         "\nmolecule argon 2\natom Ar Ar 0\nend\n";
}

void run(const char *engine) {
  require(engine != nullptr && engine[0] != '\0', "nblib engine path is empty");
  require(setenv("RGPOT_NBLIB_ENGINE", engine, 1) == 0,
          "could not set nblib engine path");
  char error[1024] = {};
  const std::string topology = argon_topology();
  rgpot_md_session *session = rgpot_md_session_create(
      "nblib", nullptr, topology.data(), topology.size(), nullptr, error,
      sizeof(error));
  if (session == nullptr) {
    throw std::runtime_error(error);
  }
  struct SessionGuard {
    rgpot_md_session *session;
    ~SessionGuard() { rgpot_md_session_destroy(session); }
  } guard{session};

  const std::array<double, 9> box{20.0, 0.0, 0.0, 0.0, 20.0,
                                  0.0, 0.0, 0.0, 20.0};
  const std::array<double, 6> positions{0.0, 0.0, 0.0, 4.2, 0.0, 0.0};
  const std::array<double, 6> velocities{0.0, 0.1, 0.0, 0.0, -0.1, 0.0};
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

  std::array<double, 6> nblib_force{};
  double nblib_energy = 0.0;
  require(rgpot_md_session_get_state(session, nullptr, nullptr,
                                     nblib_force.data(), nblib_force.size(),
                                     &nblib_energy, nullptr, error,
                                     sizeof(error)) == 0,
          error);

  const std::array<int, 2> atomic_numbers{18, 18};
  std::array<double, 6> rgpot_force{};
  double rgpot_energy = 0.0;
  int has_stress = 0;
  std::array<double, 9> stress{};
  require(rgpot_lj_eval(positions.data(), atomic_numbers.data(),
                        atomic_numbers.size(), box.data(), 0.01032, 8.0, 3.4,
                        rgpot_force.data(), &rgpot_energy, &has_stress,
                        stress.data()) == 0,
          "rgpot LJ evaluation failed");
  for (size_t component = 0; component < nblib_force.size(); ++component) {
    const double force_scale = kNblibForceRelativeTolerance *
        std::max(kComparisonTolerance, std::abs(rgpot_force[component]));
    require_near(nblib_force[component], rgpot_force[component], force_scale,
                 "nblib and rgpot LJ forces differ");
  }

  const std::array<double, 6> plane_point = positions;
  const std::array<double, 6> plane_normal{1.0, 0.0, 0.0, -1.0, 0.0, 0.0};
  std::array<double, 6> first_positions{};
  std::array<double, 6> second_positions{};
  std::array<double, 6> first_velocities{};
  std::array<double, 6> second_velocities{};
  uint64_t first_hits = 0;
  uint64_t second_hits = 0;
  const auto advance = [&](std::array<double, 6> &out_positions,
                           std::array<double, 6> &out_velocities,
                           uint64_t &hits) {
    require(rgpot_md_session_set_positions(session, positions.data(),
                                            positions.size(), error,
                                            sizeof(error)) == 0,
            error);
    require(rgpot_md_session_set_velocities(session, velocities.data(),
                                             velocities.size(), error,
                                             sizeof(error)) == 0,
            error);
    require(rgpot_md_session_step_langevin(
                session, 0.025, 1.0, 0.001, plane_point.data(),
                plane_normal.data(), nullptr, nullptr, 0, positions.size(), 4,
                17, &hits, error, sizeof(error)) == 0,
            error);
    require(rgpot_md_session_get_state(session, out_positions.data(),
                                       out_velocities.data(), nullptr,
                                       out_positions.size(), nullptr, nullptr,
                                       error, sizeof(error)) == 0,
            error);
  };
  advance(first_positions, first_velocities, first_hits);
  advance(second_positions, second_velocities, second_hits);
  require(first_hits == second_hits &&
              first_positions == second_positions &&
              first_velocities == second_velocities,
          "nblib Langevin seed is not deterministic");
}

} // namespace

int main(int argc, char **argv) {
  try {
    require(argc == 2, "usage: test_nblib_session <engine>");
    run(argv[1]);
    return 0;
  } catch (const std::exception &exception) {
    std::cerr << exception.what() << '\n';
    return 1;
  }
}
