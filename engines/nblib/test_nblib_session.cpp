// MIT License
// Copyright 2026 Rohit Goswami

#include "rgpot/ExternalPot/external_potentials.h"
#include "rgpot/ExternalPot/md_units.hpp"
#include "lj_eval.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using rgpot::md::kEvAPerKjNm;
using rgpot::md::kEvPerKj;
using rgpot::md::kKjPerEv;

// Double-precision GROMACS agrees with the closed forms to rounding; a few
// thousand ulp covers the sums and the table-free kernels.
constexpr double kUlps = 8192.0;

std::string full_number(double value) {
  std::ostringstream out;
  out.precision(std::numeric_limits<double>::max_digits10);
  out << value;
  return out.str();
}

void require(bool condition, const char *message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

void require_near(double actual, double expected, const std::string &message) {
  const double magnitude = std::max(std::abs(actual), std::abs(expected));
  const double tolerance = kUlps * std::numeric_limits<double>::epsilon() *
                           std::max(1.0, magnitude);
  if (!(std::abs(actual - expected) <= tolerance)) {
    throw std::runtime_error(message + ": got " + full_number(actual) +
                             " expected " + full_number(expected));
  }
}

// Potential-shifted 12-6 pair in kJ/mol and kJ/mol/nm, the form nblib
// evaluates inside the cutoff.
void shifted_pair(double c6, double c12, double cutoff, double r, double &energy,
                  double &dvdr) {
  const double inv_r6 = std::pow(r, -6);
  const double shift = c6 / std::pow(cutoff, 6) - c12 / std::pow(cutoff, 12);
  energy = c12 * inv_r6 * inv_r6 - c6 * inv_r6 + shift;
  dvdr = (-12.0 * c12 * inv_r6 * inv_r6 + 6.0 * c6 * inv_r6) / r;
}

constexpr double kEpsilonEv = 0.01032;
constexpr double kSigmaAngstrom = 3.4;
constexpr double kCutoffNm = 0.8;

double argon_c6() {
  return 4.0 * kEpsilonEv * kKjPerEv * std::pow(kSigmaAngstrom * 0.1, 6);
}

double argon_c12() {
  return 4.0 * kEpsilonEv * kKjPerEv * std::pow(kSigmaAngstrom * 0.1, 12);
}

// std::to_string keeps six decimals, which turns c12 = 1.0e-5 kJ/mol nm^12
// into 0.000010; the full digits keep the argon well depth.
std::string argon_topology() {
  return "nblib 1\ncutoff " + full_number(kCutoffNm) +
         "\ncoulomb none\natomtype Ar 39.948 " + full_number(argon_c6()) + " " +
         full_number(argon_c12()) + "\nmolecule argon 2\natom Ar Ar 0\nend\n";
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

  // The pair sits at 4.2 Angstrom, near the well, so a wrong c12 shows in
  // the force at once: c12 = 0.000010 puts the repulsion off by orders.
  double energy_kj = 0.0;
  double dvdr_kj = 0.0;
  shifted_pair(argon_c6(), argon_c12(), kCutoffNm, 0.42, energy_kj, dvdr_kj);
  require_near(nblib_energy, energy_kj * kEvPerKj, "nblib argon energy");
  require_near(nblib_force[0], dvdr_kj * kEvAPerKjNm, "nblib argon force on 0");
  require_near(nblib_force[3], -dvdr_kj * kEvAPerKjNm, "nblib argon force on 1");
  for (const size_t component : {1u, 2u, 4u, 5u}) {
    require_near(nblib_force[component], 0.0, "nblib argon transverse force");
  }

  const std::array<int, 2> atomic_numbers{18, 18};
  std::array<double, 6> rgpot_force{};
  double rgpot_energy = 0.0;
  int has_stress = 0;
  std::array<double, 9> stress{};
  require(rgpot_lj_eval(positions.data(), atomic_numbers.data(),
                        atomic_numbers.size(), box.data(), kEpsilonEv, 8.0,
                        kSigmaAngstrom, rgpot_force.data(), &rgpot_energy,
                        &has_stress, stress.data()) == 0,
          "rgpot LJ evaluation failed");
  for (size_t component = 0; component < nblib_force.size(); ++component) {
    require_near(nblib_force[component], rgpot_force[component],
                 "nblib and rgpot LJ forces differ");
  }

  std::array<double, 9> box_back{};
  require(rgpot_md_session_get_periodic_box(session, box_back.data(), error,
                                            sizeof(error)) == 0,
          error);
  for (size_t index = 0; index < box.size(); ++index) {
    require_near(box_back[index], box[index], "periodic box round trip");
  }

  // nblib has no neighbour callbacks and no wall map of its own: the
  // neighbour slots answer with the typed status, not with an error.
  rgpot_nlist_request request{};
  request.bytes = sizeof(request);
  size_t count = 0;
  size_t needed = 0;
  require(rgpot_md_session_get_neighbors(session, &request, nullptr, &count,
                                         &needed, error, sizeof(error)) ==
              RGPOT_MD_NLIST_UNSUPPORTED,
          "nblib get_neighbors must report RGPOT_MD_NLIST_UNSUPPORTED");
  require(std::string(error).find("neighbor") != std::string::npos,
          "the unsupported neighbour status names the missing slot");
  rgpot_nlist_soa pairs{};
  require(rgpot_md_session_set_neighbors(session, &request, &pairs, 0, error,
                                         sizeof(error)) ==
              RGPOT_MD_NLIST_UNSUPPORTED,
          "nblib set_neighbors must report RGPOT_MD_NLIST_UNSUPPORTED");

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

// This atom has no interaction, so the plane multiplier is 0.
// The buffer starts at -1, and every slot must hold that zero.
void check_recorded_multiplier(const char *engine) {
  require(setenv("RGPOT_NBLIB_ENGINE", engine, 1) == 0,
          "could not set nblib engine path");
  const std::string topology =
      "nblib 1\ncutoff 0.4\ncoulomb none\natomtype X 1 0 0\n"
      "molecule one 1\natom A X 0\nend\n";
  char error[1024] = {};
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
  const std::array<double, 3> positions{1.0, 2.0, 3.0};
  const std::array<double, 3> velocities{};
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

  const double plane_point[3] = {};
  const double plane_normal[3] = {1.0, 0.0, 0.0};
  double recorded[2];
  recorded[0] = -1.0;
  recorded[1] = -1.0;
  rgpot_md_langevin_request request{};
  request.struct_size = sizeof(request);
  request.kT = 0.025;
  request.friction = 1.0;
  request.time_step = 0.001;
  request.steps = 2;
  request.seed = 1;
  request.n3 = positions.size();
  request.plane_point = plane_point;
  request.plane_normal = plane_normal;
  request.wall_mode = RGPOT_MD_WALL_REFLECT;
  rgpot_md_langevin_result result{};
  result.struct_size = sizeof(result);
  result.constraint_force = recorded;
  result.constraint_force_len = 2;
  require(rgpot_md_session_run_langevin(session, &request, &result, error,
                                        sizeof(error)) == 0,
          error);
  for (const double value : recorded) {
    require_near(value, 0.0, "nblib recording constraint force");
  }
}

} // namespace

int main(int argc, char **argv) {
  try {
    require(argc == 2, "usage: test_nblib_session <engine>");
    run(argv[1]);
    check_recorded_multiplier(argv[1]);
    return 0;
  } catch (const std::exception &exception) {
    std::cerr << exception.what() << '\n';
    return 1;
  }
}
