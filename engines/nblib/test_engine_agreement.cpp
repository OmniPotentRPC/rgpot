// MIT License
// Copyright 2026 Rohit Goswami

// Cross-engine checks for the shared MD session ABI. Argon Lennard-Jones
// energies and forces are compared with the potential-shifted 12-6 formula
// GROMACS nblib and the OpenMM session both evaluate. A two-particle
// harmonic dimer is held on one hyperplane so the transverse motion is
// two-dimensional; the kinetic temperature and the mean constraint force
// are compared between the engines and with the quadrature of that plane.

#include "rgpot/ExternalPot/external_potentials.h"
#include "rgpot/ExternalPot/md_units.hpp"

#include <openmm/CustomNonbondedForce.h>
#include <openmm/HarmonicBondForce.h>
#include <openmm/System.h>
#include <openmm/serialization/XmlSerializer.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using rgpot::md::kEvAPerKjNm;
using rgpot::md::kEvPerKj;
using rgpot::md::kKjPerEv;

constexpr int kSigmaMultiple = 6;
constexpr double kRoundoff = 8192.0;

void require(bool condition, const std::string &message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

std::string full_number(double value) {
  std::ostringstream out;
  out.precision(std::numeric_limits<double>::max_digits10);
  out << value;
  return out.str();
}

double roundoff_scale(double magnitude) {
  return kRoundoff * std::numeric_limits<double>::epsilon() *
         std::max(1.0, std::abs(magnitude));
}

void require_near(double actual, double expected, const std::string &message) {
  const double scale = std::max(std::abs(actual), std::abs(expected));
  const double tolerance = roundoff_scale(scale);
  if (!(std::abs(actual - expected) <= tolerance)) {
    throw std::runtime_error(message + ": got " + full_number(actual) +
                             " expected " + full_number(expected));
  }
}

void require_finite(double value, const std::string &message) {
  require(std::isfinite(value), message);
}

struct Moments {
  double mean = 0.0;
  double stderr = 0.0;
};

Moments moments_of(const std::vector<double> &samples) {
  require(samples.size() > 1, "a mean needs more than one sample");
  double sum = 0.0;
  for (const double value : samples) {
    require_finite(value, "a sample is not finite");
    sum += value;
  }
  const double mean = sum / static_cast<double>(samples.size());
  double scatter = 0.0;
  for (const double value : samples) {
    const double delta = value - mean;
    scatter += delta * delta;
  }
  const double variance =
      scatter / static_cast<double>(samples.size() - 1);
  return {mean, std::sqrt(variance / static_cast<double>(samples.size()))};
}

void require_within_se(double actual, double expected, double stderr,
                       const std::string &message) {
  require_finite(actual, message);
  require_finite(expected, message);
  require_finite(stderr, message);
  require(stderr > 0.0, message + " has no sample scatter");
  const double allowance = static_cast<double>(kSigmaMultiple) * stderr +
                           roundoff_scale(std::max(std::abs(actual),
                                                   std::abs(expected)));
  if (!(std::abs(actual - expected) <= allowance)) {
    throw std::runtime_error(message + ": got " + full_number(actual) +
                             " expected " + full_number(expected) +
                             " stderr " + full_number(stderr));
  }
}

struct OwnedSession {
  rgpot_md_session *session = nullptr;
  explicit OwnedSession(rgpot_md_session *value) : session(value) {}
  OwnedSession(const OwnedSession &) = delete;
  OwnedSession &operator=(const OwnedSession &) = delete;
  ~OwnedSession() { rgpot_md_session_destroy(session); }
};

OwnedSession open_session(const char *engine, const char *library,
                          const std::string &spec, const char *options) {
  char error[1024] = {};
  rgpot_md_session *session = rgpot_md_session_create(
      engine, library, spec.data(), spec.size(), options, error, sizeof(error));
  if (session == nullptr) {
    throw std::runtime_error(std::string(engine) + ": " + error);
  }
  return OwnedSession(session);
}

void call_ok(int status, const char *error, const std::string &what) {
  if (status != 0) {
    throw std::runtime_error(what + ": " + error);
  }
}

struct Snapshot {
  std::vector<double> positions;
  std::vector<double> velocities;
  std::vector<double> forces;
  double potential = 0.0;
  double kinetic = 0.0;
};

Snapshot read_state(rgpot_md_session *session, std::size_t n3) {
  Snapshot out;
  out.positions.resize(n3);
  out.velocities.resize(n3);
  out.forces.resize(n3);
  char error[1024] = {};
  call_ok(rgpot_md_session_get_state(session, out.positions.data(),
                                     out.velocities.data(), out.forces.data(),
                                     n3, &out.potential, &out.kinetic, error,
                                     sizeof(error)),
          error, "get_state");
  require_finite(out.potential, "potential energy is not finite");
  require_finite(out.kinetic, "kinetic energy is not finite");
  for (std::size_t i = 0; i < n3; ++i) {
    require_finite(out.positions[i], "a position is not finite");
    require_finite(out.velocities[i], "a velocity is not finite");
    require_finite(out.forces[i], "a force is not finite");
  }
  return out;
}

void set_box(rgpot_md_session *session, const std::vector<double> &box) {
  char error[1024] = {};
  call_ok(rgpot_md_session_set_periodic_box(session, box.data(), error,
                                            sizeof(error)),
          error, "set_periodic_box");
}

void set_positions(rgpot_md_session *session,
                   const std::vector<double> &positions) {
  char error[1024] = {};
  call_ok(rgpot_md_session_set_positions(session, positions.data(),
                                         positions.size(), error, sizeof(error)),
          error, "set_positions");
}

void set_velocities(rgpot_md_session *session,
                    const std::vector<double> &velocities) {
  char error[1024] = {};
  call_ok(rgpot_md_session_set_velocities(session, velocities.data(),
                                          velocities.size(), error,
                                          sizeof(error)),
          error, "set_velocities");
}

void initialize(rgpot_md_session *session) {
  char error[1024] = {};
  call_ok(rgpot_md_session_initialize(session, error, sizeof(error)), error,
          "initialize");
}

std::uint64_t step_langevin(rgpot_md_session *session, double kT,
                            double friction, double dt,
                            const std::vector<double> &point,
                            const std::vector<double> &normal,
                            const std::vector<double> &wall_points,
                            const std::vector<double> &wall_normals,
                            std::size_t steps, std::uint64_t seed) {
  char error[1024] = {};
  std::uint64_t hits = 0;
  const std::size_t n3 = point.size();
  const std::size_t n_walls = n3 == 0 ? 0 : wall_points.size() / n3;
  call_ok(rgpot_md_session_step_langevin(
              session, kT, friction, dt, point.data(), normal.data(),
              n_walls == 0 ? nullptr : wall_points.data(),
              n_walls == 0 ? nullptr : wall_normals.data(), n_walls, n3, steps,
              seed, &hits, error, sizeof(error)),
          error, "step_langevin");
  return hits;
}

// Shifted 12-6 in kJ/mol and kJ/mol/nm. The shift is the GROMACS
// potential-shift modifier: V(r) - V(rc) inside the cutoff, and zero outside.
double minimum_image(double delta, double length) {
  return delta - length * std::nearbyint(delta / length);
}

void shifted_lj(const std::vector<double> &positions_nm, double c6, double c12,
                double cutoff_nm, double box_nm, double &energy_kj,
                std::vector<double> &forces_kj_nm) {
  const std::size_t atoms = positions_nm.size() / 3;
  energy_kj = 0.0;
  forces_kj_nm.assign(positions_nm.size(), 0.0);
  const double rc2 = cutoff_nm * cutoff_nm;
  const double shift = c6 / std::pow(cutoff_nm, 6) - c12 / std::pow(cutoff_nm, 12);
  for (std::size_t i = 0; i < atoms; ++i) {
    for (std::size_t j = i + 1; j < atoms; ++j) {
      const double dx = minimum_image(positions_nm[3 * j] - positions_nm[3 * i], box_nm);
      const double dy =
          minimum_image(positions_nm[3 * j + 1] - positions_nm[3 * i + 1], box_nm);
      const double dz =
          minimum_image(positions_nm[3 * j + 2] - positions_nm[3 * i + 2], box_nm);
      const double r2 = dx * dx + dy * dy + dz * dz;
      if (!(r2 < rc2) || !(r2 > 0.0)) {
        continue;
      }
      const double r = std::sqrt(r2);
      const double inv_r2 = 1.0 / r2;
      const double inv_r6 = inv_r2 * inv_r2 * inv_r2;
      const double inv_r12 = inv_r6 * inv_r6;
      energy_kj += c12 * inv_r12 - c6 * inv_r6 + shift;
      const double dV_dr = -12.0 * c12 * inv_r12 / r + 6.0 * c6 * inv_r6 / r;
      const double scale = dV_dr / r;
      forces_kj_nm[3 * i] += scale * dx;
      forces_kj_nm[3 * i + 1] += scale * dy;
      forces_kj_nm[3 * i + 2] += scale * dz;
      forces_kj_nm[3 * j] -= scale * dx;
      forces_kj_nm[3 * j + 1] -= scale * dy;
      forces_kj_nm[3 * j + 2] -= scale * dz;
    }
  }
}

std::string argon_topology(double c6, double c12, double cutoff_nm,
                           int atoms) {
  std::string text = "nblib 1\ncutoff " + full_number(cutoff_nm) +
                     "\ncoulomb none\natomtype Ar 39.948 " + full_number(c6) +
                     " " + full_number(c12) + "\nmolecule argon 1\n";
  for (int atom = 0; atom < atoms; ++atom) {
    text += "atom Ar" + std::to_string(atom) + " Ar 0\n";
  }
  text += "end\n";
  return text;
}

std::string argon_openmm_xml(double epsilon_kj, double sigma_nm,
                             double cutoff_nm, double box_nm, int atoms) {
  OpenMM::System system;
  for (int atom = 0; atom < atoms; ++atom) {
    system.addParticle(39.948);
  }
  system.setDefaultPeriodicBoxVectors(OpenMM::Vec3(box_nm, 0.0, 0.0),
                                      OpenMM::Vec3(0.0, box_nm, 0.0),
                                      OpenMM::Vec3(0.0, 0.0, box_nm));
  // Same shifted 12-6 the OpenMM Lennard-Jones engine evaluates.
  const char *expression =
      "4*sqrt(eps1*eps2)*((sqrt(sig1*sig2)/r)^12-(sqrt(sig1*sig2)/r)^6)"
      "-4*sqrt(eps1*eps2)*((sqrt(sig1*sig2)/rc)^12-(sqrt(sig1*sig2)/rc)^6)";
  auto *force = new OpenMM::CustomNonbondedForce(expression);
  force->addGlobalParameter("rc", cutoff_nm);
  force->addPerParticleParameter("eps");
  force->addPerParticleParameter("sig");
  for (int atom = 0; atom < atoms; ++atom) {
    force->addParticle({epsilon_kj, sigma_nm});
  }
  force->setNonbondedMethod(OpenMM::CustomNonbondedForce::CutoffPeriodic);
  force->setCutoffDistance(cutoff_nm);
  system.addForce(force);
  std::ostringstream xml;
  OpenMM::XmlSerializer::serialize<OpenMM::System>(&system, "System", xml);
  return xml.str();
}

void compare_snapshot_to_lj(const Snapshot &state, double c6, double c12,
                            double cutoff_nm, double box_nm,
                            const std::string &label) {
  std::vector<double> positions_nm(state.positions.size());
  for (std::size_t i = 0; i < state.positions.size(); ++i) {
    positions_nm[i] = state.positions[i] * rgpot::md::kNmPerAngstrom;
  }
  double energy_kj = 0.0;
  std::vector<double> forces_kj;
  shifted_lj(positions_nm, c6, c12, cutoff_nm, box_nm, energy_kj, forces_kj);
  require_near(state.potential, energy_kj * kEvPerKj, label + " energy");
  for (std::size_t i = 0; i < forces_kj.size(); ++i) {
    require_near(state.forces[i], forces_kj[i] * kEvAPerKjNm,
                 label + " force " + std::to_string(i));
  }
}

void prepare(rgpot_md_session *session, const std::vector<double> &box,
             const std::vector<double> &positions,
             const std::vector<double> &velocities) {
  set_box(session, box);
  set_positions(session, positions);
  set_velocities(session, velocities);
  initialize(session);
}

void check_argon(const char *nblib_library, const char *openmm_library) {
  constexpr double epsilon_ev = 0.01032;
  constexpr double sigma_angstrom = 3.4;
  constexpr double cutoff_angstrom = 8.0;
  constexpr double box_angstrom = 30.0;
  const double epsilon_kj = epsilon_ev * kKjPerEv;
  const double sigma_nm = sigma_angstrom * rgpot::md::kNmPerAngstrom;
  const double cutoff_nm = cutoff_angstrom * rgpot::md::kNmPerAngstrom;
  const double box_nm = box_angstrom * rgpot::md::kNmPerAngstrom;
  const double c6 = 4.0 * epsilon_kj * std::pow(sigma_nm, 6);
  const double c12 = 4.0 * epsilon_kj * std::pow(sigma_nm, 12);
  constexpr int atoms = 3;
  const std::vector<double> box{box_angstrom, 0.0, 0.0, 0.0, box_angstrom,
                                0.0,          0.0, 0.0, box_angstrom};
  // Two distances inside the cutoff and one pair outside it.
  const std::vector<double> inside{0.0, 0.0, 0.0, 4.2, 0.0, 0.0, 0.0, 5.1, 0.3};
  const std::vector<double> outside{0.0, 0.0, 0.0, 12.0, 0.0,
                                    0.0, 0.0, 12.0, 0.0};
  // Direct separation is past the cutoff. The nearest periodic image is not.
  const std::vector<double> image{1.0, 2.0, 3.0, 28.0, 2.4, 3.2, 1.5, 14.0, 3.0};
  const std::vector<double> zero(static_cast<std::size_t>(3 * atoms), 0.0);
  const std::string topology = argon_topology(c6, c12, cutoff_nm, atoms);
  const std::string xml =
      argon_openmm_xml(epsilon_kj, sigma_nm, cutoff_nm, box_nm, atoms);

  OwnedSession nblib =
      open_session("nblib", nblib_library, topology, nullptr);
  OwnedSession openmm =
      open_session("openmm", openmm_library, xml, "Reference");
  for (const auto &geometry : {inside, outside, image, inside}) {
    prepare(nblib.session, box, geometry, zero);
    prepare(openmm.session, box, geometry, zero);
    const Snapshot nblib_state = read_state(nblib.session, geometry.size());
    const Snapshot openmm_state = read_state(openmm.session, geometry.size());
    compare_snapshot_to_lj(nblib_state, c6, c12, cutoff_nm, box_nm, "nblib argon");
    compare_snapshot_to_lj(openmm_state, c6, c12, cutoff_nm, box_nm,
                           "OpenMM argon");
    require_near(nblib_state.potential, openmm_state.potential,
                 "argon energy nblib vs OpenMM");
    for (std::size_t i = 0; i < geometry.size(); ++i) {
      require_near(nblib_state.forces[i], openmm_state.forces[i],
                   "argon force nblib vs OpenMM");
    }
  }
}

std::string dimer_topology(double k_kj, double r0_nm, double cutoff_nm) {
  return "nblib 1\ncutoff " + full_number(cutoff_nm) +
         "\ncoulomb none\natomtype A 39.948 0 0\nmolecule dimer 1\n"
         "atom A1 A 0\natom A2 A 0\nbond harmonic A1 A2 " +
         full_number(k_kj) + " " + full_number(r0_nm) + "\nend\n";
}

std::string dimer_openmm_xml(double k_kj, double r0_nm, double box_nm) {
  OpenMM::System system;
  system.addParticle(39.948);
  system.addParticle(39.948);
  system.setDefaultPeriodicBoxVectors(OpenMM::Vec3(box_nm, 0.0, 0.0),
                                      OpenMM::Vec3(0.0, box_nm, 0.0),
                                      OpenMM::Vec3(0.0, 0.0, box_nm));
  auto *bond = new OpenMM::HarmonicBondForce();
  bond->addBond(0, 1, r0_nm, k_kj);
  system.addForce(bond);
  std::ostringstream xml;
  OpenMM::XmlSerializer::serialize<OpenMM::System>(&system, "System", xml);
  return xml.str();
}

// lambda = k (r - r0) xi / r for the equal-mass plane x1 - x2 + xi = 0.
// The transverse coordinates are integrated in polar radius.
double quadrature_mean_lambda(double k_kj, double r0_nm, double xi_nm,
                              double kT_kj) {
  const auto simpson = [&](int panels) {
    require(panels % 2 == 0, "Simpson quadrature needs an even panel count");
    const double thermal = std::sqrt(kT_kj / k_kj);
    const double rho_max = std::abs(xi_nm) + thermal * 16.0;
    const double d_rho = rho_max / static_cast<double>(panels);
    double weighted = 0.0;
    double weight = 0.0;
    for (int panel = 0; panel <= panels; ++panel) {
      const double rho = d_rho * static_cast<double>(panel);
      const double r = std::sqrt(xi_nm * xi_nm + rho * rho);
      const double stretch = r - r0_nm;
      const double boltzmann =
          std::exp(-(0.5 * k_kj * stretch * stretch) / kT_kj) * rho;
      double coeff = 2.0;
      if (panel == 0 || panel == panels) {
        coeff = 1.0;
      } else if (panel % 2 == 1) {
        coeff = 4.0;
      }
      const double sample = coeff * boltzmann;
      weight += sample;
      weighted += sample * (k_kj * stretch * xi_nm / r);
    }
    require(weight > 0.0, "constraint quadrature weight vanished");
    return weighted / weight;
  };
  const double coarse = simpson(8000);
  const double fine = simpson(16000);
  require_near(coarse, fine, "constraint quadrature is not settled");
  return fine;
}

double constraint_lambda(const std::vector<double> &forces,
                         const std::vector<double> &masses,
                         const std::vector<double> &normal) {
  double along = 0.0;
  double weighted = 0.0;
  for (std::size_t atom = 0; atom < masses.size(); ++atom) {
    for (int axis = 0; axis < 3; ++axis) {
      const std::size_t k = static_cast<std::size_t>(3 * atom + axis);
      along += normal[k] * forces[k] / masses[atom];
      weighted += normal[k] * normal[k] / masses[atom];
    }
  }
  require(weighted > 0.0, "constraint normal has no inverse-mass weight");
  return along / weighted;
}

void check_constrained_dimer(const char *nblib_library,
                             const char *openmm_library) {
  constexpr double k_kj = 800.0;
  constexpr double r0_nm = 0.30;
  constexpr double xi_nm = 0.36;
  constexpr double cutoff_nm = 0.8;
  constexpr double box_nm = 3.0;
  constexpr double kT_ev = 0.025851;
  constexpr double friction = 8.0;
  constexpr double dt = 0.001;
  constexpr std::size_t burn = 4000;
  constexpr std::size_t stride = 500;
  constexpr std::size_t samples = 48;
  constexpr double mass = 39.948;
  const double kT_kj = kT_ev * kKjPerEv;
  const double box_angstrom = box_nm / rgpot::md::kNmPerAngstrom;
  const double xi_angstrom = xi_nm / rgpot::md::kNmPerAngstrom;
  const std::vector<double> box{box_angstrom, 0.0, 0.0, 0.0, box_angstrom,
                                0.0,          0.0, 0.0, box_angstrom};
  const std::vector<double> positions{0.0, 0.0, 0.0, xi_angstrom, 0.0, 0.0};
  const std::vector<double> zero(6, 0.0);
  const std::vector<double> normal{1.0, 0.0, 0.0, -1.0, 0.0, 0.0};
  const std::vector<double> masses{mass, mass};
  const double analytic =
      quadrature_mean_lambda(k_kj, r0_nm, xi_nm, kT_kj) * kEvAPerKjNm;

  const auto run = [&](const char *engine, const char *library,
                       const std::string &spec, std::uint64_t seed) {
    OwnedSession session =
        open_session(engine, library, spec, engine[0] == 'o' ? "Reference" : nullptr);
    prepare(session.session, box, positions, zero);
    const Snapshot initial = read_state(session.session, 6);
    const double r_nm = xi_nm;
    const double energy_kj = 0.5 * k_kj * (r_nm - r0_nm) * (r_nm - r0_nm);
    const double force_kj = k_kj * (r_nm - r0_nm);
    require_near(initial.potential, energy_kj * kEvPerKj,
                 std::string(engine) + " harmonic energy");
    require_near(initial.forces[0], force_kj * kEvAPerKjNm,
                 std::string(engine) + " harmonic force");
    require_near(initial.forces[3], -force_kj * kEvAPerKjNm,
                 std::string(engine) + " harmonic reaction");

    step_langevin(session.session, kT_ev, friction, dt, positions, normal, {},
                  {}, burn, seed);
    std::vector<double> temperatures;
    std::vector<double> lambdas;
    temperatures.reserve(samples);
    lambdas.reserve(samples);
    for (std::size_t sample = 0; sample < samples; ++sample) {
      step_langevin(session.session, kT_ev, friction, dt, positions, normal, {},
                    {}, stride, seed + 1 + sample);
      const Snapshot state = read_state(session.session, 6);
      const double separation =
          state.positions[0] - state.positions[3];
      require_near(separation, -xi_angstrom,
                   std::string(engine) + " left the hyperplane");
      // Five degrees of freedom: six Cartesian coordinates, one constraint.
      temperatures.push_back(2.0 * state.kinetic / 5.0);
      lambdas.push_back(constraint_lambda(state.forces, masses, normal));
    }
    return std::pair<Moments, Moments>{moments_of(temperatures),
                                       moments_of(lambdas)};
  };

  const auto nblib = run("nblib", nblib_library,
                         dimer_topology(k_kj, r0_nm, cutoff_nm), 17);
  const auto openmm =
      run("openmm", openmm_library, dimer_openmm_xml(k_kj, r0_nm, box_nm), 19);
  require_within_se(nblib.first.mean, kT_ev, nblib.first.stderr,
                    "nblib temperature");
  require_within_se(openmm.first.mean, kT_ev, openmm.first.stderr,
                    "OpenMM temperature");
  require_within_se(nblib.first.mean, openmm.first.mean,
                    std::hypot(nblib.first.stderr, openmm.first.stderr),
                    "temperature nblib vs OpenMM");
  require_within_se(nblib.second.mean, analytic, nblib.second.stderr,
                    "nblib mean constraint force");
  require_within_se(openmm.second.mean, analytic, openmm.second.stderr,
                    "OpenMM mean constraint force");
  require_within_se(nblib.second.mean, openmm.second.mean,
                    std::hypot(nblib.second.stderr, openmm.second.stderr),
                    "mean constraint force nblib vs OpenMM");
}

void check_reflection_keeps_the_pair(const char *nblib_library,
                                     const char *openmm_library) {
  constexpr double epsilon_ev = 0.01032;
  constexpr double sigma_angstrom = 3.4;
  constexpr double cutoff_angstrom = 8.0;
  constexpr double box_angstrom = 30.0;
  const double epsilon_kj = epsilon_ev * kKjPerEv;
  const double sigma_nm = sigma_angstrom * rgpot::md::kNmPerAngstrom;
  const double cutoff_nm = cutoff_angstrom * rgpot::md::kNmPerAngstrom;
  const double box_nm = box_angstrom * rgpot::md::kNmPerAngstrom;
  const double c6 = 4.0 * epsilon_kj * std::pow(sigma_nm, 6);
  const double c12 = 4.0 * epsilon_kj * std::pow(sigma_nm, 12);
  const std::vector<double> box{box_angstrom, 0.0, 0.0, 0.0, box_angstrom,
                                0.0,          0.0, 0.0, box_angstrom};
  const std::vector<double> positions{0.0, 0.0, 0.0, 4.2, 0.0, 0.0};
  std::vector<double> velocities{0.0, 0.0, 80.0, 0.0, 0.0, 0.0};
  const std::vector<double> plane{0.0, 0.0, 0.0, 4.2, 0.0, 0.0};
  const std::vector<double> plane_normal{0.0, 1.0, 0.0, 0.0, -1.0, 0.0};
  const std::vector<double> wall_point{0.0, 0.0, 0.2, 0.0, 0.0, 0.0};
  const std::vector<double> wall_normal{0.0, 0.0, 1.0, 0.0, 0.0, 0.0};
  const std::string topology = argon_topology(c6, c12, cutoff_nm, 2);
  const std::string xml =
      argon_openmm_xml(epsilon_kj, sigma_nm, cutoff_nm, box_nm, 2);
  for (const auto &engine : {std::pair<const char *, const char *>{"nblib", nblib_library},
                             {"openmm", openmm_library}}) {
    OwnedSession session = open_session(
        engine.first, engine.second, engine.first[0] == 'n' ? topology : xml,
        engine.first[0] == 'n' ? nullptr : "Reference");
    prepare(session.session, box, positions, velocities);
    const std::uint64_t hits =
        step_langevin(session.session, 0.02, 0.0, 0.01, plane, plane_normal,
                      wall_point, wall_normal, 1, 3);
    require(hits > 0, std::string(engine.first) + " wall was not hit");
    const Snapshot state = read_state(session.session, positions.size());
    compare_snapshot_to_lj(state, c6, c12, cutoff_nm, box_nm,
                           std::string(engine.first) + " after reflection");
  }
}

void check_nblib_reseed(const char *nblib_library) {
  const std::string topology =
      "nblib 1\ncutoff 0.5\ncoulomb none\natomtype A 12 0 0\n"
      "molecule mono 1\natom A A 0\nend\n";
  OwnedSession session = open_session("nblib", nblib_library, topology, nullptr);
  const std::vector<double> box{20.0, 0.0, 0.0, 0.0, 20.0, 0.0, 0.0, 0.0, 20.0};
  const std::vector<double> positions{1.0, 2.0, 0.0};
  const std::vector<double> velocities(3, 0.0);
  const std::vector<double> normal{0.0, 0.0, 1.0};
  const auto once = [&](std::uint64_t seed) {
    set_box(session.session, box);
    set_positions(session.session, positions);
    set_velocities(session.session, velocities);
    initialize(session.session);
    step_langevin(session.session, 0.03, 1.0, 0.002, positions, normal, {}, {},
                  1, seed);
    return read_state(session.session, 3);
  };
  const Snapshot first = once(11);
  const Snapshot second = once(11);
  for (std::size_t i = 0; i < 3; ++i) {
    require(first.positions[i] == second.positions[i] &&
                first.velocities[i] == second.velocities[i],
            "nblib Langevin reseed changed the one-atom trajectory");
  }
  const Snapshot other = once(12);
  require(first.velocities != other.velocities,
          "distinct nblib seeds produced the same one-atom velocity");
}

} // namespace

int main(int argc, char **argv) {
  try {
    require(argc == 3, "usage: test_engine_agreement <nblib-engine> <openmm-engine>");
    check_argon(argv[1], argv[2]);
    check_constrained_dimer(argv[1], argv[2]);
    check_reflection_keeps_the_pair(argv[1], argv[2]);
    check_nblib_reseed(argv[1]);
    std::cout << "nblib and OpenMM agree on argon, temperature, and the "
                 "constrained mean force\n";
    return 0;
  } catch (const std::exception &exception) {
    std::cerr << exception.what() << '\n';
    return 1;
  }
}
