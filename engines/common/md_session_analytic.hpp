#pragma once
// MIT License
// Copyright 2026 Rohit Goswami

// Analytic checks of the generic MD session ABI, written once for every
// engine. The model is a chain of three atoms joined by harmonic bonds of
// rest length zero, A-B with k1 and B-C with k2, in rgpot units (Angstrom,
// eV, ps, amu). Its potential depends on the bond vectors only, each Cartesian
// component being a pair of independent Gaussians, so everything below has a
// closed form:
//
//   - On a plane n . x = c with n_A + n_B + n_C = 0 the six bond coordinates
//     keep five free ones, so <U> = 5/2 kT + F(c), and the eight momentum
//     degrees of freedom left by the plane give <K> = 4 kT.
//   - The plane coordinate sigma = n . x = -n_A r1 + n_C r2 is Gaussian with
//     variance kT s, s = n_A^2/k1 + n_C^2/k2, so the free energy is
//     F(c) = c^2 / (2 s) and the mean constraint force on the plane is
//     dF/dc = c / s.
//
// Each engine session, built from its own spec, must reproduce these within
// the batch-means standard error and must agree with the other engines within
// the pooled error. Seeded replay and the continuation of a stream across
// calls are exact.

#include "rgpot/ExternalPot/external_potentials.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace rgpot::analytic {

constexpr double kTElectronVolt = 0.025851;
constexpr double kSigmas = 5.0;

struct Chain {
  double k1 = 0.4;                        // eV / Angstrom^2
  double k2 = 0.64;                       // eV / Angstrom^2
  std::array<double, 3> masses{4.0, 39.9, 12.0};
  std::array<double, 3> normal{1.0, -2.0, 1.0}; // x components, sum zero
  double offset = 0.3;                    // Angstrom

  [[nodiscard]] double normalNorm() const {
    return std::sqrt(normal[0] * normal[0] + normal[1] * normal[1] +
                     normal[2] * normal[2]);
  }
  [[nodiscard]] std::vector<double> planeNormal() const {
    const double norm = normalNorm();
    std::vector<double> out(9, 0.0);
    for (std::size_t atom = 0; atom < 3; ++atom) {
      out[3 * atom] = normal[atom] / norm;
    }
    return out;
  }
  [[nodiscard]] std::vector<double> planePoint() const {
    std::vector<double> n = planeNormal();
    for (double &value : n) {
      value *= offset;
    }
    return n;
  }
  [[nodiscard]] double spread() const {
    const double norm = normalNorm();
    const double a = normal[0] / norm;
    const double c = normal[2] / norm;
    return a * a / k1 + c * c / k2;
  }
  // <U> in eV: five bond coordinates share kT/2 each on the plane, and the
  // offset stores F(c) = c^2 / (2 s) in the coordinate it fixes.
  [[nodiscard]] double meanPotential(double kT) const {
    return 2.5 * kT + offset * offset / (2.0 * spread());
  }
  // dF/dc in eV / Angstrom.
  [[nodiscard]] double meanConstraintForce() const {
    return offset / spread();
  }
};

inline void require(bool condition, const std::string &message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

struct Mean {
  double value = 0.0;
  double error = 0.0;
};

// Mean and standard error of a correlated series by batch means.
inline Mean batchMean(const std::vector<double> &series,
                      std::size_t batches = 40) {
  require(series.size() >= 2 * batches, "series too short for batch means");
  const std::size_t size = series.size() / batches;
  std::vector<double> means;
  for (std::size_t b = 0; b < batches; ++b) {
    means.push_back(std::accumulate(series.begin() + static_cast<long>(b * size),
                                    series.begin() +
                                        static_cast<long>((b + 1) * size),
                                    0.0) /
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

inline std::string number(double value) {
  std::ostringstream out;
  out.precision(10);
  out << value;
  return out.str();
}

inline void requireWithin(double actual, double expected, double error,
                          const std::string &what) {
  require(std::isfinite(actual) && std::isfinite(expected) && error > 0.0,
          what + ": not finite or no scatter");
  if (!(std::abs(actual - expected) <= kSigmas * error)) {
    throw std::runtime_error(what + ": got " + number(actual) + ", expected " +
                             number(expected) + " +- " + number(error));
  }
}

struct Session {
  rgpot_md_session *handle = nullptr;
  explicit Session(rgpot_md_session *value) : handle(value) {}
  Session(const Session &) = delete;
  Session &operator=(const Session &) = delete;
  ~Session() { rgpot_md_session_destroy(handle); }
};

inline void check(int status, const char *error, const std::string &what) {
  if (status != 0) {
    throw std::runtime_error(what + ": " + error);
  }
}

struct Sampled {
  Mean kinetic;     // eV
  Mean potential;   // eV
  Mean constraint;  // eV / Angstrom
};

class Harness {
public:
  Harness(rgpot_md_session *session, const Chain &chain)
      : session_(session), chain_(chain) {}

  void prepare(const std::vector<double> &box) {
    char error[512] = {};
    check(rgpot_md_session_set_periodic_box(session_, box.data(), error,
                                            sizeof(error)),
          error, "set_periodic_box");
    reset();
  }

  void reset() {
    char error[512] = {};
    const std::vector<double> start = chain_.planePoint();
    check(rgpot_md_session_set_positions(session_, start.data(), start.size(),
                                         error, sizeof(error)),
          error, "set_positions");
    const std::vector<double> rest(9, 0.0);
    check(rgpot_md_session_set_velocities(session_, rest.data(), rest.size(),
                                          error, sizeof(error)),
          error, "set_velocities");
    check(rgpot_md_session_initialize(session_, error, sizeof(error)), error,
          "initialize");
  }

  // One call of constrained Langevin dynamics; returns the constraint force
  // of every step.
  std::vector<double> run(std::size_t steps, std::uint64_t seed,
                          double friction, double dt, bool plane = true) {
    const std::vector<double> point = chain_.planePoint();
    const std::vector<double> normal = chain_.planeNormal();
    rgpot_md_langevin_request request{};
    request.struct_size = sizeof(request);
    request.kT = kTElectronVolt;
    request.friction = friction;
    request.time_step = dt;
    request.steps = steps;
    request.seed = seed;
    request.n3 = 9;
    if (plane) {
      request.plane_point = point.data();
      request.plane_normal = normal.data();
    }
    std::vector<double> lambda(plane ? steps : 0, 0.0);
    rgpot_md_langevin_result result{};
    result.struct_size = sizeof(result);
    result.constraint_force = plane ? lambda.data() : nullptr;
    result.constraint_force_len = lambda.size();
    char error[512] = {};
    check(rgpot_md_session_run_langevin(session_, &request, &result, error,
                                        sizeof(error)),
          error, "run_langevin");
    return lambda;
  }

  struct Energies {
    double potential = 0.0;
    double kinetic = 0.0;
  };
  Energies energies() {
    char error[512] = {};
    Energies out;
    check(rgpot_md_session_get_energy(session_, &out.potential, &out.kinetic,
                                      error, sizeof(error)),
          error, "get_energy");
    return out;
  }

  std::vector<double> positions() {
    char error[512] = {};
    std::vector<double> out(9);
    check(rgpot_md_session_get_positions(session_, out.data(), out.size(),
                                         error, sizeof(error)),
          error, "get_positions");
    return out;
  }

  std::vector<double> velocities() {
    char error[512] = {};
    std::vector<double> out(9);
    check(rgpot_md_session_get_velocities(session_, out.data(), out.size(),
                                          error, sizeof(error)),
          error, "get_velocities");
    return out;
  }

  // Equipartition and the mean constraint force on the plane. One seed names
  // one stream for the whole run, so every call continues it.
  Sampled sample(std::size_t samples, std::size_t stride, std::uint64_t seed) {
    constexpr double friction = 10.0;
    constexpr double dt = 0.002;
    reset();
    run(5000, seed, friction, dt);
    std::vector<double> kinetic;
    std::vector<double> potential;
    std::vector<double> constraint;
    for (std::size_t sample = 0; sample < samples; ++sample) {
      const std::vector<double> lambda = run(stride, seed, friction, dt);
      constraint.push_back(std::accumulate(lambda.begin(), lambda.end(), 0.0) /
                           static_cast<double>(lambda.size()));
      const Energies state = energies();
      kinetic.push_back(state.kinetic);
      potential.push_back(state.potential);
      const std::vector<double> x = positions();
      const std::vector<double> n = chain_.planeNormal();
      double sigma = 0.0;
      for (std::size_t k = 0; k < 9; ++k) {
        sigma += n[k] * x[k];
      }
      require(std::abs(sigma - chain_.offset) < 1.0e-9,
              "the run left the plane");
    }
    return {batchMean(kinetic), batchMean(potential), batchMean(constraint)};
  }

  // The same seed after the same state replays, one call or two continue.
  void checkReplay() {
    constexpr double friction = 10.0;
    constexpr double dt = 0.002;
    auto trajectory = [&](const std::vector<std::size_t> &chunks,
                          std::uint64_t seed) {
      reset();
      std::vector<double> lambdas;
      for (const std::size_t steps : chunks) {
        const std::vector<double> part = run(steps, seed, friction, dt);
        lambdas.insert(lambdas.end(), part.begin(), part.end());
      }
      std::vector<double> out = positions();
      const std::vector<double> v = velocities();
      out.insert(out.end(), v.begin(), v.end());
      out.insert(out.end(), lambdas.begin(), lambdas.end());
      return out;
    };
    const std::vector<double> whole = trajectory({60}, 41);
    const std::vector<double> again = trajectory({60}, 41);
    const std::vector<double> split = trajectory({25, 10, 25}, 41);
    const std::vector<double> other = trajectory({60}, 42);
    for (std::size_t k = 0; k < whole.size(); ++k) {
      require(whole[k] == again[k], "a seeded replay differs");
      require(std::abs(split[k] - whole[k]) <=
                  1.0e-9 * std::max(1.0, std::abs(whole[k])),
              "a run split in chunks differs from one run");
    }
    bool differs = false;
    for (std::size_t k = 0; k < whole.size(); ++k) {
      differs = differs || whole[k] != other[k];
    }
    require(differs, "two seeds gave one trajectory");
  }

  // Maxwell-Boltzmann reseeding: on the plane n . v = 0, replayable, and with
  // the thermal kinetic energy of the eight momentum degrees of freedom.
  void checkReseed() {
    const std::vector<double> normal = chain_.planeNormal();
    char error[512] = {};
    std::vector<double> kinetic;
    for (std::uint64_t seed = 1; seed <= 4000; ++seed) {
      check(rgpot_md_session_reseed_velocities(session_, kTElectronVolt, seed,
                                               normal.data(), normal.size(),
                                               error, sizeof(error)),
            error, "reseed_velocities");
      const std::vector<double> v = velocities();
      double along = 0.0;
      for (std::size_t k = 0; k < 9; ++k) {
        along += normal[k] * v[k];
      }
      require(std::abs(along) < 1.0e-9, "reseeded velocities left the plane");
      kinetic.push_back(energies().kinetic);
      if (seed == 1) {
        check(rgpot_md_session_reseed_velocities(session_, kTElectronVolt, 1,
                                                 normal.data(), normal.size(),
                                                 error, sizeof(error)),
              error, "reseed_velocities");
        require(velocities() == v, "a reseed with one seed differs");
      }
    }
    const Mean mean = batchMean(kinetic, 20);
    requireWithin(mean.value, 4.0 * kTElectronVolt, mean.error,
                  "reseeded kinetic energy");
  }

private:
  rgpot_md_session *session_;
  Chain chain_;
};

inline void checkAgainstAnalytic(const Sampled &sampled, const Chain &chain,
                                 const std::string &label) {
  requireWithin(sampled.kinetic.value, 4.0 * kTElectronVolt,
                sampled.kinetic.error, label + " kinetic energy");
  requireWithin(sampled.potential.value,
                chain.meanPotential(kTElectronVolt),
                sampled.potential.error, label + " potential energy");
  requireWithin(sampled.constraint.value, chain.meanConstraintForce(),
                sampled.constraint.error, label + " mean constraint force");
}

inline void checkAgreement(const Sampled &a, const Sampled &b,
                           const std::string &label) {
  const auto pooled = [](const Mean &x, const Mean &y) {
    return std::hypot(x.error, y.error);
  };
  requireWithin(a.kinetic.value, b.kinetic.value,
                pooled(a.kinetic, b.kinetic), label + " kinetic energy");
  requireWithin(a.potential.value, b.potential.value,
                pooled(a.potential, b.potential), label + " potential energy");
  requireWithin(a.constraint.value, b.constraint.value,
                pooled(a.constraint, b.constraint),
                label + " mean constraint force");
}

} // namespace rgpot::analytic
