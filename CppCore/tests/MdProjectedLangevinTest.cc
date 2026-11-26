// MIT License
// Copyright 2023--present rgpot developers
//
// The projected BAOAB driver against closed forms: equipartition in a
// harmonic well, the constraint force on a hyperplane against the derivative
// of the free energy, seeded replay, walls, and Maxwell-Boltzmann velocities.
// The units are the caller's: amu, nm, ps and kJ/mol.

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <tuple>
#include <vector>

#include "rgpot/MDSession/ProjectedLangevin.hpp"

namespace {

using rgpot::md::ForceFunction;
using rgpot::md::LangevinResult;
using rgpot::md::LangevinSettings;
using rgpot::md::PlaneSpec;
using rgpot::md::ProjectedLangevin;
using rgpot::md::WallMode;
using rgpot::md::WallSpec;

constexpr double kT = 2.4943387854; // 300 K in kJ/mol

struct Mean {
  double value = 0.0;
  double error = 0.0;
};

// Mean and standard error of a correlated series by batch means.
Mean batchMean(const std::vector<double> &series, std::size_t batches = 40) {
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

// Three atoms joined by harmonic bonds of rest length zero, A-B (k1) and
// B-C (k2): V = k1/2 |xB - xA|^2 + k2/2 |xC - xB|^2.
struct Chain {
  double k1 = 1000.0;
  double k2 = 1600.0;
  std::vector<double> atomMass{4.0, 39.9, 12.0};

  [[nodiscard]] std::vector<double> masses() const {
    std::vector<double> out;
    for (const double mass : atomMass) {
      out.insert(out.end(), {mass, mass, mass});
    }
    return out;
  }

  double operator()(const double *x, double *force) const {
    double energy = 0.0;
    std::fill(force, force + 9, 0.0);
    for (std::size_t d = 0; d < 3; ++d) {
      const double ab = x[3 + d] - x[d];
      const double bc = x[6 + d] - x[3 + d];
      energy += 0.5 * k1 * ab * ab + 0.5 * k2 * bc * bc;
      force[d] += k1 * ab;
      force[3 + d] += -k1 * ab + k2 * bc;
      force[6 + d] += -k2 * bc;
    }
    return energy;
  }
};

double kinetic(const std::vector<double> &masses, const double *v) {
  double sum = 0.0;
  for (std::size_t k = 0; k < masses.size(); ++k) {
    sum += 0.5 * masses[k] * v[k] * v[k];
  }
  return sum;
}

} // namespace

TEST_CASE("SplitMix64 is a pure function of the seed", "[md][session]") {
  rgpot::md::SplitMix64 first(1);
  CHECK(first.nextU64() == 0x910a2dec89025cc1ULL);
  CHECK(first.nextU64() == 0xbeeb8da1658eec67ULL);
  rgpot::md::SplitMix64 a(42);
  rgpot::md::SplitMix64 b(42);
  for (int i = 0; i < 100; ++i) {
    CHECK(a.nextNormal() == b.nextNormal());
  }
  rgpot::md::SplitMix64 normal(7);
  std::vector<double> draws;
  for (int i = 0; i < 200000; ++i) {
    draws.push_back(normal.nextNormal());
  }
  const Mean mean = batchMean(draws);
  CHECK(std::abs(mean.value) < 5.0 * mean.error);
  double square = 0.0;
  for (const double value : draws) {
    square += value * value;
  }
  CHECK(std::abs(square / static_cast<double>(draws.size()) - 1.0) < 0.02);
}

TEST_CASE("unconstrained Langevin dynamics obeys equipartition",
          "[md][session]") {
  const Chain chain;
  const std::vector<double> masses = chain.masses();
  const ProjectedLangevin driver(masses);
  std::vector<double> x{0.0, 0.0, 0.0, 0.05, 0.0, 0.0, 0.0, 0.04, 0.0};
  std::vector<double> v(9, 0.0);
  LangevinSettings settings;
  settings.kT = kT;
  settings.friction = 5.0;
  settings.timeStep = 0.001;
  settings.steps = 2000;
  settings.seed = 11;
  driver.run(settings, nullptr, {}, x.data(), v.data(), chain);

  std::vector<double> kinetics;
  std::vector<double> potentials;
  settings.steps = 100;
  for (std::uint64_t sample = 0; sample < 4000; ++sample) {
    settings.seed = 1000 + sample;
    const LangevinResult result =
        driver.run(settings, nullptr, {}, x.data(), v.data(), chain);
    kinetics.push_back(kinetic(masses, v.data()));
    potentials.push_back(result.potential);
  }
  // Nine kinetic and six harmonic degrees of freedom.
  const Mean ke = batchMean(kinetics);
  const Mean pe = batchMean(potentials);
  CHECK(std::abs(ke.value - 4.5 * kT) < 5.0 * ke.error + 2.0e-3 * kT);
  CHECK(std::abs(pe.value - 3.0 * kT) < 5.0 * pe.error + 2.0e-3 * kT);
}

TEST_CASE("the constraint force is dF/dc of the harmonic free energy",
          "[md][session]") {
  const Chain chain;
  const std::vector<double> masses = chain.masses();
  const ProjectedLangevin driver(masses);

  // A normal with no net translation part, oblique to the axes, so the
  // constraint force fluctuates and depends on the masses.
  const std::array<double, 9> normal{0.3, -0.4, 0.2, 0.1, 0.5, -0.3,
                                     -0.4, -0.1, 0.1};
  double norm2 = 0.0;
  for (const double value : normal) {
    norm2 += value * value;
  }
  const double norm = std::sqrt(norm2);
  PlaneSpec plane;
  plane.normal.assign(normal.begin(), normal.end());
  plane.point = {0.0, 0.0, 0.0, 0.12, 0.05, -0.04, 0.2, 0.0, 0.1};
  double offset = 0.0;
  for (std::size_t k = 0; k < 9; ++k) {
    offset += normal[k] / norm * plane.point[k];
  }
  // F(c) = c^2 / (2 s), s = |n_A|^2 / k1 + |n_C|^2 / k2, for unit n with
  // n_A + n_B + n_C = 0.
  double s = 0.0;
  for (std::size_t d = 0; d < 3; ++d) {
    s += normal[d] * normal[d] / (norm2 * chain.k1) +
         normal[6 + d] * normal[6 + d] / (norm2 * chain.k2);
  }
  const double expected = offset / s;
  REQUIRE(std::abs(expected) > 1.0);

  std::vector<double> x = plane.point;
  std::vector<double> v(9, 0.0);
  LangevinSettings settings;
  settings.kT = kT;
  settings.friction = 5.0;
  settings.timeStep = 0.001;
  settings.steps = 4000;
  settings.seed = 3;
  driver.run(settings, &plane, {}, x.data(), v.data(), chain);
  settings.steps = 400000;
  settings.seed = 4;
  settings.recordConstraintForce = true;
  const LangevinResult result =
      driver.run(settings, &plane, {}, x.data(), v.data(), chain);
  REQUIRE(result.constraintForce.size() == settings.steps);
  const Mean force = batchMean(result.constraintForce);
  CHECK(force.error > 0.0);
  CHECK(std::abs(force.value - expected) < 5.0 * force.error);

  // The trajectory stays on the plane with tangent velocities.
  double sigma = 0.0;
  double along = 0.0;
  for (std::size_t k = 0; k < 9; ++k) {
    sigma += normal[k] / norm * (x[k] - plane.point[k]);
    along += normal[k] / norm * v[k];
  }
  CHECK(std::abs(sigma) < 1.0e-12);
  CHECK(std::abs(along) < 1.0e-12);

  // Degrees of freedom: nine minus one constraint.
  std::vector<double> kinetics;
  settings.recordConstraintForce = false;
  settings.steps = 100;
  for (std::uint64_t sample = 0; sample < 4000; ++sample) {
    settings.seed = 9000 + sample;
    driver.run(settings, &plane, {}, x.data(), v.data(), chain);
    kinetics.push_back(kinetic(masses, v.data()));
  }
  const Mean ke = batchMean(kinetics);
  CHECK(std::abs(ke.value - 4.0 * kT) < 5.0 * ke.error + 2.0e-3 * kT);
}

TEST_CASE("a seeded run replays exactly", "[md][session]") {
  const Chain chain;
  const ProjectedLangevin driver(chain.masses());
  PlaneSpec plane;
  plane.normal = {0.3, -0.4, 0.2, 0.1, 0.5, -0.3, -0.4, -0.1, 0.1};
  plane.point = {0.0, 0.0, 0.0, 0.12, 0.05, -0.04, 0.2, 0.0, 0.1};
  LangevinSettings settings;
  settings.kT = kT;
  settings.friction = 2.0;
  settings.timeStep = 0.002;
  settings.steps = 300;
  settings.recordConstraintForce = true;

  const auto run = [&](std::uint64_t seed) {
    std::vector<double> x = plane.point;
    std::vector<double> v(9, 0.0);
    settings.seed = seed;
    const LangevinResult result =
        driver.run(settings, &plane, {}, x.data(), v.data(), chain);
    return std::make_tuple(x, v, result.constraintForce);
  };
  const auto first = run(17);
  const auto second = run(17);
  const auto other = run(18);
  CHECK(std::get<0>(first) == std::get<0>(second));
  CHECK(std::get<1>(first) == std::get<1>(second));
  CHECK(std::get<2>(first) == std::get<2>(second));
  CHECK(std::get<0>(first) != std::get<0>(other));
  // The seed zero is a seed, not a request for entropy.
  CHECK(std::get<0>(run(0)) == std::get<0>(run(0)));
}

TEST_CASE("a hard wall reflects or rejects and keeps the sampling",
          "[md][session]") {
  // A single particle in three harmonic wells bounded by the wall x <= 0.05.
  // The density is the Gaussian cut at the wall.
  const std::vector<double> masses(3, 39.9);
  const ProjectedLangevin driver(masses);
  const double k = 1000.0;
  const ForceFunction well = [k](const double *x, double *force) {
    double energy = 0.0;
    for (int d = 0; d < 3; ++d) {
      energy += 0.5 * k * x[d] * x[d];
      force[d] = -k * x[d];
    }
    return energy;
  };
  const double sigma = std::sqrt(kT / k);
  const double wallAt = 0.5 * sigma;
  WallSpec wall;
  wall.point = {wallAt, 0.0, 0.0};
  wall.normal = {1.0, 0.0, 0.0};

  // The truncated normal: mean and second moment of x for x <= a.
  const double a = wallAt / sigma;
  const double pdf = std::exp(-0.5 * a * a) / std::sqrt(2.0 * 3.14159265358979323846);
  const double cdf = 0.5 * std::erfc(-a / std::sqrt(2.0));
  const double expectedMean = -sigma * pdf / cdf;
  const double expectedSecond = sigma * sigma * (1.0 - a * pdf / cdf);

  for (const WallMode mode : {WallMode::Reflect, WallMode::Reject}) {
    std::vector<double> x{0.0, 0.0, 0.0};
    std::vector<double> v{0.0, 0.0, 0.0};
    LangevinSettings settings;
    settings.kT = kT;
    settings.friction = 5.0;
    settings.timeStep = 0.0005;
    settings.wallMode = mode;
    settings.steps = 4000;
    settings.seed = 5;
    driver.run(settings, nullptr, {wall}, x.data(), v.data(), well);
    std::vector<double> firsts;
    std::vector<double> seconds;
    std::uint64_t hits = 0;
    settings.steps = 100;
    for (std::uint64_t sample = 0; sample < 8000; ++sample) {
      settings.seed = 500 + sample;
      const LangevinResult result =
          driver.run(settings, nullptr, {wall}, x.data(), v.data(), well);
      hits += result.wallHits;
      REQUIRE(x[0] <= wallAt + 1.0e-12);
      firsts.push_back(x[0]);
      seconds.push_back(x[0] * x[0]);
    }
    CHECK(hits > 0);
    const Mean first = batchMean(firsts);
    const Mean second = batchMean(seconds);
    // A hard wall costs a bias of the order of the step length at the wall.
    const double bias = 0.05 * sigma;
    CHECK(std::abs(first.value - expectedMean) < 5.0 * first.error + bias);
    CHECK(std::abs(second.value - expectedSecond) <
          5.0 * second.error + bias * sigma);
  }

  std::vector<double> outside{0.1, 0.0, 0.0};
  std::vector<double> v{0.0, 0.0, 0.0};
  LangevinSettings settings;
  settings.kT = kT;
  settings.timeStep = 0.001;
  settings.steps = 1;
  CHECK_THROWS_AS(
      driver.run(settings, nullptr, {wall}, outside.data(), v.data(), well),
      std::invalid_argument);
}

TEST_CASE("walls act on the plane in the mass metric", "[md][session]") {
  // Two atoms of unequal mass in a plane that fixes x1 + x2 and a wall
  // that bounds x1 alone; the wall's normal has a part along the plane's.
  const std::vector<double> masses{4.0, 4.0, 4.0, 39.9, 39.9, 39.9};
  const ProjectedLangevin driver(masses);
  PlaneSpec plane;
  plane.normal = {1.0, 0.0, 0.0, 1.0, 0.0, 0.0};
  plane.point = {0.0, 0.0, 0.0, 0.2, 0.0, 0.0};
  WallSpec wall;
  wall.point = {0.1, 0.0, 0.0, 0.0, 0.0, 0.0};
  wall.normal = {1.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  const ForceFunction free = [](const double *, double *force) {
    std::fill(force, force + 6, 0.0);
    return 0.0;
  };
  std::vector<double> x{0.05, 0.0, 0.0, 0.15, 0.0, 0.0};
  std::vector<double> v{2.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  LangevinSettings settings;
  settings.kT = kT;
  settings.friction = 0.0;
  settings.timeStep = 0.002;
  settings.steps = 500;
  settings.seed = 8;
  const LangevinResult result =
      driver.run(settings, &plane, {wall}, x.data(), v.data(), free);
  // The tangent velocity of the start is v1 = -v2 with the mass-metric
  // projection of (2, 0); free motion and reflection both keep its energy.
  const double tangent = 2.0 * (1.0 / 39.9) / (1.0 / 4.0 + 1.0 / 39.9);
  const double energy = 0.5 * (4.0 + 39.9) * tangent * tangent;
  CHECK(result.wallHits > 0);
  CHECK(v[0] < 0.0);
  CHECK(kinetic(masses, v.data()) == Catch::Approx(energy).epsilon(1.0e-9));
  CHECK(x[0] <= 0.1 + 1.0e-12);
  CHECK(std::abs(x[0] + x[3] - 0.2) < 1.0e-12);
  CHECK(std::abs(v[0] + v[3]) < 1.0e-12);
}

TEST_CASE("Maxwell-Boltzmann velocities have the thermal spread",
          "[md][session]") {
  const Chain chain;
  const std::vector<double> masses = chain.masses();
  const std::array<double, 9> normal{0.3, -0.4, 0.2, 0.1, 0.5, -0.3,
                                     -0.4, -0.1, 0.1};
  std::vector<double> kinetics;
  std::vector<double> first;
  std::vector<double> projected;
  for (std::uint64_t seed = 0; seed < 20000; ++seed) {
    const std::vector<double> free =
        rgpot::md::maxwellBoltzmannVelocities(masses, kT, seed, nullptr);
    kinetics.push_back(kinetic(masses, free.data()));
    first.push_back(free[3] * free[3] * masses[3] / kT);
    const std::vector<double> tangent =
        rgpot::md::maxwellBoltzmannVelocities(masses, kT, seed, normal.data());
    double along = 0.0;
    for (std::size_t k = 0; k < 9; ++k) {
      along += normal[k] * tangent[k];
    }
    REQUIRE(std::abs(along) < 1.0e-12);
    projected.push_back(kinetic(masses, tangent.data()));
  }
  const Mean ke = batchMean(kinetics);
  CHECK(std::abs(ke.value - 4.5 * kT) < 5.0 * ke.error);
  const Mean one = batchMean(first);
  CHECK(std::abs(one.value - 1.0) < 5.0 * one.error);
  const Mean tangentKe = batchMean(projected);
  CHECK(std::abs(tangentKe.value - 4.0 * kT) < 5.0 * tangentKe.error);

  CHECK(rgpot::md::maxwellBoltzmannVelocities(masses, kT, 5, nullptr) ==
        rgpot::md::maxwellBoltzmannVelocities(masses, kT, 5, nullptr));
  const std::vector<double> frozen = rgpot::md::maxwellBoltzmannVelocities(
      {0.0, 0.0, 0.0, 1.0, 1.0, 1.0}, kT, 3, nullptr);
  CHECK(frozen[0] == 0.0);
  CHECK(frozen[3] != 0.0);
}

TEST_CASE("the driver rejects invalid input", "[md][session]") {
  const Chain chain;
  const ProjectedLangevin driver(chain.masses());
  std::vector<double> x(9, 0.0);
  std::vector<double> v(9, 0.0);
  LangevinSettings settings;
  settings.kT = kT;
  settings.timeStep = 0.001;
  settings.steps = 1;
  PlaneSpec plane;
  plane.point.assign(9, 0.0);
  plane.normal.assign(9, 0.0);
  CHECK_THROWS_AS(driver.run(settings, &plane, {}, x.data(), v.data(), chain),
                  std::invalid_argument);
  plane.normal.assign(8, 1.0);
  CHECK_THROWS_AS(driver.run(settings, &plane, {}, x.data(), v.data(), chain),
                  std::invalid_argument);
  settings.kT = 0.0;
  CHECK_THROWS_AS(driver.run(settings, nullptr, {}, x.data(), v.data(), chain),
                  std::invalid_argument);
  CHECK_THROWS_AS(ProjectedLangevin({0.0, 0.0, 0.0}), std::invalid_argument);
  CHECK_THROWS_AS(ProjectedLangevin({1.0, -1.0}), std::invalid_argument);
  settings.kT = kT;
  settings.steps = 0;
  CHECK(driver.run(settings, nullptr, {}, x.data(), v.data(), chain)
            .wallHits == 0);
}

TEST_CASE("a seeded stream continues across calls", "[md][session]") {
  const Chain chain;
  const std::vector<double> masses = chain.masses();
  const ProjectedLangevin driver(masses);
  PlaneSpec plane;
  plane.point.assign(9, 0.0);
  plane.normal = {1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0};
  const ForceFunction forces = [&](const double *x, double *f) {
    return chain(x, f);
  };
  LangevinSettings settings;
  settings.kT = kT;
  settings.friction = 5.0;
  settings.timeStep = 0.001;

  auto run = [&](const std::vector<std::uint64_t> &chunks, std::uint64_t seed) {
    std::vector<double> x(9, 0.0);
    x[3] = 0.01;
    std::vector<double> v(9, 0.0);
    rgpot::md::SeededStream stream;
    for (const std::uint64_t steps : chunks) {
      settings.steps = steps;
      driver.run(settings, &plane, {}, x.data(), v.data(), forces,
                 stream.forSeed(seed));
    }
    x.insert(x.end(), v.begin(), v.end());
    return x;
  };

  const auto whole = run({7}, 29);
  const auto split = run({3, 1, 3}, 29);
  const auto odd = run({1, 1, 1, 1, 1, 1, 1}, 29);
  const auto other = run({7}, 30);
  REQUIRE(whole.size() == split.size());
  for (std::size_t k = 0; k < whole.size(); ++k) {
    // Each call projects its start again, which moves the last bits only.
    CHECK(split[k] == Catch::Approx(whole[k]).margin(1.0e-13));
    CHECK(odd[k] == Catch::Approx(whole[k]).margin(1.0e-13));
  }
  CHECK(other != whole);

  // Ending the stream makes the next call with the same seed replay.
  std::vector<double> x(9, 0.0);
  std::vector<double> v(9, 0.0);
  settings.steps = 3;
  rgpot::md::SeededStream stream;
  driver.run(settings, &plane, {}, x.data(), v.data(), forces,
             stream.forSeed(29));
  const std::vector<double> first = x;
  x.assign(9, 0.0);
  v.assign(9, 0.0);
  stream.reset();
  driver.run(settings, &plane, {}, x.data(), v.data(), forces,
             stream.forSeed(29));
  CHECK(x == first);
}

TEST_CASE("the first kick uses the force at the constrained position",
          "[md][session]") {
  // U = 100 x y on the plane x = 0: the tangent force is zero there, while
  // the force at the raw start (x = 0.1) would push y.
  const ProjectedLangevin driver({1.0, 1.0, 1.0});
  PlaneSpec plane;
  plane.point = {0.0, 0.0, 0.0};
  plane.normal = {1.0, 0.0, 0.0};
  const ForceFunction forces = [](const double *x, double *f) {
    f[0] = -100.0 * x[1];
    f[1] = -100.0 * x[0];
    f[2] = 0.0;
    return 100.0 * x[0] * x[1];
  };
  LangevinSettings settings;
  settings.kT = kT;
  settings.timeStep = 0.001;
  settings.steps = 1;
  std::vector<double> x{0.1, 0.0, 0.0};
  std::vector<double> v{0.0, 0.0, 0.0};
  driver.run(settings, &plane, {}, x.data(), v.data(), forces);
  CHECK(x[0] == Catch::Approx(0.0).margin(1.0e-15));
  CHECK(x[1] == Catch::Approx(0.0).margin(1.0e-15));
  CHECK(v[1] == Catch::Approx(0.0).margin(1.0e-15));
}

TEST_CASE("a state outside many bounces is returned inside", "[md][session]") {
  const ProjectedLangevin driver({1.0, 1.0, 1.0});
  PlaneSpec plane;
  plane.point = {0.0, 0.0, 0.0};
  plane.normal = {1.0, 0.0, 0.0};
  const std::vector<WallSpec> walls{
      {{0.0, 1.0, 0.0}, {0.0, 1.0, 0.0}},
      {{0.0, -1.0, 0.0}, {0.0, -1.0, 0.0}}};
  const ForceFunction free = [](const double *, double *f) {
    std::fill(f, f + 3, 0.0);
    return 0.0;
  };
  for (const WallMode mode : {WallMode::Reflect}) {
    LangevinSettings settings;
    settings.kT = kT;
    settings.timeStep = 1.0;
    settings.steps = 1;
    settings.wallMode = mode;
    std::vector<double> x{0.0, 0.0, 0.0};
    std::vector<double> v{0.0, 100.0, 0.0};
    const LangevinResult result =
        driver.run(settings, &plane, walls, x.data(), v.data(), free);
    CHECK(std::abs(x[1]) <= 1.0 + 1.0e-12);
    CHECK(result.wallHits >= 40);
    CHECK(std::abs(v[1]) == Catch::Approx(100.0));
  }
}

TEST_CASE("distance constraints hold and remove their degrees of freedom",
          "[md][session]") {
  // A rigid dimer in a harmonic trap with the plane z1 + z2 = 0: six
  // coordinates less the plane and the bond leave four kinetic degrees of
  // freedom, so <K> = 2 kT.
  const std::vector<double> masses{1.0, 1.0, 1.0, 3.0, 3.0, 3.0};
  const ProjectedLangevin driver(masses, {{0, 1, 0.2}});
  PlaneSpec plane;
  plane.point.assign(6, 0.0);
  plane.normal = {0.0, 0.0, 1.0, 0.0, 0.0, 1.0};
  const ForceFunction trap = [](const double *x, double *f) {
    double energy = 0.0;
    for (std::size_t k = 0; k < 6; ++k) {
      f[k] = -50.0 * x[k];
      energy += 25.0 * x[k] * x[k];
    }
    return energy;
  };
  LangevinSettings settings;
  settings.kT = kT;
  settings.friction = 5.0;
  settings.timeStep = 0.0005;
  settings.steps = 20;
  std::vector<double> x{-0.1, 0.0, 0.0, 0.1, 0.0, 0.0};
  std::vector<double> v{-1.0, 0.0, 0.0, 1.0, 0.0, 0.0};
  rgpot::md::SeededStream stream;
  std::vector<double> kinetics;
  for (int sample = 0; sample < 8000; ++sample) {
    driver.run(settings, &plane, {}, x.data(), v.data(), trap,
               stream.forSeed(3));
    double length2 = 0.0;
    for (std::size_t d = 0; d < 3; ++d) {
      length2 += (x[d] - x[3 + d]) * (x[d] - x[3 + d]);
    }
    REQUIRE(std::sqrt(length2) == Catch::Approx(0.2).margin(1.0e-10));
    REQUIRE(std::abs(x[2] + x[5]) < 1.0e-10);
    const double relative =
        (x[0] - x[3]) * (v[0] - v[3]) + (x[1] - x[4]) * (v[1] - v[4]) +
        (x[2] - x[5]) * (v[2] - v[5]);
    REQUIRE(std::abs(relative) < 1.0e-9);
    kinetics.push_back(kinetic(masses, v.data()));
  }
  const Mean mean = batchMean(kinetics);
  INFO("<K> = " << mean.value << " +- " << mean.error);
  CHECK(std::abs(mean.value - 2.0 * kT) < 5.0 * mean.error);
}

TEST_CASE("the plane multiplier accounts for distance constraints",
          "[md][session]") {
  // A bonded pair under a uniform force, on the plane x0 + x1 = 0. The
  // multiplier is the plane entry of -(J M^-1 J^T)^-1 J M^-1 F with the
  // two rows n and g, which the test solves in closed form for 2x2.
  const std::vector<double> masses{2.0, 2.0, 2.0, 5.0, 5.0, 5.0};
  const ProjectedLangevin driver(masses, {{0, 1, 0.5}});
  PlaneSpec plane;
  plane.point.assign(6, 0.0);
  const double root = std::sqrt(2.0);
  plane.normal = {1.0 / root, 0.0, 0.0, 1.0 / root, 0.0, 0.0};
  const std::vector<double> field{0.7, -1.3, 0.2, -0.4, 0.9, 1.1};
  const ForceFunction uniform = [&](const double *, double *f) {
    std::copy(field.begin(), field.end(), f);
    return 0.0;
  };
  // On the start, x0 = (0.2, 0.1, 0), x1 = (-0.2, ...) with |d| = 0.5.
  const double dy = std::sqrt(0.25 - 0.16);
  std::vector<double> x{0.2, 0.0, 0.0, -0.2, -dy, 0.0};
  std::vector<double> v(6, 0.0);
  LangevinSettings settings;
  settings.kT = kT;
  settings.timeStep = 1.0e-9;
  settings.steps = 1;
  settings.recordConstraintForce = true;
  const LangevinResult result =
      driver.run(settings, &plane, {}, x.data(), v.data(), uniform);
  REQUIRE(result.constraintForce.size() == 1);

  const std::array<double, 6> n{1.0 / root, 0.0, 0.0, 1.0 / root, 0.0, 0.0};
  const std::array<double, 6> g{0.4, dy, 0.0, -0.4, -dy, 0.0};
  const std::array<double, 6> inverse{0.5, 0.5, 0.5, 0.2, 0.2, 0.2};
  auto gram = [&](const std::array<double, 6> &a,
                  const std::array<double, 6> &b) {
    double sum = 0.0;
    for (std::size_t k = 0; k < 6; ++k) {
      sum += a[k] * inverse[k] * b[k];
    }
    return sum;
  };
  std::array<double, 6> f{};
  std::copy(field.begin(), field.end(), f.begin());
  const double a11 = gram(n, n);
  const double a12 = gram(n, g);
  const double a22 = gram(g, g);
  const double b1 = -gram(n, f);
  const double b2 = -gram(g, f);
  const double lambda = (b1 * a22 - b2 * a12) / (a11 * a22 - a12 * a12);
  CHECK(result.constraintForce[0] == Catch::Approx(lambda).epsilon(1.0e-6));
}

TEST_CASE("a rigid rotor keeps its energy without friction",
          "[md][session]") {
  // A free dumbbell turns by 0.01 rad per step. Dropping the constraint
  // impulse of the drift would shorten its velocity on every step.
  const std::vector<double> masses{1.0, 1.0, 1.0, 16.0, 16.0, 16.0};
  const double length = 0.1;
  const ProjectedLangevin driver(
      masses, {rgpot::md::DistanceConstraint{0, 1, length}});
  const ForceFunction none = [](const double *, double *f) {
    std::fill(f, f + 6, 0.0);
    return 0.0;
  };
  std::vector<double> x{length, 0.0, 0.0, 0.0, 0.0, 0.0};
  std::vector<double> v{0.0, 16.0, 0.0, 0.0, -1.0, 0.0};
  const auto kinetic = [&] {
    double sum = 0.0;
    for (std::size_t k = 0; k < 6; ++k) {
      sum += 0.5 * masses[k] * v[k] * v[k];
    }
    return sum;
  };
  LangevinSettings settings;
  settings.kT = kT;
  settings.friction = 0.0;
  settings.timeStep = 0.01 * length / 17.0;
  settings.steps = 2000;
  const double start = kinetic();
  driver.run(settings, nullptr, {}, x.data(), v.data(), none);
  CHECK(kinetic() == Catch::Approx(start).epsilon(1.0e-4));
}
