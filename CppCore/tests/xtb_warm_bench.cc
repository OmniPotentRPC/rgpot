// MIT License — cold, warm, and loose-SCC GFN2 single points on water clusters.
//
// One calculator is kept. A warm call moves one oxygen by 1/16 Angstrom
// and reuses the SCC wavefunction. A cold call is the same geometry after
// reset(), so xtb builds its own guess. The new arm is that warm step at
// xtb accuracy 50 while the RMS force stays above 0.05 eV/Angstrom.
// The timed calls exclude the first load of the parametrisation.
//
// Run serial, under hostname bench:
//   sudo unshare --uts bash -c 'hostname bench && OMP_NUM_THREADS=1 taskset -c 0 ./xtb_warm_bench --repeats 6'

#include "rgpot/XTBPot/XTBPot.hpp"
#include "rgpot/types/AtomMatrix.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef _WIN32
#include <unistd.h>
#endif

using rgpot::types::AtomMatrix;
using SteadyClock = std::chrono::steady_clock;

namespace {

void force_serial() {
#ifdef _WIN32
  _putenv_s("OMP_NUM_THREADS", "1");
  _putenv_s("MKL_NUM_THREADS", "1");
  _putenv_s("OPENBLAS_NUM_THREADS", "1");
  _putenv_s("BLIS_NUM_THREADS", "1");
  _putenv_s("OMP_DYNAMIC", "FALSE");
#else
  setenv("OMP_NUM_THREADS", "1", 1);
  setenv("MKL_NUM_THREADS", "1", 1);
  setenv("OPENBLAS_NUM_THREADS", "1", 1);
  setenv("BLIS_NUM_THREADS", "1", 1);
  setenv("OMP_DYNAMIC", "false", 1);
#endif
}

std::string hostname() {
#ifdef _WIN32
  return "unknown";
#else
  char host[256] = {};
  if (gethostname(host, sizeof host - 1) != 0)
    return "unknown";
  return host;
#endif
}

// Regular dodecahedron, then the first `molecules` vertices scaled so the
// shortest oxygen-oxygen contact is 2.8 Angstrom. Hydrogens point outward.
void water_cluster(int molecules, std::vector<double> &positions,
                   std::vector<int> &numbers) {
  const double phi = (1.0 + std::sqrt(5.0)) / 2.0;
  const double inv_phi = 1.0 / phi;
  std::vector<std::array<double, 3>> vertices;
  vertices.reserve(20);
  for (double x : {1.0, -1.0})
    for (double y : {1.0, -1.0})
      for (double z : {1.0, -1.0})
        vertices.push_back({x, y, z});
  for (double sy : {1.0, -1.0})
    for (double sz : {1.0, -1.0})
      vertices.push_back({0.0, sy * inv_phi, sz * phi});
  for (double sx : {1.0, -1.0})
    for (double sy : {1.0, -1.0})
      vertices.push_back({sx * inv_phi, sy * phi, 0.0});
  for (double sx : {1.0, -1.0})
    for (double sz : {1.0, -1.0})
      vertices.push_back({sx * phi, 0.0, sz * inv_phi});
  if (static_cast<int>(vertices.size()) != 20 || molecules > 20 ||
      molecules < 1) {
    throw std::runtime_error("water cluster size is outside 1..20");
  }
  vertices.resize(static_cast<size_t>(molecules));

  double nearest = 1.0e300;
  for (size_t i = 0; i < vertices.size(); ++i) {
    for (size_t j = i + 1; j < vertices.size(); ++j) {
      double d2 = 0.0;
      for (int axis = 0; axis < 3; ++axis) {
        const double d = vertices[i][axis] - vertices[j][axis];
        d2 += d * d;
      }
      nearest = std::min(nearest, std::sqrt(d2));
    }
  }
  const double scale = 2.8 / nearest;
  for (auto &vertex : vertices)
    for (double &coordinate : vertex)
      coordinate *= scale;

  positions.clear();
  numbers.clear();
  const double bond = 0.9572;
  const double half = (104.52 * std::acos(-1.0) / 180.0) / 2.0;
  const double along = std::cos(half);
  const double aside = std::sin(half);
  for (const auto &oxygen : vertices) {
    const double norm =
        std::sqrt(oxygen[0] * oxygen[0] + oxygen[1] * oxygen[1] +
                  oxygen[2] * oxygen[2]);
    const double radial[3] = {oxygen[0] / norm, oxygen[1] / norm,
                              oxygen[2] / norm};
    const double up_x = std::abs(radial[2]) < 0.9 ? 0.0 : 1.0;
    const double up_z = std::abs(radial[2]) < 0.9 ? 1.0 : 0.0;
    const double up[3] = {up_x, 0.0, up_z};
    double tangent[3] = {radial[1] * up[2] - radial[2] * up[1],
                         radial[2] * up[0] - radial[0] * up[2],
                         radial[0] * up[1] - radial[1] * up[0]};
    const double tangent_norm =
        std::sqrt(tangent[0] * tangent[0] + tangent[1] * tangent[1] +
                  tangent[2] * tangent[2]);
    for (double &component : tangent)
      component /= tangent_norm;
    auto push_atom = [&](int number, double x, double y, double z) {
      positions.push_back(x);
      positions.push_back(y);
      positions.push_back(z);
      numbers.push_back(number);
    };
    push_atom(8, oxygen[0], oxygen[1], oxygen[2]);
    for (double sign : {1.0, -1.0}) {
      const double hx =
          oxygen[0] + bond * (along * radial[0] + sign * aside * tangent[0]);
      const double hy =
          oxygen[1] + bond * (along * radial[1] + sign * aside * tangent[1]);
      const double hz =
          oxygen[2] + bond * (along * radial[2] + sign * aside * tangent[2]);
      push_atom(1, hx, hy, hz);
    }
  }
}

double min_pair(const std::vector<double> &positions) {
  const size_t n = positions.size() / 3;
  double best = 1.0e300;
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = i + 1; j < n; ++j) {
      double d2 = 0.0;
      for (int axis = 0; axis < 3; ++axis) {
        const double d = positions[3 * i + axis] - positions[3 * j + axis];
        d2 += d * d;
      }
      best = std::min(best, std::sqrt(d2));
    }
  }
  return best;
}

AtomMatrix as_matrix(const std::vector<double> &flat) {
  const size_t n = flat.size() / 3;
  AtomMatrix matrix(n, 3);
  for (size_t i = 0; i < n; ++i)
    for (size_t axis = 0; axis < 3; ++axis)
      matrix(i, axis) = flat[3 * i + axis];
  return matrix;
}

struct Arm {
  std::vector<double> samples_ms;
  int reused = 0;
  double last_accuracy = 0.0;
};

double mean_of(const std::vector<double> &samples) {
  double sum = 0.0;
  for (double sample : samples)
    sum += sample;
  return samples.empty() ? 0.0 : sum / static_cast<double>(samples.size());
}

void print_samples(const char *name, const Arm &arm) {
  std::cout << name << "_ms";
  for (double sample : arm.samples_ms)
    std::cout << ' ' << sample;
  std::cout << "\n" << name << "_mean_ms " << mean_of(arm.samples_ms) << "\n";
  std::cout << name << "_reused " << arm.reused << "\n";
}

Arm time_arm(rgpot::XTBPot &pot, const std::vector<AtomMatrix> &frames,
             const std::vector<int> &numbers,
             const std::array<std::array<double, 3>, 3> &box, bool cold) {
  Arm arm;
  arm.samples_ms.reserve(frames.size());
  for (const AtomMatrix &frame : frames) {
    if (cold)
      pot.reset();
    const auto started = SteadyClock::now();
    auto [energy, forces, variance] = pot(frame, numbers, box);
    const auto stopped = SteadyClock::now();
    (void)energy;
    (void)forces;
    (void)variance;
    arm.samples_ms.push_back(
        std::chrono::duration<double, std::milli>(stopped - started).count());
    if (pot.reusedWavefunction())
      arm.reused += 1;
    arm.last_accuracy = pot.lastAccuracy();
  }
  return arm;
}

} // namespace

int main(int argc, char **argv) {
  force_serial();
  int repeats = 6;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--repeats" && i + 1 < argc)
      repeats = std::atoi(argv[++i]);
  }
  if (repeats < 1) {
    std::cerr << "--repeats must be positive\n";
    return 2;
  }

  std::cout.setf(std::ios::fixed);
  std::cout.precision(3);
  const char *threads = std::getenv("OMP_NUM_THREADS");
  std::cout << "xtb_warm_bench\n"
            << "hostname " << hostname() << "\n"
            << "omp_threads " << (threads ? threads : "unset") << "\n"
            << "repeats " << repeats << "\n"
            << "accuracy 1\n"
            << "warm_move_A 1\n"
            << "loose_accuracy 50\n"
            << "adaptive_force_eV_A 0.05\n";

  const std::array<std::array<double, 3>, 3> box = {
      {{80.0, 0.0, 0.0}, {0.0, 80.0, 0.0}, {0.0, 0.0, 80.0}}};
  int failed = 0;
  try {
    for (int molecules : {16, 20}) {
      std::vector<double> flat;
      std::vector<int> numbers;
      water_cluster(molecules, flat, numbers);
      const AtomMatrix base = as_matrix(flat);
      std::vector<double> stepped = flat;
      stepped[0] += 0.0625;
      const AtomMatrix step = as_matrix(stepped);
      std::vector<AtomMatrix> warm_frames;
      warm_frames.reserve(static_cast<size_t>(repeats));
      for (int i = 0; i < repeats; ++i)
        warm_frames.push_back(i % 2 == 0 ? step : base);
      std::vector<AtomMatrix> cold_frames(static_cast<size_t>(repeats), base);

      rgpot::XTBConfig cfg;
      cfg.method = rgpot::GFNMethod::GFN2xTB;
      rgpot::XTBConfig adaptive = cfg;
      adaptive.loose_accuracy = 50.0;
      adaptive.adaptive_force = 0.05;
      rgpot::XTBPot warm(cfg);
      rgpot::XTBPot cold(cfg);
      rgpot::XTBPot scheduled(adaptive);
      auto [e0, f0, v0] = warm(base, numbers, box);
      auto [e1, f1, v1] = cold(base, numbers, box);
      auto [e2, f2, v2] = scheduled(base, numbers, box);
      (void)e0;
      (void)v0;
      (void)e1;
      (void)v1;
      (void)e2;
      (void)v2;
      double force_sq = 0.0;
      for (size_t i = 0; i < f0.rows(); ++i)
        for (size_t j = 0; j < f0.cols(); ++j)
          force_sq += f0(i, j) * f0(i, j);
      const double force_rms =
          std::sqrt(force_sq / static_cast<double>(f0.rows() * f0.cols()));

      const Arm warm_arm = time_arm(warm, warm_frames, numbers, box, false);
      const Arm cold_arm = time_arm(cold, cold_frames, numbers, box, true);
      const Arm new_arm = time_arm(scheduled, warm_frames, numbers, box, false);
      std::cout << "molecules " << molecules << " n_atoms " << numbers.size()
                << " min_pair_A " << min_pair(flat) << " force_rms_eV_A "
                << force_rms << "\n";
      print_samples("cold", cold_arm);
      print_samples("warm", warm_arm);
      print_samples("new", new_arm);
      std::cout << "new_accuracy " << new_arm.last_accuracy << "\n";
      if (cold_arm.reused != 0 || warm_arm.reused != repeats ||
          new_arm.reused != repeats)
        failed = 1;
    }
  } catch (const std::exception &ex) {
    std::cerr << ex.what() << "\n";
    return 1;
  }
  return failed;
}
