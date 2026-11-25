// MIT License
// Copyright 2023--present rgpot developers
//
// GuptaPot checks: dimer at r0 matching rgpot_gupta bit-for-bit, analytic
// forces against central differences on a 13-atom cluster and a periodic fcc
// cell, translation and rotation invariance, stress against strain finite
// differences, PerAtom / Staged / Kept / C ABI parity.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <catch2/catch_all.hpp>

#include "rgpot/Gupta/GuptaPot.hpp"
#include "rgpot/Gupta/gupta.h"
#include "rgpot/KeptPotential.hpp"
#include "rgpot/PotHelpers.hpp"
#include "rgpot/stress.hpp"
#include "rgpot/types/AtomMatrix.hpp"

// Heap probe. rgpot_gupta takes its pair scratch from malloc, so the tests
// that it stays off the heap, and releases what it takes, count the calls the
// test thread makes to malloc, calloc, realloc and free. This file's own
// definitions of those four replace the allocator's for the whole executable
// and hand every request to the glibc entry points, which exist on every
// glibc, the 2.17 era conda sysroot included. Nothing here reads mallinfo2,
// so the count does not depend on which glibc builds the tests, and a thread
// is counted on its own: tcache and arena bookkeeping that glibc keeps for a
// new thread never passes through these functions.
//
// A sanitizer runtime brings its own malloc, and a non glibc host has no such
// entry points, so the probe and the tests that use it are compiled out there.
#if defined(__linux__) && defined(__GLIBC__)
#if defined(__SANITIZE_ADDRESS__) || defined(__SANITIZE_THREAD__)
#elif defined(__has_feature)
#if !__has_feature(address_sanitizer) && !__has_feature(thread_sanitizer) && \
    !__has_feature(memory_sanitizer)
#define RGPOT_HEAP_PROBE 1
#endif
#else
#define RGPOT_HEAP_PROBE 1
#endif
#endif

#ifdef RGPOT_HEAP_PROBE
#include <malloc.h>

extern "C" {
void *__libc_malloc(std::size_t size);
void *__libc_calloc(std::size_t count, std::size_t size);
void *__libc_realloc(void *ptr, std::size_t size);
void __libc_free(void *ptr);
}

namespace {

/// What the calling thread has asked of the allocator since measure_heap
/// armed it: requests, bytes still held, and the most bytes held at once.
struct HeapWindow {
  bool armed;
  long calls;
  long long live;
  long long peak;
};

__thread HeapWindow t_heap;

inline void heap_gain(void *ptr) noexcept {
  ++t_heap.calls;
  t_heap.live += static_cast<long long>(malloc_usable_size(ptr));
  if (t_heap.live > t_heap.peak) {
    t_heap.peak = t_heap.live;
  }
}

} // namespace

extern "C" {

void *malloc(std::size_t size) noexcept {
  void *ptr = __libc_malloc(size);
  if (t_heap.armed && ptr != nullptr) {
    heap_gain(ptr);
  }
  return ptr;
}

void *calloc(std::size_t count, std::size_t size) noexcept {
  void *ptr = __libc_calloc(count, size);
  if (t_heap.armed && ptr != nullptr) {
    heap_gain(ptr);
  }
  return ptr;
}

void *realloc(void *ptr, std::size_t size) noexcept {
  const long long held =
      t_heap.armed && ptr != nullptr
          ? static_cast<long long>(malloc_usable_size(ptr))
          : 0;
  void *next = __libc_realloc(ptr, size);
  if (t_heap.armed && next != nullptr) {
    t_heap.live -= held;
    heap_gain(next);
  }
  return next;
}

void free(void *ptr) noexcept {
  if (t_heap.armed && ptr != nullptr) {
    t_heap.live -= static_cast<long long>(malloc_usable_size(ptr));
  }
  __libc_free(ptr);
}

} // extern "C"
#endif

#if defined(RGPOT_HAS_RUST_CORE) && defined(__ELF__)
#include "rgpot/c_api.hpp"
#endif

using Catch::Matchers::WithinAbs;
using rgpot::types::AtomMatrix;

namespace {

constexpr int kVoigt[6][2] = {{0, 0}, {1, 1}, {2, 2}, {1, 2}, {0, 2}, {0, 1}};

std::array<std::array<double, 3>, 3> cube(double length) {
  return {{{length, 0.0, 0.0}, {0.0, length, 0.0}, {0.0, 0.0, length}}};
}

/// Atoms on a cubic grid at 1.05 spacing, no two coincident.
std::vector<double> lattice_cluster(std::size_t n) {
  std::vector<double> pos(3 * n, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    pos[3 * i] = 1.05 * static_cast<double>(i % 12);
    pos[3 * i + 1] = 1.05 * static_cast<double>((i / 12) % 12);
    pos[3 * i + 2] = 1.05 * static_cast<double>(i / 144);
  }
  return pos;
}

#ifdef RGPOT_HEAP_PROBE
/// Allocator traffic of the calling thread over one measure_heap call.
struct HeapUse {
  long calls = 0;
  long long net = 0;
  long long peak = 0;
};

/// Runs `work` with the probe armed. `work` must not use Catch2, which
/// allocates; check the returned numbers afterwards.
template <class Work> HeapUse measure_heap(Work &&work) {
  t_heap = HeapWindow{true, 0, 0, 0};
  work();
  t_heap.armed = false;
  return {t_heap.calls, t_heap.live, t_heap.peak};
}

bool probe_sees_malloc() {
  const HeapUse use = measure_heap([] {
    void *volatile block = std::malloc(256);
    std::free(block);
  });
  return use.calls == 1 && use.net == 0 && use.peak >= 256;
}
#endif

bool same_bits(double a, double b) {
  return std::memcmp(&a, &b, sizeof(double)) == 0;
}

bool same_bits(const std::vector<double> &a, const std::vector<double> &b) {
  return a.size() == b.size() &&
         (a.empty() ||
          std::memcmp(a.data(), b.data(), a.size() * sizeof(double)) == 0);
}

void require_scaled_match(double actual, double expected) {
  REQUIRE(std::isfinite(actual));
  REQUIRE(std::isfinite(expected));
  const double magnitude =
      std::max({1.0, std::abs(actual), std::abs(expected)});
  const double tolerance =
      64.0 * std::numeric_limits<double>::epsilon() * magnitude;
  REQUIRE_THAT(actual, WithinAbs(expected, tolerance));
}

struct StressEval {
  double energy = 0.0;
  double stress[9] = {};
  int has_stress = 0;
  std::vector<double> forces;
};

StressEval eval_stress(const rgpot::GuptaPot &pot,
                       const std::vector<double> &pos, const double *box) {
  StressEval e;
  e.forces.assign(pos.size(), 0.0);
  rgpot::ForceOut out{};
  out.F = e.forces.data();
  std::fill(out.stress, out.stress + 9, 7.0);
  pot.forceImpl(rgpot::ForceInput{pos.size() / 3, pos.data(), nullptr, box},
                &out);
  e.energy = out.energy;
  std::copy(out.stress, out.stress + 9, e.stress);
  e.has_stress = out.has_stress;
  return e;
}

/// Perturbed 13-atom cuboctahedral cluster scaled to nearest-neighbour spacing r0.
std::vector<double> cluster13(double r0) {
  const double s = r0 / std::sqrt(2.0);
  const double raw[13][3] = {
      {0.0, 0.0, 0.0},    {s, s, 0.0},     {s, -s, 0.0},   {-s, s, 0.0},
      {-s, -s, 0.0},      {s, 0.0, s},     {s, 0.0, -s},   {-s, 0.0, s},
      {-s, 0.0, -s},      {0.0, s, s},     {0.0, s, -s},   {0.0, -s, s},
      {0.0, -s, -s},
  };
  std::vector<double> pos(39);
  for (int i = 0; i < 13; ++i) {
    for (int k = 0; k < 3; ++k) {
      const double jitter =
          0.03 * r0 * std::sin(1.7 * static_cast<double>(3 * i + k) + 0.4);
      pos[static_cast<std::size_t>(3 * i + k)] = raw[i][k] + jitter;
    }
  }
  return pos;
}

/// Perturbed periodic fcc supercell of `cells^3` conventional cells.
std::vector<double> fcc_cell(int cells, double lattice, bool sheared,
                             double *box) {
  const double basis[4][3] = {
      {0, 0, 0}, {0.5, 0.5, 0}, {0.5, 0, 0.5}, {0, 0.5, 0.5}};
  const double side = cells * lattice;
  std::fill(box, box + 9, 0.0);
  box[0] = side;
  box[4] = side;
  box[8] = side;
  if (sheared) {
    box[3] = 0.06 * side;
    box[6] = 0.04 * side;
    box[7] = -0.05 * side;
  }
  std::vector<double> pos;
  int idx = 0;
  for (int ix = 0; ix < cells; ++ix) {
    for (int iy = 0; iy < cells; ++iy) {
      for (int iz = 0; iz < cells; ++iz) {
        for (const auto &b : basis) {
          const double fx =
              (ix + b[0]) / cells +
              0.004 * std::sin(1.3 * static_cast<double>(3 * idx) + 0.2);
          const double fy =
              (iy + b[1]) / cells +
              0.004 * std::sin(1.3 * static_cast<double>(3 * idx + 1) + 0.5);
          const double fz =
              (iz + b[2]) / cells +
              0.004 * std::sin(1.3 * static_cast<double>(3 * idx + 2) + 0.8);
          for (int c = 0; c < 3; ++c) {
            pos.push_back(fx * box[c] + fy * box[3 + c] + fz * box[6 + c]);
          }
          ++idx;
        }
      }
    }
  }
  return pos;
}

void check_forces_central_difference(const rgpot::GuptaPot &pot,
                                     const std::vector<double> &pos,
                                     const double *box, double h, double tol) {
  const std::size_t n = pos.size() / 3;
  const StressEval base = eval_stress(pot, pos, box);
  double net[3] = {0.0, 0.0, 0.0};
  for (std::size_t i = 0; i < n; ++i) {
    for (int k = 0; k < 3; ++k) {
      net[k] += base.forces[3 * i + static_cast<std::size_t>(k)];
      std::vector<double> plus = pos;
      std::vector<double> minus = pos;
      plus[3 * i + static_cast<std::size_t>(k)] += h;
      minus[3 * i + static_cast<std::size_t>(k)] -= h;
      const double e_plus = eval_stress(pot, plus, box).energy;
      const double e_minus = eval_stress(pot, minus, box).energy;
      const double numeric = -(e_plus - e_minus) / (2.0 * h);
      const double analytic = base.forces[3 * i + static_cast<std::size_t>(k)];
      INFO("atom " << i << " coord " << k);
      REQUIRE_THAT(analytic, WithinAbs(numeric, tol * (1.0 + std::abs(numeric))));
    }
  }
  for (double component : net) {
    REQUIRE_THAT(component, WithinAbs(0.0, 1e-10));
  }
}

} // namespace

TEST_CASE("GuptaPot dimer at r0 gives 2A - 2xi and matches rgpot_gupta bit for bit",
          "[Gupta]") {
  const double a = 0.1477;
  const double xi = 0.8900;
  const double p = 9.689;
  const double q = 4.602;
  const double r0 = 1.0;

  const std::vector<double> pos = {0.0, 0.0, 0.0, 1.0, 0.0, 0.0};
  const unsigned char slot[2] = {0, 0};
  double c_energy = 0.0;
  double c_grad[6] = {};
  rgpot_gupta(pos.data(), 2, slot, 1, &a, &xi, &p, &q, &r0, &c_energy, c_grad);
  std::vector<double> c_forces(6);
  for (std::size_t k = 0; k < 6; ++k) {
    c_forces[k] = -c_grad[k];
  }

  const rgpot::GuptaConfig cfg{
      .A = a, .xi = xi, .p = p, .q = q, .r0 = r0, .cutoff = 0.0};
  rgpot::GuptaPot pot(cfg);
  REQUIRE(pot.getType() == rgpot::PotType::Gupta);

  const AtomMatrix mat{{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}};
  const std::vector<int> types{1, 1};
  auto [energy, forces, variance] = pot(mat, types, cube(40.0));
  REQUIRE(variance == 0.0);
  REQUIRE(energy == Catch::Approx(2.0 * a - 2.0 * xi).margin(1e-12));
  REQUIRE(energy == Catch::Approx(-1.4846).margin(1e-12));
  REQUIRE(same_bits(energy, c_energy));
  std::vector<double> got_forces(forces.data(), forces.data() + 6);
  REQUIRE(same_bits(got_forces, c_forces));

  // Periodic and finite-cutoff path also matches rgpot_gupta at r0 bit-for-bit.
  rgpot::GuptaPot periodic_pot({.A = a,
                                .xi = xi,
                                .p = p,
                                .q = q,
                                .r0 = r0,
                                .cutoff = 5.0,
                                .periodic = true,
                                .stress = true});
  const double box[9] = {40.0, 0.0, 0.0, 0.0, 40.0, 0.0, 0.0, 0.0, 40.0};
  const StressEval per = eval_stress(periodic_pot, pos, box);
  REQUIRE(per.has_stress == 1);
  REQUIRE(same_bits(per.energy, c_energy));
  REQUIRE(same_bits(per.energy, c_energy));
  REQUIRE(same_bits(per.forces[0], c_forces[0]));
  REQUIRE(same_bits(per.forces[3], c_forces[3]));
  REQUIRE(per.forces == c_forces);
}

TEST_CASE("GuptaPot degenerate pairs match all-pairs and neighbour paths",
          "[Gupta]") {
  rgpot::GuptaConfig base{
      .A = 0.2061, .xi = 0.0, .p = 10.229, .q = 4.036, .r0 = 2.5};
  rgpot::GuptaConfig bounded = base;
  bounded.cutoff = 4.0 * base.r0;

  auto compare_paths = [&](const std::vector<double> &positions) {
    const StressEval all_pairs =
        eval_stress(rgpot::GuptaPot(base), positions, nullptr);
    const StressEval neighbours =
        eval_stress(rgpot::GuptaPot(bounded), positions, nullptr);
    require_scaled_match(all_pairs.energy, neighbours.energy);
    REQUIRE(all_pairs.forces.size() == neighbours.forces.size());
    for (std::size_t k = 0; k < all_pairs.forces.size(); ++k) {
      require_scaled_match(all_pairs.forces[k], neighbours.forces[k]);
    }
    return all_pairs;
  };

  SECTION("zero hopping dimer") {
    const std::vector<double> positions = {
        0.0, 0.0, 0.0, base.r0, 0.0, 0.0};
    const StressEval result = compare_paths(positions);
    const double analytic_force = -(base.p / base.r0) * 2.0 * base.A;
    require_scaled_match(result.energy, 2.0 * base.A);
    require_scaled_match(result.forces[0], analytic_force);
    require_scaled_match(result.forces[3], -analytic_force);
    require_scaled_match(analytic_force, -1.68655752);
  }

  base.xi = 0.8900;
  bounded.xi = base.xi;

  SECTION("coincident dimer") {
    const std::vector<double> positions(6, 0.0);
    const StressEval all_pairs =
        eval_stress(rgpot::GuptaPot(base), positions, nullptr);
    // The free cluster delegates to rgpot_gupta (+infinity) and poisons forces.
    REQUIRE(std::isinf(all_pairs.energy));
    REQUIRE(all_pairs.energy > 0.0);
    for (double force : all_pairs.forces) {
      REQUIRE(std::isnan(force));
    }
    // The neighbour walk poisons both energy and forces across all modes.
    for (const auto mode : {rgpot::nlist::Mode::Auto,
                            rgpot::nlist::Mode::Stored,
                            rgpot::nlist::Mode::Stream}) {
      INFO("mode " << static_cast<int>(mode));
      rgpot::GuptaPot pot(bounded);
      pot.setNeighbourMode(mode);
      const StressEval neighbours = eval_stress(pot, positions, nullptr);
      REQUIRE(std::isnan(neighbours.energy));
      for (double force : neighbours.forces) {
        REQUIRE(std::isnan(force));
      }
    }
    // Periodic stress and per-atom evaluation also poison energy, forces,
    // stress, per-atom energy, and per-atom virial.
    rgpot::GuptaConfig per_cfg = bounded;
    per_cfg.periodic = true;
    per_cfg.stress = true;
    const double box[9] = {20.0, 0.0, 0.0, 0.0, 20.0, 0.0, 0.0, 0.0, 20.0};
    const rgpot::GuptaPot per_pot(per_cfg);
    std::vector<double> pa_forces(6, 0.0);
    double pa_e[2] = {};
    double pa_v[12] = {};
    rgpot::ForceOut pa_out{};
    pa_out.F = pa_forces.data();
    per_pot.forcePerAtom(rgpot::ForceInput{2, positions.data(), nullptr, box},
                         &pa_out, rgpot::PerAtomOut{pa_e, pa_v});
    REQUIRE(std::isnan(pa_out.energy));
    for (double force : pa_forces) {
      REQUIRE(std::isnan(force));
    }
    for (double s : pa_out.stress) {
      REQUIRE(std::isnan(s));
    }
    REQUIRE(std::isnan(pa_e[0]));
    REQUIRE(std::isnan(pa_e[1]));
    for (double v : pa_v) {
      REQUIRE(std::isnan(v));
    }
  }

  SECTION("trimer with one coincident pair") {
    const std::vector<double> positions = {
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, base.r0, 0.0, 0.0};
    const StressEval all_pairs =
        eval_stress(rgpot::GuptaPot(base), positions, nullptr);
    const StressEval neighbours =
        eval_stress(rgpot::GuptaPot(bounded), positions, nullptr);
    REQUIRE(std::isinf(all_pairs.energy));
    REQUIRE(all_pairs.energy > 0.0);
    for (double force : all_pairs.forces) {
      REQUIRE(std::isnan(force));
    }
    REQUIRE(std::isnan(neighbours.energy));
    for (double force : neighbours.forces) {
      REQUIRE(std::isnan(force));
    }
  }

  SECTION("staged coincidence") {
    const rgpot::GuptaPot pot(bounded);
    const double pos[6] = {};
    const int atmnrs[2] = {1, 1};
    const int rows[1] = {0};
    const int count[2] = {1, 0};
    const int neigh0[1] = {1};
    const int *neighs[2] = {neigh0, nullptr};
    rgpot::HalfList list;
    list.nAtoms = 2;
    list.nOwned = 2;
    list.pos = pos;
    list.atmnrs = atmnrs;
    list.nRows = 1;
    list.rows = rows;
    list.count = count;
    list.neighbours = neighs;
    double dens[2] = {};
    pot.densityStage(list, dens);
    REQUIRE(std::isnan(dens[0]));
    REQUIRE(std::isnan(dens[1]));
    double forces[6] = {};
    rgpot::StageTally tally;
    pot.forceStage(list, dens, forces, &tally);
    REQUIRE(std::isnan(tally.energy));
    for (double component : forces) {
      REQUIRE(std::isnan(component));
    }
  }
}

TEST_CASE("GuptaPot forces match central differences near a coincidence",
          "[Gupta]") {
  rgpot::GuptaConfig cfg{
      .A = 0.2061, .xi = 0.8900, .p = 10.229, .q = 4.036, .r0 = 2.5};
  const double gap = std::ldexp(cfg.r0, -8);
  const std::vector<double> pos = {0.0, 0.0, 0.0, gap, 0.0, 0.0};
  rgpot::GuptaConfig cut = cfg;
  cut.cutoff = 4.0 * cfg.r0;
  check_forces_central_difference(rgpot::GuptaPot(cut), pos, nullptr, 1e-6,
                                  1e-5);

  const unsigned char slot[2] = {0, 0};
  double c_energy = 0.0;
  double c_grad[6] = {};
  rgpot_gupta(pos.data(), 2, slot, 1, &cfg.A, &cfg.xi, &cfg.p, &cfg.q, &cfg.r0,
              &c_energy, c_grad);
  const StressEval got = eval_stress(rgpot::GuptaPot(cut), pos, nullptr);
  require_scaled_match(got.energy, c_energy);
  require_scaled_match(got.forces[0], -c_grad[0]);
  require_scaled_match(got.forces[3], -c_grad[3]);
}

TEST_CASE("GuptaPot propagates a NaN coordinate on the neighbour path",
          "[Gupta]") {
  // The free-cluster kernel is covered by rgpot-pair. A positive cutoff
  // takes the neighbour walk, which used to drop a NaN separation and
  // leave a finite energy and zero force.
  const rgpot::GuptaConfig cfg{.A = 0.2061,
                               .xi = 1.790,
                               .p = 10.229,
                               .q = 4.036,
                               .r0 = 2.884,
                               .cutoff = 6.0};
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const std::vector<double> pos = {
      nan, 0.0, 0.0, 1.0, 0.0, 0.0, 0.2, 1.1, 0.0};
  const StressEval got = eval_stress(rgpot::GuptaPot(cfg), pos, nullptr);
  REQUIRE(std::isnan(got.energy));
  REQUIRE(got.forces.size() == pos.size());
  for (double component : got.forces) {
    REQUIRE(std::isnan(component));
  }

  // A partner in a non-adjacent bin of a 1000^3 box still reaches the NaN
  // atom across Auto, Stored, and Stream modes.
  const double box[9] = {1000.0, 0.0, 0.0, 0.0, 1000.0, 0.0, 0.0, 0.0, 1000.0};
  const std::vector<double> far = {nan, 0.0, 0.0, 500.0, 0.0, 0.0};
  for (const auto mode : {rgpot::nlist::Mode::Auto, rgpot::nlist::Mode::Stored,
                          rgpot::nlist::Mode::Stream}) {
    INFO("mode " << static_cast<int>(mode));
    rgpot::GuptaPot pot(cfg);
    pot.setNeighbourMode(mode);
    const StressEval cross = eval_stress(pot, far, box);
    REQUIRE_FALSE(std::isfinite(cross.energy));
    REQUIRE(cross.forces.size() == far.size());
    for (double component : cross.forces) {
      REQUIRE_FALSE(std::isfinite(component));
    }
  }

  const rgpot::GuptaPot pot(cfg);
  const double staged_pos[6] = {nan, 0.0, 0.0, 1.0, 0.0, 0.0};
  const int atmnrs[2] = {1, 1};
  const int rows[1] = {0};
  const int count[2] = {1, 0};
  const int neigh0[1] = {1};
  const int *neighs[2] = {neigh0, nullptr};
  rgpot::HalfList list;
  list.nAtoms = 2;
  list.nOwned = 2;
  list.pos = staged_pos;
  list.atmnrs = atmnrs;
  list.nRows = 1;
  list.rows = rows;
  list.count = count;
  list.neighbours = neighs;
  double dens[2] = {};
  pot.densityStage(list, dens);
  REQUIRE(std::isnan(dens[0]));
  REQUIRE(std::isnan(dens[1]));
  double forces[6] = {};
  rgpot::StageTally tally;
  pot.forceStage(list, dens, forces, &tally);
  REQUIRE(std::isnan(tally.energy));
  for (double component : forces) {
    REQUIRE(std::isnan(component));
  }
}

TEST_CASE("GuptaPot zero density stays finite and a NaN density propagates",
          "[Gupta]") {
  const rgpot::GuptaPot pot({.A = 0.2061,
                             .xi = 0.8900,
                             .p = 10.229,
                             .q = 4.036,
                             .r0 = 2.5,
                             .cutoff = 8.0});
  const double pos[6] = {0.0, 0.0, 0.0, 2.5, 0.0, 0.0};
  const int atmnrs[2] = {1, 1};
  const int rows[1] = {0};
  const int count[2] = {1, 0};
  const int neigh0[1] = {1};
  const int *neighs[2] = {neigh0, nullptr};
  rgpot::HalfList list;
  list.nAtoms = 2;
  list.nOwned = 2;
  list.pos = pos;
  list.atmnrs = atmnrs;
  list.nRows = 1;
  list.rows = rows;
  list.count = count;
  list.neighbours = neighs;

  double dens[2] = {};
  double forces[6] = {};
  rgpot::StageTally tally;
  pot.forceStage(list, dens, forces, &tally);
  REQUIRE(std::isfinite(tally.energy));
  REQUIRE(std::isfinite(forces[0]));
  REQUIRE(forces[0] != 0.0);

  const double nan = std::numeric_limits<double>::quiet_NaN();
  dens[0] = nan;
  dens[1] = nan;
  std::fill(forces, forces + 6, 0.0);
  tally = {};
  pot.forceStage(list, dens, forces, &tally);
  REQUIRE(std::isnan(tally.energy));
  REQUIRE(std::isnan(forces[0]));
  REQUIRE(std::isnan(forces[3]));
}

TEST_CASE("GuptaPot forces match central differences on a 13-atom cluster",
          "[Gupta]") {
  const rgpot::GuptaConfig cfg{
      .A = 0.2061, .xi = 1.790, .p = 10.229, .q = 4.036, .r0 = 2.884};
  const auto pos = cluster13(cfg.r0);

  SECTION("unbounded all-pairs matches C kernel bit for bit") {
    rgpot::GuptaPot pot(cfg);
    check_forces_central_difference(pot, pos, nullptr, 1e-6, 1e-8);

    const std::vector<unsigned char> slot(13, 0);
    double c_energy = 0.0;
    std::vector<double> c_grad(39, 0.0);
    rgpot_gupta(pos.data(), 13, slot.data(), 1, &cfg.A, &cfg.xi, &cfg.p,
                &cfg.q, &cfg.r0, &c_energy, c_grad.data());
    std::vector<double> c_forces(39);
    for (std::size_t k = 0; k < 39; ++k) {
      c_forces[k] = -c_grad[k];
    }
    const StressEval got = eval_stress(pot, pos, nullptr);
    REQUIRE(same_bits(got.energy, c_energy));
    REQUIRE(same_bits(got.forces, c_forces));
  }

  SECTION("truncated cluster") {
    rgpot::GuptaConfig cut_cfg = cfg;
    cut_cfg.cutoff = 4.5;
    rgpot::GuptaPot pot(cut_cfg);
    check_forces_central_difference(pot, pos, nullptr, 1e-6, 1e-8);
  }
}

TEST_CASE("GuptaPot forces match central differences on a periodic fcc cell",
          "[Gupta]") {
  const double a = 4.08;
  const bool sheared = GENERATE(false, true);
  double box[9] = {};
  const auto pos = fcc_cell(2, a, sheared, box);
  const rgpot::GuptaPot pot({.A = 0.2061,
                             .xi = 1.790,
                             .p = 10.229,
                             .q = 4.036,
                             .r0 = a / std::sqrt(2.0),
                             .cutoff = 3.9,
                             .periodic = true,
                             .stress = true});
  check_forces_central_difference(pot, pos, box, 1e-5, 1e-7);
}

TEST_CASE("GuptaPot is translation and rotation invariant", "[Gupta]") {
  const rgpot::GuptaConfig cfg{.A = 0.2061,
                               .xi = 1.790,
                               .p = 10.229,
                               .q = 4.036,
                               .r0 = 2.884,
                               .cutoff = 0.0,
                               .periodic = false,
                               .stress = false};
  const rgpot::GuptaPot cluster_pot(cfg);
  const auto pos = cluster13(cfg.r0);
  const StressEval ref = eval_stress(cluster_pot, pos, nullptr);

  // Translation on the 13-atom cluster.
  const double shift[3] = {3.1, -2.4, 5.7};
  std::vector<double> translated = pos;
  for (std::size_t i = 0; i < 13; ++i) {
    for (int k = 0; k < 3; ++k) {
      translated[3 * i + static_cast<std::size_t>(k)] += shift[k];
    }
  }
  const StressEval tr = eval_stress(cluster_pot, translated, nullptr);
  REQUIRE_THAT(tr.energy, WithinAbs(ref.energy, 1e-12));
  for (std::size_t k = 0; k < 39; ++k) {
    REQUIRE_THAT(tr.forces[k], WithinAbs(ref.forces[k], 1e-12));
  }

  // Translation on a periodic cell (across periodic boundaries).
  double box[9] = {};
  const double lattice = 4.08;
  const auto fcc = fcc_cell(2, lattice, false, box);
  const rgpot::GuptaPot periodic_pot({.A = 0.2061,
                                      .xi = 1.790,
                                      .p = 10.229,
                                      .q = 4.036,
                                      .r0 = lattice / std::sqrt(2.0),
                                      .cutoff = 3.9,
                                      .periodic = true,
                                      .stress = true});
  const StressEval p_ref = eval_stress(periodic_pot, fcc, box);
  std::vector<double> fcc_shifted = fcc;
  for (std::size_t i = 0; i < fcc.size() / 3; ++i) {
    for (int k = 0; k < 3; ++k) {
      fcc_shifted[3 * i + static_cast<std::size_t>(k)] += shift[k];
    }
  }
  const StressEval p_tr = eval_stress(periodic_pot, fcc_shifted, box);
  REQUIRE_THAT(p_tr.energy, WithinAbs(p_ref.energy, 1e-12));
  for (std::size_t k = 0; k < fcc.size(); ++k) {
    REQUIRE_THAT(p_tr.forces[k], WithinAbs(p_ref.forces[k], 1e-12));
  }

  // Proper 3D rotation R = R_z(0.6) * R_y(-0.4) * R_x(0.9), det(R) == +1.
  const double cx = std::cos(0.9), sx = std::sin(0.9);
  const double cy = std::cos(-0.4), sy = std::sin(-0.4);
  const double cz = std::cos(0.6), sz = std::sin(0.6);
  const double R[3][3] = {
      {cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx},
      {sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx},
      {-sy, cy * sx, cy * cx},
  };
  std::vector<double> rotated(pos.size(), 0.0);
  for (std::size_t i = 0; i < 13; ++i) {
    for (int a = 0; a < 3; ++a) {
      double sum = 0.0;
      for (int b = 0; b < 3; ++b) {
        sum += R[a][b] * pos[3 * i + static_cast<std::size_t>(b)];
      }
      rotated[3 * i + static_cast<std::size_t>(a)] = sum;
    }
  }
  const StressEval rot = eval_stress(cluster_pot, rotated, nullptr);
  REQUIRE_THAT(rot.energy, WithinAbs(ref.energy, 1e-12));
  for (std::size_t i = 0; i < 13; ++i) {
    for (int a = 0; a < 3; ++a) {
      double expected_f = 0.0;
      for (int b = 0; b < 3; ++b) {
        expected_f += R[a][b] * ref.forces[3 * i + static_cast<std::size_t>(b)];
      }
      REQUIRE_THAT(rot.forces[3 * i + static_cast<std::size_t>(a)],
                   WithinAbs(expected_f, 1e-12));
    }
  }
}

TEST_CASE("GuptaPot stress is opt-in and matches strain finite differences",
          "[Gupta][stress]") {
  REQUIRE_FALSE(rgpot::GuptaConfig{}.stress);

  const double a = 4.08;
  double box[9] = {};
  const auto pos = fcc_cell(2, a, true, box);
  const rgpot::GuptaConfig base{.A = 0.2061,
                                .xi = 1.790,
                                .p = 10.229,
                                .q = 4.036,
                                .r0 = a / std::sqrt(2.0),
                                .cutoff = 3.9,
                                .periodic = true,
                                .stress = false};
  rgpot::GuptaConfig with_stress = base;
  with_stress.stress = true;

  const rgpot::GuptaPot plain(base);
  const rgpot::GuptaPot stressed(with_stress);
  REQUIRE_FALSE(plain.caps().stress);
  REQUIRE(stressed.caps().stress);
  REQUIRE(plain.paramsKey() != stressed.paramsKey());

  const StressEval e_plain = eval_stress(plain, pos, box);
  const StressEval e_stress = eval_stress(stressed, pos, box);
  REQUIRE(e_plain.has_stress == 0);
  REQUIRE(std::all_of(e_plain.stress, e_plain.stress + 9,
                      [](double v) { return v == 7.0; }));
  REQUIRE(e_stress.has_stress == 1);
  REQUIRE(same_bits(e_plain.energy, e_stress.energy));
  REQUIRE(same_bits(e_plain.forces, e_stress.forces));

  // Cluster with null box publishes no stress even when stress=true.
  const rgpot::GuptaPot cluster_stress({.cutoff = 0.0, .stress = true});
  const StressEval c =
      eval_stress(cluster_stress, {0.0, 0.0, 0.0, 1.0, 0.1, 0.0}, nullptr);
  REQUIRE(c.has_stress == 0);
  REQUIRE(
      std::all_of(c.stress, c.stress + 9, [](double v) { return v == 0.0; }));

  // Verify all 6 Voigt components against central strain finite difference.
  const double h = 1.0e-6;
  const double volume = rgpot::cellVolume(box);
  for (const auto &ab : kVoigt) {
    const int r_idx = ab[0];
    const int c_idx = ab[1];
    double e_side[2] = {};
    for (int side = 0; side < 2; ++side) {
      const double step = (side == 0 ? h : -h) * (r_idx == c_idx ? 1.0 : 0.5);
      std::vector<double> rows(pos);
      rows.insert(rows.end(), box, box + 9);
      for (std::size_t row = 0; row < rows.size(); row += 3) {
        const double ra = rows[row + static_cast<std::size_t>(r_idx)];
        const double rb = rows[row + static_cast<std::size_t>(c_idx)];
        rows[row + static_cast<std::size_t>(c_idx)] += step * ra;
        if (r_idx != c_idx) {
          rows[row + static_cast<std::size_t>(r_idx)] += step * rb;
        }
      }
      const std::vector<double> deformed(rows.begin(), rows.end() - 9);
      e_side[side] =
          eval_stress(stressed, deformed, rows.data() + deformed.size()).energy;
    }
    const double numeric = (e_side[0] - e_side[1]) / (2.0 * h * volume);
    INFO("component " << r_idx << c_idx);
    REQUIRE_THAT(e_stress.stress[3 * r_idx + c_idx], WithinAbs(numeric, 1e-8));
    REQUIRE(e_stress.stress[3 * c_idx + r_idx] ==
            e_stress.stress[3 * r_idx + c_idx]);
  }
}

TEST_CASE("GuptaPot C entry point matches GuptaPot and validates inputs",
          "[Gupta]") {
  const rgpot_gupta_config raw{
      .A = 0.2061,
      .xi = 1.790,
      .p = 10.229,
      .q = 4.036,
      .r0 = 2.884,
      .cutoff = 3.9,
      .periodic = 1,
  };
  double box[9] = {};
  const auto pos = fcc_cell(2, 4.08, false, box);
  const int32_t natoms = static_cast<int32_t>(pos.size() / 3);
  double c_energy = 0.0;
  std::vector<double> c_forces(pos.size(), 0.0);
  REQUIRE(rgpot_gupta_energy_forces(&raw, natoms, pos.data(), box, &c_energy,
                                    c_forces.data()) == 0);

  const rgpot::GuptaPot pot({.A = raw.A,
                             .xi = raw.xi,
                             .p = raw.p,
                             .q = raw.q,
                             .r0 = raw.r0,
                             .cutoff = raw.cutoff,
                             .periodic = true});
  const StressEval cpp = eval_stress(pot, pos, box);
  REQUIRE(same_bits(c_energy, cpp.energy));
  REQUIRE(same_bits(c_forces, cpp.forces));

  // Bad inputs return 1.
  REQUIRE(rgpot_gupta_energy_forces(nullptr, natoms, pos.data(), box,
                                    &c_energy, c_forces.data()) == 1);
  REQUIRE(rgpot_gupta_energy_forces(&raw, -1, pos.data(), box, &c_energy,
                                    c_forces.data()) == 1);
  REQUIRE(rgpot_gupta_energy_forces(&raw, natoms, pos.data(), nullptr,
                                    &c_energy, c_forces.data()) == 1);

  // Degenerate / edge cases.
  REQUIRE_THROWS_AS(rgpot::GuptaPot({.r0 = 0.0}), std::invalid_argument);
  REQUIRE_THROWS_AS(rgpot::GuptaPot({.cutoff = 0.0, .periodic = true}),
                    std::invalid_argument);

  // Coincident atoms: the free cluster is +infinity from rgpot_gupta, and a
  // cutoff poisons the energy; both poison the forces with NaN.
  rgpot_gupta_config open = raw;
  open.cutoff = 0.0;
  open.periodic = 0;
  const double stacked[6] = {};
  double stacked_energy = 0.0;
  double stacked_forces[6] = {};
  REQUIRE(rgpot_gupta_energy_forces(&open, 2, stacked, nullptr, &stacked_energy,
                                    stacked_forces) == 0);
  REQUIRE(std::isinf(stacked_energy));
  REQUIRE(stacked_energy > 0.0);
  for (double component : stacked_forces) {
    REQUIRE(std::isnan(component));
  }
  open.cutoff = 4.0;
  stacked_energy = 0.0;
  std::fill(stacked_forces, stacked_forces + 6, 0.0);
  REQUIRE(rgpot_gupta_energy_forces(&open, 2, stacked, nullptr, &stacked_energy,
                                    stacked_forces) == 0);
  REQUIRE(std::isnan(stacked_energy));
  for (double component : stacked_forces) {
    REQUIRE(std::isnan(component));
  }

  // Isolated atom beyond cutoff has rho_i = 0 and produces finite zero force.
  const rgpot::GuptaPot cut_pot({.cutoff = 2.5});
  const std::vector<double> three = {
      0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 20.0, 0.0, 0.0};
  const StressEval iso = eval_stress(cut_pot, three, nullptr);
  REQUIRE(std::isfinite(iso.energy));
  REQUIRE(iso.forces[6] == 0.0);
  REQUIRE(iso.forces[7] == 0.0);
  REQUIRE(iso.forces[8] == 0.0);
}

TEST_CASE("GuptaPot C entry point stores the reason of every failure",
          "[Gupta]") {
  rgpot_gupta_config raw{
      .A = 0.2061,
      .xi = 1.790,
      .p = 10.229,
      .q = 4.036,
      .r0 = 2.884,
      .cutoff = 0.0,
      .periodic = 0,
  };
  const double pos[6] = {0.0, 0.0, 0.0, 2.884, 0.0, 0.0};
  double energy = 0.0;
  double forces[6] = {};
  const auto fails_with = [](const rgpot_gupta_config *cfg, int32_t natoms,
                             const double *positions, double *e, double *f,
                             const std::string &message) {
#if defined(RGPOT_HAS_RUST_CORE) && defined(__ELF__)
    rgpot_set_last_error("stale message from an earlier call");
#endif
    REQUIRE(rgpot_gupta_energy_forces(cfg, natoms, positions, nullptr, e, f) ==
            1);
    REQUIRE(std::string(rgpot::forceLastError()) == message);
#if defined(RGPOT_HAS_RUST_CORE) && defined(__ELF__)
    REQUIRE(std::string(rgpot_last_error()) == message);
#endif
  };

  fails_with(nullptr, 2, pos, &energy, forces,
             "invalid parameter: cfg is null");
  fails_with(&raw, -1, pos, &energy, forces,
             "invalid parameter: natoms is negative");
  fails_with(&raw, 2, pos, nullptr, forces,
             "invalid parameter: energy is null");
  fails_with(&raw, 2, nullptr, &energy, forces,
             "invalid parameter: positions is null");
  fails_with(&raw, 2, pos, &energy, nullptr,
             "invalid parameter: forces is null");

  // A configuration the constructor refuses reaches the caller as its
  // exception text.
  rgpot_gupta_config bad_r0 = raw;
  bad_r0.r0 = 0.0;
  fails_with(&bad_r0, 2, pos, &energy, forces,
             "Gupta r0 must be positive and finite");
  rgpot_gupta_config periodic_uncut = raw;
  periodic_uncut.periodic = 1;
  fails_with(&periodic_uncut, 2, pos, &energy, forces,
             "periodic Gupta needs a positive cutoff");

  // A success leaves the stored reason in place.
  REQUIRE(rgpot_gupta_energy_forces(&raw, 2, pos, nullptr, &energy, forces) ==
          0);
  REQUIRE(std::string(rgpot::forceLastError()) ==
          "periodic Gupta needs a positive cutoff");
}

#ifdef RGPOT_HEAP_PROBE
TEST_CASE("rgpot_gupta thread-local buffers grow and hold steady at fixed N",
          "[Gupta][heap]") {
  REQUIRE(probe_sees_malloc());
  const rgpot::GuptaConfig cfg{};
  constexpr std::size_t small_n = 64;
  constexpr std::size_t large_n = 1500;
  const std::vector<double> small_pos = lattice_cluster(small_n);
  const std::vector<double> large_pos = lattice_cluster(large_n);
  const std::vector<unsigned char> small_slot(small_n, 0);
  const std::vector<unsigned char> large_slot(large_n, 0);

  struct Traffic {
    HeapUse first_small;
    HeapUse steady_small;
    HeapUse grow_to_large;
    HeapUse steady_large;
  };
  Traffic use{};
  std::thread worker([&] {
    std::vector<double> small_grad(3 * small_n, 0.0);
    std::vector<double> large_grad(3 * large_n, 0.0);
    double energy = 0.0;
    auto call_small = [&] {
      rgpot_gupta(small_pos.data(), small_n, small_slot.data(), 1, &cfg.A, &cfg.xi,
                  &cfg.p, &cfg.q, &cfg.r0, &energy, small_grad.data());
    };
    auto call_large = [&] {
      rgpot_gupta(large_pos.data(), large_n, large_slot.data(), 1, &cfg.A, &cfg.xi,
                  &cfg.p, &cfg.q, &cfg.r0, &energy, large_grad.data());
    };
    use.first_small = measure_heap(call_small);
    use.steady_small = measure_heap([&] {
      call_small();
      call_small();
    });
    use.grow_to_large = measure_heap(call_large);
    use.steady_large = measure_heap([&] {
      call_large();
      call_large();
      call_large();
    });
    REQUIRE(std::isfinite(energy));
  });
  worker.join();

  REQUIRE(use.first_small.calls > 0);
  REQUIRE(use.first_small.net > 0);
  REQUIRE(use.steady_small.calls == 0);
  REQUIRE(use.steady_small.net == 0);
  REQUIRE(use.grow_to_large.calls > 0);
  REQUIRE(use.grow_to_large.net > 0);
  REQUIRE(use.steady_large.calls == 0);
  REQUIRE(use.steady_large.net == 0);
}

TEST_CASE("rgpot_gupta stays correct after a shrink and regrow",
          "[Gupta][heap]") {
  REQUIRE(probe_sees_malloc());
  const rgpot::GuptaConfig cfg{};
  constexpr std::size_t large_n = 1500;
  constexpr std::size_t small_n = 64;
  const std::vector<double> large_pos = lattice_cluster(large_n);
  const std::vector<double> small_pos = lattice_cluster(small_n);
  const std::vector<unsigned char> large_slot(large_n, 0);
  const std::vector<unsigned char> small_slot(small_n, 0);

  std::vector<double> first_large_grad(3 * large_n, 0.0);
  std::vector<double> regrown_large_grad(3 * large_n, 0.0);
  std::vector<double> small_grad(3 * small_n, 0.0);
  double first_large_energy = 0.0;
  double regrown_large_energy = 0.0;
  double small_energy = 0.0;

  rgpot_gupta(large_pos.data(), large_n, large_slot.data(), 1, &cfg.A, &cfg.xi, &cfg.p,
              &cfg.q, &cfg.r0, &first_large_energy, first_large_grad.data());
  rgpot_gupta(small_pos.data(), small_n, small_slot.data(), 1, &cfg.A, &cfg.xi, &cfg.p,
              &cfg.q, &cfg.r0, &small_energy, small_grad.data());
  const HeapUse regrow_use = measure_heap([&] {
    rgpot_gupta(large_pos.data(), large_n, large_slot.data(), 1, &cfg.A, &cfg.xi,
                &cfg.p, &cfg.q, &cfg.r0, &regrown_large_energy,
                regrown_large_grad.data());
  });

  REQUIRE(regrow_use.calls == 0);
  REQUIRE(regrow_use.net == 0);
  REQUIRE(same_bits(regrown_large_energy, first_large_energy));
  REQUIRE(same_bits(regrown_large_grad, first_large_grad));

  // GuptaPot delegation (cutoff <= 0) uses the same thread-local buffers.
  const rgpot::GuptaPot direct(cfg);
  std::vector<double> forces(3 * large_n, 0.0);
  rgpot::ForceOut out{};
  out.F = forces.data();
  const rgpot::ForceInput in{large_n, large_pos.data(), nullptr, nullptr};
  direct.forceImpl(in, &out);
  const HeapUse steady = measure_heap([&] { direct.forceImpl(in, &out); });
  REQUIRE(steady.calls == 0);
  REQUIRE(steady.net == 0);
  REQUIRE(same_bits(out.energy, first_large_energy));
}
#endif

TEST_CASE("rgpot_gupta large call matches the explicit wide cutoff",
          "[Gupta]") {
  constexpr std::size_t n = 1500;
  const std::vector<double> pos = lattice_cluster(n);
  const rgpot::GuptaConfig cfg{};
  const std::vector<unsigned char> slot(n, 0);
  std::vector<double> grad(3 * n, 0.0);
  double c_energy = 0.0;
  rgpot_gupta(pos.data(), n, slot.data(), 1, &cfg.A, &cfg.xi, &cfg.p, &cfg.q,
              &cfg.r0, &c_energy, grad.data());
  const rgpot::GuptaPot direct(cfg);
  const StressEval open = eval_stress(direct, pos, nullptr);
  rgpot::GuptaConfig wide = cfg;
  wide.cutoff = 1.0e5;
  const StressEval ref = eval_stress(rgpot::GuptaPot(wide), pos, nullptr);
  require_scaled_match(open.energy, c_energy);
  require_scaled_match(open.energy, ref.energy);
  for (std::size_t k = 0; k < 3 * n; ++k) {
    require_scaled_match(open.forces[k], -grad[k]);
    require_scaled_match(open.forces[k], ref.forces[k]);
  }
}
