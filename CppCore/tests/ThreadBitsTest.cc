// MIT License
// Copyright 2023--present rgpot developers

// Energy and forces of every threaded kernel must match the one-thread
// call bit for bit. Sizes sit below and above the point where the pair
// walk splits across threads.

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <tuple>
#include <vector>

#include <catch2/catch_all.hpp>
#include <omp.h>

#include "rgpot/EAM/EAMPot.hpp"
#include "rgpot/LennardJones/LJClusterPot.hpp"
#include "rgpot/LennardJones/LJPot.hpp"
#include "rgpot/Morse/MorsePot.hpp"
#include "rgpot/StillingerWeber/SWSi.hpp"
#include "rgpot/SuttonChen/SuttonChenPot.hpp"
#include "rgpot/Tersoff/TersoffSiC.hpp"
#include "rgpot/ZBL/ZBLPot.hpp"
#include "rgpot/fortran/FortranPots.hpp"
#include "rgpot/types/AtomMatrix.hpp"

namespace {

struct Geom {
  rgpot::types::AtomMatrix pos;
  std::vector<int> types;
  std::array<std::array<double, 3>, 3> box{};
};

std::uint64_t bits(double x) {
  std::uint64_t u = 0;
  std::memcpy(&u, &x, sizeof(u));
  return u;
}

void set_threads(int n) {
  const std::string text = std::to_string(n);
  setenv("OMP_NUM_THREADS", text.c_str(), 1);
  omp_set_num_threads(n);
}

Geom lattice(int n, int z, double a, int per_cell, const double basis[][3]) {
  int cells = (n + per_cell - 1) / per_cell;
  int nz = 1;
  for (int k = 1; k * k * k <= cells; ++k) {
    if (cells % k == 0) {
      nz = k;
    }
  }
  int rest = cells / nz;
  int ny = 1;
  for (int k = 1; k * k <= rest; ++k) {
    if (rest % k == 0) {
      ny = k;
    }
  }
  int nx = rest / ny;
  Geom g;
  g.pos = rgpot::types::AtomMatrix(static_cast<std::size_t>(n), 3);
  g.types.assign(static_cast<std::size_t>(n), z);
  int filled = 0;
  for (int ix = 0; ix < nx && filled < n; ++ix) {
    for (int iy = 0; iy < ny && filled < n; ++iy) {
      for (int iz = 0; iz < nz && filled < n; ++iz) {
        for (int b = 0; b < per_cell && filled < n; ++b) {
          const double wobble = 0.015 * std::sin(1.3 * filled);
          g.pos(static_cast<std::size_t>(filled), 0) = (ix + basis[b][0]) * a + wobble;
          g.pos(static_cast<std::size_t>(filled), 1) = (iy + basis[b][1]) * a;
          g.pos(static_cast<std::size_t>(filled), 2) = (iz + basis[b][2]) * a + 0.01 * filled;
          ++filled;
        }
      }
    }
  }
  g.box[0][0] = nx * a;
  g.box[1][1] = ny * a;
  g.box[2][2] = nz * a;
  return g;
}

Geom fcc(int n, int z, double a) {
  static const double basis[4][3] = {{0, 0, 0}, {0, 0.5, 0.5}, {0.5, 0, 0.5}, {0.5, 0.5, 0}};
  return lattice(n, z, a, 4, basis);
}

Geom diamond(int n, int z, double a) {
  static const double basis[8][3] = {
      {0, 0, 0},          {0.5, 0.5, 0},      {0.5, 0, 0.5},      {0, 0.5, 0.5},
      {0.25, 0.25, 0.25}, {0.75, 0.75, 0.25}, {0.75, 0.25, 0.75}, {0.25, 0.75, 0.75}};
  return lattice(n, z, a, 8, basis);
}

struct Outcome {
  double energy;
  std::vector<double> force;
};

template <typename Pot>
Outcome run(Pot &pot, const Geom &g) {
  auto [energy, forces, variance] = pot(g.pos, g.types, g.box);
  (void)variance;
  Outcome out;
  out.energy = energy;
  const double *data = forces.data();
  out.force.assign(data, data + g.types.size() * 3);
  return out;
}

void expect_same(const Outcome &got, const Outcome &ref) {
  REQUIRE(bits(got.energy) == bits(ref.energy));
  REQUIRE(got.force.size() == ref.force.size());
  for (std::size_t i = 0; i < got.force.size(); ++i) {
    REQUIRE(bits(got.force[i]) == bits(ref.force[i]));
  }
}

template <typename Make>
void check(Make make, const Geom &g) {
  set_threads(1);
  auto serial = make();
  const Outcome one = run(serial, g);
  for (int threads : {2, 4, 8}) {
    set_threads(threads);
    auto parallel = make();
    expect_same(run(parallel, g), one);
  }
  set_threads(1);
}

} // namespace

TEST_CASE("threaded kernels match the one-thread bits", "[threads][bits]") {
  check([] { return rgpot::LJPot{}; }, fcc(256, 18, 4.05));
  check([] { return rgpot::LJPot{}; }, fcc(1024, 18, 4.05));
  check([] { return rgpot::LJClusterPot{}; }, fcc(64, 18, 3.5));
  check([] { return rgpot::LJClusterPot{}; }, fcc(256, 18, 3.5));
  check([] { return rgpot::MorsePot{}; }, fcc(256, 78, 3.92));
  check([] { return rgpot::MorsePot{}; }, fcc(1024, 78, 3.92));
  check([] { return rgpot::ZBLPot{}; }, fcc(64, 14, 2.0));
  check([] { return rgpot::ZBLPot{}; }, fcc(256, 14, 2.0));
  rgpot::SuttonChenConfig sc;
  sc.periodic = true;
  sc.cutoff = 8.0;
  sc.a = 4.08;
  check([&] { return rgpot::SuttonChenPot{sc}; }, fcc(256, 79, 4.08));
  check([] { return rgpot::StillingerWeberSi{}; }, diamond(64, 14, 5.43));
  check([] { return rgpot::StillingerWeberSi{}; }, diamond(256, 14, 5.43));
  check([] { return rgpot::TersoffSiC{}; }, diamond(64, 14, 5.43));
  check([] { return rgpot::TersoffSiC{}; }, diamond(256, 14, 5.43));
  check([] { return rgpot::EAMPot{}; }, fcc(64, 13, 4.05));
  check([] { return rgpot::EAMPot{}; }, fcc(256, 13, 4.05));
  check([] { return rgpot::fortranpots::EAMAlPot{}; }, fcc(256, 13, 4.05));
  check([] { return rgpot::fortranpots::EAMAlPot{}; }, fcc(1024, 13, 4.05));
  check([] { return rgpot::fortranpots::SWPot{}; }, diamond(64, 14, 5.43));
  check([] { return rgpot::fortranpots::SWPot{}; }, diamond(256, 14, 5.43));
  check([] { return rgpot::fortranpots::TersoffPot{}; }, diamond(64, 14, 5.43));
  check([] { return rgpot::fortranpots::TersoffPot{}; }, diamond(256, 14, 5.43));
}
