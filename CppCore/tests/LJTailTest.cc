// MIT License
// Copyright 2023--present rgpot developers
//
// LJConfig::tail_correction on a dense periodic fluid. The corrected energy
// and pressure at a short cutoff, with and without the quintic switch, are
// compared with a brute-force sum over periodic images out to a cutoff many
// times larger. The per-atom outputs, the single-atom path and the closed
// form of the switch window are checked on their own.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <numbers>
#include <stdexcept>
#include <vector>

#include <catch2/catch_all.hpp>

#include "rgpot/LennardJones/LJPot.hpp"
#include "rgpot/PairSwitch.hpp"
#include "rgpot/stress.hpp"

using Catch::Matchers::WithinRel;

namespace {

struct SplitMix64 {
  std::uint64_t s;
  std::uint64_t next() {
    std::uint64_t z = (s += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
  }
  double uniform() { return static_cast<double>(next() >> 11) * 0x1.0p-53; }
};

struct Fluid {
  std::size_t n = 0;
  double side = 0.0;
  std::vector<double> pos; ///< Row-major, 3 per atom.
  std::array<double, 9> box{};
  [[nodiscard]] double volume() const { return side * side * side; }
};

// Random sequential insertion of soft spheres at number density rho in a
// cubic box, with every pair at least d_min apart under the minimum image.
// The result is a disordered dense fluid with no lattice order, so g(r) is
// flat well inside half the box.
Fluid denseFluid(std::size_t n, double rho, double d_min, std::uint64_t seed) {
  Fluid f;
  f.n = n;
  f.side = std::cbrt(static_cast<double>(n) / rho);
  f.box = {f.side, 0.0, 0.0, 0.0, f.side, 0.0, 0.0, 0.0, f.side};
  SplitMix64 rng{seed};
  const double dmin2 = d_min * d_min;
  while (f.pos.size() < 3 * n) {
    const double x = rng.uniform() * f.side;
    const double y = rng.uniform() * f.side;
    const double z = rng.uniform() * f.side;
    bool clear = true;
    for (std::size_t j = 0; j < f.pos.size() / 3 && clear; ++j) {
      double dx = x - f.pos[3 * j];
      double dy = y - f.pos[3 * j + 1];
      double dz = z - f.pos[3 * j + 2];
      dx -= f.side * std::round(dx / f.side);
      dy -= f.side * std::round(dy / f.side);
      dz -= f.side * std::round(dz / f.side);
      clear = dx * dx + dy * dy + dz * dz >= dmin2;
    }
    if (clear) {
      f.pos.insert(f.pos.end(), {x, y, z});
    }
  }
  return f;
}

struct Totals {
  double energy = 0.0;
  double pressure = 0.0; ///< Virial part, (1 / 3V) sum r . F.
};

// Direct sum of the bare 12-6 term over every periodic image within r_max.
// Nothing here uses the library or its neighbour list.
Totals bruteForce(const Fluid &f, double u0, double psi, double r_max) {
  const int m = static_cast<int>(std::ceil(r_max / f.side)) + 1;
  const double r2_max = r_max * r_max;
  const double psi2 = psi * psi;
  double e = 0.0;
  double w = 0.0;
  for (std::size_t i = 0; i < f.n; ++i) {
    for (std::size_t j = i; j < f.n; ++j) {
      const double wt = i == j ? 0.5 : 1.0;
      const double bx = f.pos[3 * i] - f.pos[3 * j];
      const double by = f.pos[3 * i + 1] - f.pos[3 * j + 1];
      const double bz = f.pos[3 * i + 2] - f.pos[3 * j + 2];
      for (int ix = -m; ix <= m; ++ix) {
        const double dx = bx + f.side * ix;
        if (dx * dx > r2_max) {
          continue;
        }
        for (int iy = -m; iy <= m; ++iy) {
          const double dy = by + f.side * iy;
          if (dx * dx + dy * dy > r2_max) {
            continue;
          }
          for (int iz = -m; iz <= m; ++iz) {
            const double dz = bz + f.side * iz;
            const double r2 = dx * dx + dy * dy + dz * dz;
            if (r2 > r2_max || r2 == 0.0) {
              continue;
            }
            const double a = std::pow(psi2 / r2, 3.0);
            e += wt * 4.0 * u0 * a * (a - 1.0);
            w += wt * 24.0 * u0 * a * (2.0 * a - 1.0);
          }
        }
      }
    }
  }
  return {e, w / (3.0 * f.volume())};
}

// Standard continuum tail of the unswitched 12-6 term beyond r_c for n atoms
// in volume v (Allen and Tildesley), as written for LAMMPS lj/cut.
Totals standardTail(std::size_t n, double v, double u0, double psi,
                    double r_c) {
  const double pi = std::numbers::pi;
  const double s3 = std::pow(psi / r_c, 3.0);
  const double s9 = s3 * s3 * s3;
  const double nn = static_cast<double>(n) * static_cast<double>(n);
  const double s_vol = u0 * std::pow(psi, 3.0);
  return {(8.0 * pi / 3.0) * nn / v * s_vol * (s9 / 3.0 - s3),
          (16.0 * pi / 3.0) * nn / (v * v) * s_vol * (2.0 * s9 / 3.0 - s3)};
}

struct Evaluated {
  double energy = 0.0;
  double pressure = 0.0;
  bool has_stress = false;
};

Evaluated evaluate(const rgpot::LJPot &pot, const Fluid &f,
                   const int *z = nullptr) {
  std::vector<double> forces(3 * f.n, 0.0);
  rgpot::ForceOut out{forces.data(), 0.0, 0.0, {}, 0};
  pot.forceImpl({f.n, f.pos.data(), z, f.box.data()}, &out);
  return {out.energy, -(out.stress[0] + out.stress[4] + out.stress[8]) / 3.0,
          out.has_stress != 0};
}

constexpr double kU0 = 0.0104;
constexpr double kPsi = 3.40;

// Reduced density 0.8 sigma^-3 and a 343-atom cube. The cutoffs below stay
// under half the side, which the minimum image needs.
const Fluid &fluid() {
  static const Fluid f =
      denseFluid(343, 0.8 / std::pow(kPsi, 3.0), 0.85 * kPsi, 20260519);
  return f;
}

} // namespace

TEST_CASE("LJPot tail correction matches a large-cutoff sum on a dense fluid",
          "[LJPot][tail][fluid]") {
  const Fluid &f = fluid();
  const double r_ref = 3.0 * f.side;
  REQUIRE(r_ref > 4.0 * kPsi);
  const Totals sum = bruteForce(f, kU0, kPsi, r_ref);
  const Totals far_tail = standardTail(f.n, f.volume(), kU0, kPsi, r_ref);
  const Totals ref{sum.energy + far_tail.energy,
                   sum.pressure + far_tail.pressure};
  INFO("brute-force reference cutoff " << r_ref << ", energy " << ref.energy
                                       << ", pressure " << ref.pressure);
  INFO("far tail beyond the reference cutoff: energy "
       << far_tail.energy << ", pressure " << far_tail.pressure);

  struct Case {
    const char *name;
    double cutoff_sigma;
    double width_sigma;
  };
  for (const Case c : {Case{"unswitched", 3.5, 0.0}, Case{"switched", 3.5, 0.7},
                       Case{"switched narrow", 3.5, 0.25}}) {
    DYNAMIC_SECTION(c.name) {
      const double rc = c.cutoff_sigma * kPsi;
      const double w = c.width_sigma * kPsi;
      REQUIRE(rc < 0.5 * f.side);
      const rgpot::LJConfig plain{
          .u0 = kU0, .cutoff = rc, .psi = kPsi, .switch_width = w};
      rgpot::LJConfig with_tail = plain;
      with_tail.tail_correction = true;

      const Evaluated corrected = evaluate(rgpot::LJPot(with_tail), f);
      REQUIRE(corrected.has_stress);

      // The same cutoff and switch with no correction and no shift.
      Totals truncated;
      if (w == 0.0) {
        truncated = bruteForce(f, kU0, kPsi, rc);
      } else {
        const Evaluated bare = evaluate(rgpot::LJPot(plain), f);
        truncated = {bare.energy, bare.pressure};
      }
      // The tail of the full 12-6 term beyond rc. This alone is what a
      // correction blind to the switch would add.
      const Totals beyond = standardTail(f.n, f.volume(), kU0, kPsi, rc);

      const double e_err = std::abs(corrected.energy - ref.energy);
      const double p_err = std::abs(corrected.pressure - ref.pressure);
      const double e_gap = std::abs(truncated.energy - ref.energy);
      const double p_gap = std::abs(truncated.pressure - ref.pressure);
      const double e_blind =
          std::abs(truncated.energy + beyond.energy - ref.energy);
      const double p_blind =
          std::abs(truncated.pressure + beyond.pressure - ref.pressure);
      INFO("cutoff " << rc << ", switch width " << w);
      INFO("energy: reference " << ref.energy << ", corrected "
                                << corrected.energy << ", truncated "
                                << truncated.energy);
      INFO("pressure: reference " << ref.pressure << ", corrected "
                                  << corrected.pressure << ", truncated "
                                  << truncated.pressure);
      INFO("energy error " << e_err << " of a " << e_gap
                           << " correction, pressure error " << p_err
                           << " of a " << p_gap << " correction");
      INFO("a correction blind to the switch leaves energy error "
           << e_blind << " and pressure error " << p_blind);
      CHECK(e_err < 0.02 * e_gap);
      CHECK(p_err < 0.05 * p_gap);
      if (w != 0.0) {
        // The comparison has power: dropping the switch window would fail it.
        CHECK(e_blind > 10.0 * e_err);
        CHECK(p_blind > 10.0 * p_err);
      }
    }
  }
}

namespace {

// Composite Simpson rule on [lo, hi] with an even number of panels.
template <typename F>
double simpson(F &&fn, double lo, double hi, int panels = 4000) {
  const double h = (hi - lo) / panels;
  double sum = fn(lo) + fn(hi);
  for (int k = 1; k < panels; ++k) {
    sum += (k % 2 == 1 ? 4.0 : 2.0) * fn(lo + k * h);
  }
  return sum * h / 3.0;
}

double bareV(double u0, double psi, double r) {
  const double a = std::pow(psi / r, 6.0);
  return 4.0 * u0 * a * (a - 1.0);
}

double bareDV(double u0, double psi, double r) {
  const double a = std::pow(psi / r, 6.0);
  return (4.0 * u0 / r) * (6.0 * a - 12.0 * a * a);
}

} // namespace

TEST_CASE(
    "LJPot tail of a single atom is the closed form plus the switch window",
    "[LJPot][tail]") {
  // One atom in a large cell has no pair, so the result is the correction
  // alone. The switch window is integrated numerically here from the bare
  // term and the QuinticSwitch of PairSwitch.hpp, not from the closed form.
  const double side = 60.0;
  const double volume = side * side * side;
  const std::array<double, 9> box{side, 0.0, 0.0, 0.0, side,
                                  0.0,  0.0, 0.0, side};
  const std::array<double, 3> pos{1.0, 2.0, 3.0};
  const std::array<int, 1> z{29};

  struct Case {
    double cutoff;
    double width;
    double tol;
  };
  // The closed form of the window expands a polynomial in r, so its error
  // grows like (r_on / width)^5. The tolerances follow that, for a platform
  // whose long double is no wider than double.
  for (const Case c :
       {Case{11.9, 0.0, 1e-14}, Case{11.9, 2.38, 1e-8}, Case{11.9, 0.85, 1e-7},
        Case{8.5, 4.25, 1e-8}, Case{15.0, 1.0, 1e-8}, Case{9.0, 0.36, 1e-6},
        Case{7.0, 3.0, 1e-8}}) {
    DYNAMIC_SECTION("cutoff " << c.cutoff << " width " << c.width) {
      const rgpot::LJConfig cfg{.u0 = kU0,
                                .cutoff = c.cutoff,
                                .psi = kPsi,
                                .switch_width = c.width,
                                .tail_correction = true};
      const rgpot::LJPot pot(cfg);
      std::vector<double> forces(3, 1.0);
      rgpot::ForceOut out{forces.data(), 0.0, 0.0, {}, 0};
      pot.forceImpl({1, pos.data(), z.data(), box.data()}, &out);
      REQUIRE(out.has_stress == 1);
      CHECK(forces[0] == 0.0);
      CHECK(forces[1] == 0.0);
      CHECK(forces[2] == 0.0);

      const Totals beyond = standardTail(1, volume, kU0, kPsi, c.cutoff);
      double want_e = beyond.energy;
      double want_p = beyond.pressure;
      if (c.width != 0.0) {
        const auto sw = rgpot::QuinticSwitch::endingAt(c.cutoff, c.width);
        const double r_on = sw.r_on;
        const double window_e = simpson(
            [&](double r) {
              return r * r * bareV(kU0, kPsi, r) * (1.0 - sw(r).s);
            },
            r_on, c.cutoff);
        // The force the switch removes is V' (1 - S) - V S'.
        const double window_dphi = simpson(
            [&](double r) {
              return r * r * r *
                     (bareDV(kU0, kPsi, r) * (1.0 - sw(r).s) -
                      bareV(kU0, kPsi, r) * sw(r).dsdr);
            },
            r_on, c.cutoff);
        want_e += 2.0 * std::numbers::pi / volume * window_e;
        want_p +=
            2.0 * std::numbers::pi / (volume * volume) * (-window_dphi / 3.0);
      }
      const double got_p =
          -(out.stress[0] + out.stress[4] + out.stress[8]) / 3.0;
      INFO("energy " << out.energy << " against " << want_e << ", pressure "
                     << got_p << " against " << want_p);
      CHECK_THAT(out.energy, WithinRel(want_e, c.tol));
      CHECK_THAT(got_p, WithinRel(want_p, c.tol));
      for (const int k : {1, 2, 3, 5, 6, 7}) {
        CHECK(out.stress[k] == 0.0);
      }
      CHECK(out.stress[0] == out.stress[4]);
      CHECK(out.stress[4] == out.stress[8]);
    }
  }
}

TEST_CASE("LJPot tail correction leaves forces unchanged and ignores species",
          "[LJPot][tail][fluid]") {
  const Fluid &f = fluid();
  const double rc = 3.5 * kPsi;
  std::vector<int> mixed(f.n);
  std::vector<int> single(f.n, 18);
  for (std::size_t i = 0; i < f.n; ++i) {
    mixed[i] = std::array<int, 4>{1, 6, 8, 14}[i % 4];
  }
  for (const double w : {0.0, 0.7 * kPsi}) {
    const rgpot::LJPot off(rgpot::LJConfig{
        .u0 = kU0, .cutoff = rc, .psi = kPsi, .switch_width = w});
    const rgpot::LJPot on(rgpot::LJConfig{.u0 = kU0,
                                          .cutoff = rc,
                                          .psi = kPsi,
                                          .switch_width = w,
                                          .tail_correction = true});
    std::vector<double> f_off(3 * f.n), f_on(3 * f.n), f_mixed(3 * f.n);
    rgpot::ForceOut o_off{f_off.data(), 0.0, 0.0, {}, 0},
        o_on{f_on.data(), 0.0, 0.0, {}, 0},
        o_mixed{f_mixed.data(), 0.0, 0.0, {}, 0};
    off.forceImpl({f.n, f.pos.data(), single.data(), f.box.data()}, &o_off);
    on.forceImpl({f.n, f.pos.data(), single.data(), f.box.data()}, &o_on);
    on.forceImpl({f.n, f.pos.data(), mixed.data(), f.box.data()}, &o_mixed);
    REQUIRE(f_off == f_on);
    REQUIRE(f_on == f_mixed);
    REQUIRE(o_on.energy == o_mixed.energy);
    REQUIRE(std::equal(std::begin(o_on.stress), std::end(o_on.stress),
                       std::begin(o_mixed.stress)));
  }
}

TEST_CASE("LJPot per-atom tail sums to the energy and the virial of the stress",
          "[LJPot][tail][peratom]") {
  const Fluid &f = fluid();
  const double rc = 3.5 * kPsi;
  const std::array<double, 3> lone{1.0, 2.0, 3.0};
  const double lone_side = 50.0;
  const std::array<double, 9> lone_box{lone_side, 0.0, 0.0, 0.0,      lone_side,
                                       0.0,       0.0, 0.0, lone_side};

  struct System {
    const char *name;
    std::size_t n;
    const double *pos;
    const double *box;
    double volume;
  };
  const System systems[] = {
      {"fluid", f.n, f.pos.data(), f.box.data(), f.volume()},
      {"single atom", 1, lone.data(), lone_box.data(),
       lone_side * lone_side * lone_side}};

  for (const System &sys : systems) {
    for (const double w : {0.0, 0.7 * kPsi}) {
      for (const bool tail : {false, true}) {
        DYNAMIC_SECTION(sys.name << ", width " << w
                                 << (tail ? ", tail" : ", no tail")) {
          const rgpot::LJPot pot(rgpot::LJConfig{.u0 = kU0,
                                                 .cutoff = rc,
                                                 .psi = kPsi,
                                                 .switch_width = w,
                                                 .tail_correction = tail});
          const rgpot::ForceInput in{sys.n, sys.pos, nullptr, sys.box};
          std::vector<double> forces(3 * sys.n), plain_forces(3 * sys.n);
          std::vector<double> e_atom(sys.n), w_atom(6 * sys.n);
          rgpot::ForceOut out{forces.data(), 0.0, 0.0, {}, 0};
          rgpot::ForceOut plain{plain_forces.data(), 0.0, 0.0, {}, 0};
          pot.forcePerAtom(in, &out, {e_atom.data(), w_atom.data()});
          pot.forceImpl(in, &plain);
          REQUIRE(out.has_stress == 1);
          CHECK_THAT(out.energy, WithinRel(plain.energy, 1e-12));

          double e_sum = 0.0;
          std::array<double, 6> w_sum{};
          for (std::size_t i = 0; i < sys.n; ++i) {
            e_sum += e_atom[i];
            for (int c = 0; c < 6; ++c) {
              w_sum[c] += w_atom[6 * i + c];
            }
          }
          // Voigt xx yy zz yz xz xy against the row-major stress.
          const std::array<int, 6> slot{0, 4, 8, 5, 2, 1};
          double scale = 0.0;
          for (int c = 0; c < 6; ++c) {
            scale = std::max(scale, std::abs(out.stress[slot[c]] * sys.volume));
          }
          INFO("energy " << out.energy << ", per-atom sum " << e_sum);
          CHECK_THAT(e_sum, WithinRel(out.energy, 1e-11));
          for (int c = 0; c < 6; ++c) {
            INFO("component " << c << ": sum of atoms " << w_sum[c]
                              << ", -V stress "
                              << -sys.volume * out.stress[slot[c]]);
            CHECK(std::abs(w_sum[c] + sys.volume * out.stress[slot[c]]) <=
                  1e-10 * scale);
          }
        }
      }
    }
  }
}

TEST_CASE(
    "LJPot tail correction with a switch needs a window inside the cutoff",
    "[LJPot][tail]") {
  using rgpot::LJConfig;
  using rgpot::LJPot;
  CHECK_THROWS_AS(LJPot(LJConfig{.cutoff = 5.0,
                                 .switch_width = 5.0,
                                 .tail_correction = true}),
                  std::invalid_argument);
  CHECK_THROWS_AS(LJPot(LJConfig{.cutoff = 5.0,
                                 .switch_width = 6.0,
                                 .tail_correction = true}),
                  std::invalid_argument);
  CHECK_NOTHROW(LJPot(
      LJConfig{.cutoff = 5.0, .switch_width = 4.9, .tail_correction = true}));
  CHECK_NOTHROW(LJPot(LJConfig{.cutoff = 5.0, .switch_width = 5.0}));
  CHECK(LJPot(LJConfig{
                  .cutoff = 5.0, .switch_width = 2.0, .tail_correction = true})
            .paramsKey() !=
        LJPot(LJConfig{.cutoff = 5.0, .switch_width = 2.0}).paramsKey());
}
