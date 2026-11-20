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
  // The window is integrated in u = (r - r_on) / width over [0, 1], so it
  // avoids the cancellation of expanding (r - r_on)^k around 0.
  for (const Case c :
       {Case{11.9, 0.0, 1e-14}, Case{11.9, 2.38, 1e-12},
        Case{11.9, 0.85, 1e-12}, Case{8.5, 4.25, 1e-12},
        Case{15.0, 1.0, 1e-12}, Case{9.0, 0.36, 1e-12},
        Case{7.0, 3.0, 1e-12}}) {
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

TEST_CASE("LJPot switch window integral stays accurate for narrow widths",
          "[LJPot][tail][narrow]") {
  // At rc = 15 and L = 50, a narrow switch window w << r_on makes the
  // monomial expansion in r cancel like (r_on / w)^6. Check the window
  // contribution (with the unswitched tail subtracted) and the total tail
  // at w = 1e-2, 1e-3 and 1e-6 against an independent power series in
  // u = (r - r_on) / w and adaptive Simpson quadrature in long double.
  const double rc = 15.0;
  const double side = 50.0;
  const double volume = side * side * side;
  const std::array<double, 9> box{side, 0.0, 0.0, 0.0, side,
                                  0.0,  0.0, 0.0, side};
  const std::array<double, 3> pos{1.0, 2.0, 3.0};
  const std::array<int, 1> z{18};

  const rgpot::LJPot unswitched(rgpot::LJConfig{
      .u0 = 1.0, .cutoff = rc, .psi = 1.0, .tail_correction = true});
  std::vector<double> f0(3, 0.0);
  rgpot::ForceOut out0{f0.data(), 0.0, 0.0, {}, 0};
  unswitched.forceImpl({1, pos.data(), z.data(), box.data()}, &out0);

  // Adaptive Simpson quadrature on [0, 1] in long double, where the
  // integrand (1 + alpha * u)^n * u^3 * (10 - 15 u + 6 u^2) is positive.
  auto quad01 = [](auto &&fn) {
    auto panel = [&](long double a, long double b) {
      const long double m = 0.5L * (a + b);
      return (b - a) / 6.0L * (fn(a) + 4.0L * fn(m) + fn(b));
    };
    auto adapt = [&](auto &self, long double a, long double b, long double whole,
                     long double tol, int depth) -> long double {
      const long double m = 0.5L * (a + b);
      const long double left = panel(a, m);
      const long double right = panel(m, b);
      const long double sum = left + right;
      if (depth <= 0 || std::abs(sum - whole) <= 15.0L * tol) {
        return sum + (sum - whole) / 15.0L;
      }
      return self(self, a, m, left, 0.5L * tol, depth - 1) +
             self(self, m, b, right, 0.5L * tol, depth - 1);
    };
    const long double whole = panel(0.0L, 1.0L);
    return adapt(adapt, 0.0L, 1.0L, whole, 1e-18L, 20);
  };

  // Independent power series of integral_0^1 (1 + alpha u)^n (1 - S(u)) du,
  // with int_0^1 u^k (10 u^3 - 15 u^4 + 6 u^5) du = 10/(k+4) - 15/(k+5) + 6/(k+6).
  auto seriesWindow = [](int n, long double r_on, long double w) {
    const long double alpha = w / r_on;
    long double binom = 1.0L;
    long double pow_a = 1.0L;
    long double sum = 0.0L;
    for (int k = 0; k < 64; ++k) {
      const auto kk = static_cast<long double>(k);
      const long double moment =
          10.0L / (kk + 4.0L) - 15.0L / (kk + 5.0L) + 6.0L / (kk + 6.0L);
      const long double term = binom * pow_a * moment;
      sum += term;
      if (std::abs(term) <= 1e-20L * std::abs(sum)) {
        break;
      }
      binom *= static_cast<long double>(n - k) / (kk + 1.0L);
      pow_a *= alpha;
    }
    return w * std::pow(r_on, static_cast<long double>(n)) * sum;
  };

  for (const double w : {1e-2, 1e-3, 1e-6}) {
    DYNAMIC_SECTION("width " << w) {
      const rgpot::LJPot pot(rgpot::LJConfig{.u0 = 1.0,
                                             .cutoff = rc,
                                             .psi = 1.0,
                                             .switch_width = w,
                                             .tail_correction = true});
      std::vector<double> forces(3, 0.0);
      rgpot::ForceOut out{forces.data(), 0.0, 0.0, {}, 0};
      pot.forceImpl({1, pos.data(), z.data(), box.data()}, &out);

      const auto r_on = static_cast<long double>(rc - w);
      const auto wl = static_cast<long double>(w);
      const long double alpha = wl / r_on;
      auto quadWindow = [&](int n) {
        const long double i01 = quad01([&](long double u) {
          const long double one_minus_s =
              u * u * u * (10.0L + u * (-15.0L + 6.0L * u));
          return std::pow(1.0L + alpha * u, static_cast<long double>(n)) *
                 one_minus_s;
        });
        return wl * std::pow(r_on, static_cast<long double>(n)) * i01;
      };

      const long double i_m10_q = quadWindow(-10);
      const long double i_m4_q = quadWindow(-4);
      const long double i_m10_s = seriesWindow(-10, r_on, wl);
      const long double i_m4_s = seriesWindow(-4, r_on, wl);
      CHECK(std::abs(i_m10_q - i_m10_s) <= 1e-13L * std::abs(i_m10_s));
      CHECK(std::abs(i_m4_q - i_m4_s) <= 1e-13L * std::abs(i_m4_s));

      const auto scale =
          static_cast<long double>(2.0 * std::numbers::pi / volume);
      const auto want_window =
          static_cast<double>(scale * 4.0L * (i_m10_q - i_m4_q));
      const double got_window = out.energy - out0.energy;
      const double want_total = out0.energy + want_window;
      const double got_p =
          -(out.stress[0] + out.stress[4] + out.stress[8]) / 3.0;
      const double want_p = want_total / volume;
      INFO("w = " << w << ": i_m10_q = " << static_cast<double>(i_m10_q)
                  << ", i_m10_s = " << static_cast<double>(i_m10_s)
                  << ", i_m4_q = " << static_cast<double>(i_m4_q)
                  << ", i_m4_s = " << static_cast<double>(i_m4_s)
                  << ", got_window = " << got_window
                  << ", want_window = " << want_window);
      CHECK(got_window < 0.0);
      if (w >= 1e-3) {
        CHECK(std::abs(got_window - want_window) <=
              1e-3 * std::abs(want_window));
      }
      CHECK(std::abs(out.energy - want_total) <= 1e-14 * std::abs(want_total));
      CHECK(std::abs(got_p - want_p) <= 1e-14 * std::abs(want_p));
    }
  }
}

TEST_CASE("LJPot rotated and standard periodic cells give the same tail",
          "[LJPot][tail][periodic]") {
  const double side = 50.0;
  const std::array<double, 9> std_box{side, 0.0,  0.0, 0.0, side,
                                      0.0,  0.0,  0.0, side};
  // 90-degree rotation in the xy plane: a = (0, L, 0), b = (-L, 0, 0),
  // c = (0, 0, L), with det(H) = L^3 > 0 and zero x/y diagonal entries.
  const std::array<double, 9> rot_box{0.0,   side, 0.0, -side, 0.0,
                                      0.0,   0.0,  0.0, side};

  // Single atom: pure tail contribution.
  {
    const std::array<double, 3> pos_std{1.0, 2.0, 3.0};
    const std::array<double, 3> pos_rot{-2.0, 1.0, 3.0};
    const std::array<int, 1> z{18};
    for (const double w : {0.0, 1.5}) {
      const rgpot::LJPot pot(rgpot::LJConfig{.u0 = 1.0,
                                             .cutoff = 15.0,
                                             .psi = 1.0,
                                             .switch_width = w,
                                             .tail_correction = true});
      std::vector<double> f_std(3), f_rot(3), e_std(1), e_rot(1), v_std(6),
          v_rot(6);
      rgpot::ForceOut out_std{f_std.data(), 0.0, 0.0, {}, 0};
      rgpot::ForceOut out_rot{f_rot.data(), 0.0, 0.0, {}, 0};
      pot.forcePerAtom({1, pos_std.data(), z.data(), std_box.data()}, &out_std,
                       {e_std.data(), v_std.data()});
      pot.forcePerAtom({1, pos_rot.data(), z.data(), rot_box.data()}, &out_rot,
                       {e_rot.data(), v_rot.data()});
      REQUIRE(out_std.energy < 0.0);
      CHECK(out_rot.energy == out_std.energy);
      CHECK(e_rot == e_std);
      CHECK(v_rot == v_std);
      CHECK(std::equal(std::begin(out_rot.stress), std::end(out_rot.stress),
                       std::begin(out_std.stress)));
    }
  }

  // Multi-atom periodic system: rotated cell matches standard cell in energy,
  // per-atom energy, and hydrostatic pressure, with and without tail.
  {
    const std::size_t n = 4;
    const std::vector<double> pos_std{
        0.2, 0.5,  1.0,  // atom 0
        1.5, -0.3, 1.2,  // atom 1
        0.1, 49.1, 0.8,  // atom 2 (wraps across y in std, x in rot)
        48.9, 0.4, 49.2  // atom 3 (wraps across x and z)
    };
    std::vector<double> pos_rot(3 * n);
    for (std::size_t i = 0; i < n; ++i) {
      // r_rot = (-y, x, z) maps fractional coordinates identically into rot_box.
      pos_rot[3 * i] = -pos_std[3 * i + 1];
      pos_rot[3 * i + 1] = pos_std[3 * i];
      pos_rot[3 * i + 2] = pos_std[3 * i + 2];
    }
    const std::vector<int> z(n, 18);
    const rgpot::LJPot pot_off(rgpot::LJConfig{
        .u0 = 1.0, .cutoff = 15.0, .psi = 1.0, .tail_correction = false});
    const rgpot::LJPot pot_on(rgpot::LJConfig{
        .u0 = 1.0, .cutoff = 15.0, .psi = 1.0, .tail_correction = true});

    std::vector<double> f_std_off(3 * n), f_rot_off(3 * n);
    std::vector<double> f_std_on(3 * n), f_rot_on(3 * n);
    std::vector<double> ea_std(n), ea_rot(n), va_std(6 * n), va_rot(6 * n);
    rgpot::ForceOut o_std_off{f_std_off.data(), 0.0, 0.0, {}, 0};
    rgpot::ForceOut o_rot_off{f_rot_off.data(), 0.0, 0.0, {}, 0};
    rgpot::ForceOut o_std_on{f_std_on.data(), 0.0, 0.0, {}, 0};
    rgpot::ForceOut o_rot_on{f_rot_on.data(), 0.0, 0.0, {}, 0};

    pot_off.forceImpl({n, pos_std.data(), z.data(), std_box.data()}, &o_std_off);
    pot_off.forceImpl({n, pos_rot.data(), z.data(), rot_box.data()}, &o_rot_off);
    pot_on.forcePerAtom({n, pos_std.data(), z.data(), std_box.data()}, &o_std_on,
                        {ea_std.data(), va_std.data()});
    pot_on.forcePerAtom({n, pos_rot.data(), z.data(), rot_box.data()}, &o_rot_on,
                        {ea_rot.data(), va_rot.data()});

    const double tail_std = o_std_on.energy - o_std_off.energy;
    const double tail_rot = o_rot_on.energy - o_rot_off.energy;
    const Totals want_tail =
        standardTail(n, side * side * side, 1.0, 1.0, 15.0);
    // All 6 pairs lie inside the 15 A cutoff, so turning on tail_correction
    // adds 6 * V(rc) (removing the shift) plus want_tail.energy.
    CHECK_THAT(tail_std - 6.0 * bareV(1.0, 1.0, 15.0),
               WithinRel(want_tail.energy, 1e-9));
    CHECK_THAT(tail_rot, WithinRel(tail_std, 1e-14));
    CHECK_THAT(o_rot_on.energy, WithinRel(o_std_on.energy, 1e-12));
    for (std::size_t i = 0; i < n; ++i) {
      CHECK_THAT(ea_rot[i], WithinRel(ea_std[i], 1e-12));
    }
    const double p_std =
        -(o_std_on.stress[0] + o_std_on.stress[4] + o_std_on.stress[8]) / 3.0;
    const double p_rot =
        -(o_rot_on.stress[0] + o_rot_on.stress[4] + o_rot_on.stress[8]) / 3.0;
    CHECK_THAT(p_rot, WithinRel(p_std, 1e-12));
  }

  // Single atom at rc = 2.5, L = 20: E_tail = -3.24e-5 eV.
  {
    const double l20 = 20.0;
    const std::array<double, 9> box20_std{l20, 0.0, 0.0, 0.0, l20,
                                          0.0, 0.0, 0.0, l20};
    const std::array<double, 9> box20_rot{0.0,  l20, 0.0, -l20, 0.0,
                                          0.0,  0.0, 0.0, l20};
    const std::array<double, 3> p{0.0, 0.0, 0.0};
    const std::array<int, 1> z{18};
    const rgpot::LJPot pot(rgpot::LJConfig{
        .u0 = 1.0, .cutoff = 2.5, .psi = 1.0, .tail_correction = true});
    std::vector<double> f_s(3), f_r(3);
    rgpot::ForceOut o_s{f_s.data(), 0.0, 0.0, {}, 0};
    rgpot::ForceOut o_r{f_r.data(), 0.0, 0.0, {}, 0};
    pot.forceImpl({1, p.data(), z.data(), box20_std.data()}, &o_s);
    pot.forceImpl({1, p.data(), z.data(), box20_rot.data()}, &o_r);
    CHECK_THAT(o_s.energy,
               WithinRel(standardTail(1, l20 * l20 * l20, 1.0, 1.0, 2.5).energy,
                         1e-14));
    CHECK(o_r.energy == o_s.energy);
    CHECK(std::equal(std::begin(o_r.stress), std::end(o_r.stress),
                     std::begin(o_s.stress)));
  }
}
