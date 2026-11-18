// MIT License
// Copyright 2023--present rgpot developers
//
// A periodic EAM evaluation whose cutoff passes half a face spacing has
// more than one image of a pair inside the cutoff. The energy and the
// forces are the sum of those images. The reference is that sum, each
// image evaluated as its own non-periodic pair.

#include <cmath>
#include <sstream>
#include <string>
#include <vector>

#include <catch2/catch_all.hpp>

#include "rgpot/EAMTable/EAMTablePot.hpp"

using Catch::Matchers::WithinAbs;

namespace {

std::string pair_table() {
  // phi(r) = (5 - r)^2 on knots one angstrom apart, stored as r phi.
  // Embedding and density are zero, so the energy is the pair sum.
  std::ostringstream s;
  s << "pair\npair\npair\n1 Cu\n5 1 6 1 5\n29 63.546 1 fcc\n";
  for (int m = 0; m < 5; ++m) {
    s << "0 ";
  }
  s << '\n';
  for (int m = 0; m < 6; ++m) {
    s << "0 ";
  }
  s << '\n';
  for (int m = 0; m < 6; ++m) {
    const double r = static_cast<double>(m);
    s << (r * (5.0 - r) * (5.0 - r)) << ' ';
  }
  s << '\n';
  return s.str();
}

rgpot::EAMTablePot pot(bool periodic) {
  std::istringstream in(pair_table());
  return rgpot::EAMTablePot(
      rgpot::read_eam_table(in, rgpot::EAMTableFormat::Setfl), periodic, false);
}

struct Eval {
  double energy = 0.0;
  std::vector<double> forces;
};

Eval run(const rgpot::EAMTablePot &p, const std::vector<double> &pos,
         const double *box) {
  Eval e;
  e.forces.assign(pos.size(), 0.0);
  rgpot::ForceOut out{};
  out.F = e.forces.data();
  const std::vector<int> z(pos.size() / 3, 29);
  p.forceImpl(rgpot::ForceInput{z.size(), pos.data(), z.data(), box}, &out);
  e.energy = out.energy;
  return e;
}

/// Cartesian vector of lattice image (h, k, l). Rows of `box` are the
/// lattice vectors.
void image(const double *box, int h, int k, int l, double &x, double &y,
           double &z) {
  x = h * box[0] + k * box[3] + l * box[6];
  y = h * box[1] + k * box[4] + l * box[7];
  z = h * box[2] + k * box[5] + l * box[8];
}

double pair_energy(const rgpot::EAMTablePot &free, double r) {
  const double far[9] = {40, 0, 0, 0, 40, 0, 0, 0, 40};
  const std::vector<double> pos{0, 0, 0, r, 0, 0};
  return run(free, pos, far).energy;
}

/// Sum of every image with 0 < r < cutoff. Distinct atoms take every
/// lattice translation. An atom takes every nonzero translation once.
double brute_energy(const rgpot::EAMTablePot &free, const std::vector<double> &pos,
                    const double *box, double cutoff) {
  const int n = static_cast<int>(pos.size() / 3);
  const double cut2 = cutoff * cutoff;
  double energy = 0.0;
  for (int i = 0; i < n; ++i) {
    for (int j = i; j < n; ++j) {
      for (int h = -4; h <= 4; ++h) {
        for (int k = -4; k <= 4; ++k) {
          for (int l = -4; l <= 4; ++l) {
            if (i == j && h == 0 && k == 0 && l == 0) {
              continue;
            }
            double sx = 0, sy = 0, sz = 0;
            image(box, h, k, l, sx, sy, sz);
            const double dx = pos[3 * i] - (pos[3 * j] + sx);
            const double dy = pos[3 * i + 1] - (pos[3 * j + 1] + sy);
            const double dz = pos[3 * i + 2] - (pos[3 * j + 2] + sz);
            const double r2 = dx * dx + dy * dy + dz * dz;
            if (!(r2 > 0.0) || !(r2 < cut2)) {
              continue;
            }
            energy += pair_energy(free, std::sqrt(r2));
          }
        }
      }
    }
  }
  return energy;
}

void check(const std::vector<double> &pos, const double box[9]) {
  const rgpot::EAMTablePot periodic = pot(true);
  const rgpot::EAMTablePot free = pot(false);
  const Eval got = run(periodic, pos, box);
  const double want = brute_energy(free, pos, box, 5.0);
  REQUIRE_THAT(got.energy, WithinAbs(want, 1e-8));
  std::vector<double> moved = pos;
  for (std::size_t c = 0; c < pos.size(); ++c) {
    constexpr double step = 1e-5;
    moved[c] += step;
    const double up = brute_energy(free, moved, box, 5.0);
    moved[c] -= 2.0 * step;
    const double down = brute_energy(free, moved, box, 5.0);
    moved[c] += step;
    const double force = -(up - down) / (2.0 * step);
    REQUIRE_THAT(got.forces[c], WithinAbs(force, 1e-4));
  }
}

} // namespace

TEST_CASE("EAM sums every image once the cutoff passes half a face",
          "[EAMTable][nlist]") {
  // Orthogonal. Half of 6 is 3, and the cutoff is 5. The pair at 1.7 has
  // a second image at 4.3, and both sit inside the cutoff.
  const double ortho[9] = {6, 0, 0, 0, 6, 0, 0, 0, 6};
  check({0.2, 0.3, 0.4, 1.9, 0.5, 0.2}, ortho);

  // One atom. The six face images at 4 angstroms are inside the cutoff.
  check({0.4, 0.5, 0.6}, ortho);

  // Sheared. The same cutoff still passes half of every face.
  const double shear[9] = {6, 0, 0, 1.2, 5.5, 0, 0.3, -0.4, 5.8};
  check({0.2, 0.3, 0.1, 2.1, 0.6, 0.4}, shear);
}
