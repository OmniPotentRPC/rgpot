// MIT License
// Copyright 2023--present rgpot developers
//
// The orthorhombic pair list must match the fused scan's pair content.
// Energies and forces of LJ and Morse go through that list; a minimizer
// that saw a different force would take a different number of steps.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#ifdef RGPOT_HAS_FAST_NLIST
#include "rgpot/nlist/fastnlist.h"
#endif

#include <catch2/catch_all.hpp>

#include "rgpot/LennardJones/LJClusterPot.hpp"
#include "rgpot/LennardJones/LJPot.hpp"
#include "rgpot/Morse/MorsePot.hpp"
#include "rgpot/nlist/PairListCache.hpp"
#include "rgpot/nlist/vesin_visit.hpp"

using Catch::Matchers::WithinAbs;

namespace {

double max_abs(const std::vector<double> &a, const std::vector<double> &b) {
  double worst = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    worst = std::max(worst, std::abs(a[i] - b[i]));
  }
  return worst;
}

void lj_accumulate(int32_t i, int32_t j, double dx, double dy, double dz, double r2,
                   double u0, double psi, double cutoff_u, std::vector<double> &f,
                   double &energy) {
  const double inv_r2 = 1.0 / r2;
  const double sr2 = (psi * psi) * inv_r2;
  const double a = sr2 * sr2 * sr2;
  const double b = 4.0 * u0 * a;
  energy += b * (a - 1.0) - cutoff_u;
  const double scale = 6.0 * b * inv_r2 * (2.0 * a - 1.0);
  f[3 * i] += scale * dx;
  f[3 * i + 1] += scale * dy;
  f[3 * i + 2] += scale * dz;
  f[3 * j] -= scale * dx;
  f[3 * j + 1] -= scale * dy;
  f[3 * j + 2] -= scale * dz;
}

void morse_accumulate(int32_t i, int32_t j, double dx, double dy, double dz, double r2,
                      double De, double a, double re, double shift, std::vector<double> &f,
                      double &energy) {
  const double r = std::sqrt(r2);
  const double gap = 1.0 - std::exp(-a * (r - re));
  energy += De * gap * gap - De - shift;
  const double scale = (2.0 * De * a) * gap * (gap - 1.0) / r;
  f[3 * i] += scale * dx;
  f[3 * i + 1] += scale * dy;
  f[3 * i + 2] += scale * dz;
  f[3 * j] -= scale * dx;
  f[3 * j + 1] -= scale * dy;
  f[3 * j + 2] -= scale * dz;
}

struct Oracle {
  double energy{0.0};
  std::vector<double> forces;
};

Oracle lj_oracle(const std::vector<double> &pos, const double *box,
                 const std::array<bool, 3> &periodic, double cutoff) {
  Oracle out;
  out.forces.assign(pos.size(), 0.0);
  double w[3] = {box[0], box[4], box[8]};
  double inv[3] = {0, 0, 0};
  for (int k = 0; k < 3; ++k) {
    if (periodic[static_cast<std::size_t>(k)] && w[k] != 0.0) {
      inv[k] = 1.0 / w[k];
    }
  }
  const double a = std::pow(1.0 / cutoff, 6.0);
  const double shift = 4.0 * a * (a - 1.0);
  vesin::cpu::brute_force_visit_only(
      pos.data(), pos.size() / 3, w, inv, cutoff * cutoff,
      [&](int32_t i, int32_t j, double dx, double dy, double dz, double r2) {
        lj_accumulate(i, j, -dx, -dy, -dz, r2, 1.0, 1.0, shift, out.forces, out.energy);
      });
  return out;
}

Oracle morse_oracle(const std::vector<double> &pos, const double *box, double cutoff) {
  Oracle out;
  out.forces.assign(pos.size(), 0.0);
  const double De = 0.7102, a = 1.6047, re = 2.897;
  const double gap = 1.0 - std::exp(-a * (cutoff - re));
  const double shift = De * gap * gap - De;
  double w[3] = {box[0], box[4], box[8]};
  double inv[3] = {1.0 / w[0], 1.0 / w[1], 1.0 / w[2]};
  vesin::cpu::brute_force_visit_only(
      pos.data(), pos.size() / 3, w, inv, cutoff * cutoff,
      [&](int32_t i, int32_t j, double dx, double dy, double dz, double r2) {
        morse_accumulate(i, j, -dx, -dy, -dz, r2, De, a, re, shift, out.forces, out.energy);
      });
  return out;
}

// Pair energies are reduced in eight partial sums per atom, the same
// tree on every instruction set. The 343-atom Pt island contributes
// about 3e4 Morse terms. That blocked total differs from this
// sequential scan by about 2e-11, on both the hardware-FMA path and
// the software product, while the forces still agree to 1e-14. A
// missing neighbour is a few tenths of an eV, so the wider energy
// bound still rejects a wrong list.
constexpr double kPtMorseEnergy = 1e-9;

void require_close(double energy, const std::vector<double> &forces, const Oracle &ref,
                   double energy_abs = 1e-12) {
  REQUIRE_THAT(energy - ref.energy, WithinAbs(0.0, energy_abs));
  REQUIRE_THAT(max_abs(forces, ref.forces), WithinAbs(0.0, 1e-12));
}

std::vector<double> slab() {
  // 16 atoms. Two sit across the x boundary so the minimum image is the
  // short one, and the z span is longer than the cutoff.
  std::vector<double> pos;
  for (int i = 0; i < 14; ++i) {
    const double t = static_cast<double>(i);
    pos.push_back(std::fmod(t * 1.7, 18.0));
    pos.push_back(std::fmod(t * 2.3, 18.0));
    pos.push_back(4.0 + 0.15 * t);
  }
  pos.insert(pos.end(), {0.15, 9.0, 6.0, 17.6, 9.2, 6.4});
  return pos;
}

bool load_con(const std::string &path, std::vector<double> &pos, double box[9]) {
  std::ifstream in(path);
  if (!in) {
    return false;
  }
  std::string line;
  std::getline(in, line);
  double lx = 0, ly = 0, lz = 0, a = 0, b = 0, c = 0;
  in >> lx >> ly >> lz >> a >> b >> c;
  int ntypes = 0, nat = 0;
  in >> ntypes >> nat;
  if (nat < 1) {
    return false;
  }
  std::getline(in, line);
  while (std::getline(in, line)) {
    if (line.find("Coordinates of Component") != std::string::npos) {
      break;
    }
  }
  pos.assign(static_cast<std::size_t>(nat) * 3, 0.0);
  for (int i = 0; i < nat; ++i) {
    int fixed = 0, idx = 0;
    if (!(in >> pos[3 * i] >> pos[3 * i + 1] >> pos[3 * i + 2] >> fixed >> idx)) {
      return false;
    }
  }
  box[0] = lx;
  box[1] = 0;
  box[2] = 0;
  box[3] = 0;
  box[4] = ly;
  box[5] = 0;
  box[6] = 0;
  box[7] = 0;
  box[8] = lz;
  return true;
}

bool load_xyz(const std::string &path, std::vector<double> &pos) {
  std::ifstream in(path);
  if (!in) {
    return false;
  }
  pos.clear();
  double x = 0, y = 0, z = 0;
  while (in >> x >> y >> z) {
    pos.push_back(x);
    pos.push_back(y);
    pos.push_back(z);
  }
  return !pos.empty() && pos.size() % 3 == 0;
}

#ifdef RGPOT_HAS_FAST_NLIST
std::size_t pair_count(const std::vector<double> &pos, const double *box,
                       const int32_t periodic[3], double list_cutoff) {
  const std::size_t n = pos.size() / 3;
  const std::size_t cap = n < 2 ? 0 : n * (n - 1) / 2;
  std::vector<int32_t> ij(std::max<std::size_t>(cap, 1) * 2);
  std::size_t count = 0;
  const int rc = rgpot_fast_pairs(pos.data(), n, box, periodic, list_cutoff, ij.data(), cap,
                                  &count);
  REQUIRE(rc == 0);
  return count;
}
#endif

/// Nearest integer, half away from zero. Adding 0.5 and taking ``floor``
/// is not this operation: a value one ulp under a half rounds the sum up.
double anint_away(double x) {
  const double ax = std::abs(x);
  const double whole = std::trunc(ax);
  const double frac = ax - whole;
  const double mag = frac < 0.5 ? whole : whole + 1.0;
  return std::copysign(mag, x);
}

/// Shortest of the rounded fractional image and the 26 neighbours.
/// The inverse matches the cofactor layout in ``rgpot_water_h``. Ties keep
/// the rounded image.
void image26(const double cell[9], const double d[3], double out[3]) {
  auto at = [&](int row, int col) { return cell[col * 3 + row]; };
  double inv[9];
  inv[0] = at(1, 1) * at(2, 2) - at(1, 2) * at(2, 1);
  inv[1] = at(1, 2) * at(2, 0) - at(1, 0) * at(2, 2);
  inv[2] = at(1, 0) * at(2, 1) - at(1, 1) * at(2, 0);
  inv[3] = at(0, 2) * at(2, 1) - at(0, 1) * at(2, 2);
  inv[4] = at(0, 0) * at(2, 2) - at(0, 2) * at(2, 0);
  inv[5] = at(0, 1) * at(2, 0) - at(0, 0) * at(2, 1);
  inv[6] = at(0, 1) * at(1, 2) - at(0, 2) * at(1, 1);
  inv[7] = at(0, 2) * at(1, 0) - at(0, 0) * at(1, 2);
  inv[8] = at(0, 0) * at(1, 1) - at(0, 1) * at(1, 0);
  const double det = at(0, 0) * inv[0] + at(0, 1) * inv[1] + at(0, 2) * inv[2];
  for (double &v : inv) {
    v /= det;
  }
  double s[3];
  for (int row = 0; row < 3; ++row) {
    s[row] = inv[row] * d[0] + inv[3 + row] * d[1] + inv[6 + row] * d[2];
    s[row] -= anint_away(s[row]);
  }
  auto apply = [&](int a, int b, int c, double dest[3]) {
    const double f[3] = {s[0] + static_cast<double>(a), s[1] + static_cast<double>(b),
                         s[2] + static_cast<double>(c)};
    for (int row = 0; row < 3; ++row) {
      dest[row] = at(row, 0) * f[0] + at(row, 1) * f[1] + at(row, 2) * f[2];
    }
  };
  apply(0, 0, 0, out);
  double best = out[0] * out[0] + out[1] * out[1] + out[2] * out[2];
  for (int a = -1; a <= 1; ++a) {
    for (int b = -1; b <= 1; ++b) {
      for (int c = -1; c <= 1; ++c) {
        double trial[3];
        apply(a, b, c, trial);
        const double span = trial[0] * trial[0] + trial[1] * trial[1] + trial[2] * trial[2];
        if (span < best) {
          best = span;
          out[0] = trial[0];
          out[1] = trial[1];
          out[2] = trial[2];
        }
      }
    }
  }
}

} // namespace

extern "C" int rgpot_water_h_force(int natoms, const double *positions,
                                   const int *atomic_numbers, const double *cell,
                                   double *forces, double *energy);

TEST_CASE("LJPot matches the fused scan on a wrapping slab", "[fastnlist][lj]") {
  const auto base = slab();
  const double box[9] = {18, 0, 0, 0, 18, 0, 0, 0, 30};
  rgpot::LJPot pot(rgpot::LJConfig{.cutoff = 6.0});
  const std::vector<int> types(base.size() / 3, 1);
  std::vector<double> pos = base;
  std::vector<double> forces(base.size(), 0.0);

  for (int step = 0; step < 4; ++step) {
    if (step > 0) {
      for (std::size_t k = 0; k < pos.size(); k += 3) {
        const double idx = static_cast<double>(k / 3);
        pos[k] = base[k] + 0.02 * std::sin(0.3 * step + idx);
        pos[k + 1] = base[k + 1] + 0.02 * std::sin(0.4 * step + idx);
        pos[k + 2] = base[k + 2] + 0.02 * std::sin(0.5 * step + idx);
      }
    }
    rgpot::ForceInput in{.nAtoms = pos.size() / 3,
                         .pos = pos.data(),
                         .atmnrs = types.data(),
                         .box = box};
    rgpot::ForceOut out{.F = forces.data(), .energy = 0.0, .variance = 0.0};
    pot.forceImpl(in, &out);
    const auto ref = lj_oracle(pos, box, {{true, true, true}}, 6.0);
    require_close(out.energy, forces, ref);
  }
}

TEST_CASE("MorsePot matches the fused scan across the cell boundary", "[fastnlist][morse]") {
  const auto pos = slab();
  const double box[9] = {20, 0, 0, 0, 20, 0, 0, 0, 30};
  rgpot::MorsePot pot;
  const std::vector<int> types(pos.size() / 3, 78);
  std::vector<double> forces(pos.size(), 0.0);
  rgpot::ForceInput in{.nAtoms = pos.size() / 3,
                       .pos = pos.data(),
                       .atmnrs = types.data(),
                       .box = box};
  rgpot::ForceOut out{.F = forces.data(), .energy = 0.0, .variance = 0.0};
  pot.forceImpl(in, &out);
  const auto ref = morse_oracle(pos, box, 9.5);
  require_close(out.energy, forces, ref);

  // Warm reuse: a second evaluation of the same geometry must not move
  // the energy. That is the cached minimage walk, not a second search
  // with a different pair set.
  rgpot::ForceOut again{.F = forces.data(), .energy = 0.0, .variance = 0.0};
  pot.forceImpl(in, &again);
  REQUIRE_THAT(again.energy - out.energy, WithinAbs(0.0, 0.0));
}

TEST_CASE("LJClusterPot matches the fused scan with open boundaries", "[fastnlist][ljcluster]") {
  const auto pos = slab();
  const double box[9] = {100, 0, 0, 0, 100, 0, 0, 0, 100};
  rgpot::LJClusterPot pot(rgpot::LJClusterConfig{.cutoff = 6.0});
  const std::vector<int> types(pos.size() / 3, 1);
  std::vector<double> forces(pos.size(), 0.0);
  rgpot::ForceInput in{.nAtoms = pos.size() / 3,
                       .pos = pos.data(),
                       .atmnrs = types.data(),
                       .box = box};
  rgpot::ForceOut out{.F = forces.data(), .energy = 0.0, .variance = 0.0};
  pot.forceImpl(in, &out);
  const auto ref = lj_oracle(pos, box, {{false, false, false}}, 6.0);
  require_close(out.energy, forces, ref);
}

TEST_CASE("Partial periodic list matches the fused scan", "[fastnlist]") {
  const auto base = slab();
  const double box[9] = {18, 0, 0, 0, 18, 0, 0, 0, 30};
  rgpot::nlist::CachedPairList::Options opt;
  opt.cutoff = 6.0;
  opt.skin = 1.0;
  opt.periodic = {{true, true, false}};

  auto run = [&](const std::vector<double> &pos) {
    double energy = 0.0;
    std::vector<double> forces(pos.size(), 0.0);
    rgpot::nlist::PairListCache::global().evaluate(
        pos.data(), pos.size() / 3, box, opt,
        [&](int32_t i, int32_t j, double dx, double dy, double dz, double r2) {
          lj_accumulate(i, j, dx, dy, dz, r2, 1.0, 1.0, 0.0, forces, energy);
        });
    const auto ref = lj_oracle(pos, box, opt.periodic, 6.0);
    // The oracle subtracts the cutoff shift. This direct walk does not,
    // so compare forces only: the shift is a constant on every in-cutoff
    // pair and does not change the force.
    REQUIRE_THAT(max_abs(forces, ref.forces), WithinAbs(0.0, 1e-12));
    (void)energy;
  };

  run(base);
  std::vector<double> moved = base;
  for (std::size_t k = 0; k < moved.size(); ++k) {
    moved[k] += 0.02 * std::sin(0.2 * static_cast<double>(k));
  }
  run(moved);
}

TEST_CASE("A skewed cell stays on the fused scan", "[fastnlist]") {
  const auto pos = slab();
  const double box[9] = {18, 0.4, 0, 0, 18, 0, 0, 0.2, 30};
  rgpot::LJPot pot(rgpot::LJConfig{.cutoff = 6.0});
  const std::vector<int> types(pos.size() / 3, 1);
  std::vector<double> forces(pos.size(), 0.0);
  rgpot::ForceInput in{.nAtoms = pos.size() / 3,
                       .pos = pos.data(),
                       .atmnrs = types.data(),
                       .box = box};
  rgpot::ForceOut out{.F = forces.data(), .energy = 0.0, .variance = 0.0};
  pot.forceImpl(in, &out);
  // Diagonal fold, which is what the fused scan applies when the cell
  // is not orthorhombic.
  const auto ref = lj_oracle(pos, box, {{true, true, true}}, 6.0);
  require_close(out.energy, forces, ref);
}

TEST_CASE("A wide cell returns a short linkcell list", "[fastnlist][linkcell]") {
  // 5x5x5 lattice, 25 A box, list cutoff 7 A. Three cutoffs fit in the
  // box, so the search must not hand back every unordered pair.
  std::vector<double> pos;
  for (int i = 0; i < 5; ++i) {
    for (int j = 0; j < 5; ++j) {
      for (int k = 0; k < 5; ++k) {
        pos.push_back((i + 0.5) * 5.0);
        pos.push_back((j + 0.5) * 5.0);
        pos.push_back((k + 0.5) * 5.0);
      }
    }
  }
  const double box[9] = {25, 0, 0, 0, 25, 0, 0, 0, 25};
  const std::size_t n = pos.size() / 3;
#ifdef RGPOT_HAS_FAST_NLIST
  const int32_t periodic[3] = {1, 1, 1};
  const std::size_t count = pair_count(pos, box, periodic, 7.0);
  REQUIRE(count > 0);
  REQUIRE(count < n * (n - 1) / 2);
#endif

  rgpot::LJPot pot(rgpot::LJConfig{.cutoff = 6.0});
  const std::vector<int> types(n, 1);
  std::vector<double> forces(pos.size(), 0.0);
  rgpot::ForceInput in{.nAtoms = n, .pos = pos.data(), .atmnrs = types.data(), .box = box};
  rgpot::ForceOut out{.F = forces.data(), .energy = 0.0, .variance = 0.0};
  pot.forceImpl(in, &out);
  const auto ref = lj_oracle(pos, box, {{true, true, true}}, 6.0);
  require_close(out.energy, forces, ref);

  std::vector<double> moved = pos;
  for (std::size_t a = 0; a < n; ++a) {
    moved[3 * a] += 0.02 * std::sin(0.3 * static_cast<double>(a));
  }
  rgpot::ForceInput moved_in{.nAtoms = n,
                             .pos = moved.data(),
                             .atmnrs = types.data(),
                             .box = box};
  pot.forceImpl(moved_in, &out);
  const auto moved_ref = lj_oracle(moved, box, {{true, true, true}}, 6.0);
  require_close(out.energy, forces, moved_ref);
}

#ifdef RGPOT_HAS_FAST_NLIST
TEST_CASE("A narrow cell is every unordered pair", "[fastnlist]") {
  const auto pos = slab();
  const double box[9] = {18, 0, 0, 0, 18, 0, 0, 0, 30};
  const int32_t periodic[3] = {1, 1, 1};
  const std::size_t n = pos.size() / 3;
  // List cutoff 7 A, box 18 A: fewer than three bins, so the buffer is
  // the full pair set and the caller filters.
  REQUIRE(pair_count(pos, box, periodic, 7.0) == n * (n - 1) / 2);
}

TEST_CASE("A narrow cell past the old pair cap is every unordered pair", "[fastnlist]") {
  // The client bridge places 10k atoms in a 10 A box under the default
  // Lennard-Jones cutoff of 15 A. A count just above the old 4096 guard
  // is enough: that cell is one bin, and linkcell must not build its stencil.
  constexpr int n = 4100;
  std::vector<double> pos;
  pos.reserve(static_cast<std::size_t>(n) * 3);
  for (int i = 0; i < n; ++i) {
    const double t = static_cast<double>(i);
    pos.push_back(std::fmod(t * 0.37, 10.0));
    pos.push_back(std::fmod(t * 0.53, 10.0));
    pos.push_back(std::fmod(t * 0.71, 10.0));
  }
  const double box[9] = {10, 0, 0, 0, 10, 0, 0, 0, 10};
  const int32_t periodic[3] = {1, 1, 1};
  REQUIRE(pair_count(pos, box, periodic, 15.0) ==
          static_cast<std::size_t>(n) * static_cast<std::size_t>(n - 1) / 2);

  rgpot::nlist::CachedPairList::Options opt;
  opt.cutoff = 15.0;
  std::size_t visited = 0;
  rgpot::nlist::PairListCache::global().evaluate(
      pos.data(), static_cast<std::size_t>(n), box, opt,
      [&](int32_t, int32_t, double, double, double, double) { ++visited; });
  REQUIRE(visited == static_cast<std::size_t>(n) * static_cast<std::size_t>(n - 1) / 2);
}
#endif

TEST_CASE("Optbench LJ clusters and the Pt island match the fused scan", "[fastnlist][optbench]") {
  std::vector<double> lj38;
  double box38[9]{};
  REQUIRE(load_con("benchmarks/data/lj38.con", lj38, box38));
  REQUIRE(lj38.size() / 3 == 38);
  rgpot::LJClusterPot lj(rgpot::LJClusterConfig{.cutoff = 15.0});
  const std::vector<int> h(lj38.size() / 3, 1);
  std::vector<double> forces(lj38.size(), 0.0);
  rgpot::ForceInput in{.nAtoms = lj38.size() / 3,
                       .pos = lj38.data(),
                       .atmnrs = h.data(),
                       .box = box38};
  rgpot::ForceOut out{.F = forces.data(), .energy = 0.0, .variance = 0.0};
  lj.forceImpl(in, &out);
  const auto ref38 = lj_oracle(lj38, box38, {{false, false, false}}, 15.0);
  require_close(out.energy, forces, ref38);
  REQUIRE_THAT(out.energy, WithinAbs(67118.522074988403, 0.0));

  std::vector<double> lj75;
  REQUIRE(load_xyz("benchmarks/data/lj75.xyz", lj75));
  REQUIRE(lj75.size() / 3 == 75);
  const double box100[9] = {100, 0, 0, 0, 100, 0, 0, 0, 100};
  forces.assign(lj75.size(), 0.0);
  const std::vector<int> h75(lj75.size() / 3, 1);
  rgpot::ForceInput in75{.nAtoms = lj75.size() / 3,
                         .pos = lj75.data(),
                         .atmnrs = h75.data(),
                         .box = box100};
  out.F = forces.data();
  lj.forceImpl(in75, &out);
  require_close(out.energy, forces, lj_oracle(lj75, box100, {{false, false, false}}, 15.0));
  REQUIRE_THAT(out.energy, WithinAbs(22735480889.731754, 0.0));

  std::vector<double> lj150;
  REQUIRE(load_xyz("benchmarks/data/lj150.xyz", lj150));
  REQUIRE(lj150.size() / 3 == 150);
  forces.assign(lj150.size(), 0.0);
  const std::vector<int> h150(lj150.size() / 3, 1);
  rgpot::ForceInput in150{.nAtoms = lj150.size() / 3,
                          .pos = lj150.data(),
                          .atmnrs = h150.data(),
                          .box = box100};
  out.F = forces.data();
  lj.forceImpl(in150, &out);
  require_close(out.energy, forces, lj_oracle(lj150, box100, {{false, false, false}}, 15.0));
  REQUIRE_THAT(out.energy, WithinAbs(152095727010.16843, 0.0));

  std::vector<double> pt;
  double boxpt[9]{};
  REQUIRE(load_con("benchmarks/data/pt-island.con", pt, boxpt));
  REQUIRE(pt.size() / 3 == 343);
  rgpot::MorsePot morse;
  forces.assign(pt.size(), 0.0);
  const std::vector<int> zpt(pt.size() / 3, 78);
  rgpot::ForceInput inpt{.nAtoms = pt.size() / 3,
                         .pos = pt.data(),
                         .atmnrs = zpt.data(),
                         .box = boxpt};
  out.F = forces.data();
  morse.forceImpl(inpt, &out);
  require_close(out.energy, forces, morse_oracle(pt, boxpt, 9.5), kPtMorseEnergy);
  REQUIRE_THAT(out.energy, WithinAbs(-1775.7911601735484, kPtMorseEnergy));
  rgpot::ForceOut again{.F = forces.data(), .energy = 0.0, .variance = 0.0};
  morse.forceImpl(inpt, &again);
  REQUIRE_THAT(again.energy - out.energy, WithinAbs(0.0, 0.0));
}

TEST_CASE("Water plus H wraps every leg with minimage", "[fastnlist][water]") {
  constexpr int nmol = 2;
  constexpr int nat = 3 * nmol + 1;
  std::vector<double> pos(3 * nat, 0.0);
  std::vector<int> z(nat, 1);
  const double roh = 0.9572;
  const double half = 0.5 * 104.52 * std::acos(-1.0) / 180.0;
  const double armx = roh * std::sin(half);
  const double army = roh * std::cos(half);
  for (int m = 0; m < nmol; ++m) {
    const double ox = 3.0 + m * 3.2;
    const int h1 = 2 * m;
    const int h2 = 2 * m + 1;
    const int o = 2 * nmol + m;
    pos[3 * h1] = ox + armx;
    pos[3 * h1 + 1] = 3.0 + army;
    pos[3 * h1 + 2] = 3.0;
    pos[3 * h2] = ox - armx;
    pos[3 * h2 + 1] = 3.0 + army;
    pos[3 * h2 + 2] = 3.0;
    pos[3 * o] = ox;
    pos[3 * o + 1] = 3.0;
    pos[3 * o + 2] = 3.0;
    z[o] = 8;
  }
  pos[3 * (nat - 1)] = 8.0;
  pos[3 * (nat - 1) + 1] = 4.0;
  pos[3 * (nat - 1) + 2] = 3.2;
  const double cell[9] = {12, 0, 0, 0, 12, 0, 0, 0, 12};
  std::vector<double> forces(pos.size(), 0.0);
  double energy = 0.0;
  REQUIRE(rgpot_water_h_force(nat, pos.data(), z.data(), cell, forces.data(), &energy) == 0);
  double again = 0.0;
  REQUIRE(rgpot_water_h_force(nat, pos.data(), z.data(), cell, forces.data(), &again) == 0);
  REQUIRE_THAT(again - energy, WithinAbs(0.0, 0.0));
  // A zero cell is the isolated molecule. The same coordinates must
  // still return a finite energy, and a second call must agree.
  const double open_cell[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
  double isolated = 0.0;
  REQUIRE(rgpot_water_h_force(nat, pos.data(), z.data(), open_cell, forces.data(), &isolated) ==
          0);
  double isolated_again = 0.0;
  REQUIRE(rgpot_water_h_force(nat, pos.data(), z.data(), open_cell, forces.data(),
                              &isolated_again) == 0);
  REQUIRE_THAT(isolated_again - isolated, WithinAbs(0.0, 0.0));
  REQUIRE(std::isfinite(isolated));
}

TEST_CASE("A rigid shift keeps the Pt list", "[fastnlist][morse]") {
  std::vector<double> base;
  double box[9]{};
  REQUIRE(load_con("benchmarks/data/pt-island.con", base, box));
  rgpot::MorsePot pot;
  const std::vector<int> z(base.size() / 3, 78);
  std::vector<double> forces(base.size(), 0.0);
  rgpot::ForceInput in{.nAtoms = base.size() / 3,
                       .pos = base.data(),
                       .atmnrs = z.data(),
                       .box = box};
  rgpot::ForceOut out{.F = forces.data(), .energy = 0.0, .variance = 0.0};
  pot.forceImpl(in, &out);
  const double built = out.energy;

  std::vector<double> shifted = base;
  for (std::size_t k = 0; k < shifted.size(); k += 3) {
    shifted[k] += 2.0;
  }
  rgpot::ForceInput in_shift{.nAtoms = shifted.size() / 3,
                             .pos = shifted.data(),
                             .atmnrs = z.data(),
                             .box = box};
  pot.forceImpl(in_shift, &out);
  require_close(out.energy, forces, morse_oracle(shifted, box, 9.5), kPtMorseEnergy);
  REQUIRE_THAT(out.energy - built, WithinAbs(0.0, 0.0));

  std::vector<double> pulled = base;
  pulled[0] += 2.0;
  pulled[1] -= 1.5;
  rgpot::ForceInput in_pull{.nAtoms = pulled.size() / 3,
                            .pos = pulled.data(),
                            .atmnrs = z.data(),
                            .box = box};
  pot.forceImpl(in_pull, &out);
  require_close(out.energy, forces, morse_oracle(pulled, box, 9.5), kPtMorseEnergy);
}

TEST_CASE("A non-rigid move inside the skin matches a fresh scan", "[fastnlist][lj]") {
  // Skin is 1 A. Atom 0 moves 0.6 A, past half a skin, and atom 1 moves
  // the other way. The displacement cloud's diagonal stays under the skin,
  // so a cached list is still the right pair set.
  const double box[9] = {36, 0, 0, 0, 36, 0, 0, 0, 36};
  std::vector<double> base = {0, 0, 0, 5.5, 0, 0, 0, 5.5, 0, 10, 10, 10, 20, 1, 1, 1, 20, 2};
  rgpot::LJPot pot(rgpot::LJConfig{.cutoff = 6.0});
  const std::vector<int> types(base.size() / 3, 1);
  std::vector<double> forces(base.size(), 0.0);
  rgpot::ForceInput in{.nAtoms = base.size() / 3,
                       .pos = base.data(),
                       .atmnrs = types.data(),
                       .box = box};
  rgpot::ForceOut out{.F = forces.data(), .energy = 0.0, .variance = 0.0};
  pot.forceImpl(in, &out);
  require_close(out.energy, forces, lj_oracle(base, box, {{true, true, true}}, 6.0));

  std::vector<double> moved = base;
  const double shift[18] = {0.6, 0, 0, -0.2, 0, 0, 0, 0.1, 0, 0, 0, -0.1, 0.05, 0, 0, 0, -0.05, 0};
  for (std::size_t k = 0; k < moved.size(); ++k) {
    moved[k] += shift[k];
  }
  double minx = shift[0];
  double maxx = shift[0];
  double miny = shift[1];
  double maxy = shift[1];
  double minz = shift[2];
  double maxz = shift[2];
  for (std::size_t a = 1; a < moved.size() / 3; ++a) {
    minx = std::min(minx, shift[3 * a]);
    maxx = std::max(maxx, shift[3 * a]);
    miny = std::min(miny, shift[3 * a + 1]);
    maxy = std::max(maxy, shift[3 * a + 1]);
    minz = std::min(minz, shift[3 * a + 2]);
    maxz = std::max(maxz, shift[3 * a + 2]);
  }
  const double ex = maxx - minx;
  const double ey = maxy - miny;
  const double ez = maxz - minz;
  REQUIRE(ex * ex + ey * ey + ez * ez < 1.0);
  REQUIRE(std::abs(shift[0]) > 0.5);

  rgpot::ForceInput in_moved{.nAtoms = moved.size() / 3,
                             .pos = moved.data(),
                             .atmnrs = types.data(),
                             .box = box};
  pot.forceImpl(in_moved, &out);
  require_close(out.energy, forces, lj_oracle(moved, box, {{true, true, true}}, 6.0));
}

TEST_CASE("Skewed WaterH matches the 26-image oracle", "[fastnlist][water]") {
  // a = (8, 0, 0), b = (7.2, 0.8, 0), c = (0, 0, 8). The oxygen at
  // (4, 0.3, 1) has fractional length squared 17.09. A neighbour in the
  // shell of 26 has length squared 11.49.
  const double cell[9] = {8, 0, 0, 7.2, 0.8, 0, 0, 0, 8};
  const int nat = 4;
  std::vector<double> pos = {8.36, 0.99, 0.2, 6.84, 0.99, -0.1, 4.0, 0.3, 1.0, 0, 0, 0};
  const std::vector<int> z = {1, 1, 8, 1};
  double oxygen[3];
  image26(cell, pos.data() + 6, oxygen);
  const double raw2 = pos[6] * pos[6] + pos[7] * pos[7] + pos[8] * pos[8];
  const double near2 = oxygen[0] * oxygen[0] + oxygen[1] * oxygen[1] + oxygen[2] * oxygen[2];
  REQUIRE(near2 < raw2);

  std::vector<double> forces(pos.size(), 0.0);
  double energy = 0.0;
  REQUIRE(rgpot_water_h_force(nat, pos.data(), z.data(), cell, forces.data(), &energy) == 0);

  std::vector<double> open = pos;
  for (int site = 0; site < nat - 1; ++site) {
    double image[3];
    const double d[3] = {pos[3 * site] - pos[9], pos[3 * site + 1] - pos[10],
                         pos[3 * site + 2] - pos[11]};
    image26(cell, d, image);
    open[3 * site] = pos[9] + image[0];
    open[3 * site + 1] = pos[10] + image[1];
    open[3 * site + 2] = pos[11] + image[2];
  }
  const double zero[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<double> open_forces(pos.size(), 0.0);
  double open_energy = 0.0;
  REQUIRE(rgpot_water_h_force(nat, open.data(), z.data(), zero, open_forces.data(), &open_energy) ==
          0);
  REQUIRE_THAT(energy - open_energy, WithinAbs(0.0, 0.0));
  REQUIRE_THAT(max_abs(forces, open_forces), WithinAbs(0.0, 0.0));
}
