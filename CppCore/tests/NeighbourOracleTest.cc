// MIT License
// Copyright 2023--present rgpot developers
//
// The stored and streaming cell lists, with a skin, against an all-pairs
// minimum image. The reference does not bin atoms. A skin move just inside
// and just outside half the skin is compared with a fresh build.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <map>
#include <numeric>
#include <random>
#include <utility>
#include <vector>

#include <catch2/catch_all.hpp>

#include "rgpot/LennardJones/LJPot.hpp"
#include "rgpot/nlist/MinimumImage.hpp"
#include "rgpot/nlist/Neighbours.hpp"

namespace {

constexpr double kPi = 3.14159265358979323846;

void setThreads(const char *value) {
#if defined(_WIN32)
  _putenv_s("OMP_NUM_THREADS", value == nullptr ? "" : value);
#else
  if (value == nullptr || value[0] == '\0') {
    unsetenv("OMP_NUM_THREADS");
  } else {
    setenv("OMP_NUM_THREADS", value, 1);
  }
#endif
}

struct Geom {
  double dx = 0;
  double dy = 0;
  double dz = 0;
  double r2 = 0;
};

using Pairs = std::map<std::pair<int, int>, Geom>;

double shortestPeriodic(const double cell[9], const std::array<bool, 3> &periodic) {
  double best = std::numeric_limits<double>::infinity();
  for (int i = -2; i <= 2; ++i) {
    if (!periodic[0] && i != 0) {
      continue;
    }
    for (int j = -2; j <= 2; ++j) {
      if (!periodic[1] && j != 0) {
        continue;
      }
      for (int k = -2; k <= 2; ++k) {
        if (!periodic[2] && k != 0) {
          continue;
        }
        if (i == 0 && j == 0 && k == 0) {
          continue;
        }
        const double x = i * cell[0] + j * cell[3] + k * cell[6];
        const double y = i * cell[1] + j * cell[4] + k * cell[7];
        const double z = i * cell[2] + j * cell[5] + k * cell[8];
        best = std::min(best, x * x + y * y + z * z);
      }
    }
  }
  return std::sqrt(best);
}

void atFraction(const double cell[9], double fi, double fj, double fk, double out[3]) {
  for (int c = 0; c < 3; ++c) {
    out[c] = fi * cell[c] + fj * cell[3 + c] + fk * cell[6 + c];
  }
}

Pairs fromList(rgpot::nlist::Neighbours &list, std::size_t n) {
  Pairs out;
  list.forDirected(0, static_cast<std::int64_t>(n),
                   [&](std::int32_t i, std::int32_t j, double dx, double dy, double dz, double r2) {
                     const auto key = std::make_pair(static_cast<int>(i), static_cast<int>(j));
                     INFO("duplicate pair " << i << "," << j);
                     REQUIRE(out.find(key) == out.end());
                     out.emplace(key, Geom{dx, dy, dz, r2});
                   });
  return out;
}

Pairs referencePairs(const double cell[9], const std::array<bool, 3> &periodic,
                     const std::vector<double> &pos, double search2) {
  const rgpot::nlist::MinimumImage image(cell, periodic);
  const auto n = pos.size() / 3;
  Pairs out;
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      if (i == j) {
        continue;
      }
      double dx = pos[3 * i] - pos[3 * j];
      double dy = pos[3 * i + 1] - pos[3 * j + 1];
      double dz = pos[3 * i + 2] - pos[3 * j + 2];
      image.fold(dx, dy, dz);
      const double r2 = dx * dx + dy * dy + dz * dz;
      if (r2 <= search2) {
        out.emplace(std::make_pair(static_cast<int>(i), static_cast<int>(j)),
                    Geom{dx, dy, dz, r2});
      }
    }
  }
  return out;
}

void expectMatches(const Pairs &got, const Pairs &ref, double search2) {
  const double band = 1e-8 * std::max(1.0, search2);
  for (const auto &item : ref) {
    if (!(item.second.r2 < search2 - band)) {
      continue;
    }
    INFO("missing " << item.first.first << "," << item.first.second
                    << " r2=" << item.second.r2);
    REQUIRE(got.find(item.first) != got.end());
  }
  for (const auto &item : got) {
    const auto it = ref.find(item.first);
    INFO("extra " << item.first.first << "," << item.first.second
                  << " r2=" << item.second.r2);
    REQUIRE(it != ref.end());
    const double scale = std::max(1.0, it->second.r2);
    INFO("distance " << item.first.first << "," << item.first.second);
    REQUIRE(std::abs(item.second.r2 - it->second.r2) <= 1e-8 * scale);
  }
}

void checkList(const double cell[9], const std::array<bool, 3> &periodic,
               const std::vector<double> &pos, double cutoff, double skin) {
  const double search = cutoff + skin;
  const double search2 = search * search;
  const auto ref = referencePairs(cell, periodic, pos, search2);
  for (const auto mode : {rgpot::nlist::Mode::Stored, rgpot::nlist::Mode::Stream}) {
    rgpot::nlist::Spec spec;
    spec.cutoff = cutoff;
    spec.skin = skin;
    spec.periodic = periodic;
    spec.image = rgpot::nlist::Image::Round;
    spec.mode = mode;
    rgpot::nlist::Neighbours list;
    list.update(pos.data(), pos.size() / 3, cell, spec);
    if (mode == rgpot::nlist::Mode::Stored) {
      REQUIRE_FALSE(list.streaming());
    }
    expectMatches(fromList(list, pos.size() / 3), ref, search2);
  }
}

std::vector<double> positions(const double cell[9], std::mt19937_64 &rng, int nrandom) {
  std::uniform_real_distribution<double> frac(-0.05, 1.05);
  std::vector<double> pos;
  auto push = [&](double fi, double fj, double fk) {
    double p[3];
    atFraction(cell, fi, fj, fk, p);
    pos.insert(pos.end(), p, p + 3);
  };
  // Faces of the parallelepiped, including the far face.
  push(0.0, 0.2, 0.3);
  push(1.0, 0.2, 0.35);
  push(0.4, 0.0, 0.6);
  push(0.45, 1.0, 0.55);
  push(0.15, 0.7, 0.0);
  push(0.2, 0.75, 1.0);
  for (int i = 0; i < nrandom; ++i) {
    push(frac(rng), frac(rng), frac(rng));
  }
  // One pair along the first row, inside half of that row.
  push(0.1, 0.5, 0.5);
  push(0.1 + 0.2, 0.5, 0.5);
  return pos;
}

void monoclinic(double degrees, double cell[9]) {
  const double rad = degrees * kPi / 180.0;
  const double L = 10.0;
  cell[0] = L;
  cell[1] = 0.0;
  cell[2] = 0.0;
  cell[3] = 0.0;
  cell[4] = L;
  cell[5] = 0.0;
  cell[6] = L * std::cos(rad);
  cell[7] = 0.0;
  cell[8] = L * std::sin(rad);
}

bool samePairs(const Pairs &a, const Pairs &b) {
  if (a.size() != b.size()) {
    return false;
  }
  for (const auto &item : a) {
    const auto it = b.find(item.first);
    if (it == b.end()) {
      return false;
    }
    const double scale = std::max(1.0, item.second.r2);
    if (std::abs(item.second.r2 - it->second.r2) > 1e-9 * scale + 1e-12) {
      return false;
    }
  }
  return true;
}

// Displacement stored in `pos` relative to `ref`, on the side of the skin
// test the caller asks for. The comparison is the one `withinSkin` uses,
// after the coordinates have been written.
void placeSkin(std::vector<double> &pos, const std::vector<double> &ref, std::size_t atom,
               double ux, double uy, double uz, double skin, bool inside) {
  const double limit2 = 0.25 * skin * skin;
  const double u2 = ux * ux + uy * uy + uz * uz;
  double scale = std::sqrt(limit2 / u2);
  for (int n = 0; n < 100000; ++n) {
    pos[3 * atom] = ref[3 * atom] + ux * scale;
    pos[3 * atom + 1] = ref[3 * atom + 1] + uy * scale;
    pos[3 * atom + 2] = ref[3 * atom + 2] + uz * scale;
    const double dx = pos[3 * atom] - ref[3 * atom];
    const double dy = pos[3 * atom + 1] - ref[3 * atom + 1];
    const double dz = pos[3 * atom + 2] - ref[3 * atom + 2];
    const bool over = dx * dx + dy * dy + dz * dz > limit2;
    if (inside != over) {
      return;
    }
    scale = std::nextafter(scale, inside ? 0.0 : std::numeric_limits<double>::infinity());
  }
}

} // namespace

TEST_CASE("cell lists match an all-pairs minimum image", "[nlist][oracle]") {
  setThreads("1");
  double ortho[9] = {12.0, 0, 0, 0, 10.0, 0, 0, 0, 14.0};
  double mono[9];
  monoclinic(100.0, mono);
  const double skew[9] = {12.0, 0.0, 0.0, 11.2, 1.5, 0.0, 10.4, 0.8, 1.2};
  const double thin[9] = {1.0, 1.0, 0.0, 0.0, 5.0e-10, 0.0, 0.0, 0.0, 1.0};
  const double *cells[4] = {ortho, mono, skew, thin};
  const int masks[] = {1, 2, 4, 3, 5, 6, 7};
  std::mt19937_64 rng(19);
  for (const double *cell : cells) {
    const auto pos = positions(cell, rng, 8);
    for (int mask : masks) {
      const std::array<bool, 3> periodic{(mask & 1) != 0, (mask & 2) != 0,
                                         (mask & 4) != 0};
      const double shortest = shortestPeriodic(cell, periodic);
      REQUIRE(std::isfinite(shortest));
      REQUIRE(shortest > 0.0);
      const double near_half = 0.49 * shortest;
      INFO("mask=" << mask << " cutoff=" << near_half);
      checkList(cell, periodic, pos, near_half, 0.05 * shortest);
      if (shortest < 1.0) {
        // The long sides still have a cell list. Half of a length-1 row.
        checkList(cell, periodic, pos, 0.49, 0.05);
      }
    }
  }
}

TEST_CASE("a wide orthogonal cell keeps pairs the slab test could drop",
          "[nlist][oracle]") {
  setThreads("1");
  // Forty angstroms and a search of 6 give at least five bins. The
  // rebuild skips a neighbour cell only when every atom in it is past
  // the search. These two sit on either side of a bin face, inside it.
  const double cell[9] = {40.0, 0, 0, 0, 40.0, 0, 0, 0, 40.0};
  std::vector<double> pos;
  auto add = [&](double x, double y, double z) {
    pos.push_back(x);
    pos.push_back(y);
    pos.push_back(z);
  };
  add(6.2, 20.0, 20.0);
  add(7.0, 20.0, 20.0);
  std::mt19937_64 rng(41);
  std::uniform_real_distribution<double> u(0.0, 40.0);
  for (int i = 0; i < 80; ++i) {
    add(u(rng), u(rng), u(rng));
  }
  const std::array<bool, 3> periodic{true, true, true};
  checkList(cell, periodic, pos, 5.0, 1.0);
}

TEST_CASE("an open axis of a sheared cell follows the atoms' fractional extent",
          "[nlist][oracle]") {
  setThreads("1");
  double mono[9];
  monoclinic(100.0, mono);
  // Two clusters many cutoffs apart along the open z axis, plus a pair
  // that meets across the periodic x face. A single slab would still
  // find these pairs; bins along the extent must not drop them.
  std::vector<double> pos;
  auto add = [&](double x, double y, double z) {
    pos.push_back(x);
    pos.push_back(y);
    pos.push_back(z);
  };
  add(0.05, 1.0, 0.2);
  add(0.4, 1.1, 0.5);
  add(9.9, 1.05, 0.3);
  add(2.0, 3.0, 40.0);
  add(2.2, 3.1, 40.4);
  add(1.5, 2.8, 41.0);
  add(5.0, 5.0, 20.0);
  const std::array<bool, 3> periodic{true, true, false};
  checkList(mono, periodic, pos, 2.5, 0.4);
}

TEST_CASE("a skin move matches a fresh neighbour list", "[nlist][oracle][skin]") {
  setThreads("1");
  const double ortho[9] = {16.0, 0, 0, 0, 14.0, 0, 0, 0, 18.0};
  double mono[9];
  monoclinic(100.0, mono);
  const double *cells[2] = {ortho, mono};
  const double cutoff = 3.0;
  const double skin = 1.0;
  std::mt19937_64 rng(23);
  std::uniform_real_distribution<double> box01(0.05, 0.95);
  std::uniform_real_distribution<double> ang(0.0, 2.0 * kPi);
  std::uniform_real_distribution<double> z(-1.0, 1.0);
  for (const double *cell : cells) {
    std::vector<double> pos;
    for (int i = 0; i < 10; ++i) {
      double p[3];
      atFraction(cell, box01(rng), box01(rng), box01(rng), p);
      pos.insert(pos.end(), p, p + 3);
    }
    const std::array<bool, 3> periodic{true, true, true};
    rgpot::nlist::Spec spec;
    spec.cutoff = cutoff;
    spec.skin = skin;
    spec.periodic = periodic;
    spec.image = rgpot::nlist::Image::Round;
    spec.mode = rgpot::nlist::Mode::Stored;
    rgpot::nlist::Neighbours kept;
    kept.update(pos.data(), pos.size() / 3, cell, spec);
    REQUIRE_FALSE(kept.reused());

    std::vector<double> inside = pos;
    for (std::size_t a = 0; a < inside.size() / 3; ++a) {
      const double theta = ang(rng);
      const double uz = z(rng);
      const double xy = std::sqrt(std::max(0.0, 1.0 - uz * uz));
      placeSkin(inside, pos, a, xy * std::cos(theta), xy * std::sin(theta), uz, skin, true);
    }
    kept.update(inside.data(), inside.size() / 3, cell, spec);
    REQUIRE(kept.reused());
    // The stored shell is the old cutoff-plus-skin set. A fresh build
    // drops pairs that left that shell. Every pair now inside the cutoff
    // has to still be there, and every stored distance has to be the
    // minimum image of the moved atoms.
    const auto kept_pairs = fromList(kept, inside.size() / 3);
    const double search2 = (cutoff + skin) * (cutoff + skin);
    const auto ref_in = referencePairs(cell, periodic, inside, search2);
    const double cutoff2 = cutoff * cutoff;
    const double band = 1e-8 * std::max(1.0, cutoff2);
    for (const auto &item : ref_in) {
      if (!(item.second.r2 < cutoff2 - band)) {
        continue;
      }
      INFO("skin missed " << item.first.first << "," << item.first.second);
      REQUIRE(kept_pairs.find(item.first) != kept_pairs.end());
    }
    const rgpot::nlist::MinimumImage image(cell, periodic);
    for (const auto &item : kept_pairs) {
      const int i = item.first.first;
      const int j = item.first.second;
      double dx = inside[3 * i] - inside[3 * j];
      double dy = inside[3 * i + 1] - inside[3 * j + 1];
      double dz = inside[3 * i + 2] - inside[3 * j + 2];
      image.fold(dx, dy, dz);
      const double r2 = dx * dx + dy * dy + dz * dz;
      const double scale = std::max(1.0, r2);
      INFO("skin distance " << i << "," << j);
      REQUIRE(std::abs(item.second.r2 - r2) <= 1e-8 * scale);
    }
    // The stream folds every stencil pair. On the moved geometry it has
    // to match the all-pairs search set, including after the fractional
    // bound skips pairs that cannot be inside it.
    spec.mode = rgpot::nlist::Mode::Stream;
    rgpot::nlist::Neighbours streamed;
    streamed.update(inside.data(), inside.size() / 3, cell, spec);
    REQUIRE(streamed.streaming());
    expectMatches(fromList(streamed, inside.size() / 3), ref_in, search2);
    spec.mode = rgpot::nlist::Mode::Stored;

    std::vector<double> outside = inside;
    const double theta = ang(rng);
    const double uz = z(rng);
    const double xy = std::sqrt(std::max(0.0, 1.0 - uz * uz));
    // From the original reference, not from the inside pose. The kept
    // list still measures the skin against the first build.
    placeSkin(outside, pos, 0, xy * std::cos(theta), xy * std::sin(theta), uz, skin, false);
    kept.update(outside.data(), outside.size() / 3, cell, spec);
    REQUIRE_FALSE(kept.reused());
    rgpot::nlist::Neighbours fresh_out;
    fresh_out.update(outside.data(), outside.size() / 3, cell, spec);
    REQUIRE(samePairs(fromList(kept, outside.size() / 3),
                      fromList(fresh_out, outside.size() / 3)));
  }
}

TEST_CASE("shuffled atom indices keep each row on its atom", "[nlist][oracle][order]") {
  setThreads("1");
  const double ortho[9] = {22, 0, 0, 0, 18, 0, 0, 0, 24};
  double shear[9];
  monoclinic(100.0, shear);
  const double tri[9] = {16, 0.4, 0.2, 0.6, 18, -0.3, 0.2, -0.4, 20};
  const double *cells[3] = {ortho, shear, tri};
  const double cutoff = 4.0;
  const double skin = 1.0;
  const double search2 = (cutoff + skin) * (cutoff + skin);
  std::mt19937_64 rng(91);
  std::uniform_real_distribution<double> frac(0.0, 1.0);
  for (const double *cell : cells) {
    std::vector<double> ordered;
    for (int i = 0; i < 24; ++i) {
      const double f0 = frac(rng);
      const double f1 = frac(rng);
      const double f2 = frac(rng);
      for (int c = 0; c < 3; ++c) {
        ordered.push_back(f0 * cell[c] + f1 * cell[3 + c] + f2 * cell[6 + c]);
      }
    }
    std::vector<int> perm(ordered.size() / 3);
    std::iota(perm.begin(), perm.end(), 0);
    std::shuffle(perm.begin(), perm.end(), rng);
    std::vector<double> pos(ordered.size());
    for (std::size_t i = 0; i < perm.size(); ++i) {
      for (int c = 0; c < 3; ++c) {
        pos[3 * i + c] = ordered[3 * static_cast<std::size_t>(perm[i]) + c];
      }
    }
    const std::array<bool, 3> periodic{true, true, true};
    rgpot::nlist::Spec spec;
    spec.cutoff = cutoff;
    spec.skin = skin;
    spec.periodic = periodic;
    spec.image = rgpot::nlist::Image::Round;
    spec.mode = rgpot::nlist::Mode::Stored;
    rgpot::nlist::Neighbours list;
    list.update(pos.data(), pos.size() / 3, cell, spec);
    REQUIRE_FALSE(list.streaming());
    const auto full = fromList(list, pos.size() / 3);
    const auto ref = referencePairs(cell, periodic, pos, search2);
    expectMatches(full, ref, search2);
    std::map<int, int> degree;
    list.forHalf(0, static_cast<std::int64_t>(pos.size() / 3),
                 [&](std::int32_t i, std::int32_t j, double, double, double, double) {
                   REQUIRE(j > i);
                   ++degree[i];
                   ++degree[j];
                 });
    for (const auto &item : full) {
      const int i = item.first.first;
      const int j = item.first.second;
      if (j > i) {
        REQUIRE(degree[i] > 0);
      }
    }
    const rgpot::nlist::MinimumImage image(cell, periodic);
    list.forDirected(0, static_cast<std::int64_t>(pos.size() / 3),
                     [&](std::int32_t i, std::int32_t j, double dx, double dy, double dz, double) {
                       double ex = pos[3 * i] - pos[3 * j];
                       double ey = pos[3 * i + 1] - pos[3 * j + 1];
                       double ez = pos[3 * i + 2] - pos[3 * j + 2];
                       image.fold(ex, ey, ez);
                       const double scale = std::max(1.0, ex * ex + ey * ey + ez * ez);
                       REQUIRE(std::abs(dx - ex) <= 1e-8 * std::sqrt(scale));
                       REQUIRE(std::abs(dy - ey) <= 1e-8 * std::sqrt(scale));
                       REQUIRE(std::abs(dz - ez) <= 1e-8 * std::sqrt(scale));
                     });

    rgpot::LJPot pot(rgpot::LJConfig{.cutoff = cutoff});
    const std::vector<int> types(pos.size() / 3, 1);
    std::vector<double> forces(pos.size(), 0.0);
    rgpot::ForceOut out{};
    out.F = forces.data();
    pot.forceImpl(rgpot::ForceInput{.nAtoms = pos.size() / 3,
                                   .pos = pos.data(),
                                   .atmnrs = types.data(),
                                   .box = cell},
                  &out);
    double energy = 0.0;
    std::vector<double> ref_f(pos.size(), 0.0);
    const double cut2 = cutoff * cutoff;
    const double shift_r2 = cutoff * cutoff;
    const double shift_sr2 = 1.0 / shift_r2;
    const double shift_a = shift_sr2 * shift_sr2 * shift_sr2;
    const double shift = 4.0 * shift_a * (shift_a - 1.0);
    for (std::size_t i = 0; i < pos.size() / 3; ++i) {
      for (std::size_t j = i + 1; j < pos.size() / 3; ++j) {
        double dx = pos[3 * i] - pos[3 * j];
        double dy = pos[3 * i + 1] - pos[3 * j + 1];
        double dz = pos[3 * i + 2] - pos[3 * j + 2];
        image.fold(dx, dy, dz);
        const double r2 = dx * dx + dy * dy + dz * dz;
        if (!(r2 > 0.0) || r2 > cut2) {
          continue;
        }
        const double inv_r2 = 1.0 / r2;
        const double sr2 = inv_r2;
        const double a = sr2 * sr2 * sr2;
        const double b = 4.0 * a;
        const double pair_e = b * (a - 1.0) - shift;
        const double fscale = (6.0 * b * inv_r2) * (2.0 * a - 1.0);
        energy += pair_e;
        ref_f[3 * i] += fscale * dx;
        ref_f[3 * i + 1] += fscale * dy;
        ref_f[3 * i + 2] += fscale * dz;
        ref_f[3 * j] -= fscale * dx;
        ref_f[3 * j + 1] -= fscale * dy;
        ref_f[3 * j + 2] -= fscale * dz;
      }
    }
    REQUIRE(std::abs(out.energy - energy) <= 1e-8 * std::max(1.0, std::abs(energy)));
    for (std::size_t k = 0; k < forces.size(); ++k) {
      REQUIRE(std::abs(forces[k] - ref_f[k]) <= 1e-6 * std::max(1.0, std::abs(ref_f[k])));
    }

    std::vector<double> moved = pos;
    for (std::size_t a = 0; a < moved.size() / 3; ++a) {
      moved[3 * a] += 0.2;
    }
    std::fill(forces.begin(), forces.end(), 0.0);
    out.energy = 0.0;
    pot.forceImpl(rgpot::ForceInput{.nAtoms = moved.size() / 3,
                                   .pos = moved.data(),
                                   .atmnrs = types.data(),
                                   .box = cell},
                  &out);
    REQUIRE(pot.listReused());
    double moved_e = 0.0;
    std::vector<double> moved_f(moved.size(), 0.0);
    for (std::size_t i = 0; i < moved.size() / 3; ++i) {
      for (std::size_t j = i + 1; j < moved.size() / 3; ++j) {
        double dx = moved[3 * i] - moved[3 * j];
        double dy = moved[3 * i + 1] - moved[3 * j + 1];
        double dz = moved[3 * i + 2] - moved[3 * j + 2];
        image.fold(dx, dy, dz);
        const double r2 = dx * dx + dy * dy + dz * dz;
        if (!(r2 > 0.0) || r2 > cut2) {
          continue;
        }
        const double inv_r2 = 1.0 / r2;
        const double a = inv_r2 * inv_r2 * inv_r2;
        const double b = 4.0 * a;
        moved_e += b * (a - 1.0) - shift;
        const double fscale = (6.0 * b * inv_r2) * (2.0 * a - 1.0);
        moved_f[3 * i] += fscale * dx;
        moved_f[3 * i + 1] += fscale * dy;
        moved_f[3 * i + 2] += fscale * dz;
        moved_f[3 * j] -= fscale * dx;
        moved_f[3 * j + 1] -= fscale * dy;
        moved_f[3 * j + 2] -= fscale * dz;
      }
    }
    REQUIRE(std::abs(out.energy - moved_e) <= 1e-8 * std::max(1.0, std::abs(moved_e)));
    for (std::size_t k = 0; k < forces.size(); ++k) {
      REQUIRE(std::abs(forces[k] - moved_f[k]) <= 1e-6 * std::max(1.0, std::abs(moved_f[k])));
    }
  }
}
