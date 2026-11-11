// MIT License
// Copyright 2023--present rgpot developers
//
// The short-cutoff walk may use a cell shift in place of the round
// minimum image, and Morse may evaluate exp on a batch. Both are the
// same operations as the scalar walk, so the energy and the forces match
// it to the last bit. Stream order is not the scalar pair order; those
// bits are pinned to the values recorded before the change.

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include <catch2/catch_all.hpp>

#include "rgpot/LennardJones/LJClusterPot.hpp"
#include "rgpot/LennardJones/LJPot.hpp"
#include "rgpot/Morse/MorsePot.hpp"
#include "rgpot/simd/exp.hpp"
#include "rgpot/stress.hpp"

namespace {

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

double round_visit(double t) {
  if (!std::isfinite(t)) {
    return 0.0;
  }
  const double shifted = t + std::copysign(0.5, t);
  if (!std::isfinite(shifted) || shifted >= 9223372036854775808.0 ||
      shifted <= -9223372036854775808.0) {
    return 0.0;
  }
  return static_cast<double>(static_cast<long long>(shifted));
}

void fold_round(double &dx, double &dy, double &dz, const double *box, bool periodic) {
  if (!periodic) {
    return;
  }
  const double Lx = box[0];
  const double Ly = box[4];
  const double Lz = box[8];
  if (Lx > 0.0) {
    dx -= Lx * round_visit(dx / Lx);
  }
  if (Ly > 0.0) {
    dy -= Ly * round_visit(dy / Ly);
  }
  if (Lz > 0.0) {
    dz -= Lz * round_visit(dz / Lz);
  }
}

std::uint64_t bits(double x) {
  std::uint64_t u = 0;
  std::memcpy(&u, &x, sizeof(u));
  return u;
}

std::uint64_t hash_state(double energy, const std::vector<double> &force) {
  std::uint64_t h = bits(energy);
  for (double v : force) {
    h ^= bits(v) + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
  }
  return h;
}

std::vector<double> fcc(int n, double a) {
  const double b[4][3] = {{0, 0, 0}, {0, 0.5, 0.5}, {0.5, 0, 0.5}, {0.5, 0.5, 0}};
  std::vector<double> pos;
  for (int ix = 0; ix < n; ++ix) {
    for (int iy = 0; iy < n; ++iy) {
      for (int iz = 0; iz < n; ++iz) {
        for (const auto &p : b) {
          pos.push_back((ix + p[0]) * a);
          pos.push_back((iy + p[1]) * a);
          pos.push_back((iz + p[2]) * a);
        }
      }
    }
  }
  return pos;
}

void nudge(std::vector<double> &pos) {
  for (std::size_t i = 0; i < pos.size(); i += 3) {
    pos[i] += 0.11 * std::sin(0.3 * static_cast<double>(i));
    pos[i + 1] += 0.07 * std::cos(0.2 * static_cast<double>(i));
    pos[i + 2] += 0.05 * std::sin(0.1 * static_cast<double>(i));
  }
}

struct Oracle {
  double energy = 0;
  std::vector<double> force;
  double stress[9]{};
};

void lj_accumulate(std::int32_t i, std::int32_t j, double dx, double dy, double dz, double r2,
                   double psi, double u0, double shift, std::vector<double> &force, double &energy,
                   double strain[6]) {
  const double inv = 1.0 / r2;
  const double sr2 = (psi * psi) * inv;
  const double a = (sr2 * sr2) * sr2;
  const double b = (4.0 * u0) * a;
  energy += (b * (a - 1.0)) - shift;
  const double scale = ((6.0 * b) * inv) * ((2.0 * a) - 1.0);
  const double fx = scale * dx;
  const double fy = scale * dy;
  const double fz = scale * dz;
  force[3 * static_cast<std::size_t>(i)] += fx;
  force[3 * static_cast<std::size_t>(i) + 1] += fy;
  force[3 * static_cast<std::size_t>(i) + 2] += fz;
  force[3 * static_cast<std::size_t>(j)] -= fx;
  force[3 * static_cast<std::size_t>(j) + 1] -= fy;
  force[3 * static_cast<std::size_t>(j) + 2] -= fz;
  rgpot::accumulatePairStrain(strain, scale, dx, dy, dz);
}

Oracle lj_direct(const std::vector<double> &pos, const double *box, bool periodic, double cutoff,
                 double psi, double u0) {
  const double a6 = std::pow(psi / cutoff, 6.0);
  const double shift = 4.0 * u0 * a6 * (a6 - 1.0);
  const double cutoff2 = cutoff * cutoff;
  const std::size_t n = pos.size() / 3;
  Oracle out;
  out.force.assign(pos.size(), 0.0);
  double strain[6] = {};
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = i + 1; j < n; ++j) {
      double dx = pos[3 * i] - pos[3 * j];
      double dy = pos[3 * i + 1] - pos[3 * j + 1];
      double dz = pos[3 * i + 2] - pos[3 * j + 2];
      fold_round(dx, dy, dz, box, periodic);
      const double r2 = dx * dx + dy * dy + dz * dz;
      if (!(r2 > 0.0) || r2 > cutoff2) {
        continue;
      }
      lj_accumulate(static_cast<std::int32_t>(i), static_cast<std::int32_t>(j), dx, dy, dz, r2,
                    psi, u0, shift, out.force, out.energy, strain);
    }
  }
  const double volume = rgpot::cellVolume(box);
  rgpot::ForceOut pub{nullptr, out.energy, 0.0, {}, 0};
  rgpot::publishCauchyStress(&pub, strain, volume);
  std::memcpy(out.stress, pub.stress, sizeof(out.stress));
  return out;
}

Oracle morse_direct(const std::vector<double> &pos, const double *box, const rgpot::MorseConfig &cfg) {
  const double cutoff2 = cfg.cutoff * cfg.cutoff;
  const double dcut = 1.0 - rgpot::simd::exp_fixed(-cfg.a * (cfg.cutoff - cfg.re));
  const double shift = cfg.De * dcut * dcut - cfg.De;
  const double two = 2.0 * cfg.De * cfg.a;
  const std::size_t n = pos.size() / 3;
  Oracle out;
  out.force.assign(pos.size(), 0.0);
  double strain[6] = {};
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = i + 1; j < n; ++j) {
      double dx = pos[3 * i] - pos[3 * j];
      double dy = pos[3 * i + 1] - pos[3 * j + 1];
      double dz = pos[3 * i + 2] - pos[3 * j + 2];
      fold_round(dx, dy, dz, box, true);
      const double r2 = dx * dx + dy * dy + dz * dz;
      if (!(r2 > 0.0) || r2 > cutoff2) {
        continue;
      }
      const double r = std::sqrt(r2);
      const double d = 1.0 - rgpot::simd::exp_fixed(-cfg.a * (r - cfg.re));
      const double bare = cfg.De * d * d - cfg.De;
      const double pair_e = bare - shift;
      const double fscale = two * d * (d - 1.0) / r;
      out.energy += pair_e;
      const double fx = fscale * dx;
      const double fy = fscale * dy;
      const double fz = fscale * dz;
      out.force[3 * i] += fx;
      out.force[3 * i + 1] += fy;
      out.force[3 * i + 2] += fz;
      out.force[3 * j] -= fx;
      out.force[3 * j + 1] -= fy;
      out.force[3 * j + 2] -= fz;
      rgpot::accumulatePairStrain(strain, fscale, dx, dy, dz);
    }
  }
  rgpot::ForceOut pub{nullptr, out.energy, 0.0, {}, 0};
  rgpot::publishCauchyStress(&pub, strain, rgpot::cellVolume(box));
  std::memcpy(out.stress, pub.stress, sizeof(out.stress));
  return out;
}

void require_bits(double energy, const std::vector<double> &force, const double stress[9],
                  const Oracle &ref) {
  REQUIRE(std::memcmp(&energy, &ref.energy, sizeof(double)) == 0);
  REQUIRE(force.size() == ref.force.size());
  REQUIRE(std::memcmp(force.data(), ref.force.data(), force.size() * sizeof(double)) == 0);
  REQUIRE(std::memcmp(stress, ref.stress, 9 * sizeof(double)) == 0);
}

} // namespace

TEST_CASE("stored LJ matches a direct pair sum to the last bit", "[nlist][bits][lj]") {
  setThreads("");
  auto pos = fcc(3, 4.05);
  nudge(pos);
  const double L = 3 * 4.05;
  const double box[9] = {L, 0, 0, 0, L, 0, 0, 0, L};
  const std::size_t n = pos.size() / 3;
  const std::vector<int> z(n, 13);
  rgpot::LJConfig cfg;
  cfg.cutoff = 6.0;
  cfg.psi = 1.0;
  cfg.u0 = 1.0;
  rgpot::LJPot pot(cfg);
  pot.setNeighbourMode(rgpot::nlist::Mode::Stored);
  std::vector<double> force(pos.size());
  rgpot::ForceInput in{n, pos.data(), z.data(), box};
  rgpot::ForceOut out{force.data(), 0, 0, {}, 0};
  pot.forceImpl(in, &out);
  require_bits(out.energy, force, out.stress, lj_direct(pos, box, true, 6.0, 1.0, 1.0));

  // Inside half the skin the stored shifts stay the minimum image.
  pos[0] += 0.2;
  pot.forceImpl(in, &out);
  REQUIRE(pot.listReused());
  require_bits(out.energy, force, out.stress, lj_direct(pos, box, true, 6.0, 1.0, 1.0));
}

TEST_CASE("stored Morse matches a direct pair sum to the last bit", "[nlist][bits][morse]") {
  setThreads("");
  auto pos = fcc(3, 4.05);
  nudge(pos);
  const double L = 3 * 4.05;
  const double box[9] = {L, 0, 0, 0, L, 0, 0, 0, L};
  const std::size_t n = pos.size() / 3;
  const std::vector<int> z(n, 78);
  rgpot::MorseConfig cfg;
  cfg.cutoff = 5.0;
  rgpot::MorsePot pot(cfg);
  pot.setNeighbourMode(rgpot::nlist::Mode::Stored);
  std::vector<double> force(pos.size());
  rgpot::ForceInput in{n, pos.data(), z.data(), box};
  rgpot::ForceOut out{force.data(), 0, 0, {}, 0};
  pot.forceImpl(in, &out);
  require_bits(out.energy, force, out.stress, morse_direct(pos, box, cfg));
}

TEST_CASE("open LJ matches a direct pair sum to the last bit", "[nlist][bits][lj]") {
  setThreads("");
  auto pos = fcc(3, 4.05);
  nudge(pos);
  const double L = 3 * 4.05;
  const double box[9] = {L, 0, 0, 0, L, 0, 0, 0, L};
  const std::size_t n = pos.size() / 3;
  const std::vector<int> z(n, 13);
  rgpot::LJClusterConfig cfg;
  cfg.cutoff = 6.0;
  cfg.psi = 1.0;
  cfg.u0 = 1.0;
  rgpot::LJClusterPot pot(cfg);
  pot.setNeighbourMode(rgpot::nlist::Mode::Stored);
  std::vector<double> force(pos.size());
  rgpot::ForceInput in{n, pos.data(), z.data(), box};
  rgpot::ForceOut out{force.data(), 0, 0, {}, 0};
  pot.forceImpl(in, &out);
  // The open cluster does not publish stress. The pair sum is the check.
  const Oracle ref = lj_direct(pos, box, false, 6.0, 1.0, 1.0);
  REQUIRE(std::memcmp(&out.energy, &ref.energy, sizeof(double)) == 0);
  REQUIRE(std::memcmp(force.data(), ref.force.data(), force.size() * sizeof(double)) == 0);
}

TEST_CASE("a short grid keeps the folded LJ bits", "[nlist][bits][lj]") {
  setThreads("");
  auto pos = fcc(3, 4.05);
  for (double &v : pos) {
    v = std::fmod(std::fabs(v), 8.0);
  }
  const double box[9] = {8, 0, 0, 0, 8, 0, 0, 0, 8};
  const std::size_t n = pos.size() / 3;
  const std::vector<int> z(n, 13);
  rgpot::LJConfig cfg;
  cfg.cutoff = 3.5;
  cfg.psi = 1.0;
  cfg.u0 = 1.0;
  rgpot::LJPot pot(cfg);
  pot.setNeighbourMode(rgpot::nlist::Mode::Stored);
  std::vector<double> force(pos.size());
  rgpot::ForceInput in{n, pos.data(), z.data(), box};
  rgpot::ForceOut out{force.data(), 0, 0, {}, 0};
  pot.forceImpl(in, &out);
  require_bits(out.energy, force, out.stress, lj_direct(pos, box, true, 3.5, 1.0, 1.0));
}

TEST_CASE("stream bits stay on the recorded short-cutoff values", "[nlist][bits]") {
  setThreads("");
  auto base = fcc(3, 4.05);
  auto moved = base;
  nudge(moved);
  const double L = 3 * 4.05;
  const double box[9] = {L, 0, 0, 0, L, 0, 0, 0, L};
  const std::size_t n = base.size() / 3;
  const std::vector<int> z(n, 13);
  std::vector<double> force(base.size());

  auto lj_hash = [&](rgpot::nlist::Mode mode, const std::vector<double> &pos, double cut) {
    rgpot::LJConfig cfg;
    cfg.cutoff = cut;
    cfg.psi = 1.0;
    cfg.u0 = 1.0;
    rgpot::LJPot pot(cfg);
    pot.setNeighbourMode(mode);
    rgpot::ForceInput in{n, pos.data(), z.data(), box};
    rgpot::ForceOut out{force.data(), 0, 0, {}, 0};
    pot.forceImpl(in, &out);
    return hash_state(out.energy, force);
  };
  // Recorded from the walk before the cell-shift list. Stream and stored
  // disagree with each other in the last bits because the pair order differs.
  REQUIRE(lj_hash(rgpot::nlist::Mode::Stored, base, 2.5) == 0x63329afc5f0492a6ull);
  REQUIRE(lj_hash(rgpot::nlist::Mode::Stream, base, 2.5) == 0x63329afc5f0492a6ull);
  REQUIRE(lj_hash(rgpot::nlist::Mode::Stored, base, 6.0) == 0x2a9d31cb6a5f3d94ull);
  REQUIRE(lj_hash(rgpot::nlist::Mode::Stream, base, 6.0) == 0x948a5509ecdfbb44ull);
  REQUIRE(lj_hash(rgpot::nlist::Mode::Stored, moved, 6.0) == 0x8bf94b4805d31a46ull);
  REQUIRE(lj_hash(rgpot::nlist::Mode::Stream, moved, 6.0) == 0x067667410a5428c0ull);

  rgpot::MorseConfig mcfg;
  mcfg.cutoff = 5.0;
  rgpot::MorsePot stored(mcfg);
  stored.setNeighbourMode(rgpot::nlist::Mode::Stored);
  rgpot::ForceInput in{n, moved.data(), z.data(), box};
  rgpot::ForceOut out{force.data(), 0, 0, {}, 0};
  stored.forceImpl(in, &out);
  REQUIRE(hash_state(out.energy, force) == 0x12a050fb80714877ull);
  rgpot::MorsePot stream(mcfg);
  stream.setNeighbourMode(rgpot::nlist::Mode::Stream);
  stream.forceImpl(in, &out);
  REQUIRE(hash_state(out.energy, force) == 0xd3d685cd7993bce2ull);
}
