// MIT License
// Copyright 2023--present rgpot developers

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <string>
#include <vector>

#include <catch2/catch_all.hpp>

#include "rgpot/EAMTable/EAMTablePot.hpp"
#include "rgpot/SuttonChen/SuttonChenPot.hpp"
#include "rgpot/SuttonChen/SuttonChenScales.hpp"

namespace {

std::uint64_t fnv(std::uint64_t h, const void *data, std::size_t len) {
  const auto *b = static_cast<const unsigned char *>(data);
  for (std::size_t i = 0; i < len; ++i) {
    h ^= b[i];
    h *= 1099511628211ULL;
  }
  return h;
}

std::vector<double> fcc(int nx, int ny, int nz, double a, double *box) {
  const double basis[4][3] = {
      {0.0, 0.0, 0.0},
      {0.0, 0.5, 0.5},
      {0.5, 0.0, 0.5},
      {0.5, 0.5, 0.0},
  };
  std::vector<double> pos;
  pos.reserve(static_cast<std::size_t>(nx * ny * nz * 12));
  for (int ix = 0; ix < nx; ++ix) {
    for (int iy = 0; iy < ny; ++iy) {
      for (int iz = 0; iz < nz; ++iz) {
        for (const auto &b : basis) {
          pos.push_back((ix + b[0]) * a);
          pos.push_back((iy + b[1]) * a);
          pos.push_back((iz + b[2]) * a);
        }
      }
    }
  }
  std::fill(box, box + 9, 0.0);
  box[0] = nx * a;
  box[4] = ny * a;
  box[8] = nz * a;
  return pos;
}

std::vector<double> pointFile(const std::string &path) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("cannot open " + path);
  }
  double head[5];
  for (double &h : head) {
    in >> h;
  }
  std::vector<double> pos;
  double x = 0.0;
  while (in >> x) {
    pos.push_back(x);
  }
  return pos;
}

std::vector<std::vector<double>> frames(const std::vector<double> &base, double sigma,
                                        std::uint64_t seed, int nframes) {
  std::vector<std::vector<double>> out(static_cast<std::size_t>(nframes), base);
  std::uint64_t state = seed ? seed : 1;
  auto next_u01 = [&state]() {
    state = state * 6364136223846793005ULL + 1ULL;
    const std::uint64_t bits = (state >> 11) | 0x3ff0000000000000ULL;
    double v = 0.0;
    std::memcpy(&v, &bits, sizeof(v));
    return v - 1.0;
  };
  for (auto &f : out) {
    for (double &x : f) {
      const double u1 = std::max(next_u01(), 1e-12);
      const double u2 = next_u01();
      const double mag = sigma * std::sqrt(-2.0 * std::log(u1));
      x += mag * std::cos(6.283185307179586 * u2);
    }
  }
  return out;
}

struct Eval {
  double energy = 0.0;
  std::vector<double> forces;
};

std::uint64_t evalBits(const Eval &e) {
  std::uint64_t bits = 1469598103934665603ULL;
  bits = fnv(bits, &e.energy, sizeof(e.energy));
  return fnv(bits, e.forces.data(), e.forces.size() * sizeof(double));
}

/// Energy and forces of the Sutton-Chen sums over every pair with r <= cutoff,
/// by the minimum image when periodic.
Eval scReference(const rgpot::SuttonChenConfig &cfg, const std::vector<double> &pos,
                 const double *box) {
  const std::size_t n = pos.size() / 3;
  const double cutoff2 = cfg.cutoff * cfg.cutoff;
  Eval r;
  r.forces.assign(pos.size(), 0.0);
  struct Pair {
    std::size_t i, j;
    double d[3];
    double r2;
  };
  std::vector<Pair> pairs;
  std::vector<double> rho(n, 0.0);
  double epair = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = i + 1; j < n; ++j) {
      Pair p{i, j, {}, 0.0};
      for (int k = 0; k < 3; ++k) {
        double d = pos[3 * i + k] - pos[3 * j + k];
        if (cfg.periodic) {
          d -= box[4 * k] * std::round(d / box[4 * k]);
        }
        p.d[k] = d;
        p.r2 += d * d;
      }
      if (p.r2 > cutoff2) {
        continue;
      }
      const double ar = cfg.a / std::sqrt(p.r2);
      rho[i] += std::pow(ar, cfg.m);
      rho[j] += std::pow(ar, cfg.m);
      epair += cfg.epsilon * std::pow(ar, cfg.n);
      pairs.push_back(p);
    }
  }
  double eemb = 0.0;
  std::vector<double> weight(n, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    if (rho[i] > 0.0) {
      eemb -= cfg.epsilon * cfg.c * std::sqrt(rho[i]);
      weight[i] = cfg.epsilon * cfg.c / (2.0 * std::sqrt(rho[i]));
    }
  }
  for (const Pair &p : pairs) {
    const double ar = cfg.a / std::sqrt(p.r2);
    const double fscale = (cfg.epsilon * cfg.n * std::pow(ar, cfg.n) -
                           cfg.m * std::pow(ar, cfg.m) * (weight[p.i] + weight[p.j])) /
                          p.r2;
    for (int k = 0; k < 3; ++k) {
      r.forces[3 * p.i + k] += fscale * p.d[k];
      r.forces[3 * p.j + k] -= fscale * p.d[k];
    }
  }
  r.energy = epair + eemb;
  return r;
}

} // namespace

TEST_CASE("EAM and SC benchmark geometries keep the stored base fingerprints",
          "[EAMTable][SuttonChen][regression]") {
  constexpr int kCalls = 8;

  SECTION("EAM 4000 Cu cell") {
    double box[9] = {};
    const std::vector<double> pos = fcc(10, 10, 10, 3.615, box);
    const int n = static_cast<int>(pos.size() / 3);
    std::vector<int> z(static_cast<std::size_t>(n), 29);
    rgpot::EAMTablePot pot({.path = "CppCore/tests/data/eam_table/Cu_u3.eam",
                            .format = rgpot::EAMTableFormat::Funcfl,
                            .periodic = true,
                            .stress = false,
                            .skin = 1.0});
    std::vector<double> f(pos.size(), 0.0);
    rgpot::ForceOut out{};
    out.F = f.data();
    const rgpot::ForceInput in{static_cast<std::size_t>(n), pos.data(), z.data(), box};
    pot.forceImpl(in, &out);
    pot.forceImpl(in, &out);
    std::uint64_t bits = 1469598103934665603ULL;
    for (int k = 0; k < kCalls; ++k) {
      pot.forceImpl(in, &out);
      bits = fnv(bits, &out.energy, sizeof(out.energy));
      bits = fnv(bits, f.data(), f.size() * sizeof(double));
    }
    REQUIRE(std::abs(out.energy - (-14160.0000090987)) < 1e-9);
    REQUIRE(bits == 0x3ee4b21bc23d1a83ULL);
  }

  SECTION("SC Pt/Au cluster panel") {
    const std::vector<double> pt38 = pointFile("CppCore/tests/data/sc108/pt38");
    const std::vector<double> pt75 = pointFile("CppCore/tests/data/sc108/pt75");
    const double scale =
        rgpot::sutton_chen_gold().a / rgpot::sutton_chen_platinum().a;
    std::vector<double> au38 = pt38;
    std::vector<double> au75 = pt75;
    for (double &x : au38) {
      x *= scale;
    }
    for (double &x : au75) {
      x *= scale;
    }
    struct Case {
      const char *name;
      const std::vector<double> *pos;
      rgpot_sc_config cfg;
      double expected_energy;
      std::uint64_t expected_bits;
    };
    const Case cases[] = {
        {"pt38", &pt38, {1.98330e-2, 34.408, 3.92, 10, 8, 40.0, 0},
         132574625.005886, 0x6c2f775a660b05d1ULL},
        {"au38", &au38, {1.27930e-2, 34.408, 4.08, 10, 8, 40.0, 0},
         84663998.4567799, 0x061e38ddc1ab66caULL},
        {"au75", &au75, {1.27930e-2, 34.408, 4.08, 10, 8, 40.0, 0},
         166200138.331864, 0xc4101c4b756c78edULL},
    };
    for (const Case &c : cases) {
      INFO(c.name);
      const auto fr = frames(*c.pos, 0.02, 7, 16);
      std::vector<double> f(c.pos->size(), 0.0);
      double e = 0.0;
      std::uint64_t bits = 1469598103934665603ULL;
      for (int k = 0; k < kCalls; ++k) {
        const auto &x =
            fr[static_cast<std::size_t>(k % static_cast<int>(fr.size()))];
        REQUIRE(rgpot_sutton_chen_energy_forces(
                    &c.cfg, static_cast<int32_t>(x.size() / 3), x.data(), nullptr, &e,
                    f.data()) == 0);
      }
      bits = fnv(bits, &e, sizeof(e));
      bits = fnv(bits, f.data(), f.size() * sizeof(double));
      REQUIRE(std::abs(e - c.expected_energy) < 1e-6);
      REQUIRE(bits == c.expected_bits);
    }
  }
}

TEST_CASE("Sutton-Chen keeps pairs at exactly the cutoff on every path",
          "[SuttonChen][regression]") {
  // A simple cubic grid of spacing 2 puts every nearest pair at r^2 = 4
  // exactly, which is cutoff^2 for cutoff 2, and every second pair past it.
  // 256 atoms is the smallest count that takes the gathered pair path.
  constexpr int kNx = 8;
  constexpr int kNy = 8;
  constexpr int kNz = 4;
  constexpr double kSpacing = 2.0;
  std::vector<double> pos;
  for (int ix = 0; ix < kNx; ++ix) {
    for (int iy = 0; iy < kNy; ++iy) {
      for (int iz = 0; iz < kNz; ++iz) {
        pos.push_back(ix * kSpacing);
        pos.push_back(iy * kSpacing);
        pos.push_back(iz * kSpacing);
      }
    }
  }
  const std::size_t n = pos.size() / 3;
  REQUIRE(n == 256);
  double box[9] = {};
  box[0] = kNx * kSpacing;
  box[4] = kNy * kSpacing;
  box[8] = kNz * kSpacing;

  for (const bool periodic : {false, true}) {
    INFO("periodic " << periodic);
    rgpot::SuttonChenConfig cfg;
    cfg.epsilon = 0.01;
    cfg.c = 30.0;
    cfg.a = 2.5;
    cfg.cutoff = kSpacing;
    cfg.periodic = periodic;
    const Eval ref = scReference(cfg, pos, box);
    REQUIRE(ref.energy != 0.0);

    rgpot::SuttonChenPot pot(cfg);
    Eval got;
    got.forces.assign(pos.size(), 0.0);
    rgpot::ForceOut out{};
    out.F = got.forces.data();
    const rgpot::ForceInput in{n, pos.data(), nullptr, periodic ? box : nullptr};
    pot.forceImpl(in, &out);
    pot.forceImpl(in, &out);

    REQUIRE(std::abs(out.energy - ref.energy) < 1e-12 * std::abs(ref.energy));
    double scale = 0.0;
    for (const double f : ref.forces) {
      scale = std::max(scale, std::abs(f));
    }
    // The open grid has surface forces; the periodic one sums to zero.
    REQUIRE((periodic || scale > 0.0));
    for (std::size_t k = 0; k < pos.size(); ++k) {
      REQUIRE(std::abs(got.forces[k] - ref.forces[k]) < 1e-12 * std::max(scale, 1.0));
    }
    if (!periodic) {
      // The energy and forces this potential gave before the pair gather.
      Eval kept;
      kept.energy = out.energy;
      kept.forces = got.forces;
      INFO("energy " << std::setprecision(15) << out.energy << " bits " << std::hex
                     << evalBits(kept));
      REQUIRE(std::abs(out.energy - (-358.277063164732)) < 1e-9);
      REQUIRE(evalBits(kept) == 0xabb545887837750eULL);
    }
  }
}

