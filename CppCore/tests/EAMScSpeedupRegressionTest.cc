// MIT License
// Copyright 2023--present rgpot developers

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
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
