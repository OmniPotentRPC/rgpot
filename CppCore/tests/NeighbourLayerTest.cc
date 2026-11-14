// MIT License
// Copyright 2023--present rgpot developers

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <random>
#include <utility>
#include <vector>

#include <catch2/catch_all.hpp>

#include "rgpot/EAM/EAMPot.hpp"
#include "rgpot/LennardJones/LJPot.hpp"
#include "rgpot/StillingerWeber/SWSi.hpp"
#include "rgpot/SuttonChen/SuttonChenPot.hpp"
#include "rgpot/Tersoff/TersoffSiC.hpp"
#include "rgpot/nlist/MinimumImage.hpp"
#include "rgpot/nlist/Neighbours.hpp"

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

bool close(double a, double b) {
  const double scale = std::max(1.0, std::max(std::abs(a), std::abs(b)));
  return std::abs(a - b) <= 1e-9 * scale;
}

std::uint64_t bits(double v) {
  std::uint64_t out = 0;
  std::memcpy(&out, &v, sizeof(out));
  return out;
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

std::vector<double> diamond(int n, double a) {
  const double b[8][3] = {{0, 0, 0},         {0.5, 0.5, 0},     {0.5, 0, 0.5},
                          {0, 0.5, 0.5},     {0.25, 0.25, 0.25},{0.75, 0.75, 0.25},
                          {0.75, 0.25, 0.75},{0.25, 0.75, 0.75}};
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

template <typename Pot>
void checkModes(Pot &stored, Pot &stream, const std::vector<double> &pos,
                const std::vector<int> &z, double length) {
  rgpot::types::AtomMatrix xyz(z.size(), 3);
  for (std::size_t i = 0; i < z.size(); ++i) {
    xyz(i, 0) = pos[3 * i];
    xyz(i, 1) = pos[3 * i + 1];
    xyz(i, 2) = pos[3 * i + 2];
  }
  const std::array<std::array<double, 3>, 3> box{{
      {length, 0, 0},
      {0, length, 0},
      {0, 0, length},
  }};
  const auto a = stored(xyz, z, box);
  const auto b = stream(xyz, z, box);
  const auto c = stream(xyz, z, box);
  REQUIRE(close(std::get<0>(a), std::get<0>(b)));
  REQUIRE(bits(std::get<0>(b)) == bits(std::get<0>(c)));
  const auto &fa = std::get<1>(a);
  const auto &fb = std::get<1>(b);
  const auto &fc = std::get<1>(c);
  for (std::size_t i = 0; i < z.size() * 3; ++i) {
    REQUIRE(close(fa.data()[i], fb.data()[i]));
    REQUIRE(bits(fb.data()[i]) == bits(fc.data()[i]));
  }
}

} // namespace

TEST_CASE("neighbour modes agree for the classical potentials", "[nlist]") {
  setThreads("");
  const auto metal = fcc(3, 4.05);
  const std::vector<int> al(metal.size() / 3, 13);
  rgpot::EAMPot eam_s;
  rgpot::EAMPot eam_w;
  eam_s.setNeighbourMode(rgpot::nlist::Mode::Stored);
  eam_w.setNeighbourMode(rgpot::nlist::Mode::Stream);
  checkModes(eam_s, eam_w, metal, al, 3 * 4.05);

  rgpot::LJConfig ljcfg;
  ljcfg.cutoff = 2.5;
  ljcfg.psi = 1.0;
  rgpot::LJPot lj_s(ljcfg);
  rgpot::LJPot lj_w(ljcfg);
  lj_s.setNeighbourMode(rgpot::nlist::Mode::Stored);
  lj_w.setNeighbourMode(rgpot::nlist::Mode::Stream);
  const double a_lj = std::pow(2.0, 1.0 / 6.0) * std::sqrt(2.0);
  const auto ljpos = fcc(3, a_lj);
  const std::vector<int> ar(ljpos.size() / 3, 18);
  checkModes(lj_s, lj_w, ljpos, ar, 3 * a_lj);

  rgpot::SuttonChenConfig sc;
  sc.cutoff = 3.0;
  sc.periodic = true;
  rgpot::SuttonChenPot sc_s(sc);
  rgpot::SuttonChenPot sc_w(sc);
  sc_s.setNeighbourMode(rgpot::nlist::Mode::Stored);
  sc_w.setNeighbourMode(rgpot::nlist::Mode::Stream);
  const auto scpos = fcc(3, std::sqrt(2.0));
  const std::vector<int> pt(scpos.size() / 3, 78);
  checkModes(sc_s, sc_w, scpos, pt, 3 * std::sqrt(2.0));

  const auto si = diamond(2, 5.431);
  const std::vector<int> zsi(si.size() / 3, 14);
  rgpot::StillingerWeberSi sw_s;
  rgpot::StillingerWeberSi sw_w;
  sw_s.setNeighbourMode(rgpot::nlist::Mode::Stored);
  sw_w.setNeighbourMode(rgpot::nlist::Mode::Stream);
  checkModes(sw_s, sw_w, si, zsi, 2 * 5.431);

  rgpot::TersoffSiC te_s;
  rgpot::TersoffSiC te_w;
  te_s.setNeighbourMode(rgpot::nlist::Mode::Stored);
  te_w.setNeighbourMode(rgpot::nlist::Mode::Stream);
  checkModes(te_s, te_w, si, zsi, 2 * 5.431);
}

template <typename Pot>
void checkThreads(Pot &&serial_pot, Pot &&parallel_pot,
                  const std::vector<double> &pos, const std::vector<int> &z,
                  double length) {
  rgpot::types::AtomMatrix xyz(z.size(), 3);
  for (std::size_t i = 0; i < z.size(); ++i) {
    xyz(i, 0) = pos[3 * i];
    xyz(i, 1) = pos[3 * i + 1];
    xyz(i, 2) = pos[3 * i + 2];
  }
  const std::array<std::array<double, 3>, 3> box{{
      {length, 0, 0},
      {0, length, 0},
      {0, 0, length},
  }};
  setThreads("");
  const auto serial = serial_pot(xyz, z, box);
  setThreads("2");
  const auto a = parallel_pot(xyz, z, box);
  const auto b = parallel_pot(xyz, z, box);
  REQUIRE(bits(std::get<0>(a)) == bits(std::get<0>(b)));
  REQUIRE(close(std::get<0>(serial), std::get<0>(a)));
  const auto &fs = std::get<1>(serial);
  const auto &fa = std::get<1>(a);
  const auto &fb = std::get<1>(b);
  for (std::size_t i = 0; i < z.size() * 3; ++i) {
    REQUIRE(close(fs.data()[i], fa.data()[i]));
    REQUIRE(bits(fa.data()[i]) == bits(fb.data()[i]));
  }
  setThreads("");
}

TEST_CASE("threaded classical forces repeat and stay near one thread",
          "[nlist]") {
  const auto metal = fcc(3, 4.05);
  const std::vector<int> al(metal.size() / 3, 13);
  rgpot::EAMPot eam_one;
  rgpot::EAMPot eam_two;
  eam_one.setNeighbourMode(rgpot::nlist::Mode::Stored);
  eam_two.setNeighbourMode(rgpot::nlist::Mode::Stored);
  checkThreads(std::move(eam_one), std::move(eam_two), metal, al, 3 * 4.05);
  rgpot::EAMPot eam_one_s;
  rgpot::EAMPot eam_two_s;
  eam_one_s.setNeighbourMode(rgpot::nlist::Mode::Stream);
  eam_two_s.setNeighbourMode(rgpot::nlist::Mode::Stream);
  checkThreads(std::move(eam_one_s), std::move(eam_two_s), metal, al, 3 * 4.05);

  rgpot::LJConfig ljcfg;
  ljcfg.cutoff = 2.5;
  ljcfg.psi = 1.0;
  const double a_lj = std::pow(2.0, 1.0 / 6.0) * std::sqrt(2.0);
  const auto ljpos = fcc(3, a_lj);
  const std::vector<int> ar(ljpos.size() / 3, 18);
  rgpot::LJPot lj_one(ljcfg);
  rgpot::LJPot lj_two(ljcfg);
  lj_one.setNeighbourMode(rgpot::nlist::Mode::Stored);
  lj_two.setNeighbourMode(rgpot::nlist::Mode::Stored);
  checkThreads(std::move(lj_one), std::move(lj_two), ljpos, ar, 3 * a_lj);
  rgpot::LJPot lj_one_s(ljcfg);
  rgpot::LJPot lj_two_s(ljcfg);
  lj_one_s.setNeighbourMode(rgpot::nlist::Mode::Stream);
  lj_two_s.setNeighbourMode(rgpot::nlist::Mode::Stream);
  checkThreads(std::move(lj_one_s), std::move(lj_two_s), ljpos, ar, 3 * a_lj);

  rgpot::SuttonChenConfig sc;
  sc.cutoff = 3.0;
  sc.periodic = true;
  const auto scpos = fcc(3, std::sqrt(2.0));
  const std::vector<int> pt(scpos.size() / 3, 78);
  rgpot::SuttonChenPot sc_one(sc);
  rgpot::SuttonChenPot sc_two(sc);
  sc_one.setNeighbourMode(rgpot::nlist::Mode::Stored);
  sc_two.setNeighbourMode(rgpot::nlist::Mode::Stored);
  checkThreads(std::move(sc_one), std::move(sc_two), scpos, pt,
               3 * std::sqrt(2.0));
  rgpot::SuttonChenPot sc_one_s(sc);
  rgpot::SuttonChenPot sc_two_s(sc);
  sc_one_s.setNeighbourMode(rgpot::nlist::Mode::Stream);
  sc_two_s.setNeighbourMode(rgpot::nlist::Mode::Stream);
  checkThreads(std::move(sc_one_s), std::move(sc_two_s), scpos, pt,
               3 * std::sqrt(2.0));

  const auto si = diamond(2, 5.431);
  const std::vector<int> zsi(si.size() / 3, 14);
  rgpot::StillingerWeberSi sw_one;
  rgpot::StillingerWeberSi sw_two;
  sw_one.setNeighbourMode(rgpot::nlist::Mode::Stored);
  sw_two.setNeighbourMode(rgpot::nlist::Mode::Stored);
  checkThreads(std::move(sw_one), std::move(sw_two), si, zsi, 2 * 5.431);
  rgpot::StillingerWeberSi sw_one_s;
  rgpot::StillingerWeberSi sw_two_s;
  sw_one_s.setNeighbourMode(rgpot::nlist::Mode::Stream);
  sw_two_s.setNeighbourMode(rgpot::nlist::Mode::Stream);
  checkThreads(std::move(sw_one_s), std::move(sw_two_s), si, zsi, 2 * 5.431);

  rgpot::TersoffSiC te_one;
  rgpot::TersoffSiC te_two;
  te_one.setNeighbourMode(rgpot::nlist::Mode::Stored);
  te_two.setNeighbourMode(rgpot::nlist::Mode::Stored);
  checkThreads(std::move(te_one), std::move(te_two), si, zsi, 2 * 5.431);
  rgpot::TersoffSiC te_one_s;
  rgpot::TersoffSiC te_two_s;
  te_one_s.setNeighbourMode(rgpot::nlist::Mode::Stream);
  te_two_s.setNeighbourMode(rgpot::nlist::Mode::Stream);
  checkThreads(std::move(te_one_s), std::move(te_two_s), si, zsi, 2 * 5.431);
}

namespace {

double roundVisit(double t) {
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

void foldDelta(double &dx, double &dy, double &dz, double lx, double ly, double lz, bool periodic) {
  if (!periodic) {
    return;
  }
  dx -= lx * roundVisit(dx / lx);
  dy -= ly * roundVisit(dy / ly);
  dz -= lz * roundVisit(dz / lz);
}

using PairMap = std::map<std::pair<int, int>, double>;

PairMap directPairs(const std::vector<double> &pos, double lx, double ly, double lz, double cutoff,
                    bool periodic) {
  const double cut2 = cutoff * cutoff;
  const int n = static_cast<int>(pos.size() / 3);
  PairMap out;
  for (int i = 0; i < n; ++i) {
    for (int j = i + 1; j < n; ++j) {
      double dx = pos[3 * i] - pos[3 * j];
      double dy = pos[3 * i + 1] - pos[3 * j + 1];
      double dz = pos[3 * i + 2] - pos[3 * j + 2];
      foldDelta(dx, dy, dz, lx, ly, lz, periodic);
      const double r2 = dx * dx + dy * dy + dz * dz;
      if (r2 <= cut2) {
        out[{i, j}] = r2;
      }
    }
  }
  return out;
}

std::vector<double> fccBlock(int nx, int ny, int nz, double a) {
  const double b[4][3] = {{0, 0, 0}, {0, 0.5, 0.5}, {0.5, 0, 0.5}, {0.5, 0.5, 0}};
  std::vector<double> pos;
  for (int ix = 0; ix < nx; ++ix) {
    for (int iy = 0; iy < ny; ++iy) {
      for (int iz = 0; iz < nz; ++iz) {
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

PairMap walkedPairs(rgpot::nlist::Neighbours &cells, std::size_t n, double cutoff) {
  const double cut2 = cutoff * cutoff;
  PairMap out;
  cells.forHalf(0, static_cast<std::int64_t>(n),
                [&](std::int32_t i, std::int32_t j, double, double, double, double r2) {
                  if (r2 <= cut2) {
                    const int lo = std::min(i, j);
                    const int hi = std::max(i, j);
                    out[{lo, hi}] = r2;
                  }
                });
  return out;
}

void requireSamePairs(const PairMap &got, const PairMap &ref) {
  REQUIRE(got.size() == ref.size());
  for (const auto &item : ref) {
    const auto it = got.find(item.first);
    REQUIRE(it != got.end());
    const double scale = std::max(1.0, std::abs(item.second));
    REQUIRE(std::abs(it->second - item.second) <= 1e-12 * scale);
  }
}

void loadCells(rgpot::nlist::Neighbours &cells, const std::vector<double> &pos, double lx,
               double ly, double lz, double cutoff, bool periodic, rgpot::nlist::Mode mode) {
  const double box[9] = {lx, 0, 0, 0, ly, 0, 0, 0, lz};
  rgpot::nlist::Spec spec;
  spec.cutoff = cutoff;
  spec.skin = 0;
  spec.periodic = {periodic, periodic, periodic};
  spec.image = rgpot::nlist::Image::Round;
  spec.mode = mode;
  cells.update(pos.data(), pos.size() / 3, box, spec);
}

} // namespace

TEST_CASE("a short cutoff stream matches the direct pair sum", "[nlist][stream][direct]") {
  setThreads("1");
  const double a = 4.05;
  const double cutoff = 2.5;
  auto pos = fccBlock(10, 4, 4, a);
  const double lx = 10 * a;
  const double ly = 4 * a;
  const double lz = 4 * a;
  const auto none = directPairs(pos, lx, ly, lz, cutoff, false);
  REQUIRE(none.empty());

  rgpot::nlist::Neighbours streamed;
  loadCells(streamed, pos, lx, ly, lz, cutoff, false, rgpot::nlist::Mode::Stream);
  REQUIRE(streamed.streaming());
  REQUIRE(streamed.slots() == 0);
  requireSamePairs(walkedPairs(streamed, pos.size() / 3, cutoff), none);

  // One pair inside the cutoff. The rest of the lattice stays outside it.
  pos[3] = 1.2;
  pos[4] = 0;
  pos[5] = 0;
  const auto close = directPairs(pos, lx, ly, lz, cutoff, false);
  REQUIRE(close.count({0, 1}) == 1);

  rgpot::nlist::Neighbours again;
  loadCells(again, pos, lx, ly, lz, cutoff, false, rgpot::nlist::Mode::Stream);
  requireSamePairs(walkedPairs(again, pos.size() / 3, cutoff), close);

  rgpot::nlist::Neighbours stored;
  loadCells(stored, pos, lx, ly, lz, cutoff, false, rgpot::nlist::Mode::Stored);
  REQUIRE(stored.mode() == rgpot::nlist::Mode::Stored);
  requireSamePairs(walkedPairs(stored, pos.size() / 3, cutoff), close);

  rgpot::nlist::Neighbours automatic;
  loadCells(automatic, pos, lx, ly, lz, cutoff, false, rgpot::nlist::Mode::Auto);
  REQUIRE(automatic.mode() == rgpot::nlist::Mode::Stored);
  REQUIRE(!automatic.streaming());
  requireSamePairs(walkedPairs(automatic, pos.size() / 3, cutoff), close);

  setThreads("4");
  std::vector<PairMap> parts(4);
  again.forCellTeam(true, 4,
                    [&](int thread, std::int32_t i, std::int32_t j, double, double, double,
                        double r2) {
                      if (r2 <= cutoff * cutoff) {
                        const int lo = std::min(i, j);
                        const int hi = std::max(i, j);
                        parts[static_cast<std::size_t>(thread)][{lo, hi}] = r2;
                      }
                    });
  PairMap merged;
  for (const auto &part : parts) {
    for (const auto &item : part) {
      REQUIRE(merged.insert(item).second);
    }
  }
  requireSamePairs(merged, close);
}

TEST_CASE("periodic and long-cutoff streams match the direct pair sum", "[nlist][stream][direct]") {
  setThreads("1");
  const double a = 4.05;
  auto pos = fccBlock(10, 4, 4, a);
  const double lx = 10 * a;
  const double ly = 4 * a;
  const double lz = 4 * a;
  // Image of atom 0 across the x boundary, inside a 2.5 A cutoff.
  pos[3] = lx - 0.4;
  pos[4] = 0;
  pos[5] = 0;
  const double cutoff = 2.5;
  const auto boundary = directPairs(pos, lx, ly, lz, cutoff, true);
  REQUIRE(boundary.count({0, 1}) == 1);
  rgpot::nlist::Neighbours cells;
  loadCells(cells, pos, lx, ly, lz, cutoff, true, rgpot::nlist::Mode::Stream);
  REQUIRE(cells.streaming());
  requireSamePairs(walkedPairs(cells, pos.size() / 3, cutoff), boundary);

  const auto wide = fccBlock(4, 4, 4, a);
  const double side = 4 * a;
  const double long_cut = 6.0;
  const auto many = directPairs(wide, side, side, side, long_cut, true);
  REQUIRE(many.size() > 100);
  rgpot::nlist::Neighbours tiles;
  loadCells(tiles, wide, side, side, side, long_cut, true, rgpot::nlist::Mode::Stream);
  requireSamePairs(walkedPairs(tiles, wide.size() / 3, long_cut), many);
}

bool sameBits(double a, double b) {
  return std::memcmp(&a, &b, sizeof(double)) == 0;
}

TEST_CASE("orthorhombic image index matches the careful truncation", "[nlist][round]") {
  auto check = [](double t) {
    const double got = rgpot::nlist::orthorhombicImageIndex(t);
    const double want = roundVisit(t);
    REQUIRE(sameBits(got, want));
  };
  check(0.0);
  check(-0.0);
  check(0.5);
  check(-0.5);
  check(1.5);
  check(-1.5);
  check(2.0);
  check(-2.0);
  check(std::nextafter(0.5, 0.0));
  check(std::nextafter(0.5, 1.0));
  check(std::nextafter(-0.5, 0.0));
  check(std::nextafter(-0.5, -1.0));
  check(std::numeric_limits<double>::quiet_NaN());
  check(std::numeric_limits<double>::infinity());
  check(-std::numeric_limits<double>::infinity());
  check(0x1p62);
  check(-0x1p62);
  check(std::nextafter(0x1p62, 0.0));
  check(std::nextafter(-0x1p62, 0.0));
  check(0x1p63);
  check(-0x1p63);
  check(0x1p70);
  check(-0x1p70);
  check(1e300);
  check(-1e300);
  for (int i = -4096; i <= 4096; ++i) {
    check(static_cast<double>(i) / 64.0);
    check(std::ldexp(static_cast<double>(i), 40));
    check(std::ldexp(static_cast<double>(i), 50));
  }

  // The pair walk must hand the kernel those same displacements.
  setThreads("");
  const double side = 4.0;
  const double cutoff = 3.0;
  std::vector<double> pos = {0.0, 0.0, 0.0, side * 0.5, 0.2, -0.3};
  for (double frac : {0.0, 0.25, 0.5, -0.5, 0.75, 1.25, -1.25, 2.5, -3.5}) {
    pos[3] = frac * side;
    for (auto mode : {rgpot::nlist::Mode::Stored, rgpot::nlist::Mode::Stream}) {
      rgpot::nlist::Neighbours cells;
      loadCells(cells, pos, side, side, side, cutoff, true, mode);
      auto check = [&] {
        int seen = 0;
        cells.forHalf(0, 2, [&](std::int32_t i, std::int32_t j, double dx, double dy, double dz,
                                double r2) {
          double ex = pos[3 * i] - pos[3 * j];
          double ey = pos[3 * i + 1] - pos[3 * j + 1];
          double ez = pos[3 * i + 2] - pos[3 * j + 2];
          foldDelta(ex, ey, ez, side, side, side, true);
          REQUIRE(sameBits(dx, ex));
          REQUIRE(sameBits(dy, ey));
          REQUIRE(sameBits(dz, ez));
          REQUIRE(sameBits(r2, ex * ex + ey * ey + ez * ez));
          ++seen;
        });
        REQUIRE(seen == 1);
      };
      check();
      // The second call reuses the list. The stored index has to reproduce
      // the same displacement, including an image outside {-1, 0, 1}.
      if (mode == rgpot::nlist::Mode::Stored) {
        loadCells(cells, pos, side, side, side, cutoff, true, mode);
        REQUIRE(cells.reused());
        check();
      }
    }
  }

  // One pair, so the force sum has no ordering ambiguity. The half-box
  // tie dx/L = -0.5 rounds away from zero, so the folded x component is
  // +side/2 rather than the negative image.
  rgpot::LJConfig cfg;
  cfg.cutoff = cutoff;
  cfg.psi = 1.0;
  cfg.u0 = 1.0;
  rgpot::LJPot pot(cfg);
  pos = {0.0, 0.0, 0.0, side * 0.5, 0.0, 0.0};
  const std::vector<int> types = {18, 18};
  const double box[9] = {side, 0, 0, 0, side, 0, 0, 0, side};
  std::vector<double> forces(6, 0.0);
  rgpot::ForceInput fi{.nAtoms = 2, .pos = pos.data(), .atmnrs = types.data(), .box = box};
  rgpot::ForceOut fo{.F = forces.data(),
                     .energy = 0.0,
                     .variance = 0.0,
                     .stress = {},
                     .has_stress = 0};
  pot.forceImpl(fi, &fo);

  double dx = pos[0] - pos[3];
  double dy = 0;
  double dz = 0;
  foldDelta(dx, dy, dz, side, side, side, true);
  REQUIRE(sameBits(dx, side * 0.5));
  const double r2 = dx * dx + dy * dy + dz * dz;
  const double psi2 = cfg.psi * cfg.psi;
  const double four_u0 = 4.0 * cfg.u0;
  const double shift_a = std::pow(cfg.psi / cfg.cutoff, 6.0);
  const double shift = four_u0 * shift_a * (shift_a - 1.0);
  const double inv_r2 = 1.0 / r2;
  const double sr2 = psi2 * inv_r2;
  const double sr4 = sr2 * sr2;
  const double a = sr4 * sr2;
  const double b = four_u0 * a;
  const double a_m1 = a - 1.0;
  const double v = b * a_m1;
  const double two_a = 2.0 * a;
  const double two_a_m1 = two_a - 1.0;
  const double six_b = 6.0 * b;
  const double fscale = (six_b * inv_r2) * two_a_m1;
  const double energy = v - shift;
  REQUIRE(sameBits(fo.energy, energy));
  REQUIRE(sameBits(forces[0], fscale * dx));
  REQUIRE(sameBits(forces[1], 0.0));
  REQUIRE(sameBits(forces[2], 0.0));
  REQUIRE(sameBits(forces[3], -fscale * dx));
  REQUIRE(sameBits(forces[4], 0.0));
  REQUIRE(sameBits(forces[5], 0.0));
}

TEST_CASE("a skin move recomputes an image the stored index would miss",
          "[nlist][round][skin]") {
  setThreads("1");
  const double side = 10.0;
  const double cutoff = 6.0;
  const double skin = 1.0;
  // dx/L = -0.501 rounds to -1. A shift of 0.04 A, inside half the skin,
  // crosses to -0.497 and rounds to 0. The stored index from the first
  // call is the wrong image for the second.
  std::vector<double> pos = {0.0, 0.0, 0.0, 5.01, 0.0, 0.0};
  const double box[9] = {side, 0, 0, 0, side, 0, 0, 0, side};
  rgpot::nlist::Spec spec;
  spec.cutoff = cutoff;
  spec.skin = skin;
  spec.periodic = {true, true, true};
  spec.image = rgpot::nlist::Image::Round;
  spec.mode = rgpot::nlist::Mode::Stored;
  rgpot::nlist::Neighbours cells;
  cells.update(pos.data(), 2, box, spec);
  REQUIRE_FALSE(cells.reused());
  pos[0] = 0.04;
  cells.update(pos.data(), 2, box, spec);
  REQUIRE(cells.reused());
  int seen = 0;
  cells.forHalf(0, 2, [&](std::int32_t i, std::int32_t j, double dx, double dy, double dz,
                          double r2) {
    double ex = pos[3 * i] - pos[3 * j];
    double ey = pos[3 * i + 1] - pos[3 * j + 1];
    double ez = pos[3 * i + 2] - pos[3 * j + 2];
    foldDelta(ex, ey, ez, side, side, side, true);
    REQUIRE(sameBits(dx, ex));
    REQUIRE(sameBits(dy, ey));
    REQUIRE(sameBits(dz, ez));
    REQUIRE(sameBits(r2, ex * ex + ey * ey + ez * ez));
    // Index 0 leaves the raw displacement. The index stored at the first
    // call was -1, which would have moved x by a full box length.
    REQUIRE(sameBits(dx, pos[3 * i] - pos[3 * j]));
    ++seen;
  });
  REQUIRE(seen == 1);
}

void expectStoredMatchesRounding(rgpot::nlist::Neighbours &cells, const std::vector<double> &pos,
                                 double lx, double ly, double lz, bool px, bool py, bool pz) {
  const double invx = px && lx > 0.0 ? 1.0 / lx : 0.0;
  const double invy = py && ly > 0.0 ? 1.0 / ly : 0.0;
  const double invz = pz && lz > 0.0 ? 1.0 / lz : 0.0;
  int seen = 0;
  cells.forDirected(0, static_cast<std::int64_t>(pos.size() / 3),
                    [&](std::int32_t i, std::int32_t j, double dx, double dy, double dz, double r2) {
                      double ex = pos[3 * i] - pos[3 * j];
                      double ey = pos[3 * i + 1] - pos[3 * j + 1];
                      double ez = pos[3 * i + 2] - pos[3 * j + 2];
                      if (px) {
                        ex -= lx * rgpot::nlist::orthorhombicImageIndex(ex * invx);
                      }
                      if (py) {
                        ey -= ly * rgpot::nlist::orthorhombicImageIndex(ey * invy);
                      }
                      if (pz) {
                        ez -= lz * rgpot::nlist::orthorhombicImageIndex(ez * invz);
                      }
                      REQUIRE(sameBits(dx, ex));
                      REQUIRE(sameBits(dy, ey));
                      REQUIRE(sameBits(dz, ez));
                      REQUIRE(sameBits(r2, ex * ex + ey * ey + ez * ez));
                      ++seen;
                    });
  REQUIRE(seen > 0);
}

TEST_CASE("a skin move keeps a stored image only when rounding would",
          "[nlist][round][skin]") {
  setThreads("1");
  std::mt19937_64 rng(17);
  std::uniform_real_distribution<double> unit(0.0, 1.0);
  std::uniform_real_distribution<double> kick(-0.04, 0.04);
  const auto perturb = [&](std::vector<double> &pos) {
    for (double &x : pos) {
      x += kick(rng);
    }
  };
  const auto run = [&](std::vector<double> pos, double lx, double ly, double lz, double cutoff,
                       double skin, bool px, bool py, bool pz) {
    const double box[9] = {lx, 0, 0, 0, ly, 0, 0, 0, lz};
    rgpot::nlist::Spec spec;
    spec.cutoff = cutoff;
    spec.skin = skin;
    spec.periodic = {px, py, pz};
    spec.image = rgpot::nlist::Image::Round;
    spec.mode = rgpot::nlist::Mode::Stored;
    rgpot::nlist::Neighbours cells;
    const auto base = pos;
    cells.update(base.data(), base.size() / 3, box, spec);
    REQUIRE_FALSE(cells.reused());
    for (int sample = 0; sample < 6; ++sample) {
      pos = base;
      perturb(pos);
      cells.update(pos.data(), pos.size() / 3, box, spec);
      REQUIRE(cells.reused());
      expectStoredMatchesRounding(cells, pos, lx, ly, lz, px, py, pz);
    }
  };
  // Wide cell: every stored pair is far from a half-box face.
  {
    std::vector<double> pos;
    for (int i = 0; i < 48; ++i) {
      pos.push_back(unit(rng) * 36.0);
      pos.push_back(unit(rng) * 36.0);
      pos.push_back(unit(rng) * 36.0);
    }
    run(pos, 40.0, 40.0, 40.0, 6.0, 0.5, true, true, true);
  }
  // Narrow cell, with atoms on the faces, so some images can change.
  {
    std::vector<double> pos;
    for (int i = 0; i < 24; ++i) {
      pos.push_back(unit(rng) * 10.0);
      pos.push_back(unit(rng) * 10.0);
      pos.push_back(unit(rng) * 10.0);
    }
    pos[0] = 0.02;
    pos[1] = 0.02;
    pos[2] = 0.02;
    pos[3] = 5.01;
    pos[4] = 0.03;
    pos[5] = 0.04;
    pos[6] = 9.98;
    pos[7] = 5.02;
    pos[8] = 0.01;
    run(pos, 10.0, 10.0, 10.0, 5.5, 0.8, true, true, true);
    run(pos, 10.0, 12.0, 9.0, 5.0, 0.6, true, false, true);
  }
}

void expectStoredMatchesFold(rgpot::nlist::Neighbours &cells,
                             const std::vector<double> &pos, const double *box,
                             const std::array<bool, 3> &periodic) {
  const rgpot::nlist::MinimumImage image(box, periodic);
  int seen = 0;
  cells.forDirected(0, static_cast<std::int64_t>(pos.size() / 3),
                    [&](std::int32_t i, std::int32_t j, double dx, double dy,
                        double dz, double r2) {
                      double ex = pos[static_cast<std::size_t>(i) * 3] -
                                  pos[static_cast<std::size_t>(j) * 3];
                      double ey = pos[static_cast<std::size_t>(i) * 3 + 1] -
                                  pos[static_cast<std::size_t>(j) * 3 + 1];
                      double ez = pos[static_cast<std::size_t>(i) * 3 + 2] -
                                  pos[static_cast<std::size_t>(j) * 3 + 2];
                      image.fold(ex, ey, ez);
                      REQUIRE(sameBits(dx, ex));
                      REQUIRE(sameBits(dy, ey));
                      REQUIRE(sameBits(dz, ez));
                      REQUIRE(sameBits(r2, ex * ex + ey * ey + ez * ez));
                      ++seen;
                    });
  REQUIRE(seen > 0);
}

TEST_CASE("rounded triclinic coefficients match the minimum-image fold",
          "[nlist][general][skin]") {
  setThreads("1");
  // Mild monoclinic tilt. A short pair sits well inside the Voronoi cell,
  // so the rounded image is the one the search returns and a skin move
  // cannot change it.
  const double side = 40.0;
  const double beta = 100.0 * 3.14159265358979323846 / 180.0;
  const double box[9] = {
      side, 0, 0, 0, side, 0, side * std::cos(beta), 0, side * std::sin(beta)};
  const std::array<bool, 3> periodic{{true, true, true}};
  rgpot::nlist::MinimumImage image(box, periodic);
  const double dx = 3.1;
  const double dy = -1.4;
  const double dz = 0.7;
  std::int32_t n[3] = {};
  REQUIRE(image.tryRounded(dx, dy, dz, n));
  const double shortest = image.shortestPeriodic();
  REQUIRE(shortest > 20.0);
  REQUIRE(image.roundedHolds(dx, dy, dz, n, 1.0, shortest));
  std::mt19937_64 rng(11);
  std::uniform_real_distribution<double> unit(-1.0, 1.0);
  for (int sample = 0; sample < 32; ++sample) {
    double x = unit(rng);
    double y = unit(rng);
    double z = unit(rng);
    const double norm = std::sqrt(x * x + y * y + z * z);
    const double scale = norm > 0.0 ? 0.45 / norm : 0.0;
    const double px = dx + x * scale;
    const double py = dy + y * scale;
    const double pz = dz + z * scale;
    double folded_x = px;
    double folded_y = py;
    double folded_z = pz;
    image.fold(folded_x, folded_y, folded_z);
    double replay_x = px;
    double replay_y = py;
    double replay_z = pz;
    image.subtractRounded(n[0], n[1], n[2], replay_x, replay_y, replay_z);
    REQUIRE(sameBits(replay_x, folded_x));
    REQUIRE(sameBits(replay_y, folded_y));
    REQUIRE(sameBits(replay_z, folded_z));
  }
}

TEST_CASE("a stored general list matches the fold after a skin move",
          "[nlist][general][skin]") {
  setThreads("1");
  std::mt19937_64 rng(29);
  std::uniform_real_distribution<double> unit(-1.0, 1.0);
  const auto shift_atoms = [&](std::vector<double> &pos,
                               const std::vector<double> &base,
                               double half_skin) {
    const double cap = 0.45 * half_skin;
    for (std::size_t a = 0; a < base.size() / 3; ++a) {
      double x = unit(rng);
      double y = unit(rng);
      double z = unit(rng);
      const double norm = std::sqrt(x * x + y * y + z * z);
      const double scale = norm > 0.0 ? cap / norm : 0.0;
      pos[3 * a] = base[3 * a] + x * scale;
      pos[3 * a + 1] = base[3 * a + 1] + y * scale;
      pos[3 * a + 2] = base[3 * a + 2] + z * scale;
    }
  };
  const auto run = [&](std::vector<double> pos, const double *box,
                       double cutoff, double skin,
                       std::array<bool, 3> periodic) {
    rgpot::nlist::Spec spec;
    spec.cutoff = cutoff;
    spec.skin = skin;
    spec.periodic = periodic;
    spec.image = rgpot::nlist::Image::Round;
    spec.mode = rgpot::nlist::Mode::Stored;
    rgpot::nlist::Neighbours cells;
    const auto base = pos;
    cells.update(base.data(), base.size() / 3, box, spec);
    REQUIRE_FALSE(cells.reused());
    REQUIRE_FALSE(cells.streaming());
    expectStoredMatchesFold(cells, base, box, periodic);
    cells.update(base.data(), base.size() / 3, box, spec);
    REQUIRE(cells.reused());
    expectStoredMatchesFold(cells, base, box, periodic);
    for (int sample = 0; sample < 4; ++sample) {
      pos = base;
      shift_atoms(pos, base, 0.5 * skin);
      cells.update(pos.data(), pos.size() / 3, box, spec);
      REQUIRE(cells.reused());
      expectStoredMatchesFold(cells, pos, box, periodic);
    }
  };

  const double side = 36.0;
  const double beta = 100.0 * 3.14159265358979323846 / 180.0;
  const double mild[9] = {
      side, 0, 0, 0, side, 0, side * std::cos(beta), 0, side * std::sin(beta)};
  std::vector<double> pos;
  std::uniform_real_distribution<double> wide(0.0, side);
  for (int i = 0; i < 40; ++i) {
    pos.push_back(wide(rng));
    pos.push_back(wide(rng));
    pos.push_back(wide(rng));
  }
  // Faces of the cell, where a minimum image can sit near a boundary.
  pos[0] = 0.01;
  pos[1] = 0.02;
  pos[2] = 0.01;
  pos[3] = side - 0.02;
  pos[4] = 0.03;
  pos[5] = side - 0.04;
  pos[6] = 0.015;
  pos[7] = side - 0.01;
  pos[8] = 0.02;
  run(pos, mild, 6.0, 0.8, {true, true, true});
  run(pos, mild, 6.0, 0.8, {true, false, true});

  // Strong tilt. Rounding in fractional coordinates is not always the
  // minimum image here; the stored walk still has to match the search.
  const double skew[9] = {12.0, 0.0, 0.0, 11.2, 1.5, 0.0, 10.4, 0.8, 1.2};
  std::vector<double> tight;
  std::uniform_real_distribution<double> span(0.0, 8.0);
  for (int i = 0; i < 18; ++i) {
    tight.push_back(span(rng));
    tight.push_back(span(rng) * 0.4);
    tight.push_back(span(rng) * 0.3);
  }
  tight[0] = 0.002;
  tight[1] = 0.001;
  tight[2] = 0.002;
  tight[3] = 11.5;
  tight[4] = 0.05;
  tight[5] = 0.04;
  tight[6] = 0.01;
  tight[7] = 1.2;
  tight[8] = 0.9;
  run(tight, skew, 3.5, 0.4, {true, true, true});
}

TEST_CASE("a rebuild inside the same cells matches a fresh list", "[nlist][cells]") {
  setThreads("1");
  const double side = 40.0;
  const double cutoff = 6.0;
  const double skin = 1.0;
  const double box[9] = {side, 0, 0, 0, side, 0, 0, 0, side};
  // Cell width is side/5 = 8. An atom at the middle of a cell can move
  // by more than half the skin and still occupy that cell.
  std::vector<double> pos = {20.0, 20.0, 20.0, 22.5, 20.4, 19.7, 4.2, 12.1, 28.3};
  rgpot::nlist::Spec spec;
  spec.cutoff = cutoff;
  spec.skin = skin;
  spec.periodic = {true, true, true};
  spec.image = rgpot::nlist::Image::Round;
  spec.mode = rgpot::nlist::Mode::Stored;
  rgpot::nlist::Neighbours kept;
  kept.update(pos.data(), pos.size() / 3, box, spec);
  REQUIRE_FALSE(kept.reused());
  pos[0] = 20.6;
  pos[3] = 22.1;
  kept.update(pos.data(), pos.size() / 3, box, spec);
  REQUIRE_FALSE(kept.reused());
  rgpot::nlist::Neighbours fresh;
  fresh.update(pos.data(), pos.size() / 3, box, spec);
  std::vector<std::uint64_t> a;
  std::vector<std::uint64_t> b;
  auto take = [](rgpot::nlist::Neighbours &cells, std::vector<std::uint64_t> &out) {
    cells.forDirected(0, 3, [&](std::int32_t i, std::int32_t j, double dx, double dy, double dz,
                                double r2) {
      out.push_back(static_cast<std::uint64_t>(static_cast<std::uint32_t>(i)) << 32 |
                    static_cast<std::uint32_t>(j));
      out.push_back(bits(dx));
      out.push_back(bits(dy));
      out.push_back(bits(dz));
      out.push_back(bits(r2));
    });
  };
  take(kept, a);
  take(fresh, b);
  REQUIRE(a == b);
  REQUIRE(!a.empty());
}

std::vector<std::uint64_t> directedBits(rgpot::nlist::Neighbours &cells, std::size_t n) {
  std::vector<std::uint64_t> out;
  cells.forDirected(0, static_cast<std::int64_t>(n),
                    [&](std::int32_t i, std::int32_t j, double dx, double dy, double dz,
                        double r2) {
                      out.push_back(static_cast<std::uint64_t>(static_cast<std::uint32_t>(i))
                                        << 32 |
                                    static_cast<std::uint32_t>(j));
                      out.push_back(bits(dx));
                      out.push_back(bits(dy));
                      out.push_back(bits(dz));
                      out.push_back(bits(r2));
                    });
  return out;
}

std::vector<double> boxAtoms(int count, double lo, double hi, std::uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<double> u(lo, hi);
  std::vector<double> pos(static_cast<std::size_t>(count) * 3);
  for (double &x : pos) {
    x = u(rng);
  }
  return pos;
}

// search = cutoff + skin. The cell count is floor(L / search), reduced
// while a cell would be shorter than the search.
void expectGuard(const std::vector<double> &pos, double side, double cutoff, double skin) {
  const double box[9] = {side, 0, 0, 0, side, 0, 0, 0, side};
  rgpot::nlist::Spec spec;
  spec.cutoff = cutoff;
  spec.skin = skin;
  spec.periodic = {true, true, true};
  spec.image = rgpot::nlist::Image::Round;
  spec.mode = rgpot::nlist::Mode::Stored;
  const auto n = pos.size() / 3;
  setThreads("1");
  rgpot::nlist::Neighbours one;
  one.update(pos.data(), n, box, spec);
  REQUIRE_FALSE(one.reused());
  REQUIRE_FALSE(one.streaming());
  expectStoredMatchesRounding(one, pos, side, side, side, true, true, true);
  const auto bits1 = directedBits(one, n);
  setThreads("2");
  rgpot::nlist::Neighbours two;
  two.update(pos.data(), n, box, spec);
  REQUIRE(directedBits(two, n) == bits1);
  expectStoredMatchesRounding(two, pos, side, side, side, true, true, true);
}

void expectSwitch(std::vector<double> pos, double side, double cutoff, double skin,
                  double shift) {
  const double box[9] = {side, 0, 0, 0, side, 0, 0, 0, side};
  rgpot::nlist::Spec spec;
  spec.cutoff = cutoff;
  spec.skin = skin;
  spec.periodic = {true, true, true};
  spec.image = rgpot::nlist::Image::Round;
  spec.mode = rgpot::nlist::Mode::Stored;
  const auto n = pos.size() / 3;
  setThreads("1");
  rgpot::nlist::Neighbours kept;
  kept.update(pos.data(), n, box, spec);
  REQUIRE_FALSE(kept.reused());
  expectStoredMatchesRounding(kept, pos, side, side, side, true, true, true);
  for (std::size_t k = 0; k < pos.size(); k += 3) {
    pos[k] += shift;
  }
  kept.update(pos.data(), n, box, spec);
  REQUIRE_FALSE(kept.reused());
  rgpot::nlist::Neighbours fresh;
  fresh.update(pos.data(), n, box, spec);
  REQUIRE(directedBits(kept, n) == directedBits(fresh, n));
  expectStoredMatchesRounding(kept, pos, side, side, side, true, true, true);
  expectStoredMatchesRounding(fresh, pos, side, side, side, true, true, true);
  setThreads("2");
  rgpot::nlist::Neighbours threaded;
  threaded.update(pos.data(), n, box, spec);
  REQUIRE(directedBits(threaded, n) == directedBits(fresh, n));
}

TEST_CASE("the cell-shift guard matches the fold on both sides", "[nlist][round][cells]") {
  const double cutoff = 5.0;
  const double skin = 1.0;
  // Two and three cells stay on the original fold.
  expectGuard(boxAtoms(40, 0.05, 13.9, 3), 14.0, cutoff, skin);
  expectGuard(boxAtoms(48, 0.05, 19.9, 5), 20.0, cutoff, skin);
  // Four cells. Inside the box the stencil image is the unique nearest
  // one. The same atoms then leave the box, and the pairs still match
  // the original fold and a list built from scratch.
  auto four = boxAtoms(64, 0.05, 25.9, 7);
  expectGuard(four, 26.0, cutoff, skin);
  expectSwitch(four, 26.0, cutoff, skin, 4.0);
  // Eight cells. w = 6.25 and the outside margin is 11.25. A shift of 13
  // pushes an atom at 49.5 past that margin, onto the original fold.
  auto eight = boxAtoms(80, 0.2, 49.5, 11);
  eight[0] = 0.2;
  eight[3] = 49.5;
  expectGuard(eight, 50.0, cutoff, skin);
  expectSwitch(eight, 50.0, cutoff, skin, 13.0);
}

TEST_CASE("a neighbour mode outside the enumerators still stores", "[nlist][mode]") {
  // A mode byte that is not Auto, Stored, or Stream is not a request to
  // stream. Streaming that byte used to change a warm call from the stored
  // list to the spline walk.
  const double pos[] = {0.2, 0.1, 0.3, 2.4, 0.2, 0.4, 0.3, 2.5, 0.2, 2.2, 2.1, 0.5};
  const double box[9] = {12, 0, 0, 0, 12, 0, 0, 0, 12};
  rgpot::nlist::Spec spec;
  spec.cutoff = 4.0;
  spec.skin = 0.5;
  spec.image = rgpot::nlist::Image::Round;
  spec.periodic = {true, true, true};
  spec.mode = static_cast<rgpot::nlist::Mode>(47);
  rgpot::nlist::Neighbours garbage;
  garbage.update(pos, 4, box, spec);
  spec.mode = rgpot::nlist::Mode::Auto;
  rgpot::nlist::Neighbours automatic;
  automatic.update(pos, 4, box, spec);
  REQUIRE_FALSE(garbage.streaming());
  REQUIRE(garbage.slots() > 0);
  REQUIRE(garbage.slots() == automatic.slots());
}
