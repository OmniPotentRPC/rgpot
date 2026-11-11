// MIT License
// Copyright 2023--present rgpot developers

#include <array>
#include <cmath>
#include <utility>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <catch2/catch_all.hpp>

#include "rgpot/EAM/EAMPot.hpp"
#include "rgpot/LennardJones/LJPot.hpp"
#include "rgpot/StillingerWeber/SWSi.hpp"
#include "rgpot/SuttonChen/SuttonChenPot.hpp"
#include "rgpot/Tersoff/TersoffSiC.hpp"
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
