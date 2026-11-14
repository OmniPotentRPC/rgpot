// MIT License
// Copyright 2023--present rgpot developers

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
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

namespace {

/// Atom pair and integer lattice image: the vector is r_i - r_j - n H.
using ImageKey = std::array<std::int64_t, 5>;

struct Lattice {
  double h[9];
  double dual[9];
};

Lattice lattice(const double box[9]) {
  Lattice l{};
  std::memcpy(l.h, box, sizeof(l.h));
  const double *a = box;
  const double *b = box + 3;
  const double *c = box + 6;
  const double bc[3] = {b[1] * c[2] - b[2] * c[1], b[2] * c[0] - b[0] * c[2],
                        b[0] * c[1] - b[1] * c[0]};
  const double ca[3] = {c[1] * a[2] - c[2] * a[1], c[2] * a[0] - c[0] * a[2],
                        c[0] * a[1] - c[1] * a[0]};
  const double ab[3] = {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
                        a[0] * b[1] - a[1] * b[0]};
  const double vol = a[0] * bc[0] + a[1] * bc[1] + a[2] * bc[2];
  for (int k = 0; k < 3; ++k) {
    l.dual[k] = bc[k] / vol;
    l.dual[3 + k] = ca[k] / vol;
    l.dual[6 + k] = ab[k] / vol;
  }
  return l;
}

/// Coefficients of `v` on the rows of the cell.
void rowsOf(const Lattice &l, const double v[3], double f[3]) {
  for (int k = 0; k < 3; ++k) {
    f[k] = l.dual[3 * k] * v[0] + l.dual[3 * k + 1] * v[1] + l.dual[3 * k + 2] * v[2];
  }
}

/// Every image of every partner within `search`, with its squared length.
/// The image range on row k is exact: a vector shorter than `search` has
/// a coefficient within `search |b_k|` of the fractional separation.
std::map<ImageKey, double> allImages(const std::vector<double> &pos, const Lattice &l,
                                     const std::array<bool, 3> &periodic, double search) {
  std::map<ImageKey, double> out;
  const std::size_t n = pos.size() / 3;
  double reach[3];
  for (int k = 0; k < 3; ++k) {
    reach[k] = search * std::sqrt(l.dual[3 * k] * l.dual[3 * k] +
                                  l.dual[3 * k + 1] * l.dual[3 * k + 1] +
                                  l.dual[3 * k + 2] * l.dual[3 * k + 2]);
  }
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      const double d[3] = {pos[3 * i] - pos[3 * j], pos[3 * i + 1] - pos[3 * j + 1],
                           pos[3 * i + 2] - pos[3 * j + 2]};
      double f[3];
      rowsOf(l, d, f);
      std::int64_t lo[3] = {0, 0, 0};
      std::int64_t hi[3] = {0, 0, 0};
      for (int k = 0; k < 3; ++k) {
        if (periodic[static_cast<std::size_t>(k)]) {
          lo[k] = static_cast<std::int64_t>(std::ceil(f[k] - reach[k])) - 1;
          hi[k] = static_cast<std::int64_t>(std::floor(f[k] + reach[k])) + 1;
        }
      }
      for (std::int64_t a = lo[0]; a <= hi[0]; ++a) {
        for (std::int64_t b = lo[1]; b <= hi[1]; ++b) {
          for (std::int64_t c = lo[2]; c <= hi[2]; ++c) {
            if (i == j && a == 0 && b == 0 && c == 0) {
              continue;
            }
            double r2 = 0;
            for (int x = 0; x < 3; ++x) {
              const double v = d[x] - static_cast<double>(a) * l.h[x] -
                               static_cast<double>(b) * l.h[3 + x] -
                               static_cast<double>(c) * l.h[6 + x];
              r2 += v * v;
            }
            if (r2 <= search * search * (1 + 1e-9)) {
              out[{static_cast<std::int64_t>(i), static_cast<std::int64_t>(j), a, b, c}] = r2;
            }
          }
        }
      }
    }
  }
  return out;
}

/// The nearest image of each pair from `all`, unless a second image is as
/// near to within rounding.
std::map<ImageKey, double> nearestImages(const std::map<ImageKey, double> &all,
                                         std::map<ImageKey, double> &tied) {
  std::map<std::pair<std::int64_t, std::int64_t>, std::vector<std::pair<double, ImageKey>>> by;
  for (const auto &item : all) {
    if (item.first[0] != item.first[1]) {
      by[{item.first[0], item.first[1]}].push_back({item.second, item.first});
    }
  }
  std::map<ImageKey, double> out;
  for (auto &pair : by) {
    auto &v = pair.second;
    std::sort(v.begin(), v.end());
    if (v.size() > 1 && v[1].first <= v[0].first * (1 + 1e-9)) {
      tied[v[0].second] = v[0].first;
      tied[v[1].second] = v[1].first;
      continue;
    }
    out[v[0].second] = v[0].first;
  }
  return out;
}

/// The walk's pairs keyed by the lattice image each vector implies.
std::map<ImageKey, double> walkedImages(const rgpot::nlist::Neighbours &cells,
                                        const std::vector<double> &pos, const Lattice &l,
                                        double search) {
  std::map<ImageKey, double> out;
  cells.forDirected(0, static_cast<std::int64_t>(pos.size() / 3),
                    [&](std::int32_t i, std::int32_t j, double dx, double dy, double dz,
                        double r2) {
                      if (r2 > search * search) {
                        return;
                      }
                      const auto a = static_cast<std::size_t>(i);
                      const auto b = static_cast<std::size_t>(j);
                      const double shift[3] = {pos[3 * a] - pos[3 * b] - dx,
                                               pos[3 * a + 1] - pos[3 * b + 1] - dy,
                                               pos[3 * a + 2] - pos[3 * b + 2] - dz};
                      double f[3];
                      rowsOf(l, shift, f);
                      const ImageKey key{i, j, std::llround(f[0]), std::llround(f[1]),
                                         std::llround(f[2])};
                      REQUIRE(out.insert({key, r2}).second);
                    });
  return out;
}

/// `got` holds every pair of `ref` clear of the search radius, and nothing
/// that is not in `ref` or `fringe`.
void requireSameImages(const std::map<ImageKey, double> &got,
                       const std::map<ImageKey, double> &ref,
                       const std::map<ImageKey, double> &fringe, double search) {
  const double edge = search * search;
  for (const auto &item : ref) {
    if (item.second < edge * (1 - 1e-9)) {
      INFO("pair " << item.first[0] << " " << item.first[1] << " image " << item.first[2]
                   << " " << item.first[3] << " " << item.first[4]);
      REQUIRE(got.count(item.first) == 1);
    }
  }
  for (const auto &item : got) {
    const auto it = ref.find(item.first);
    if (it == ref.end()) {
      INFO("extra pair " << item.first[0] << " " << item.first[1]);
      REQUIRE(fringe.count(item.first) == 1);
      continue;
    }
    REQUIRE(std::abs(item.second - it->second) <= 1e-9 * std::max(1.0, it->second));
  }
}

} // namespace

TEST_CASE("sheared stored lists match every image of a brute force",
          "[nlist][stored][triclinic]") {
  setThreads("1");
  std::mt19937_64 rng(20260514u);
  const auto uniform = [&rng] { return static_cast<double>(rng() >> 11) * 0x1p-53; };
  const double cutoff = 3.2;
  const double skin = 0.4;
  const double search = cutoff + skin;
  int stored_bins = 0;
  int image_tables = 0;
  for (int trial = 0; trial < 36; ++trial) {
    const int kind = trial % 6;
    double box[9] = {};
    const double side = 9.0 + 9.0 * uniform();
    for (int k = 0; k < 3; ++k) {
      box[4 * k] = side * (0.7 + 0.6 * uniform());
    }
    // Lower-triangular tilts up to half a row, strong ones up to three
    // rows, then full rows, a thin slab and a thin needle.
    const double tilt = kind == 0 ? 0.5 : kind == 1 ? 3.0 : 1.5;
    box[3] = tilt * (2 * uniform() - 1) * box[0];
    box[6] = tilt * (2 * uniform() - 1) * box[0];
    box[7] = tilt * (2 * uniform() - 1) * box[4];
    if (kind == 2) {
      box[1] = 0.4 * (2 * uniform() - 1) * box[4];
      box[2] = 0.3 * (2 * uniform() - 1) * box[8];
      box[5] = 0.3 * (2 * uniform() - 1) * box[8];
    }
    if (kind == 3) {
      box[8] = 2.0 + 3.0 * uniform();
    }
    if (kind == 4) {
      box[0] = 2.5 + 2.0 * uniform();
      box[4] = 3.0 + 2.0 * uniform();
    }
    if (box[3] == 0.0) {
      box[3] = 0.25 * box[0];
    }
    std::array<bool, 3> periodic{{true, true, true}};
    if (kind == 5) {
      periodic[static_cast<std::size_t>(trial % 3)] = false;
    }
    const Lattice l = lattice(box);
    const double vol = std::abs(box[0] * (box[4] * box[8] - box[5] * box[7]) -
                                box[1] * (box[3] * box[8] - box[5] * box[6]) +
                                box[2] * (box[3] * box[7] - box[4] * box[6]));
    const auto n = static_cast<std::size_t>(std::clamp(vol * 0.05, 12.0, 700.0));
    // Some atoms sit outside the cell, up to a third of a row away.
    std::vector<double> pos(3 * n);
    for (std::size_t a = 0; a < n; ++a) {
      double f[3];
      for (double &v : f) {
        v = trial % 2 == 0 ? uniform() : 1.6 * uniform() - 0.3;
      }
      for (int x = 0; x < 3; ++x) {
        pos[3 * a + static_cast<std::size_t>(x)] = f[0] * box[x] + f[1] * box[3 + x] +
                                                   f[2] * box[6 + x];
      }
    }
    INFO("trial " << trial << " kind " << kind << " atoms " << n);
    const auto all = allImages(pos, l, periodic, search);
    std::map<ImageKey, double> tied;
    const auto nearest = nearestImages(all, tied);
    for (const bool images : {true, false}) {
      for (const auto mode : {rgpot::nlist::Mode::Stored, rgpot::nlist::Mode::Stream}) {
        rgpot::nlist::Spec spec;
        spec.cutoff = cutoff;
        spec.skin = skin;
        spec.periodic = periodic;
        spec.image = rgpot::nlist::Image::Round;
        spec.mode = mode;
        spec.images = images;
        rgpot::nlist::Neighbours cells;
        cells.update(pos.data(), n, box, spec);
        const auto got = walkedImages(cells, pos, l, search);
        if (cells.images()) {
          ++image_tables;
          requireSameImages(got, all, all, search);
          continue;
        }
        requireSameImages(got, nearest, tied, search);
        if (mode != rgpot::nlist::Mode::Stored) {
          continue;
        }
        REQUIRE(!cells.streaming());
        ++stored_bins;
        // The stored index is the pair set of the walk, in increasing order.
        std::map<std::pair<std::int64_t, std::int64_t>, int> index;
        const auto &off = cells.offsets();
        const auto &nbr = cells.neighbours();
        REQUIRE(off.size() == n + 1);
        for (std::size_t a = 0; a < n; ++a) {
          for (auto q = off[a]; q < off[a + 1]; ++q) {
            if (q > off[a]) {
              REQUIRE(nbr[static_cast<std::size_t>(q - 1)] < nbr[static_cast<std::size_t>(q)]);
            }
            index[{static_cast<std::int64_t>(a), nbr[static_cast<std::size_t>(q)]}] = 1;
          }
        }
        for (const auto &item : got) {
          REQUIRE(index.count({item.first[0], item.first[1]}) == 1);
        }
        for (const auto &item : nearest) {
          if (item.second < search * search * (1 - 1e-9)) {
            REQUIRE(index.count({item.first[0], item.first[1]}) == 1);
          }
        }
      }
    }
  }
  REQUIRE(stored_bins > 12);
  REQUIRE(image_tables > 6);
}
