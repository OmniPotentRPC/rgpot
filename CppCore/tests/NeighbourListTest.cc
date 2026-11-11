// MIT License
// Copyright 2023--present rgpot developers

#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <tuple>
#include <vector>

#include <catch2/catch_all.hpp>

#include "rgpot/nlist/NeighbourList.hpp"
#include "rgpot/nlist/rgpot_nlist.h"

namespace {

struct Key {
  int i, j, sa, sb, sc;
  bool operator<(const Key &o) const {
    return std::tie(i, j, sa, sb, sc) < std::tie(o.i, o.j, o.sa, o.sb, o.sc);
  }
  bool operator==(const Key &o) const {
    return std::tie(i, j, sa, sb, sc) == std::tie(o.i, o.j, o.sa, o.sb, o.sc);
  }
};

std::set<Key> brute(const std::vector<double> &pos, const double *box, double cutoff,
                    bool full, bool strict, const std::array<bool, 3> &periodic) {
  const int n = static_cast<int>(pos.size() / 3);
  const int span = 3;
  std::set<Key> out;
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      for (int sa = -span; sa <= span; ++sa) {
        if (!periodic[0] && sa != 0) {
          continue;
        }
        for (int sb = -span; sb <= span; ++sb) {
          if (!periodic[1] && sb != 0) {
            continue;
          }
          for (int sc = -span; sc <= span; ++sc) {
            if (!periodic[2] && sc != 0) {
              continue;
            }
            double vx = pos[3 * j] - pos[3 * i];
            double vy = pos[3 * j + 1] - pos[3 * i + 1];
            double vz = pos[3 * j + 2] - pos[3 * i + 2];
            vx += sa * box[0] + sb * box[3] + sc * box[6];
            vy += sa * box[1] + sb * box[4] + sc * box[7];
            vz += sa * box[2] + sb * box[5] + sc * box[8];
            const double r2 = vx * vx + vy * vy + vz * vz;
            const double cut2 = cutoff * cutoff;
            const bool inside = strict ? r2 < cut2 : r2 <= cut2;
            if (!(r2 > 0.0) || !inside) {
              continue;
            }
            if (!full) {
              if (i > j) {
                continue;
              }
              if (i == j) {
                const int sum = sa + sb + sc;
                if (sum < 0 || (sum == 0 && (sc < 0 || (sc == 0 && sb < 0))) ||
                    (sa == 0 && sb == 0 && sc == 0)) {
                  continue;
                }
              }
            }
            out.insert(Key{i, j, sa, sb, sc});
          }
        }
      }
    }
  }
  return out;
}

std::set<Key> from_table(const rgpot::nlist::Table &table) {
  std::set<Key> out;
  for (std::size_t p = 0; p < table.size(); ++p) {
    const int mask = table.mask[p];
    const int origin = table.first[p];
    if (mask == 0xff) {
      out.insert(Key{origin, table.second[p], table.shift[3 * p], table.shift[3 * p + 1],
                     table.shift[3 * p + 2]});
    } else {
      for (int lane = 0; lane < 4; ++lane) {
        if ((mask & (1 << lane)) != 0) {
          out.insert(Key{origin + lane, table.second[p], table.shift[3 * p],
                         table.shift[3 * p + 1], table.shift[3 * p + 2]});
        }
      }
    }
  }
  return out;
}

} // namespace

TEST_CASE("neighbour query matches a triclinic brute force", "[nlist][api]") {
  const double box[9] = {4.0, 0.2, 0.1, 0.3, 4.5, -0.2, 0.05, 0.15, 4.2};
  const std::vector<double> pos = {0.2, 0.3, 0.4, 3.6, 0.5, 0.2, 1.0, 3.8, 0.7,
                                   2.2, 2.1, 3.9};
  const double cutoff = 3.2;
  const std::array<bool, 3> periodic{{true, true, false}};
  for (bool full : {false, true}) {
    for (bool strict : {true, false}) {
      rgpot::nlist::Query query;
      query.cutoff = cutoff;
      query.skin = 0;
      query.shape = rgpot::nlist::Pairs{full ? rgpot::nlist::Shell::Full
                                            : rgpot::nlist::Shell::Half};
      query.cut = strict ? rgpot::nlist::Cut::Strict : rgpot::nlist::Cut::Loose;
      query.periodic = periodic;
      rgpot::nlist::Table table;
      rgpot::nlist::build(table, pos.data(), pos.size() / 3, box, query);
      const auto got = from_table(table);
      const auto ref = brute(pos, box, cutoff, full, strict, periodic);
      REQUIRE(got == ref);
      for (std::size_t p = 0; p < table.size(); ++p) {
        const int i = table.mask[p] == 0xff ? table.first[p] : table.first[p];
        (void)i;
        if (table.mask[p] != 0xff) {
          continue;
        }
        const int j = table.second[p];
        const double vx = pos[3 * j] - pos[3 * table.first[p]] +
                          table.shift[3 * p] * box[0] + table.shift[3 * p + 1] * box[3] +
                          table.shift[3 * p + 2] * box[6];
        const double vy = pos[3 * j + 1] - pos[3 * table.first[p] + 1] +
                          table.shift[3 * p] * box[1] + table.shift[3 * p + 1] * box[4] +
                          table.shift[3 * p + 2] * box[7];
        const double vz = pos[3 * j + 2] - pos[3 * table.first[p] + 2] +
                          table.shift[3 * p] * box[2] + table.shift[3 * p + 1] * box[5] +
                          table.shift[3 * p + 2] * box[8];
        REQUIRE(std::abs(vx - table.vector[3 * p]) < 1e-12);
        REQUIRE(std::abs(vy - table.vector[3 * p + 1]) < 1e-12);
        REQUIRE(std::abs(vz - table.vector[3 * p + 2]) < 1e-12);
      }
    }
  }
}

TEST_CASE("a chunked walk returns the stored pairs", "[nlist][api]") {
  const double box[9] = {6, 0, 0, 0, 5, 0, 1, 0, 7};
  const std::vector<double> pos = {0.5, 0.5, 0.5, 2.0, 1.0, 1.0, 4.5, 3.0, 6.0};
  rgpot::nlist::Query query;
  query.cutoff = 4.0;
  query.shape = rgpot::nlist::Pairs{rgpot::nlist::Shell::Full};
  query.cut = rgpot::nlist::Cut::Strict;
  rgpot::nlist::Table table;
  rgpot::nlist::build(table, pos.data(), pos.size() / 3, box, query);
  rgpot::nlist::Walk walk(pos.data(), pos.size() / 3, box, query);
  std::vector<std::int32_t> first(2), second(2), shift(6);
  std::vector<double> vector(6), distance(2);
  std::vector<std::uint8_t> mask(2);
  std::set<Key> got;
  bool done = false;
  while (!done) {
    rgpot::nlist::Buffers dest;
    dest.first = first;
    dest.second = second;
    dest.shift = shift;
    dest.vector = vector;
    dest.distance = distance;
    dest.mask = mask;
    const std::size_t n = walk.pull(dest, done);
    for (std::size_t p = 0; p < n; ++p) {
      got.insert(Key{first[p], second[p], shift[3 * p], shift[3 * p + 1], shift[3 * p + 2]});
    }
  }
  REQUIRE(got == from_table(table));
}

TEST_CASE("the C ABI round-trips a stored list and a stream", "[nlist][api]") {
  REQUIRE(rgpot_nlist_abi_version() == RGPOT_NLIST_ABI);
  const double box[9] = {5, 0, 0, 0, 5, 0, 0.4, 0, 5};
  const double pos[] = {0.2, 0.2, 0.2, 1.4, 0.3, 0.5, 3.0, 3.0, 1.0};
  rgpot_nlist_request request;
  rgpot_nlist_request_init(&request);
  request.cutoff = 3.0;
  REQUIRE(request.periodic[0] == 1);
  REQUIRE(request.want_distances == 1);
  REQUIRE(request.shape == 0);
  REQUIRE(request.shell == 0);
  REQUIRE(request.bytes == sizeof(request));
  std::size_t count = 0;
  std::size_t needed = 0;
  REQUIRE(rgpot_nlist_build(pos, 3, box, &request, nullptr, &count, &needed) ==
          RGPOT_NLIST_OK);
  REQUIRE(needed > 0);
  std::vector<std::int32_t> first(needed), second(needed), shift(needed * 3);
  std::vector<double> vector(needed * 3), distance(needed);
  std::vector<std::uint8_t> mask(needed);
  rgpot_nlist_soa soa{};
  soa.bytes = sizeof(soa);
  soa.first = first.data();
  soa.second = second.data();
  soa.shift = shift.data();
  soa.vector = vector.data();
  soa.distance = distance.data();
  soa.mask = mask.data();
  soa.cap = needed;
  REQUIRE(rgpot_nlist_build(pos, 3, box, &request, &soa, &count, &needed) == RGPOT_NLIST_OK);
  REQUIRE(count == needed);
  soa.cap = 1;
  REQUIRE(rgpot_nlist_build(pos, 3, box, &request, &soa, &count, &needed) == RGPOT_NLIST_SHORT);
  REQUIRE(count == 1);
  REQUIRE(needed > 1);
  REQUIRE(std::string(rgpot_nlist_strerror(RGPOT_NLIST_SHORT)).find("needed") !=
          std::string::npos);
  rgpot_nlist_request zero{};
  REQUIRE(rgpot_nlist_build(pos, 3, box, &zero, nullptr, &count, &needed) == RGPOT_NLIST_ERR_ABI);
  request.shape = 9;
  REQUIRE(rgpot_nlist_build(pos, 3, box, &request, nullptr, &count, &needed) ==
          RGPOT_NLIST_SHAPE);
  request.shape = 1;
  request.shell = 0;
  REQUIRE(rgpot_nlist_build(pos, 3, box, &request, nullptr, &count, &needed) ==
          RGPOT_NLIST_SHAPE);
  request.shape = 0;
  request.shell = 0;
  request.cutoff = -1;
  REQUIRE(rgpot_nlist_build(pos, 3, box, &request, nullptr, &count, &needed) ==
          RGPOT_NLIST_CUTOFF);
  request.cutoff = 3.0;
  soa.cap = needed;
  REQUIRE(rgpot_nlist_build(pos, 3, box, &request, &soa, &count, &needed) == RGPOT_NLIST_OK);
  rgpot_nlist_walk *walk = nullptr;
  REQUIRE(rgpot_nlist_walk_open(pos, 3, box, &request, &walk) == RGPOT_NLIST_OK);
  REQUIRE(walk != nullptr);
  std::vector<std::int32_t> streamed_first, streamed_second, streamed_shift;
  int32_t done = 0;
  while (done == 0) {
    std::int32_t chunk_first[2], chunk_second[2], chunk_shift[6];
    double chunk_vector[6], chunk_distance[2];
    std::uint8_t chunk_mask[2];
    rgpot_nlist_soa chunk{};
    chunk.bytes = sizeof(chunk);
    chunk.first = chunk_first;
    chunk.second = chunk_second;
    chunk.shift = chunk_shift;
    chunk.vector = chunk_vector;
    chunk.distance = chunk_distance;
    chunk.mask = chunk_mask;
    chunk.cap = 2;
    std::size_t n = 0;
    REQUIRE(rgpot_nlist_walk_pull(walk, &chunk, &n, &done) == RGPOT_NLIST_OK);
    streamed_first.insert(streamed_first.end(), chunk_first, chunk_first + n);
    streamed_second.insert(streamed_second.end(), chunk_second, chunk_second + n);
    streamed_shift.insert(streamed_shift.end(), chunk_shift, chunk_shift + 3 * n);
  }
  rgpot_nlist_walk_close(walk);
  REQUIRE(streamed_first.size() == first.size());
  for (std::size_t p = 0; p < first.size(); ++p) {
    REQUIRE(streamed_first[p] == first[p]);
    REQUIRE(streamed_second[p] == second[p]);
    REQUIRE(streamed_shift[3 * p] == shift[3 * p]);
    REQUIRE(streamed_shift[3 * p + 1] == shift[3 * p + 1]);
    REQUIRE(streamed_shift[3 * p + 2] == shift[3 * p + 2]);
  }
}

TEST_CASE("a rejected geometry throws and keeps the previous table", "[nlist][api]") {
  const double box[9] = {4, 0, 0, 0, 4, 0, 0, 0, 4};
  const std::vector<double> pos = {0, 0, 0, 1, 0, 0};
  rgpot::nlist::Query query;
  query.cutoff = 1.5;
  rgpot::nlist::Builder builder;
  builder.update(pos.data(), pos.size() / 3, box, query);
  const auto n = builder.table().size();
  REQUIRE(n > 0);
  query.cutoff = -1;
  REQUIRE_THROWS_AS(builder.update(pos.data(), pos.size() / 3, box, query),
                    rgpot::nlist::Error);
  REQUIRE(builder.table().size() == n);
}

TEST_CASE("a cluster of four expands to the full list", "[nlist][api]") {
  const double box[9] = {4.0, 0.2, 0.1, 0.3, 4.5, -0.2, 0.05, 0.15, 4.2};
  const std::vector<double> pos = {0.2, 0.3, 0.4, 3.6, 0.5, 0.2, 1.0, 3.8, 0.7,
                                   2.2, 2.1, 3.9};
  rgpot::nlist::Query query;
  query.cutoff = 3.2;
  query.shape = rgpot::nlist::Cluster4{};
  query.periodic = {{true, true, false}};
  rgpot::nlist::Table clustered;
  rgpot::nlist::build(clustered, pos.data(), pos.size() / 3, box, query);
  query.shape = rgpot::nlist::Pairs{rgpot::nlist::Shell::Full};
  rgpot::nlist::Table full;
  rgpot::nlist::build(full, pos.data(), pos.size() / 3, box, query);
  REQUIRE(from_table(clustered) == from_table(full));
  for (std::uint8_t mask : clustered.mask) {
    REQUIRE(mask != 0);
    REQUIRE((mask & 0xf0) == 0);
  }
}

TEST_CASE("a wide orthorhombic cell matches the brute force", "[nlist][api]") {
  // The box is many cutoffs across, so the query takes the cell-shift path.
  const double box[9] = {40, 0, 0, 0, 36, 0, 0, 0, 44};
  std::vector<double> pos;
  for (int i = 0; i < 24; ++i) {
    pos.push_back(1.5 + (i % 5) * 7.0);
    pos.push_back(2.0 + (i % 7) * 4.5);
    pos.push_back(0.5 + (i % 4) * 9.0);
  }
  const std::array<bool, 3> periodic{{true, true, true}};
  for (bool full : {false, true}) {
    rgpot::nlist::Query query;
    query.cutoff = 6.0;
    query.skin = 0;
    query.shape = rgpot::nlist::Pairs{full ? rgpot::nlist::Shell::Full
                                          : rgpot::nlist::Shell::Half};
    query.cut = rgpot::nlist::Cut::Strict;
    query.periodic = periodic;
    rgpot::nlist::Table table;
    rgpot::nlist::build(table, pos.data(), pos.size() / 3, box, query);
    REQUIRE(from_table(table) == brute(pos, box, 6.0, full, true, periodic));
  }
}

TEST_CASE("a large orthorhombic list is not an all-pairs scan", "[nlist][api][perf]") {
  // 30^3 FCC cells, the 108000-atom copper block. An all-pairs scan does
  // not finish in seconds. A cell list does, and each atom has the twelve,
  // six and twenty-four shells inside 5.
  constexpr int ncell = 30;
  constexpr double lattice = 3.615;
  const double basis[4][3] = {{0, 0, 0}, {0, 0.5, 0.5}, {0.5, 0, 0.5}, {0.5, 0.5, 0}};
  std::vector<double> pos;
  pos.reserve(static_cast<std::size_t>(ncell) * ncell * ncell * 4 * 3);
  for (int ix = 0; ix < ncell; ++ix) {
    for (int iy = 0; iy < ncell; ++iy) {
      for (int iz = 0; iz < ncell; ++iz) {
        for (const auto &b : basis) {
          pos.push_back((ix + b[0]) * lattice);
          pos.push_back((iy + b[1]) * lattice);
          pos.push_back((iz + b[2]) * lattice);
        }
      }
    }
  }
  const double edge = ncell * lattice;
  const double box[9] = {edge, 0, 0, 0, edge, 0, 0, 0, edge};
  rgpot::nlist::Query query;
  query.cutoff = 5.0;
  query.skin = 0;
  query.shape = rgpot::nlist::Pairs{rgpot::nlist::Shell::Half};
  query.cut = rgpot::nlist::Cut::Strict;
  rgpot::nlist::Table table;
  const auto t0 = std::chrono::steady_clock::now();
  rgpot::nlist::build(table, pos.data(), pos.size() / 3, box, query);
  const auto seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  REQUIRE(pos.size() / 3 == 108000);
  REQUIRE(table.size() == 108000 * 21);
  REQUIRE(seconds < 5.0);
  for (std::size_t p = 0; p < 64; ++p) {
    const int i = table.first[p];
    const int j = table.second[p];
    const double vx = pos[3 * j] - pos[3 * i] + table.shift[3 * p] * box[0];
    const double vy = pos[3 * j + 1] - pos[3 * i + 1] + table.shift[3 * p + 1] * box[4];
    const double vz = pos[3 * j + 2] - pos[3 * i + 2] + table.shift[3 * p + 2] * box[8];
    REQUIRE(std::abs(vx - table.vector[3 * p]) < 1e-9);
    REQUIRE(std::abs(vy - table.vector[3 * p + 1]) < 1e-9);
    REQUIRE(std::abs(vz - table.vector[3 * p + 2]) < 1e-9);
    REQUIRE(std::abs(std::sqrt(vx * vx + vy * vy + vz * vz) - table.distance[p]) < 1e-9);
    REQUIRE(table.distance[p] < 5.0);
    REQUIRE(table.distance[p] > 0.0);
  }
}

TEST_CASE("a skin rebuild keeps the same pairs", "[nlist][api]") {
  const double box[9] = {8, 0, 0, 0, 8, 0, 0, 0, 8};
  std::vector<double> pos = {1, 1, 1, 2, 1, 1, 1, 3, 1};
  rgpot::nlist::Query query;
  query.cutoff = 2.5;
  query.skin = 0.4;
  query.shape = rgpot::nlist::Pairs{rgpot::nlist::Shell::Half};
  query.cut = rgpot::nlist::Cut::Loose;
  rgpot::nlist::Builder builder;
  builder.update(pos.data(), pos.size() / 3, box, query);
  const auto first = builder.builds();
  pos[0] += 0.05;
  builder.update(pos.data(), pos.size() / 3, box, query);
  REQUIRE(builder.reused());
  REQUIRE(builder.builds() == first);
}

TEST_CASE("the skin rebuild follows half the skin", "[nlist][api]") {
  const double box[9] = {10, 0, 0, 0, 10, 0, 0, 0, 10};
  // The first atom sits at the origin so a displacement of one ulp past
  // half the skin is representable. Adding that ulp to a coordinate near
  // 1 rounds back to the exact half-skin position.
  const std::vector<double> origin = {0, 0, 0, 2.2, 1, 1, 1, 3.4, 1, 4, 4, 1};
  const double skin = 0.5;
  const double half = 0.5 * skin;
  rgpot::nlist::Query query;
  query.cutoff = 3.0;
  query.skin = skin;
  query.shape = rgpot::nlist::Pairs{rgpot::nlist::Shell::Full};
  query.cut = rgpot::nlist::Cut::Loose;
  query.fields = {true, true, true};

  rgpot::nlist::Builder edge;
  auto pos = origin;
  edge.update(pos.data(), pos.size() / 3, box, query);
  REQUIRE(edge.builds() == 1);
  pos[0] = half;
  edge.update(pos.data(), pos.size() / 3, box, query);
  REQUIRE(edge.reused());
  REQUIRE(edge.builds() == 1);
  pos[0] = std::nextafter(half, 1.0);
  edge.update(pos.data(), pos.size() / 3, box, query);
  REQUIRE_FALSE(edge.reused());
  REQUIRE(edge.builds() == 2);

  rgpot::nlist::Builder builder;
  pos = origin;
  builder.update(pos.data(), pos.size() / 3, box, query);
  pos[0] = 0.5 * half;
  builder.update(pos.data(), pos.size() / 3, box, query);
  REQUIRE(builder.reused());
  const auto &kept = builder.table();
  rgpot::nlist::Table fresh;
  rgpot::nlist::build(fresh, pos.data(), pos.size() / 3, box, query);
  REQUIRE(kept.size() == fresh.size());

  auto key_of = [](const rgpot::nlist::Table &table, std::size_t p) {
    return std::tuple{table.first[p], table.second[p], table.shift[3 * p],
                      table.shift[3 * p + 1], table.shift[3 * p + 2]};
  };
  std::map<std::tuple<int, int, int, int, int>, std::array<double, 4>> got;
  for (std::size_t p = 0; p < kept.size(); ++p) {
    const int i = kept.first[p];
    const int j = kept.second[p];
    const int sa = kept.shift[3 * p];
    const int sb = kept.shift[3 * p + 1];
    const int sc = kept.shift[3 * p + 2];
    const double vx = pos[3 * static_cast<std::size_t>(j)] - pos[3 * static_cast<std::size_t>(i)] +
                      sa * box[0] + sb * box[3] + sc * box[6];
    const double vy = pos[3 * static_cast<std::size_t>(j) + 1] -
                      pos[3 * static_cast<std::size_t>(i) + 1] + sa * box[1] + sb * box[4] +
                      sc * box[7];
    const double vz = pos[3 * static_cast<std::size_t>(j) + 2] -
                      pos[3 * static_cast<std::size_t>(i) + 2] + sa * box[2] + sb * box[5] +
                      sc * box[8];
    got[{i, j, sa, sb, sc}] = {vx, vy, vz, std::sqrt(vx * vx + vy * vy + vz * vz)};
  }
  for (std::size_t p = 0; p < fresh.size(); ++p) {
    const auto key = key_of(fresh, p);
    REQUIRE(got.count(key) == 1);
    REQUIRE(std::abs(got[key][0] - fresh.vector[3 * p]) < 1e-12);
    REQUIRE(std::abs(got[key][1] - fresh.vector[3 * p + 1]) < 1e-12);
    REQUIRE(std::abs(got[key][2] - fresh.vector[3 * p + 2]) < 1e-12);
    REQUIRE(std::abs(got[key][3] - fresh.distance[p]) < 1e-12);
  }
}

TEST_CASE("a positive skin with a strict cut is rejected", "[nlist][api]") {
  const double box[9] = {10, 0, 0, 0, 10, 0, 0, 0, 10};
  const double pos[] = {0, 0, 0, 1, 0, 0};
  rgpot::nlist::Query query;
  query.cutoff = 2.0;
  query.skin = 0.2;
  query.cut = rgpot::nlist::Cut::Strict;
  rgpot::nlist::Table table;
  REQUIRE_THROWS_AS(rgpot::nlist::build(table, pos, 2, box, query), rgpot::nlist::Error);
  try {
    rgpot::nlist::build(table, pos, 2, box, query);
  } catch (const rgpot::nlist::Error &err) {
    REQUIRE(err.code() == rgpot::nlist::Code::Skin);
  }
}

TEST_CASE("a batch keeps the systems apart", "[nlist][api]") {
  const double positions[] = {0, 0, 0, 1.2, 0, 0, 0, 0, 0, 1.2, 0, 0};
  const std::int32_t count[] = {2, 2};
  const double box[] = {8, 0, 0, 0, 8, 0, 0, 0, 8, 9, 0, 0, 0, 9, 0, 0, 0, 9};
  rgpot::nlist::Query query;
  query.cutoff = 2.0;
  rgpot::nlist::Systems systems{positions, count, 2, box};
  rgpot::nlist::Batch batch;
  rgpot::nlist::build(batch, systems, query);
  REQUIRE(batch.pairs.size() == 2);
  REQUIRE(batch.system[0] == 0);
  REQUIRE(batch.system[1] == 1);
  REQUIRE(batch.pairs.first[1] >= 2);
  rgpot::nlist::Table one;
  rgpot::nlist::build(one, positions, 2, box, query);
  REQUIRE(one.size() == 1);
  REQUIRE(batch.pairs.distance[0] == one.distance[0]);
}

TEST_CASE("one atom per thread matches the stored list", "[nlist][api][device]") {
  constexpr int ncell = 8;
  constexpr double lattice = 3.615;
  const double basis[4][3] = {{0, 0, 0}, {0, 0.5, 0.5}, {0.5, 0, 0.5}, {0.5, 0.5, 0}};
  std::vector<double> pos;
  for (int ix = 0; ix < ncell; ++ix) {
    for (int iy = 0; iy < ncell; ++iy) {
      for (int iz = 0; iz < ncell; ++iz) {
        for (const auto &b : basis) {
          pos.push_back((ix + b[0]) * lattice);
          pos.push_back((iy + b[1]) * lattice);
          pos.push_back((iz + b[2]) * lattice);
        }
      }
    }
  }
  const double edge = ncell * lattice;
  const double box[9] = {edge, 0, 0, 0, edge, 0, 0, 0, edge};
  rgpot::nlist::Query query;
  query.cutoff = 5.0;
  query.shape = rgpot::nlist::Pairs{rgpot::nlist::Shell::Half};
  rgpot::nlist::Table bulk;
  rgpot::nlist::Table by_atom;
  rgpot::nlist::build(bulk, pos.data(), pos.size() / 3, box, query);
  rgpot::nlist::buildByAtom(by_atom, pos.data(), pos.size() / 3, box, query);
  REQUIRE(pos.size() / 3 >= 512);
  REQUIRE(from_table(bulk) == from_table(by_atom));
}

TEST_CASE("the fortran bridge reuses the builder and refreshes vectors", "[nlist][api]") {
  double pos[] = {0, 0, 0, 1.2, 0, 0};
  const double box[9] = {8, 0, 0, 0, 8, 0, 0, 0, 8};
  const int32_t periodic[3] = {1, 1, 1};
  auto *list = rgpot_nlist_fortran_open();
  REQUIRE(list != nullptr);
  size_t needed = 0;
  int32_t reused = 1;
  REQUIRE(rgpot_nlist_fortran_build(list, pos, 2, box, 2.0, 0.4, periodic, nullptr, nullptr,
                                    nullptr, nullptr, 0, &needed, &reused) == RGPOT_NLIST_OK);
  REQUIRE(needed == 2);
  REQUIRE(reused == 0);
  std::vector<int32_t> first(needed), second(needed);
  std::vector<double> vector(needed * 3), distance(needed);
  REQUIRE(rgpot_nlist_fortran_build(list, pos, 2, box, 2.0, 0.4, periodic, first.data(),
                                    second.data(), vector.data(), distance.data(), needed, &needed,
                                    &reused) == RGPOT_NLIST_OK);
  REQUIRE(distance[0] == Catch::Approx(1.2));
  pos[0] = 0.05;
  REQUIRE(rgpot_nlist_fortran_build(list, pos, 2, box, 2.0, 0.4, periodic, first.data(),
                                    second.data(), vector.data(), distance.data(), needed, &needed,
                                    &reused) == RGPOT_NLIST_OK);
  REQUIRE(reused == 1);
  REQUIRE(distance[0] == Catch::Approx(1.15));
  const auto view_n = needed;
  REQUIRE(view_n == 2);
  rgpot_nlist_fortran_close(list);
}

TEST_CASE("eight threads match the serial list", "[nlist][api]") {
  constexpr int ncell = 14;
  constexpr double lattice = 3.615;
  const double basis[4][3] = {{0, 0, 0}, {0, 0.5, 0.5}, {0.5, 0, 0.5}, {0.5, 0.5, 0}};
  std::vector<double> pos;
  for (int ix = 0; ix < ncell; ++ix) {
    for (int iy = 0; iy < ncell; ++iy) {
      for (int iz = 0; iz < ncell; ++iz) {
        for (const auto &b : basis) {
          pos.push_back((ix + b[0]) * lattice);
          pos.push_back((iy + b[1]) * lattice);
          pos.push_back((iz + b[2]) * lattice);
        }
      }
    }
  }
  const double edge = ncell * lattice;
  const double box[9] = {edge, 0, 0, 0, edge, 0, 0, 0, edge};
  rgpot::nlist::Query query;
  query.cutoff = 5.0;
  query.shape = rgpot::nlist::Pairs{rgpot::nlist::Shell::Half};
  setenv("OMP_NUM_THREADS", "1", 1);
  rgpot::nlist::Table serial;
  rgpot::nlist::build(serial, pos.data(), pos.size() / 3, box, query);
  setenv("OMP_NUM_THREADS", "8", 1);
  rgpot::nlist::Table threaded;
  rgpot::nlist::build(threaded, pos.data(), pos.size() / 3, box, query);
  setenv("OMP_NUM_THREADS", "1", 1);
  REQUIRE(pos.size() / 3 >= 8192);
  REQUIRE(serial.size() == threaded.size());
  REQUIRE(from_table(serial) == from_table(threaded));
}

TEST_CASE("a pair-type cutoff drops the long pair and keeps the short one", "[nlist][api]") {
  const double pos[] = {0, 0, 0, 1, 0, 0, 3, 0, 0};
  const double box[9] = {20, 0, 0, 0, 20, 0, 0, 0, 20};
  const std::int32_t types[] = {0, 1, 0};
  const double matrix[] = {3.5, 1.5, 1.5, 3.5};
  rgpot::nlist::Query query;
  query.cutoff = 3.5;
  query.shape = rgpot::nlist::Pairs{rgpot::nlist::Shell::Half};
  rgpot::nlist::PairCutoffs cuts;
  cuts.types = types;
  cuts.cutoff = matrix;
  cuts.ntypes = 2;
  rgpot::nlist::Table typed;
  rgpot::nlist::build(typed, pos, 3, box, query, cuts);
  const auto got = from_table(typed);
  REQUIRE(got.count(Key{0, 1, 0, 0, 0}) == 1);
  REQUIRE(got.count(Key{0, 2, 0, 0, 0}) == 1);
  REQUIRE(got.count(Key{1, 2, 0, 0, 0}) == 0);
  REQUIRE(got.size() == 2);
  rgpot::nlist::Table scalar;
  rgpot::nlist::build(scalar, pos, 3, box, query);
  REQUIRE(from_table(scalar).count(Key{1, 2, 0, 0, 0}) == 1);

  const double same[] = {3.5, 3.5, 3.5, 3.5};
  cuts.cutoff = same;
  rgpot::nlist::Table uniform;
  rgpot::nlist::build(uniform, pos, 3, box, query, cuts);
  REQUIRE(from_table(uniform) == from_table(scalar));
  for (std::size_t p = 0; p < scalar.size(); ++p) {
    REQUIRE(uniform.distance[p] == scalar.distance[p]);
    REQUIRE(uniform.vector[3 * p] == scalar.vector[3 * p]);
  }

  std::set<Key> streamed;
  rgpot::nlist::each(pos, 3, box, query,
                     [&](std::int32_t i, std::int32_t j, int sa, int sb, int sc, double, double,
                         double, double) { streamed.insert(Key{i, j, sa, sb, sc}); },
                     cuts);
  REQUIRE(streamed == from_table(uniform));

  rgpot::nlist::Builder builder;
  builder.update(pos, 3, box, query, cuts);
  REQUIRE(builder.reused() == false);
  builder.update(pos, 3, box, query, cuts);
  REQUIRE(builder.reused() == true);
  const double tighter[] = {3.5, 0.5, 0.5, 3.5};
  cuts.cutoff = tighter;
  builder.update(pos, 3, box, query, cuts);
  REQUIRE(builder.reused() == false);
  REQUIRE(from_table(builder.table()).count(Key{0, 1, 0, 0, 0}) == 0);
}

TEST_CASE("the kernel callback matches the internal cell walk", "[nlist][api]") {
  const double pos[] = {0.1, 0.2, 0.3, 1.4, 0.2, 0.4, 0.2, 1.6, 0.5, 2.2, 1.1, 0.4};
  const double box[9] = {6, 0, 0, 0, 6, 0, 0, 0, 6};
  const std::array<bool, 3> periodic{{true, true, true}};
  rgpot::nlist::Neighbours raw;
  rgpot::nlist::updateCells(raw, pos, 4, box, 2.5, 0.4, rgpot::nlist::Image::Round,
                            rgpot::nlist::Mode::Stream, periodic);
  rgpot::nlist::Cells cells;
  cells.update(pos, 4, box, 2.5, 0.4, rgpot::nlist::Image::Round, rgpot::nlist::Mode::Stream,
               periodic);
  struct Hit {
    std::int32_t i;
    std::int32_t j;
    double dx, dy, dz, r2;
    bool operator==(const Hit &o) const {
      return i == o.i && j == o.j && dx == o.dx && dy == o.dy && dz == o.dz && r2 == o.r2;
    }
  };
  std::vector<Hit> a;
  std::vector<Hit> b;
  auto take = [](std::vector<Hit> &out) {
    return [&](std::int32_t i, std::int32_t j, double dx, double dy, double dz, double r2) {
      out.push_back(Hit{i, j, dx, dy, dz, r2});
    };
  };
  rgpot::nlist::visitHalf(raw, 0, 4, take(a));
  cells.forHalf(0, 4, take(b));
  REQUIRE(a == b);
  REQUIRE(!a.empty());

  const std::int32_t types[] = {0, 0, 0, 0};
  const double one[] = {1.0};
  rgpot::nlist::PairCutoffs cuts;
  cuts.types = types;
  cuts.cutoff = one;
  cuts.ntypes = 1;
  rgpot::nlist::Neighbours wide;
  rgpot::nlist::updateCells(wide, pos, 4, box, 2.5, 0.0, rgpot::nlist::Image::Round,
                            rgpot::nlist::Mode::Stream, periodic);
  rgpot::nlist::Cells narrow;
  narrow.update(pos, 4, box, 2.5, 0.0, rgpot::nlist::Image::Round, rgpot::nlist::Mode::Stream,
                periodic, cuts);
  std::vector<Hit> wide_hits;
  std::vector<Hit> narrow_hits;
  rgpot::nlist::visitHalf(wide, 0, 4, take(wide_hits));
  narrow.forHalf(0, 4, take(narrow_hits));
  std::vector<Hit> kept;
  for (const Hit &hit : wide_hits) {
    if (hit.r2 <= 1.0) {
      kept.push_back(hit);
    }
  }
  REQUIRE(narrow_hits == kept);
  REQUIRE(narrow_hits.size() < wide_hits.size());
}

TEST_CASE("the C callback and the pair matrix agree with build", "[nlist][api]") {
  const double pos[] = {0, 0, 0, 1, 0, 0, 3, 0, 0};
  const double box[9] = {20, 0, 0, 0, 20, 0, 0, 0, 20};
  const std::int32_t types[] = {0, 1, 0};
  const double matrix[] = {3.5, 1.5, 1.5, 3.5};
  rgpot_nlist_request request;
  rgpot_nlist_request_init(&request);
  request.cutoff = 3.5;
  request.want_shifts = 1;
  request.want_vectors = 0;
  request.want_distances = 1;
  request.types = types;
  request.pair_cutoff = matrix;
  request.ntypes = 2;
  size_t needed = 0;
  REQUIRE(rgpot_nlist_build(pos, 3, box, &request, nullptr, nullptr, &needed) == RGPOT_NLIST_OK);
  REQUIRE(needed == 2);
  struct Tally {
    int n;
  } tally{0};
  const auto rc = rgpot_nlist_each(
      pos, 3, box, &request,
      [](void *user, int32_t, int32_t, int32_t, int32_t, int32_t, double, double, double, double) {
        static_cast<Tally *>(user)->n += 1;
      },
      &tally);
  REQUIRE(rc == RGPOT_NLIST_OK);
  REQUIRE(tally.n == 2);
}

TEST_CASE("a sweep serves half and full from one skin", "[nlist][api]") {
  const double box[9] = {12, 0, 0, 0, 12, 0, 0, 0, 12};
  const std::vector<double> pos = {0.4, 0.5, 0.6, 1.5, 0.4, 0.5, 0.3, 1.7, 0.4, 2.4, 1.2, 0.5};
  rgpot::nlist::Query query;
  query.cutoff = 2.2;
  query.skin = 0.3;
  query.cut = rgpot::nlist::Cut::Loose;
  rgpot::nlist::Table half_table;
  rgpot::nlist::Table full_table;
  query.shape = rgpot::nlist::Pairs{rgpot::nlist::Shell::Half};
  rgpot::nlist::build(half_table, pos.data(), pos.size() / 3, box, query);
  query.shape = rgpot::nlist::Pairs{rgpot::nlist::Shell::Full};
  rgpot::nlist::build(full_table, pos.data(), pos.size() / 3, box, query);
  query.shape = rgpot::nlist::Pairs{};
  rgpot::nlist::Sweep sweep;
  sweep.update(pos.data(), pos.size() / 3, box, query);
  REQUIRE(sweep.reused() == false);
  auto collect = [](rgpot::nlist::Sweep &list, rgpot::nlist::Shell shell) {
    std::set<Key> keys;
    list.each(shell, [&](std::int32_t i, std::int32_t j, int sa, int sb, int sc, double, double,
                         double, double) { keys.insert(Key{i, j, sa, sb, sc}); });
    return keys;
  };
  REQUIRE(collect(sweep, rgpot::nlist::Shell::Half) == from_table(half_table));
  REQUIRE(collect(sweep, rgpot::nlist::Shell::Full) == from_table(full_table));
  std::set<Key> threaded;
  std::mutex gate;
  sweep.eachTeam(rgpot::nlist::Shell::Half, 4,
                 [&](int, std::int32_t i, std::int32_t j, int sa, int sb, int sc, double, double,
                     double, double) {
                   std::lock_guard<std::mutex> lock(gate);
                   threaded.insert(Key{i, j, sa, sb, sc});
                 });
  REQUIRE(threaded == from_table(half_table));
  std::vector<double> nudged = pos;
  nudged[0] += 0.05;
  sweep.update(nudged.data(), nudged.size() / 3, box, query);
  REQUIRE(sweep.reused() == true);
  nudged[0] += 1.0;
  sweep.update(nudged.data(), nudged.size() / 3, box, query);
  REQUIRE(sweep.reused() == false);
}
