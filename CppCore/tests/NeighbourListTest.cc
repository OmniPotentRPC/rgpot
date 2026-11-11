// MIT License
// Copyright 2023--present rgpot developers

#include <array>
#include <cmath>
#include <cstdint>
#include <set>
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
      query.format = full ? rgpot::nlist::Format::Full : rgpot::nlist::Format::Half;
      query.cut = strict ? rgpot::nlist::Cut::Strict : rgpot::nlist::Cut::Loose;
      query.periodic = periodic;
      query.mode = rgpot::nlist::Mode::Stored;
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
  query.format = rgpot::nlist::Format::Full;
  query.cut = rgpot::nlist::Cut::Strict;
  query.mode = rgpot::nlist::Mode::Stored;
  rgpot::nlist::Table table;
  rgpot::nlist::build(table, pos.data(), pos.size() / 3, box, query);
  rgpot::nlist::Walk walk(pos.data(), pos.size() / 3, box, query);
  std::vector<std::int32_t> first(2), second(2), shift(6);
  std::vector<double> vector(6), distance(2);
  std::vector<std::uint8_t> mask(2);
  std::set<Key> got;
  bool done = false;
  while (!done) {
    rgpot::nlist::OutSoA dest{first.data(),   second.data(), shift.data(),
                              vector.data(), distance.data(), mask.data()};
    const std::size_t n = walk.pull(dest, 2, done);
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
  rgpot_nlist_request request{};
  request.abi = RGPOT_NLIST_ABI;
  request.cutoff = 3.0;
  request.skin = 0;
  request.format = 0;
  request.cut = 0;
  request.periodic[0] = request.periodic[1] = request.periodic[2] = 1;
  request.want_shifts = 1;
  request.want_vectors = 1;
  request.want_distances = 1;
  request.mode = 1;
  std::size_t count = 0;
  std::size_t needed = 0;
  REQUIRE(rgpot_nlist_build(pos, 3, box, &request, nullptr, 0, &count, &needed) == 0);
  REQUIRE(needed > 0);
  std::vector<rgpot_nlist_pair> pairs(needed);
  REQUIRE(rgpot_nlist_build(pos, 3, box, &request, pairs.data(), pairs.size(), &count,
                            &needed) == 0);
  REQUIRE(count == needed);
  rgpot_nlist_walk *walk = rgpot_nlist_walk_open(pos, 3, box, &request);
  REQUIRE(walk != nullptr);
  std::vector<rgpot_nlist_pair> streamed;
  int32_t done = 0;
  while (done == 0) {
    rgpot_nlist_pair chunk[2];
    std::size_t n = 0;
    REQUIRE(rgpot_nlist_walk_pull(walk, chunk, 2, &n, &done) == 0);
    streamed.insert(streamed.end(), chunk, chunk + n);
  }
  rgpot_nlist_walk_close(walk);
  REQUIRE(streamed.size() == pairs.size());
  for (std::size_t p = 0; p < pairs.size(); ++p) {
    REQUIRE(streamed[p].first == pairs[p].first);
    REQUIRE(streamed[p].second == pairs[p].second);
    REQUIRE(streamed[p].shift[0] == pairs[p].shift[0]);
    REQUIRE(streamed[p].shift[1] == pairs[p].shift[1]);
    REQUIRE(streamed[p].shift[2] == pairs[p].shift[2]);
  }
}

TEST_CASE("a skin rebuild keeps the same pairs", "[nlist][api]") {
  const double box[9] = {8, 0, 0, 0, 8, 0, 0, 0, 8};
  std::vector<double> pos = {1, 1, 1, 2, 1, 1, 1, 3, 1};
  rgpot::nlist::Query query;
  query.cutoff = 2.5;
  query.skin = 0.4;
  query.format = rgpot::nlist::Format::Half;
  query.cut = rgpot::nlist::Cut::Loose;
  query.mode = rgpot::nlist::Mode::Stored;
  rgpot::nlist::Builder builder;
  builder.update(pos.data(), pos.size() / 3, box, query);
  const auto first = builder.builds();
  pos[0] += 0.05;
  builder.update(pos.data(), pos.size() / 3, box, query);
  REQUIRE(builder.reused());
  REQUIRE(builder.builds() == first);
}
