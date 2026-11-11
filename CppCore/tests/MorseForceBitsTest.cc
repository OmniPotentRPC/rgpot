// MIT License
// Copyright 2023--present rgpot developers
//
// Bit pins for the Morse force evaluation the RPC server runs: energy,
// Cartesian forces and the Cauchy stress. The warm call (the pair list
// already captured) is the one potserv repeats, and it has to match the
// cold call and these recorded patterns exactly.

#include <array>
#include <cstdint>
#include <cstring>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "examples/bench_systems.hpp"
#include "rgpot/Morse/MorsePot.hpp"
#include "rgpot/types/AtomMatrix.hpp"

namespace {

std::uint64_t bits(double x) {
  std::uint64_t u = 0;
  std::memcpy(&u, &x, sizeof(u));
  return u;
}

void expect_bits(rgpot::MorsePot &pot, const rgpot::types::AtomMatrix &pos,
                 const std::vector<int> &types, const double box[9],
                 std::uint64_t energy, const std::uint64_t stress[9],
                 const std::uint64_t *forces, std::size_t nforce) {
  const std::array<std::array<double, 3>, 3> cell{{
      {box[0], box[1], box[2]},
      {box[3], box[4], box[5]},
      {box[6], box[7], box[8]},
  }};
  // operator() is the entry potserv's calculate() uses.
  auto cold = pot(pos, types, cell);
  auto warm = pot(pos, types, cell);
  REQUIRE(bits(std::get<0>(cold)) == energy);
  REQUIRE(bits(std::get<0>(warm)) == energy);
  const auto &fc = std::get<1>(cold);
  const auto &fw = std::get<1>(warm);
  REQUIRE(fc.size() == nforce);
  for (std::size_t i = 0; i < nforce; ++i) {
    REQUIRE(bits(fc.data()[i]) == forces[i]);
    REQUIRE(bits(fw.data()[i]) == forces[i]);
  }

  std::vector<double> raw(nforce);
  rgpot::ForceOut out{};
  out.F = raw.data();
  const rgpot::ForceInput in{pos.rows(), pos.data(), types.data(), box};
  pot.forceImpl(in, &out);
  REQUIRE(bits(out.energy) == energy);
  REQUIRE(out.has_stress == 1);
  for (int k = 0; k < 9; ++k) {
    REQUIRE(bits(out.stress[k]) == stress[k]);
  }
  for (std::size_t i = 0; i < nforce; ++i) {
    REQUIRE(bits(raw[i]) == forces[i]);
  }
}

} // namespace

TEST_CASE("Morse pair at equilibrium keeps its recorded bits", "[MorsePot][bits]") {
  rgpot::MorsePot pot;
  const rgpot::types::AtomMatrix pos{{10.0, 10.0, 10.0}, {12.897, 10.0, 10.0}};
  const double box[9] = {40, 0, 0, 0, 40, 0, 0, 0, 40};
  const std::uint64_t stress[9] = {0x3bf448f358183c87, 0, 0, 0, 0, 0, 0, 0, 0};
  const std::uint64_t forces[6] = {0x3cdb5a0ef2363dd1, 0, 0, 0xbcdb5a0ef2363dd1, 0, 0};
  expect_bits(pot, pos, {78, 78}, box, 0xbfe6b9aad262a8ed, stress, forces, 6);
}

TEST_CASE("Morse pair across the cell keeps its recorded bits", "[MorsePot][bits]") {
  rgpot::MorsePot pot;
  const rgpot::types::AtomMatrix pos{{0.2, 0.1, -0.2}, {19.4, 0.4, 0.3}};
  const double box[9] = {20, 0, 0, 0, 21, 0, 0, 0, 22};
  const std::uint64_t stress[9] = {0xbfb1b56322c0574f, 0x3f9a9014b42082f0,
                                   0x3fa622bbeb706d1d, 0x3f9a9014b42082f0,
                                   0xbf83ec0f87186230, 0xbf909a0cf09451d2,
                                   0x3fa622bbeb706d1d, 0xbf909a0cf09451d2,
                                   0xbf9bab6ae64c885d};
  const std::uint64_t forces[6] = {0x4088f7ad75cf0f12, 0xc072b9c2185b4b4a,
                                  0xc07f3598d342d2cf, 0xc088f7ad75cf0f12,
                                  0x4072b9c2185b4b4a, 0x407f3598d342d2cf};
  expect_bits(pot, pos, {78, 78}, box, 0x40724e6cb4079cc1, stress, forces, 6);
}

TEST_CASE("Morse neb cluster keeps its recorded bits", "[MorsePot][bits]") {
  rgpot::MorsePot pot;
  const rgpot::types::AtomMatrix pos{
      {50.594227, 52.017165, 52.277482}, {51.578540, 52.642148, 51.195222},
      {51.243758, 52.919086, 52.218383}, {50.490127, 52.736378, 51.417663},
      {51.522643, 50.884535, 51.656125}, {50.927263, 51.743087, 51.272643},
      {51.240676, 50.960436, 50.573042}, {51.275727, 52.061877, 50.284160},
      {50.434110, 50.976423, 51.879062}, {51.695007, 51.919472, 52.038680},
      {51.363596, 52.199101, 53.064923}, {52.017483, 51.639074, 51.008389},
      {51.187843, 51.165217, 52.678225}};
  const double box[9] = {101.942400, 0, 0, 0, 103.142600, 0, 0, 0, 102.605500};
  const std::uint64_t stress[9] = {0xbf7e577aa5863a88, 0xbf058cf27dad2641,
                                   0x3f24540a27d89b84, 0xbf058cf27dad2641,
                                   0xbf8485ae5602ab48, 0xbf270ea903b7990d,
                                   0x3f24540a27d89b84, 0xbf270ea903b7990d,
                                   0xbf85c4089b79656c};
  const std::uint64_t forces[] = {
      0xc0a011e7dd56cd92, 0x4074dadb309af0a5, 0x408ff6a6b5ef3897, 0x40926d804b9c23e0,
      0x409d220958397292, 0xc088a3d1ce1316ea, 0x407377bf29ee0f56, 0x40a02d533ee71d3e,
      0x4088da63caf7bef8, 0xc09a2fea2fd9744a, 0x4093c68f2b9916fc, 0xc0862842222d53d8,
      0x409077e5d1b95af4, 0xc0a03bdbfae16a6b, 0x406eac8db09f857c, 0xc097a22c3358f5ce,
      0xc065b3b33581fe8c, 0xc08a6dfae5d76d0e, 0xc076314ab70bd08a, 0xc0983b57a2e97730,
      0xc0961ac949a17646, 0xc071c36ab642281e, 0x4085205959f9974b, 0xc09f3aba100eed1e,
      0xc09b7ce689c6d1b6, 0xc0951158c6abd561, 0xc037baa4926fff10, 0x40a036e765e3ce1c,
      0x406065e554f09ca0, 0x408780f124e835f2, 0x407cd06af3993b79, 0x407f306d0d0d408d,
      0x409e9d2056be00c5, 0x40a275b23b02c0f9, 0xc0704ba70f580ea4, 0xc0861d3ea71a1486,
      0x4068fab0e3af2b32, 0xc0965531dec74ab1, 0x409ac464cad99156};
  expect_bits(pot, pos, std::vector<int>(13, 1), box, 0x40bd3b1ddad55352, stress,
              forces, 39);
}

TEST_CASE("Switched Morse pair keeps its recorded bits", "[MorsePot][bits]") {
  rgpot::MorsePot pot(rgpot::MorseConfig{.cutoff = 5.0, .switch_width = 1.0});
  const rgpot::types::AtomMatrix pos{{0.0, 0.0, 0.0}, {4.3, 0.3, -0.2}};
  const double box[9] = {30, 0, 0, 0, 30, 0, 0, 0, 30};
  const std::uint64_t stress[9] = {0x3f0e5d7d3fe235a8, 0x3ed0f2b111cba6e0,
                                   0xbec698ec17ba33d7, 0x3ed0f2b111cba6e0,
                                   0x3e92eb30d2605b06, 0xbe8939966dd5ceb3,
                                   0xbec698ec17ba33d7, 0xbe8939966dd5ceb3,
                                   0x3e80d10ef3e3df23};
  const std::uint64_t forces[6] = {0x3fd746494b003c5c, 0x3f99fb1c3be87301,
                                  0xbf915212d29af757, 0xbfd746494b003c5c,
                                  0xbf99fb1c3be87301, 0x3f915212d29af757};
  expect_bits(pot, pos, {6, 8}, box, 0xbfbcee541fb473b6, stress, forces, 6);
}

TEST_CASE("Morse nanoparticle of 256 atoms keeps its recorded bits",
          "[MorsePot][bits]") {
  // 256 atoms is the first size whose warm pair walk issues four candidates
  // at a time. The hash covers energy, the nine stress components and every
  // force component (FNV-1a, offset basis, prime 1099511628211).
  rgpot::MorsePot pot;
  const auto system = rgpot_bench::nanoparticle(256, 3.92, 10.5, 78);
  rgpot::types::AtomMatrix pos(system.pos.size() / 3, 3);
  for (std::size_t i = 0; i < system.pos.size(); ++i) {
    pos.data()[i] = system.pos[i];
  }
  const std::vector<int> types(system.types.begin(), system.types.end());
  const std::array<std::array<double, 3>, 3> cell{{
      {system.box[0], system.box[1], system.box[2]},
      {system.box[3], system.box[4], system.box[5]},
      {system.box[6], system.box[7], system.box[8]},
  }};
  auto cold = pot(pos, types, cell);
  auto warm = pot(pos, types, cell);
  REQUIRE(bits(std::get<0>(cold)) == 0xc091607fa15daa85ull);
  REQUIRE(bits(std::get<0>(warm)) == 0xc091607fa15daa85ull);
  REQUIRE(std::get<1>(cold).size() == std::get<1>(warm).size());
  for (std::size_t i = 0; i < std::get<1>(warm).size(); ++i) {
    REQUIRE(bits(std::get<1>(cold).data()[i]) == bits(std::get<1>(warm).data()[i]));
  }

  std::vector<double> raw(system.pos.size());
  rgpot::ForceOut out{};
  out.F = raw.data();
  pot.forceImpl({pos.rows(), pos.data(), types.data(), system.box.data()}, &out);
  REQUIRE(bits(out.energy) == 0xc091607fa15daa85ull);
  const std::uint64_t stress[9] = {0xbf25de03f9bad5c2, 0x3f027d8edb5c135d,
                                   0x3ef4ffef41b5b3df, 0x3f027d8edb5c135d,
                                   0xbf25a221d4fe80f3, 0x3ee3d9235d736f4c,
                                   0x3ef4ffef41b5b3df, 0x3ee3d9235d736f4c,
                                   0xbf1d2d098e48b9ac};
  std::uint64_t fnv = 1469598103934665603ull;
  auto mix = [&](double x) {
    fnv ^= bits(x);
    fnv *= 1099511628211ull;
  };
  mix(out.energy);
  for (int k = 0; k < 9; ++k) {
    REQUIRE(bits(out.stress[k]) == stress[k]);
    mix(out.stress[k]);
  }
  for (double f : raw) {
    mix(f);
  }
  REQUIRE(fnv == 0x35dd6ebd1128cb9full);
  for (std::size_t i = 0; i < raw.size(); ++i) {
    REQUIRE(bits(raw[i]) == bits(std::get<1>(warm).data()[i]));
  }
}
