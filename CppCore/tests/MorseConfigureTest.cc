// MIT License
// Copyright 2023--present rgpot developers
//
// PotentialConfig.morse: the wire arm round-trips, its schema defaults are
// MorseConfig's, and configure on a Morse backend rebuilds the pot. Every
// other arm, including one this build's schema does not know, is refused
// by name.

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include <capnp/any.h>
#include <capnp/dynamic.h>
#include <capnp/message.h>
#include <capnp/schema.h>
#include <capnp/serialize.h>
#include <catch2/catch_all.hpp>

#include "rgpot/LennardJones/LJClusterPot.hpp"
#include "rgpot/LennardJones/LJPot.hpp"
#include "rgpot/Morse/MorsePot.hpp"
#include "rgpot/rpc/LJConfigure.hpp"
#include "rgpot/rpc/MorseConfigure.hpp"
#include "rgpot/rpc/PotentialConfigArm.hpp"
#include "rgpot/types/AtomMatrix.hpp"

using Catch::Matchers::ContainsSubstring;
using Catch::Matchers::WithinRel;
using rgpot::types::AtomMatrix;

namespace {

double pairEnergyAtRe(rgpot::MorsePot &pot) {
  const double re = pot.config().re;
  const AtomMatrix positions{{10.0, 10.0, 10.0}, {10.0 + re, 10.0, 10.0}};
  const std::vector<int> atmtypes{78, 78};
  const std::array<std::array<double, 3>, 3> box{
      {{40.0, 0.0, 0.0}, {0.0, 40.0, 0.0}, {0.0, 0.0, 40.0}}};
  auto [energy, forces, variance] = pot(positions, atmtypes, box);
  (void)forces;
  (void)variance;
  return energy;
}

} // namespace

TEST_CASE("PotentialConfig.morse round-trips at ordinal 10", "[morse][rpc]") {
  const auto field =
      capnp::Schema::from<::PotentialConfig>().getFieldByName("morse");
  REQUIRE(field.getProto().getOrdinal().getExplicit() == 10);

  capnp::MallocMessageBuilder builder;
  auto morse = builder.initRoot<::PotentialConfig>().initMorse();
  morse.setDe(1.25);
  morse.setA(1.5);
  morse.setRe(2.75);
  morse.setCutoff(8.0);
  const auto words = capnp::messageToFlatArray(builder);

  capnp::FlatArrayMessageReader reader(words.asPtr());
  const auto cfg = reader.getRoot<::PotentialConfig>();
  REQUIRE(cfg.which() == ::PotentialConfig::MORSE);
  const auto back = rgpot::rpc::morseConfigFrom(cfg.getMorse());
  REQUIRE(back.De == 1.25);
  REQUIRE(back.a == 1.5);
  REQUIRE(back.re == 2.75);
  REQUIRE(back.cutoff == 8.0);
}

TEST_CASE("MorseParams defaults are the MorseConfig defaults", "[morse][rpc]") {
  capnp::MallocMessageBuilder builder;
  auto cfg = builder.initRoot<::PotentialConfig>();
  cfg.initMorse();
  const auto wire = rgpot::rpc::morseConfigFrom(cfg.asReader().getMorse());
  const rgpot::MorseConfig native{};
  REQUIRE(wire.De == native.De);
  REQUIRE(wire.a == native.a);
  REQUIRE(wire.re == native.re);
  REQUIRE(wire.cutoff == native.cutoff);
}

TEST_CASE("configure with morse on the Morse backend rebuilds the pot",
          "[morse][rpc]") {
  rgpot::MorsePot current;
  const double before = pairEnergyAtRe(current);

  capnp::MallocMessageBuilder builder;
  auto cfg = builder.initRoot<::PotentialConfig>();
  cfg.initMorse().setDe(2.0 * current.config().De);
  auto applied = rgpot::rpc::configureMorse(cfg.asReader(), current);

  REQUIRE(applied.ok);
  REQUIRE(applied.replacement != nullptr);
  const auto &next = applied.replacement->config();
  REQUIRE(next.De == 2.0 * current.config().De);
  REQUIRE(next.a == current.config().a);
  REQUIRE(next.re == current.config().re);
  REQUIRE(next.cutoff == current.config().cutoff);
  REQUIRE(applied.replacement->paramsKey() != current.paramsKey());

  // At r = re the shifted pair energy is -De d^2 with d = 1 - exp(-a (rc -
  // re)), linear in De.
  const double after = pairEnergyAtRe(*applied.replacement);
  const double d = 1.0 - std::exp(-next.a * (next.cutoff - next.re));
  REQUIRE_THAT(after, WithinRel(-next.De * d * d, 1e-12));
  REQUIRE_THAT(after, WithinRel(2.0 * before, 1e-12));
}

TEST_CASE("configure on the Morse backend keeps the switch width",
          "[morse][rpc]") {
  const rgpot::MorsePot current{rgpot::MorseConfig{.switch_width = 1.0}};
  capnp::MallocMessageBuilder builder;
  builder.initRoot<::PotentialConfig>().initMorse().setCutoff(7.0);
  auto applied = rgpot::rpc::configureMorse(
      builder.getRoot<::PotentialConfig>().asReader(), current);
  REQUIRE(applied.ok);
  REQUIRE(applied.replacement->config().switch_width == 1.0);
  REQUIRE(applied.replacement->config().cutoff == 7.0);
}

TEST_CASE("configure on the Morse backend refuses a cutoff inside the switch",
          "[morse][rpc]") {
  const rgpot::MorsePot current{rgpot::MorseConfig{.switch_width = 1.0}};
  capnp::MallocMessageBuilder builder;
  builder.initRoot<::PotentialConfig>().initMorse().setCutoff(0.5);
  auto applied = rgpot::rpc::configureMorse(
      builder.getRoot<::PotentialConfig>().asReader(), current);
  REQUIRE_FALSE(applied.ok);
  REQUIRE(applied.replacement == nullptr);
  REQUIRE_THAT(applied.message, ContainsSubstring("switch width"));
}

TEST_CASE("configure on the Morse backend refuses other arms", "[morse][rpc]") {
  const rgpot::MorsePot current;

  capnp::MallocMessageBuilder none_msg;
  none_msg.initRoot<::PotentialConfig>().setNone();
  auto none = rgpot::rpc::configureMorse(
      none_msg.getRoot<::PotentialConfig>().asReader(), current);
  REQUIRE(none.ok);
  REQUIRE(none.replacement == nullptr);

  capnp::MallocMessageBuilder bad_msg;
  bad_msg.initRoot<::PotentialConfig>().initMorse().setDe(
      std::numeric_limits<double>::quiet_NaN());
  auto bad = rgpot::rpc::configureMorse(
      bad_msg.getRoot<::PotentialConfig>().asReader(), current);
  REQUIRE_FALSE(bad.ok);
  REQUIRE(bad.replacement == nullptr);

  // Every arm the schema declares, so an arm added by a repin is covered
  // without editing this test.
  std::size_t refused = 0;
  for (auto field : capnp::Schema::from<::PotentialConfig>().getUnionFields()) {
    const std::string name = field.getProto().getName().cStr();
    if (name == "none" || name == "morse") {
      continue;
    }
    DYNAMIC_SECTION("arm " << name) {
      capnp::MallocMessageBuilder message;
      capnp::DynamicStruct::Builder dynamic =
          capnp::toDynamic(message.initRoot<::PotentialConfig>());
      if (field.getType().isVoid()) {
        dynamic.set(field, capnp::VOID);
      } else {
        dynamic.init(field);
      }
      const auto cfg = message.getRoot<::PotentialConfig>().asReader();
      REQUIRE(rgpot::rpc::potentialConfigArm(cfg) == name);
      auto applied = rgpot::rpc::configureMorse(cfg, current);
      REQUIRE_FALSE(applied.ok);
      REQUIRE(applied.replacement == nullptr);
      REQUIRE_THAT(applied.message,
                   ContainsSubstring("arm " + name + " not handled"));
    }
    ++refused;
  }
  // nwchem, cpmd, metatomic, lammps, xtb, tblite, uma, lj, zbl, d3, d4,
  // mopac, skala, expr and cp2k at the 190fee6 pin.
  REQUIRE(refused >= 15);
}

TEST_CASE("an arm from a newer schema is refused by its discriminant",
          "[morse][rpc]") {
  const auto schema = capnp::Schema::from<::PotentialConfig>();
  const std::uint16_t unknown =
      static_cast<std::uint16_t>(schema.getUnionFields().size() + 7);
  const auto offset = schema.getProto().getStruct().getDiscriminantOffset();

  capnp::MallocMessageBuilder message;
  message.initRoot<::PotentialConfig>();
  auto data = message.getRoot<capnp::AnyStruct>().getDataSection();
  REQUIRE(data.size() >= (offset + 1) * 2);
  data[offset * 2] = static_cast<kj::byte>(unknown & 0xff);
  data[offset * 2 + 1] = static_cast<kj::byte>(unknown >> 8);

  const auto cfg = message.getRoot<::PotentialConfig>().asReader();
  REQUIRE(static_cast<std::uint16_t>(cfg.which()) == unknown);
  const std::string arm = rgpot::rpc::potentialConfigArm(cfg);
  REQUIRE_THAT(arm,
               ContainsSubstring("discriminant " + std::to_string(unknown)));

  auto applied = rgpot::rpc::configureMorse(cfg, rgpot::MorsePot{});
  REQUIRE_FALSE(applied.ok);
  REQUIRE(applied.replacement == nullptr);
  REQUIRE_THAT(applied.message, ContainsSubstring(arm));
}

TEST_CASE("PotentialConfig.lj round-trips tailCorrection and rebuilds LJPot",
          "[lj][rpc]") {
  const auto field =
      capnp::Schema::from<::PotentialConfig>().getFieldByName("lj");
  REQUIRE(field.getProto().getOrdinal().getExplicit() == 9);

  capnp::MallocMessageBuilder default_builder;
  default_builder.initRoot<::PotentialConfig>().initLj();
  const auto default_wire = rgpot::rpc::ljConfigFrom(
      default_builder.getRoot<::PotentialConfig>().asReader().getLj());
  const rgpot::LJConfig native{};
  REQUIRE(default_wire.u0 == native.u0);
  REQUIRE(default_wire.psi == native.psi);
  REQUIRE(default_wire.cutoff == native.cutoff);
  REQUIRE(default_wire.tail_correction == native.tail_correction);

  const rgpot::LJPot current{rgpot::LJConfig{.switch_width = 1.0}};
  capnp::MallocMessageBuilder builder;
  auto lj = builder.initRoot<::PotentialConfig>().initLj();
  lj.setU0(0.5);
  lj.setPsi(2.0);
  lj.setCutoff(6.0);
  lj.setTailCorrection(true);
  const auto words = capnp::messageToFlatArray(builder);

  capnp::FlatArrayMessageReader reader(words.asPtr());
  const auto cfg = reader.getRoot<::PotentialConfig>();
  REQUIRE(cfg.which() == ::PotentialConfig::LJ);

  auto applied = rgpot::rpc::configureLJ(cfg, current);
  REQUIRE(applied.ok);
  REQUIRE(applied.replacement != nullptr);
  const auto &next = applied.replacement->config();
  CHECK(next.u0 == 0.5);
  CHECK(next.psi == 2.0);
  CHECK(next.cutoff == 6.0);
  CHECK(next.switch_width == 1.0);
  CHECK(next.tail_correction);
  CHECK(applied.replacement->paramsKey() != current.paramsKey());

  // Refuses cutoff <= switch_width when tailCorrection is enabled.
  capnp::MallocMessageBuilder bad_window;
  auto bad_lj = bad_window.initRoot<::PotentialConfig>().initLj();
  bad_lj.setU0(0.5);
  bad_lj.setPsi(2.0);
  bad_lj.setCutoff(1.0);
  bad_lj.setTailCorrection(true);
  auto refused_window = rgpot::rpc::configureLJ(
      bad_window.getRoot<::PotentialConfig>().asReader(), current);
  CHECK_FALSE(refused_window.ok);
  CHECK(refused_window.replacement == nullptr);
  CHECK_THAT(refused_window.message,
             ContainsSubstring("switch_width < cutoff"));

  // LJClusterPot accepts lj arm with tailCorrection=false, but refuses
  // tailCorrection=true.
  const rgpot::LJClusterPot cluster_current;
  capnp::MallocMessageBuilder cluster_ok_msg;
  auto cluster_lj = cluster_ok_msg.initRoot<::PotentialConfig>().initLj();
  cluster_lj.setU0(0.75);
  cluster_lj.setPsi(1.5);
  cluster_lj.setCutoff(7.0);
  cluster_lj.setTailCorrection(false);
  auto cluster_ok = rgpot::rpc::configureLJCluster(
      cluster_ok_msg.getRoot<::PotentialConfig>().asReader(), cluster_current);
  CHECK(cluster_ok.ok);
  REQUIRE(cluster_ok.replacement != nullptr);
  CHECK(cluster_ok.replacement->config().u0 == 0.75);

  auto cluster_refused = rgpot::rpc::configureLJCluster(cfg, cluster_current);
  CHECK_FALSE(cluster_refused.ok);
  CHECK(cluster_refused.replacement == nullptr);
  CHECK_THAT(cluster_refused.message,
             ContainsSubstring("does not support tail_correction"));
}
