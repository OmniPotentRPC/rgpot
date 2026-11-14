// MIT License
// Copyright 2023--present rgpot developers
//
// PotentialConfig.morse: the wire arm round-trips, its schema defaults are
// MorseConfig's, and configure on a Morse backend rebuilds the pot.

#include <array>
#include <cmath>
#include <limits>
#include <vector>

#include <capnp/message.h>
#include <capnp/schema.h>
#include <capnp/serialize.h>
#include <catch2/catch_all.hpp>

#include "rgpot/Morse/MorsePot.hpp"
#include "rgpot/rpc/MorseConfigure.hpp"
#include "rgpot/types/AtomMatrix.hpp"

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

TEST_CASE("PotentialConfig.morse round-trips at ordinal 6", "[morse][rpc]") {
  const auto field = capnp::Schema::from<::PotentialConfig>().getFieldByName("morse");
  REQUIRE(field.getProto().getOrdinal().getExplicit() == 6);

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

TEST_CASE("configure with morse on the Morse backend rebuilds the pot", "[morse][rpc]") {
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

  // At r = re the shifted pair energy is -De d^2 with d = 1 - exp(-a (rc - re)),
  // linear in De.
  const double after = pairEnergyAtRe(*applied.replacement);
  const double d = 1.0 - std::exp(-next.a * (next.cutoff - next.re));
  REQUIRE_THAT(after, WithinRel(-next.De * d * d, 1e-12));
  REQUIRE_THAT(after, WithinRel(2.0 * before, 1e-12));
}

TEST_CASE("configure on the Morse backend keeps the switch width", "[morse][rpc]") {
  const rgpot::MorsePot current{rgpot::MorseConfig{.switch_width = 1.0}};
  capnp::MallocMessageBuilder builder;
  builder.initRoot<::PotentialConfig>().initMorse().setCutoff(7.0);
  auto applied =
      rgpot::rpc::configureMorse(builder.getRoot<::PotentialConfig>().asReader(), current);
  REQUIRE(applied.ok);
  REQUIRE(applied.replacement->config().switch_width == 1.0);
  REQUIRE(applied.replacement->config().cutoff == 7.0);
}

TEST_CASE("configure on the Morse backend refuses other arms", "[morse][rpc]") {
  const rgpot::MorsePot current;

  capnp::MallocMessageBuilder none_msg;
  none_msg.initRoot<::PotentialConfig>().setNone();
  auto none = rgpot::rpc::configureMorse(none_msg.getRoot<::PotentialConfig>().asReader(),
                                         current);
  REQUIRE(none.ok);
  REQUIRE(none.replacement == nullptr);

  capnp::MallocMessageBuilder nwchem_msg;
  nwchem_msg.initRoot<::PotentialConfig>().initNwchem();
  auto nwchem = rgpot::rpc::configureMorse(
      nwchem_msg.getRoot<::PotentialConfig>().asReader(), current);
  REQUIRE_FALSE(nwchem.ok);
  REQUIRE(nwchem.replacement == nullptr);

  capnp::MallocMessageBuilder bad_msg;
  bad_msg.initRoot<::PotentialConfig>().initMorse().setDe(
      std::numeric_limits<double>::quiet_NaN());
  auto bad =
      rgpot::rpc::configureMorse(bad_msg.getRoot<::PotentialConfig>().asReader(), current);
  REQUIRE_FALSE(bad.ok);
  REQUIRE(bad.replacement == nullptr);
}
